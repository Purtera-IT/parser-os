"""Geographic fallback site extractor.

Some deals never name a street address or a facility ("ATL-HQ-01",
"Memorial Hospital") — the only locational anchor is a bare
``City, ST ZIP`` buried in a notes file. The Yonah deal is the canonical
case: ``location Santa Fe, NM 87506`` sits in Notes.pdf, no street
address anywhere, so the regular site detectors find nothing, zero
``physical_site`` atoms are emitted, ``site_readiness`` is empty, and the
brief goes RED with "no confirmed physical site" while the 15%
site-readiness score component sits at 0.

This module is a *fallback*: it scans atoms for ``City, ST ZIP`` anchors
when the deal lacks sufficient structured site coverage, and emits
low-confidence ``physical_site`` atoms (flagged ``geo_fallback_site``,
``needs_review``) per distinct address so the deal has locational anchors
the PM can confirm — instead of a blank RED. Pure function, no I/O, no
LLM.
"""

from __future__ import annotations

import os
import re
from typing import Any

from app.core.address_parse import (
    _CITY_STATE_ZIP_RE,
    US_STATE_NAMES,
    US_STATES,
    US_STATES as _US_STATES,
    find_us_addresses_in_text,
    normalized_address_key,
)
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

_MAX_FALLBACK_SITES = 8

#: "Highland Park, MI" / "Highland Park, Michigan": a capitalised run, a comma, a state.
_CITY_STATE_MENTION_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s*,\s*([A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _is_roster_sourced(atom: Any) -> bool:
    """True when the atom came from a site-roster table extractor.

    A row lifted from a declared site-roster TABLE anchors the deal whether or
    not it carries an id column. Without this, a deal whose roster ships no ID
    column gets guessed "City, ST ZIP" sites minted on top of hundreds of real,
    fully-addressed ones.
    """
    for ref in (getattr(atom, "source_refs", None) or []):
        if "site_roster" in str(getattr(ref, "extraction_method", "") or ""):
            return True
        loc = getattr(ref, "locator", None)
        if isinstance(loc, dict) and "site_roster" in str(loc.get("extraction", "")):
            return True
    return False


def _site_location_score(val: dict[str, Any]) -> int:
    """Higher = more structured location (0 = name/id only)."""
    if not isinstance(val, dict):
        return 0
    street = str(val.get("street_address") or val.get("address") or "").strip()
    city = str(val.get("city") or "").strip()
    state = str(val.get("state") or "").strip().upper()
    zipc = str(val.get("zip") or "").strip()
    if street and city and state:
        return 3
    if city and state and zipc:
        return 2
    if city and state:
        return 2
    if val.get("site_id") or val.get("id"):
        return 1
    return 0


def _atom_site_score(atom: Any) -> int:
    """Location score for an atom, crediting its provenance.

    A roster-sourced row is a declared site, so it scores as well-structured
    even when its cells are sparse -- the document said the table lists sites.
    Scoring it on cells alone let a roster with no ID column read as a weak
    ghost, which is the opposite of the truth.
    """
    score = _site_location_score(getattr(atom, "value", None) or {})
    if _is_roster_sourced(atom):
        return max(score, 2)
    return score


def _physical_site_atoms(atoms: list[Any]) -> list[Any]:
    return [a for a in atoms if _atom_type_str(a) == "physical_site"]


def _should_skip_geo_fallback(atoms: list[Any]) -> bool:
    """Skip only when the deal already has multiple well-structured sites.

    A single weak ``physical_site`` (name-only / misparsed geo) must NOT block
    discovering additional addresses -- the MBrany failure mode. This is why a
    binary "has any real site" test is wrong here: one id-bearing ghost would
    suppress every genuine address on the deal.
    """
    sites = _physical_site_atoms(atoms)
    if not sites:
        return False
    scores = [_atom_site_score(a) for a in sites]
    high = sum(1 for s in scores if s >= 2)
    # Two well-structured sites -- skip fallback. One high + one weak ghost
    # (typed_atom under full ML) must not block geo inference (MBrany class).
    if high >= 2:
        return True
    if len(sites) >= 3 and high >= 1:
        return True
    return False


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ── Vendor / letterhead address suppression ─────────────────────────
#
# A street address in a deal is not automatically a job site. The service
# provider's own letterhead / billing address ("PurTera LLC, 11720 Amber
# Park Dr, Alpharetta GA 30009") sits in the SOW header and gets minted as a
# phantom ``physical_site`` — a job site that does not exist. A keyword list
# of vendor names can never be universal; the role of an address is a
# semantic question, so we ask a small local LLM. When the LLM is
# unreachable the gate is a NO-OP (we never drop a site on a guess).
_SITE_ROLE_CANDIDATES = ["job_site", "vendor_or_billing_address"]
# Deliberately NEUTRAL wording: it describes both roles even-handedly and lets
# the model reason from the address's own context. An instruction that *asserts*
# "a company name next to an address is letterhead" primes a small model to
# answer that way for every address (including the real job site). Tested:
# qwen2.5:3b cannot discriminate here (it parrots the prompt's emphasis);
# qwen3:14b does, stably — so this gate routes to the larger model. It is one
# call per site (a handful per deal), not the per-atom enrichment bottleneck.
_SITE_ROLE_INSTRUCTION = (
    "Classify the ROLE of this address within the deal. A job_site is a "
    "customer location where physical installation / field work is performed. "
    "A vendor_or_billing_address is the service provider's own corporate "
    "office, letterhead, or billing address (not a work location)."
)
# A small 3B model cannot make this discrimination reliably; route to the same
# capable model the rest of the pipeline uses. Overridable for ops.
_SITE_ROLE_MODEL = os.environ.get("OLLAMA_SITE_ROLE_MODEL", "qwen3:14b")
_VENDOR_DROP_CONFIDENCE = 0.6


def _stamp_decision(atom: Any, decision: Any) -> None:
    """Record WHY a site was demoted, on the atom itself (provenance, invariant
    I). Captures which tier decided (``store``/``llm``) and, when a learned
    correction drove it, that correction's id — so a PM can trace a suppression
    back to the rule that caused it, with no keyword list involved. Best-effort:
    only stamps when ``value`` is a dict, never raises."""
    try:
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            val["_decision"] = {
                "source": getattr(decision, "source", None),
                "correction_id": getattr(decision, "correction_id", None),
                "confidence": round(float(getattr(decision, "confidence", 0.0)), 3),
            }
    except Exception:  # pragma: no cover - provenance must never break a compile
        pass


def _site_address_text(atom: Any) -> tuple[str, str]:
    """Return ``(address, context)`` for a physical_site atom.

    The discriminating signal for a vendor/letterhead address (a company name
    and footer code printed next to the address) usually lives in the *source*
    text the address was lifted from, not in the terse minted site name. When a
    geo-fallback atom preserved that originating text in ``source_context``,
    hand it to the classifier so the model can see the letterhead.
    """
    val = getattr(atom, "value", None) or {}
    text = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    addr = ""
    src_ctx = ""
    if isinstance(val, dict):
        addr = str(val.get("address") or val.get("street_address") or "")
        src_ctx = str(val.get("source_context") or "")
    # Classify the richest available representation. A geo-fallback site is
    # minted from a bare "City, ST ZIP" — its own text drops the street number
    # and company name that actually mark a letterhead, so the originating
    # line (source_context) is the strongest signal and must be what the model
    # judges. A real site atom with a structured street address uses that.
    primary = addr or src_ctx or str(text)
    context = src_ctx or str(text)
    return (primary, context)


def _is_roster_site(atom: Any) -> bool:
    """Did this site come out of a locations table (site roster row)?"""
    for ref in (getattr(atom, "source_refs", None) or []):
        loc = getattr(ref, "locator", None)
        if isinstance(loc, dict) and str(loc.get("extraction") or "").startswith("site_roster"):
            return True
    v = getattr(atom, "value", None)
    if isinstance(v, dict):
        if str(v.get("kind") or "") == "physical_site" and v.get("site_id") and v.get("facility_name") and not v.get("inferred"):
            return True
    return False


def suppress_vendor_sites(
    atoms: list[Any], *, project_id: str
) -> tuple[list[Any], int]:
    """Drop ``physical_site`` atoms whose address is the vendor's own
    office / letterhead / billing address rather than a job site.

    Deterministic PurTera corporate-address ban runs first and applies even
    when the banned address is the deal's only site (PurTera HQ is never a
    job site). Semantic LLM suppression runs afterward. Safe by construction —
    the LLM path returns the atoms unchanged when:

    * the LLM is disabled / unreachable (classify_role yields ``None``), or
    * fewer than two physical_site atoms exist (never remove the deal's only
      locational anchor — except known PurTera vendor addresses above), or
    * suppression would remove *every* site (always keep at least one).
    """
    from app.core.vendor_site_ban import drop_banned_vendor_physical_sites

    atoms, det_dropped = drop_banned_vendor_physical_sites(atoms)

    # A signature-page mailing address is a party's, decided by SHAPE before
    # any model sees the sites. Otherwise the party address counts as the
    # second site, the LLM is asked to pick the vendor among two, and it can
    # pick the customer's HQ (local 010300: "2970 Brandywine Rd, Atlanta"
    # judged vendor_or_billing_address, the CDW mailing address kept, then
    # vetoed -- zero sites).
    try:
        from app.core.party_address_veto import veto_party_page_sites

        veto_party_page_sites(atoms)
    except Exception:
        pass

    # Route the address-role judgment through the universal decide() chokepoint.
    # Phase 2: the feedback store is not yet wired, so decide() is a transparent
    # pass-through to semantic_role.classify_role (same model, same instruction,
    # same result). Phase 3 seeds the global PurTera "selling-party address is
    # not a job site" correction HERE, and it then resolves from the store with
    # zero LLM cost — without this call site changing again.
    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover - defensive
        return atoms, 0

    sites = [a for a in atoms if _atom_type_str(a) == "physical_site"]
    if len(sites) < 2:
        return atoms, det_dropped

    scope = DecisionScope(deal_id=project_id or "")
    drop_ids: set[str] = set()
    # PERF: vendor-suppression exists to catch the vendor's OWN address (usually
    # 1-2 letterhead/signature addresses) leaking in as a job site. On a deal with
    # thousands of real customer sites, running one LLM call PER site is the
    # "million years" cost — and a single vendor address among thousands is
    # negligible noise anyway. So: a CHEAP store-only check runs on EVERY site
    # (instant; as the store learns vendor addresses it catches them for free),
    # and the LLM fallback is bounded to a budget. Small deals (<budget sites) are
    # unchanged; huge site rosters stay fast and complete.
    import os as _os
    try:
        llm_budget = max(0, int(_os.environ.get("SOWSMITH_VENDOR_SUPPRESS_LLM_MAX", "60")))
    except Exception:
        llm_budget = 60
    for a in sites:
        aid = getattr(a, "id", None)
        if not aid:
            continue
        # A row of a locations table ("Customer-Designated Locations", a site
        # roster) is a job site by construction -- the document listed it as a
        # place where work happens. No model judgment overrides that shape.
        if _is_roster_site(a):
            continue
        addr, context = _site_address_text(a)
        if not addr:
            continue
        # 1) store-only (no LLM): instant, free; warms over time.
        decision = decide(
            "physical_site", addr, _SITE_ROLE_CANDIDATES,
            instruction=_SITE_ROLE_INSTRUCTION, context=context,
            scope=scope, model=_SITE_ROLE_MODEL, llm=False,
        )
        # 2) bounded LLM fallback only when the store abstained AND budget remains.
        if decision.verdict is None and llm_budget > 0:
            decision = decide(
                "physical_site", addr, _SITE_ROLE_CANDIDATES,
                instruction=_SITE_ROLE_INSTRUCTION, context=context,
                scope=scope, model=_SITE_ROLE_MODEL,
            )
            llm_budget -= 1
        if (
            decision is not None
            and decision.verdict == "vendor_or_billing_address"
            and decision.confidence >= _VENDOR_DROP_CONFIDENCE
        ):
            drop_ids.add(aid)
            _stamp_decision(a, decision)

    if not drop_ids:
        return atoms, det_dropped
    # Never strip the deal down to zero sites.
    if len(drop_ids) >= len(sites):
        return atoms, det_dropped

    kept = [a for a in atoms if getattr(a, "id", None) not in drop_ids]
    return kept, det_dropped + len(drop_ids)


def _existing_address_keys(atoms: list[Any]) -> set[str]:
    keys: set[str] = set()
    for a in _physical_site_atoms(atoms):
        val = getattr(a, "value", None) or {}
        if isinstance(val, dict):
            k = normalized_address_key(val)
            if k:
                keys.add(k)
    return keys


def enrich_site_geo(atoms: list[Any]) -> int:
    """Fill missing ``city``/``state``/``zip`` on real ``physical_site`` atoms.

    ``geo_fallback_sites`` below is all-or-nothing: it mints sites only when
    the deal has none. That leaves the common middle case unserved — a site
    that IS detected but whose address arrived as one lumped string, because
    the summary table the customer wrote it in was terse::

        | AUG-DC-06 | Augusta Data Center Annex (699 Broad St, Ste 1200) | ...

    The full ``Augusta, GA 30901`` is two rows further down the same document,
    in the access-window table. Everything downstream that reasons about
    *where* a site is — site_readiness, mapping, dispatch planning — keys on
    city/state/ZIP, so a site with an address string and no city is a site
    nobody can route to.

    Runs in two passes per site: its own address text first, then any atom
    that names exactly one site (a paragraph listing three sites says nothing
    about which address belongs to which). Only ever fills blanks; never
    overwrites what an extractor already established. Pure function, no I/O,
    no LLM — mutates ``value`` in place and returns how many sites it filled.
    """
    sites = [a for a in atoms if _atom_type_str(a) == "physical_site"]
    if not sites:
        return 0

    def _fill(atom: Any, text: str) -> bool:
        val = getattr(atom, "value", None)
        if not isinstance(val, dict) or not text:
            return False
        if val.get("city") and val.get("state") and val.get("zip"):
            return False
        m = _CITY_STATE_ZIP_RE.search(str(text))
        if not m:
            return False
        city, state, zipc = m.group(1).strip(), m.group(2).upper(), m.group(3)
        if state not in _US_STATES:
            return False
        before = (val.get("city"), val.get("state"), val.get("zip"))
        val.setdefault("city", city)
        val.setdefault("state", state)
        val.setdefault("zip", zipc)
        return (val.get("city"), val.get("state"), val.get("zip")) != before

    filled = 0
    # Pass 1 — the site's own address/text.
    for s in sites:
        addr, ctx = _site_address_text(s)
        if _fill(s, addr) or _fill(s, ctx):
            filled += 1

    # Pass 2 — a sibling atom that names this site and nothing else.
    wanting = [s for s in sites
               if isinstance(getattr(s, "value", None), dict)
               and not (s.value.get("city") and s.value.get("state"))]
    if not wanting:
        return filled
    keyed: list[tuple[str, Any]] = []
    for s in wanting:
        val = s.value
        for alias in (val.get("site_id"), val.get("id"), val.get("name"),
                      val.get("facility_name")):
            if alias and len(str(alias)) >= 4:
                keyed.append((_slug(str(alias)), s))
    for atom in atoms:
        text = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
        if not text:
            continue
        hay = _slug(str(text))
        matched = {id(s): s for k, s in keyed if k and k in hay}
        if len(matched) != 1:
            continue
        if _fill(next(iter(matched.values())), text):
            filled += 1

    # Pass 2b — a "City, ST" / "City, State" mention anywhere in the deal whose
    # city the site's own name already carries. Live 000061: the site was named
    # "highland park warehouse office" and the email said "Highland Park, MI",
    # but with no ZIP neither pass above could place it, so the Deal Kit got a
    # site with no city or state. The place must exist in the reference data
    # for that state, and the site must be named by exactly one such place.
    wanting = [s for s in sites
               if isinstance(getattr(s, "value", None), dict)
               and not (s.value.get("city") and s.value.get("state"))]
    if wanting:
        from app.core.geo_reference import is_known_place as _known_place
        from app.core.address_parse import state_code as _state_code

        wanting_aliases = [
            (_slug(str(alias)), id(s))
            for s in wanting
            for alias in (s.value.get("site_id"), s.value.get("id"), s.value.get("name"),
                          s.value.get("facility_name"))
            if alias and len(str(alias)) >= 4
        ]
        mentions: set[tuple[str, str]] = set()
        for atom in atoms:
            text = str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")
            # Same rule as pass 2: a line that names two of the sites says
            # nothing about which one the place belongs to.
            hay = _slug(text)
            if len({sid for alias, sid in wanting_aliases if alias and alias in hay}) >= 2:
                continue
            for m in _CITY_STATE_MENTION_RE.finditer(text):
                st = _state_code(m.group(2))
                if not st:
                    continue
                usps_code = m.group(2).strip().isupper() and len(m.group(2).strip()) == 2
                words = m.group(1).split()
                # The capitalised run before the comma can carry lead-in words
                # ("Office In Highland Park"); the place is its longest real tail.
                for k in range(len(words)):
                    cand = " ".join(words[k:])
                    known = _known_place(cand, st)
                    # None means the reference could not be read (an installed
                    # package without its data file). That must not reject a
                    # place the author wrote with a USPS state code; a spelled
                    # out state ("Michigan") is only trusted when checked.
                    if known or (known is None and usps_code and k == 0):
                        mentions.add((cand, st))
                        break
        # What this pass saw, in the compile log: which sites wanted a place,
        # which "City, ST" mentions the deal carried, and whether the gazetteer
        # answered. 000061 (2026-09-15) placed locally and not on the worker,
        # and nothing said why.
        import logging as _logging
        _logging.getLogger(__name__).info(
            "site_geo mention pass: %d site(s) wanting a place %s; mentions %s; gazetteer %s",
            len(wanting),
            [str(s.value.get("name") or s.value.get("site_id") or "")[:40] for s in wanting][:6],
            sorted(mentions)[:6],
            "available" if _known_place("Springfield", "IL") is not None else "unavailable",
        )
        for s in wanting:
            val = s.value
            names = [_slug(str(a)) for a in (val.get("name"), val.get("facility_name"),
                                              val.get("display_name"), val.get("site_id"))
                     if a]
            names += [_slug(str(a)) for a in (val.get("names") or []) if a]
            hits = {(c, st) for c, st in mentions
                    if _slug(c) and any(re.search(rf"(?:^|_){re.escape(_slug(c))}(?:_|$)", n) for n in names)}
            places = {(_slug(c), st) for c, st in hits}
            if len(places) != 1:
                continue
            city, st = sorted(hits)[0]
            before = (val.get("city"), val.get("state"))
            if not val.get("city"):
                val["city"] = city
            if not val.get("state"):
                val["state"] = st
            if (val.get("city"), val.get("state")) != before:
                filled += 1

    # Pass 3 — the reference data closes what the text never stated.
    #
    # This runs over EVERY physical_site atom regardless of which extractor
    # made it, and that is the point: the roster extractor enriches its own
    # rows, but the docx schema registry (``table_schema_v49``) does not, so a
    # site landed with city "Oregon City" and ZIP 97045 and no state — while
    # those five digits name Oregon outright. Enriching per-extractor means
    # doing it again for the next extractor; this is the one choke point they
    # all pass through.
    #
    # Fills blanks only, and only from a ZIP or from a city that exists in
    # exactly one state. Never overwrites what a document said.
    from app.core.geo_reference import resolve as _geo_resolve

    for site in sites:
        val = getattr(site, "value", None)
        if not isinstance(val, dict):
            continue
        if val.get("city") and val.get("state"):
            continue
        ref_city, ref_state = _geo_resolve(
            city=val.get("city"), state=val.get("state"), postal_code=val.get("zip")
        )
        changed = False
        if ref_city and not val.get("city"):
            val["city"] = ref_city
            changed = True
        if ref_state and not val.get("state"):
            val["state"] = ref_state
            changed = True
        if changed:
            filled += 1

    return filled


def geo_fallback_sites(
    atoms: list[Any], *, project_id: str
) -> list[EvidenceAtom]:
    """Emit fallback ``physical_site`` atoms from address anchors in atom text.

    Returns an empty list when site coverage is already sufficient or when no
    valid geographic anchor is found.
    """
    if not atoms or _should_skip_geo_fallback(atoms):
        return []

    seen_keys = _existing_address_keys(atoms)
    out: list[EvidenceAtom] = []
    for atom in atoms:
        text = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
        if not text:
            continue
        text_s = str(text)
        for parsed_item in find_us_addresses_in_text(text_s):
            if not parsed_item.city or not parsed_item.state or parsed_item.state not in US_STATES:
                continue
            dedup_fields = {
                "street_address": parsed_item.street_address,
                "city": parsed_item.city,
                "state": parsed_item.state,
                "zip": parsed_item.zip,
            }
            addr_key = normalized_address_key(dedup_fields)
            if not addr_key or addr_key in seen_keys:
                continue
            from app.core.vendor_site_ban import is_purtera_vendor_address

            if is_purtera_vendor_address(text=text_s):
                continue
            seen_keys.add(addr_key)

            city, state, zipc = parsed_item.city, parsed_item.state, parsed_item.zip or ""
            slug = _slug(f"{city}_{state}_{zipc or parsed_item.street_address or 'site'}")
            name = (
                f"{parsed_item.street_address}, {city}, {state} {zipc}".strip(", ")
                if parsed_item.street_address
                else f"{city}, {state} {zipc}".strip()
            )
            artifact_id = getattr(atom, "artifact_id", "") or ""
            atom_id = stable_id("atm", artifact_id, "physical_site", slug)
            src_refs = list(getattr(atom, "source_refs", None) or [])
            if not src_refs:
                src_refs = [
                    SourceRef(
                        id=stable_id("src", atom_id),
                        artifact_id=artifact_id,
                        artifact_type=ArtifactType.txt,
                        filename=getattr(atom, "artifact_id", "") or "geo_fallback",
                        locator={"extraction": "site_geo_fallback"},
                        extraction_method="site_geo_fallback",
                        parser_version="site_geo_fallback_v2",
                    )
                ]
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.physical_site,
                    raw_text=name,
                    normalized_text=name.lower(),
                    value={
                        "kind": "physical_site",
                        "id": slug,
                        "site_id": slug,
                        "name": name,
                        "names": [name, city],
                        "street_address": parsed_item.street_address,
                        "address": parsed_item.street_address,
                        "city": city,
                        "state": state,
                        "zip": zipc or None,
                        "inferred": True,
                        "source_context": text_s[:600],
                    },
                    entity_keys=[f"site:{slug}"],
                    source_refs=src_refs,
                    receipts=[],
                    authority_class=AuthorityClass.machine_extractor,
                    confidence=0.5,
                    confidence_raw=0.5,
                    calibrated_confidence=0.5,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["geo_fallback_site"],
                    parser_version="site_geo_fallback_v2",
                )
            )
            if len(out) >= _MAX_FALLBACK_SITES:
                return out
    return out


# ── A place named in the documents is a site candidate ──────────────────────
#
# The extractors above mint sites from street addresses, site codes and
# "X Building"-shaped names, and the ZIP fallback from "City, ST ZIP". A deal
# whose sites are named only as "El Segundo, CA" in a list, "the Huntsville,
# Alabama facility" in prose, or a Location column in a spreadsheet got none of
# them (2026-09-15, untaught pool: 010283 named nine cities and got two,
# 010270 named its one facility four times and got none, 010307 listed
# thirteen locations in an inventory sheet and got none).
#
# Whether a named place is a job site is a judgment the lines around it
# settle -- "Locations / Cities / El Segundo, CA" is a site list; "Heading to
# Dallas, Texas" is travel; a signature's "Austin, TX" is a party's address.
# So each candidate goes through decide() on relation `geo_mention_role`
# (store first: what PMs and finished kits taught, deal then global; the
# model only when the store abstains), and a confident job_site is minted as
# an inferred, needs_review physical_site the PM can confirm.

_MENTION_RELATION = "geo_mention_role"
_MENTION_CANDIDATES = ["job_site", "mention_only"]
_MENTION_INSTRUCTION = (
    "A place named in a deal's documents, shown with the lines around it. Decide whether it is a "
    "job site -- somewhere our technicians will work on this deal: a location on a list of sites, "
    "the facility or office the work is for, the origin or destination equipment moves between -- "
    "or only a mention: where someone is travelling, a party's address in a signature or "
    "letterhead, a reference customer's city, a vendor's or reseller's office, a region named in "
    "passing, or a place the documents say is out of scope, covered by someone else, or not part of "
    "this work. If the lines do not say, answer unknown."
)
#: An inferred site is flagged needs_review; 0.8 from the model is enough to
#: put it in front of the PM (the ZIP fallback mints at 0.5 with no judge).
_MENTION_MIN_CONF = 0.8
_MENTION_CONTEXT_NEIGHBOURS = 3
#: A document naming this many places is judged once, as a list.
_LIST_MIN = 3
_LIST_INSTRUCTION = (
    "A deal's document names several places, listed here with the lines around them. Decide "
    "whether these are job sites -- a list of locations where our technicians will work on this "
    "deal (a site list, a location column, the facilities in scope) -- or only mentions: cities "
    "in a travel story, reference customers, a vendor's offices, regions named in passing. If the "
    "list mixes both or the lines do not say, answer unknown."
)
#: "5200 Lankershim Blvd Ste 200 North Hollywood, CA": a street address with its
#: city and state but no ZIP (the address parser wants the ZIP).
_STREET_CITY_STATE_RE = re.compile(
    r"\b(\d{1,6}\s+[A-Z][A-Za-z0-9.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,4}?\s+"
    r"(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Pkwy|Parkway|Hwy|Highway|Way|Ln|Lane|Ct|Court|Pl|Place|Trl|Trail|Cir|Circle)\.?"
    r"(?:\s+(?:Ste|Suite|Unit|Bldg|Building|Fl|Floor|#)\.?\s*[A-Za-z0-9\-]+)?)"
    r"\s*,?\s+([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s*,\s*([A-Z]{2})\b"
)

#: "Los Angeles CA" / "Kent WA": a capitalised run and a state code with no comma.
#: Only trusted when the gazetteer knows the place -- "Meraki MS" is a switch.
_CITY_STATE_NOCOMMA_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s+([A-Z]{2})\b(?![\-\d])"
)
#: Atom kinds whose text names a party, not a place of work.
_MENTION_SKIP_TYPES = frozenset({"stakeholder", "physical_site", "deal_metadata"})


def _mention_max() -> int:
    try:
        return max(0, int(os.environ.get("SOWSMITH_GEO_MENTION_MAX", "40")))
    except Exception:
        return 40


def _mention_llm_budget() -> int:
    try:
        return max(0, int(os.environ.get("SOWSMITH_GEO_MENTION_LLM_MAX", "40")))
    except Exception:
        return 40


def _state_code(token: str) -> str | None:
    t = str(token or "").strip()
    if len(t) == 2:
        return t.upper() if t.upper() in _US_STATES else None
    return US_STATE_NAMES.get(t.lower())


def _artifact_of(atom: Any) -> str:
    v = getattr(atom, "artifact_id", None) or getattr(atom, "source_artifact_id", None)
    if v:
        return str(v)
    for ref in getattr(atom, "source_refs", None) or []:
        a = getattr(ref, "artifact_id", None) or (ref.get("artifact_id") if isinstance(ref, dict) else None)
        if a:
            return str(a)
    return ""


def _text_of(atom: Any) -> str:
    return " ".join(str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "").split())


def _mention_candidates(atoms: list[Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Distinct places named in the documents that no site carries yet.

    Key (city, state) -- or (street address, state) for an address whose city
    already has a site, since a move has an origin and a destination in one
    town. Value: city, state, street, and every mention (atom index, as written)."""
    try:
        from app.core.geo_reference import is_known_place
    except Exception:  # pragma: no cover
        def is_known_place(city, state=None):  # type: ignore[misc]
            return None
    existing: set[tuple[str, str]] = set()
    for a in _physical_site_atoms(atoms):
        v = getattr(a, "value", None) or {}
        if isinstance(v, dict):
            c, st = str(v.get("city") or "").strip().lower(), str(v.get("state") or "").strip().upper()
            if c:
                existing.add((c, st))
            for nm in [v.get("name"), v.get("facility_name"), v.get("site_id")] + list(v.get("names") or []):
                if nm:
                    existing.add((str(nm).strip().lower(), ""))
    def _street_key(street: str) -> str:
        m = re.match(r"\s*(\d{1,6})\s+([A-Za-z0-9.'\-]+)", str(street or ""))
        return f"{m.group(1)}{m.group(2).lower()}" if m else ""

    existing_streets = set()
    for a in _physical_site_atoms(atoms):
        v = getattr(a, "value", None) or {}
        if isinstance(v, dict):
            for field in ("street_address", "address", "name", "source_context"):
                for m in re.finditer(r"\d{1,6}\s+[A-Za-z0-9.'\-]+", str(v.get(field) or "")):
                    existing_streets.add(_street_key(m.group(0)))
            for nm in v.get("names") or []:
                existing_streets.add(_street_key(str(nm)))
        for m in re.finditer(r"\d{1,6}\s+[A-Za-z0-9.'\-]+", _text_of(a)):
            existing_streets.add(_street_key(m.group(0)))
    existing_streets.discard("")
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def _offer(i: int, city: str, state: str, written: str, street: str = "") -> None:
        label = (f"{city}, {state}" if state else city).lower()
        key = (street.lower(), state) if street else (city.lower(), state)
        if not street and ((city.lower(), state) in existing or (city.lower(), "") in existing or (label, "") in existing):
            return
        if street and _street_key(street) in existing_streets:
            return
        entry = found.setdefault(key, {"city": city, "state": state, "street": street, "mentions": []})
        if len(entry["mentions"]) < 6:
            entry["mentions"].append((i, written))

    for i, atom in enumerate(atoms):
        kind = _atom_type_str(atom)
        if kind in _MENTION_SKIP_TYPES:
            continue
        text = _text_of(atom)
        if not text:
            continue
        val = getattr(atom, "value", None) or {}
        # A spreadsheet column the parser already typed as a location.
        if kind == "entity" and isinstance(val, dict) and str(val.get("entity_type") or "") == "location":
            name = str(val.get("name") or "").strip()
            if name and not name.isdigit():
                _offer(i, name, "", text)
            continue
        # A street address in a city that already has a site is a second site
        # there (010294: origin 5161 Lankershim, destination 5200 Lankershim).
        try:
            for parsed in find_us_addresses_in_text(text):
                if parsed.street_address and parsed.city and parsed.state in _US_STATES:
                    _offer(i, parsed.city, parsed.state, f"{parsed.street_address}, {parsed.city}, {parsed.state}", street=parsed.street_address)
        except Exception:
            pass
        for m in _STREET_CITY_STATE_RE.finditer(text):
            street, city, st = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            state = _state_code(st)
            if not state or is_known_place(city, state) is False:
                continue
            _offer(i, city, state, m.group(0), street=street)
        for m in _CITY_STATE_MENTION_RE.finditer(text):
            city, st = m.group(1).strip(), m.group(2).strip()
            state = _state_code(st)
            if not state:
                continue
            known = is_known_place(city, state)
            if known is False or (known is None and len(st) != 2):
                continue
            _offer(i, city, state, m.group(0))
        for m in _CITY_STATE_NOCOMMA_RE.finditer(text):
            city, st = m.group(1).strip(), m.group(2).strip()
            state = _state_code(st)
            if not state or is_known_place(city, state) is not True:
                continue
            _offer(i, city, state, m.group(0))
    return found


def _mention_context(atoms: list[Any], i: int, *, mentions: list[tuple[int, str]] | None = None,
                     siblings: list[str] | None = None) -> str:
    """What the judge reads: the document, every line that names the place,
    the other places the same document names (a list of nine cities is a site
    list), and the lines around the first mention (a heading, a row)."""
    here = atoms[i]
    art = _artifact_of(here)
    fn = ""
    for ref in getattr(here, "source_refs", None) or []:
        fn = str(getattr(ref, "filename", None) or (ref.get("filename") if isinstance(ref, dict) else "") or "")
        if fn:
            break
    fn = fn or str(getattr(here, "source_filename", "") or "")
    lines: list[str] = []
    lo, hi = max(0, i - _MENTION_CONTEXT_NEIGHBOURS), min(len(atoms), i + _MENTION_CONTEXT_NEIGHBOURS + 1)
    for j in range(lo, hi):
        a = atoms[j]
        if j != i and art and _artifact_of(a) != art:
            continue
        t = _text_of(a)
        if t:
            lines.append(("> " if j == i else "  ") + t[:200])
    head = f"document: {fn[:100]}\n" if fn else ""
    # The table a row came from: its column header says what the rows are.
    cols = None
    for j in range(max(0, i - 40), min(len(atoms), i + 40)):
        a = atoms[j]
        if art and _artifact_of(a) != art:
            continue
        v = getattr(a, "value", None) or {}
        if isinstance(v, dict) and isinstance(v.get("_columns"), list) and v["_columns"]:
            cols = [str(c) for c in v["_columns"] if str(c).strip()][:10]
            break
    if cols:
        head += "table columns: " + " | ".join(cols) + "\n"
    # The document's headings and the lines that speak of sites, wherever they sit.
    doc_lines = [_text_of(a) for a in atoms if art and _artifact_of(a) == art]
    headings = [t for t in doc_lines if t and len(t.split()) <= 4 and not _CITY_STATE_MENTION_RE.search(t)][:8]
    about = [t for t in doc_lines if re.search(r"\b(site|sites|location|locations|facility|facilities)\b", t, re.I) and len(t) > 20][:4]
    if headings:
        head += "headings in this document: " + " / ".join(h[:40] for h in headings) + "\n"
    if about:
        head += "the document says: " + " | ".join(t[:160] for t in about) + "\n"
    if siblings:
        head += f"other places this document names: {'; '.join(siblings[:12])}\n"
    if mentions and len(mentions) > 1:
        head += f"named {len(mentions)} times, e.g.:\n" + "\n".join(
            "  * " + _text_of(atoms[j])[:200] for j, _w in mentions[1:4] if 0 <= j < len(atoms)) + "\n"
    return head + "lines around it:\n" + "\n".join(lines)


def geo_mention_sites(atoms: list[Any], *, project_id: str, trace: list[dict[str, Any]] | None = None) -> list[EvidenceAtom]:
    """Mint inferred ``physical_site`` atoms for places the documents name
    that no site carries yet, when the store or the model calls them job
    sites. Never raises; returns the new atoms (not yet appended)."""
    if not atoms:
        return []
    cands = _mention_candidates(atoms)
    if not cands:
        return []
    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover
        return []
    scope = DecisionScope(deal_id=str(project_id or ""))
    llm_budget = _mention_llm_budget()
    cap = _mention_max()
    out: list[EvidenceAtom] = []
    # Which places each document names, so a list reads as a list.
    by_doc: dict[str, list[str]] = {}
    for entry in cands.values():
        for j, _w in entry["mentions"]:
            lab = f"{entry['city']}, {entry['state']}" if entry["state"] else entry["city"]
            lst = by_doc.setdefault(_artifact_of(atoms[j]), [])
            if lab not in lst:
                lst.append(lab)
    def _ask(label: str, instruction: str, context: str):
        nonlocal llm_budget
        d = decide(_MENTION_RELATION, label, list(_MENTION_CANDIDATES), instruction=instruction,
                   context=context[:1200], scope=scope, exclude_created_by=("teacher",), llm=False)
        if getattr(d, "verdict", None) is None and llm_budget > 0:
            llm_budget -= 1
            d = decide(_MENTION_RELATION, label, list(_MENTION_CANDIDATES), instruction=instruction,
                       context=context[:1200], scope=scope, exclude_created_by=("teacher",))
        return d

    # A document that names several places is judged once, as the list it
    # is: nine cities under "Locations / Cities" are a site list, and twelve
    # rows of a Location column are the sites the inventory covers. Judged one
    # by one the same list came back job_site 0.80, abstain, job_site 0.80 ...
    def _list_shaped(doc: str) -> bool:
        """Places on their own short lines, or in rows of a table: a list.
        Three places inside the sentences of a recap are three stories."""
        short = total = 0
        for entry in cands.values():
            for j, _w in entry["mentions"]:
                if _artifact_of(atoms[j]) != doc:
                    continue
                total += 1
                a = atoms[j]
                kind = _atom_type_str(a)
                v = getattr(a, "value", None) or {}
                if kind in ("entity", "raw_table_row") or (isinstance(v, dict) and "_columns" in v) or len(_text_of(a).split()) <= 6:
                    short += 1
        return total > 0 and short * 3 >= total * 2

    list_verdict: dict[str, tuple[str | None, float, str, str | None]] = {}
    for doc, labels in by_doc.items():
        if len(labels) < _LIST_MIN or not _list_shaped(doc):
            continue
        first = next((e for e in cands.values() if _artifact_of(atoms[e["mentions"][0][0]]) == doc), None)
        if first is None:
            continue
        i0 = first["mentions"][0][0]
        context = _mention_context(atoms, i0, mentions=None, siblings=labels)
        label = f"{len(labels)} places: " + "; ".join(labels[:12])
        try:
            d = _ask(label, _LIST_INSTRUCTION, context)
        except Exception:
            continue
        list_verdict[doc] = (getattr(d, "verdict", None), float(getattr(d, "confidence", 0.0) or 0.0),
                             str(getattr(d, "source", "") or ""), getattr(d, "correction_id", None))
        if trace is not None:
            trace.append({"label": label, "verdict": list_verdict[doc][0], "confidence": round(list_verdict[doc][1], 3),
                          "source": list_verdict[doc][2], "mentions": len(labels), "context": context[:400]})

    for key, entry in cands.items():
        if len(out) >= cap:
            break
        city, state, street = entry["city"], entry["state"], entry["street"]
        i, written = entry["mentions"][0]
        label = (f"{street}, {city}, {state}" if street else f"{city}, {state}") if state else city
        doc = _artifact_of(atoms[i])
        siblings = [x for x in by_doc.get(doc, []) if x != label and x != (f"{city}, {state}" if state else city)]
        context = _mention_context(atoms, i, mentions=entry["mentions"], siblings=siblings)
        lv = list_verdict.get(doc)
        if lv and lv[0] is not None:
            verdict, conf, source, corr = lv
            d = None
        else:
            try:
                d = _ask(label, _MENTION_INSTRUCTION, context)
            except Exception:
                continue
            verdict = getattr(d, "verdict", None)
            conf = float(getattr(d, "confidence", 0.0) or 0.0)
            source = str(getattr(d, "source", "") or "")
            corr = getattr(d, "correction_id", None)
        if trace is not None and d is not None:
            trace.append({"label": label, "verdict": verdict, "confidence": round(conf, 3), "source": source,
                          "mentions": len(entry["mentions"]), "context": context[:400]})
        if verdict != "job_site" or not (source == "store" or conf >= _MENTION_MIN_CONF):
            continue
        atom = atoms[i]
        artifact_id = _artifact_of(atom)
        slug = _slug(f"{street}_{city}_{state}" if street else (f"{city}_{state}" if state else city))
        atom_id = stable_id("atm", artifact_id, "physical_site", f"mention_{slug}")
        src_refs = list(getattr(atom, "source_refs", None) or []) or [
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.txt,
                filename=artifact_id or "geo_mention",
                locator={"extraction": "site_geo_mention"},
                extraction_method="site_geo_mention",
                parser_version="site_geo_mention_v1",
            )
        ]
        out.append(
            EvidenceAtom(
                id=atom_id,
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.physical_site,
                raw_text=label,
                normalized_text=label.lower(),
                value={
                    "kind": "physical_site",
                    "id": slug,
                    "site_id": slug,
                    "name": label,
                    "names": [label, city],
                    "street_address": street or None,
                    "address": street or None,
                    "city": city,
                    "state": state or None,
                    "zip": None,
                    "inferred": True,
                    "source_context": _text_of(atom)[:600],
                    "mention": written,
                    "mentions": len(entry["mentions"]),
                    "geo_mention_source": source,
                    "geo_mention_confidence": round(conf, 3),
                    "geo_mention_correction_id": corr,
                    "geo_mention_judged_as": "list" if (lv and lv[0] is not None) else "place",
                },
                entity_keys=[f"site:{slug}"],
                source_refs=src_refs,
                receipts=[],
                authority_class=AuthorityClass.machine_extractor,
                confidence=0.5,
                confidence_raw=0.5,
                calibrated_confidence=0.5,
                review_status=ReviewStatus.needs_review,
                review_flags=["geo_mention_site"],
                parser_version="site_geo_mention_v1",
            )
        )
    return out


__all__ = ["enrich_site_geo", "geo_fallback_sites", "geo_mention_sites", "suppress_vendor_sites"]
