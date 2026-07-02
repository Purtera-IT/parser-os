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
    US_STATES,
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


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


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


def _physical_site_atoms(atoms: list[Any]) -> list[Any]:
    return [a for a in atoms if _atom_type_str(a) == "physical_site"]


def _should_skip_geo_fallback(atoms: list[Any]) -> bool:
    """Skip only when the deal already has multiple well-structured sites.

    A single weak ``physical_site`` (name-only / misparsed geo) must NOT
    block discovering additional addresses — the MBrany failure mode.
    """
    sites = _physical_site_atoms(atoms)
    if not sites:
        return False
    scores = [_site_location_score(getattr(a, "value", None) or {}) for a in sites]
    high = sum(1 for s in scores if s >= 2)
    # Two well-structured sites — skip fallback. One high + one weak ghost
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


__all__ = ["geo_fallback_sites", "suppress_vendor_sites"]
