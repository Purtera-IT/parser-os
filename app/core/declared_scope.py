"""Declared scope vs. found scope — say the tension out loud.

Born on the Marion County time-clock deal (010265): the customer wrote
"We need to have 10 timeclocks installed" and "I have created SOW's for
each of the ten locations" — and the compile still reported zero sites
with no acknowledgement that anything was missing. The declaration was
extracted, then buried three ways at once: typed into the inert
``deal_metadata`` bucket, demoted to ``quoted_old_email`` authority for
arriving inside a forwarded thread's quote block, and carrying no
``quantity:`` key because "ten" is spelled out (fixed in
``entity_extraction``).

This pass compares what the documents *declare* against what the parse
*found*, and mints ``open_question`` atoms when they disagree:

* declared site count > sites identified  ->  "customer declares N
  locations; M identified" (noting when the only source is a quoted
  email, so the PM knows to confirm rather than trust);
* per-site documents referenced ("SOW's for each of the ten locations")
  with no matching artifact in the intake  ->  "referenced documents
  missing".

A silent zero and a real zero must never look the same — this is that
rule applied to site counts. The pass only ever ADDS question atoms; it
never suppresses, retypes, or promotes anything, so default compiles
without declarations are byte-identical.
"""
from __future__ import annotations

import re
from typing import Sequence

from app.core.entity_extraction import _WORD_NUMBERS
from app.core.ids import stable_id
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
)

#: How trustworthy each authority is when DECLARING a count. Quoted email is
#: enough to raise the question (that is the whole point) but the question's
#: wording flags it as unconfirmed.
_CONFIRMED_AUTHORITIES = frozenset({
    AuthorityClass.contractual_scope,
    AuthorityClass.pm_confirmed,
    AuthorityClass.customer_current_authored,
    AuthorityClass.approved_site_roster,
    AuthorityClass.vendor_quote,
    AuthorityClass.meeting_note,
})

_SITE_NOUNS = (
    r"locations?|sites?|schools?|stores?|branch(?:es)?|buildings?|"
    r"facilit(?:y|ies)|campus(?:es)?|offices?"
)

_NUM = r"(?:[0-9]{1,4}|" + "|".join(_WORD_NUMBERS) + r")"

#: "the ten locations", "10 sites", "across 12 schools", "all five branches".
_DECLARED_SITES_RE = re.compile(
    r"\b(?:the|all|across|at|for(?:\s+each\s+of(?:\s+the)?)?)?\s*"
    r"(" + _NUM + r")\s+(" + _SITE_NOUNS + r")\b",
    re.IGNORECASE,
)

#: A count of two or more takes a plural noun: "10 locations", "twelve schools".
#: A number in front of a SINGULAR noun is a name, not a count -- "consolidate
#: the 1518 location tech into the existing 1517 subnet" is store 1518, and
#: 000020 Binghamton asked the PM for a list of 1,518 site addresses because of
#: it. Grammatical number, not a list of which numbers look like IDs.
_PLURAL_SITE_NOUN_RE = re.compile(
    r"(?:locations|sites|schools|stores|branches|buildings|facilities|campuses|offices)$",
    re.IGNORECASE,
)

#: "SOW's for each of the ten locations", "a statement of work per site",
#: "individual SOWs for every school".
_PER_SITE_DOCS_RE = re.compile(
    r"\b(?:sow(?:'?s)?|statements?\s+of\s+work|scopes?\s+of\s+work)\b"
    r"[^.\n]{0,60}?\b(?:for\s+each|per\s+(?:site|location|school|store)|"
    r"for\s+every)\b",
    re.IGNORECASE,
)

#: Artifact filenames that would satisfy a per-site-SOW reference.
_SOW_FILENAME_RE = re.compile(r"\bsow\b|statement[\s_-]*of[\s_-]*work", re.IGNORECASE)


def _as_int(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        n = int(token)
    else:
        n = _WORD_NUMBERS.get(token, 0)
    return n if 0 < n <= 10_000 else None


def _declared_site_count(
    atoms: Sequence[EvidenceAtom],
) -> tuple[int, EvidenceAtom, bool] | None:
    """(count, source atom, confirmed?) for the strongest site-count claim.

    Highest declared count wins ties on authority so "ten locations" beats a
    stray "two buildings" aside; a claim from a confirmed authority beats any
    quoted one regardless of size.
    """
    best: tuple[int, int, int, EvidenceAtom] | None = None  # (confirmed, n, -idx, atom)
    for idx, atom in enumerate(atoms):
        text = atom.raw_text or ""
        for m in _DECLARED_SITES_RE.finditer(text):
            n = _as_int(m.group(1))
            if n is None or n < 2:
                continue
            if not _PLURAL_SITE_NOUN_RE.search(m.group(2)):
                continue
            confirmed = 1 if atom.authority_class in _CONFIRMED_AUTHORITIES else 0
            cand = (confirmed, n, -idx, atom)
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best is None:
        return None
    confirmed, n, _, atom = best
    return n, atom, bool(confirmed)


#: A later line that takes sites OUT: "Need Troy and Wilmington sites
#: removed." (000132), "drop the Hudson location", "Tupelo is no longer in
#: scope".
_REMOVAL_RE = re.compile(
    r"\b(?:remov(?:e|ed|ing|al)|drop(?:s|ped|ping)?|cancel(?:s|l?ed|ling)?|descop(?:e|ed|ing)|"
    r"take\s+(?:off|out)|taken\s+(?:off|out)|no\s+longer\s+(?:need(?:ed)?|required|in\s+scope|part)|"
    r"out\s+of\s+scope|not\s+(?:be\s+)?in\s+scope)\b",
    re.IGNORECASE,
)

#: "Delphos, OH" / "Wilmington, DE": a place named in the deal's own lines.
_CITY_STATE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2})\b")

_NAME_STOP = frozenset({
    "site", "sites", "location", "locations", "office", "offices", "the", "of", "and",
    "store", "school", "building", "campus", "branch", "hq", "main",
})


def _site_name_words(site_key: str, aliases: Sequence[str] = ()) -> set[str]:
    """The distinctive words of a listed site's name: ``site:troy_oh`` -> troy."""
    words: set[str] = set()
    for raw in [site_key.split(":", 1)[-1], *aliases]:
        toks = [t for t in re.split(r"[^a-z0-9]+", str(raw).lower()) if t]
        # A trailing two-letter state is the state, not the name.
        if len(toks) >= 2 and len(toks[-1]) == 2 and toks[-1].isalpha():
            toks = toks[:-1]
        toks = [t for t in toks if t not in _NAME_STOP and not t.isdigit() and len(t) >= 3]
        if toks:
            words.add(" ".join(toks))
    return words


def _removed_sites(
    atoms: Sequence[EvidenceAtom],
    listed: Sequence[dict],
    *,
    after: EvidenceAtom | None = None,
) -> tuple[set[str], set[str], list[EvidenceAtom]]:
    """Which sites a line took back out of the deal.

    Returns ``(removed place names, removed listed site keys, the removing
    atoms)``. A place counts when a removal line names it -- a listed site by
    its name, or a "City, ST" the deal's own lines name (a declared site the
    parse never listed is still one fewer to find). Only a document no
    earlier than the declaration can take a site back out of it.
    """
    places: dict[str, str] = {}
    for a in atoms:
        for m in _CITY_STATE_RE.finditer(a.raw_text or ""):
            places.setdefault(m.group(1).lower(), m.group(1))
    site_names: dict[str, set[str]] = {}
    for row in listed or []:
        key = str(row.get("site") or "")
        if key:
            site_names[key] = _site_name_words(key, row.get("aliases") or ())
            for n in site_names[key]:
                places.setdefault(n, n)

    order = None
    if after is not None:
        try:
            from app.core.cross_doc_copies import doc_key, document_order

            order = document_order(atoms)
            floor = doc_key(after, order)
        except Exception:  # pragma: no cover - ordering is a refinement
            order = None

    names: set[str] = set()
    keys: set[str] = set()
    sources: list[EvidenceAtom] = []
    if not places:
        return names, keys, sources
    # The removal has to be ABOUT the place: "Need Troy and Wilmington sites
    # removed", "drop Tupelo", "Hudson is no longer in scope" -- not "remove
    # the old APs at the Hudson office".
    place = "(?:" + "|".join(re.escape(p) for p in sorted(places, key=len, reverse=True)) + ")"
    one = place + r"(?:\s*,\s*[a-z]{2})?"
    many = one + r"(?:\s*(?:,|and|&)\s*" + one + r")*"
    verb_first = re.compile(
        r"\b(?:remov\w*|drop\w*|cancel\w*|descop\w*|tak\w+\s+(?:off|out))\s+(?:the\s+)?(" + many + r")\b",
        re.IGNORECASE,
    )
    place_first = re.compile(
        r"\b(" + many + r")\s+(?:(?:sites?|locations?|offices?|stores?|schools?|branch(?:es)?)\s+)?"
        r"(?:(?:is|are|was|were|has\s+been|have\s+been|to\s+be|needs?\s+to\s+be|should\s+be|will\s+be)\s+)?"
        r"(?:removed|dropped|cancel\w*|descoped|taken\s+(?:off|out)|no\s+longer|out\s+of\s+scope)\b",
        re.IGNORECASE,
    )
    place_re = re.compile(r"\b" + place + r"\b", re.IGNORECASE)
    for a in atoms:
        flags = a.review_flags or []
        if a is after or "cross_doc_copy" in flags or "declared_scope" in flags:
            continue
        text = a.raw_text or ""
        if not _REMOVAL_RE.search(text):
            continue
        if order is not None and doc_key(a, order) < floor:
            continue
        hit: set[str] = set()
        for rx in (verb_first, place_first):
            for m in rx.finditer(text):
                hit |= {p.group(0).lower() for p in place_re.finditer(m.group(1))}
        if not hit:
            continue
        names |= hit
        keys |= {k for k, ws in site_names.items() if ws & hit}
        sources.append(a)
    return names, keys, sources


def _found_site_count(atoms: Sequence[EvidenceAtom]) -> int:
    """How many sites the deal's site list shows.

    The question atom says "N identified", and the PM reads that against the
    live site list -- so it must be the site list's own count. Counting only
    ``site:`` entity keys on physical_site atoms disagreed with it: a site
    anchored by its value id with no ``site:`` key (a name-only roster row),
    or one surfaced under an alias, is on the list and was not counted (live
    000132: "declare 6 locations; 3 identified" beside four live sites).
    ``build_site_readiness`` is the list; the key count is the fallback when
    it cannot run.
    """
    try:
        from app.core.orbitbrief_core import build_site_readiness

        listed = build_site_readiness(atoms=list(atoms), edges=[])
        n = int((listed or {}).get("site_count") or 0)
        if n or not _physical_site_key_count(atoms):
            return n
    except Exception:  # pragma: no cover - the cross-check must not fail a compile
        pass
    return _physical_site_key_count(atoms)


def _listed_sites(atoms: Sequence[EvidenceAtom]) -> list[dict]:
    try:
        from app.core.orbitbrief_core import build_site_readiness

        return list((build_site_readiness(atoms=list(atoms), edges=[]) or {}).get("sites") or [])
    except Exception:  # pragma: no cover
        return []


def _physical_site_key_count(atoms: Sequence[EvidenceAtom]) -> int:
    slugs: set[str] = set()
    for atom in atoms:
        if getattr(atom, "atom_type", None) == AtomType.physical_site:
            for key in atom.entity_keys:
                if isinstance(key, str) and key.startswith("site:"):
                    slugs.add(key)
    return len(slugs)


def _question(
    *,
    project_id: str,
    kind: str,
    text: str,
    src: EvidenceAtom,
    structured: dict,
) -> EvidenceAtom:
    return EvidenceAtom(
        id=stable_id("declared_scope", f"{project_id}:{kind}"),
        project_id=project_id,
        artifact_id=src.artifact_id,
        atom_type=AtomType.open_question,
        raw_text=text,
        normalized_text=text.lower(),
        value={"text": text, "declared_scope": structured},
        entity_keys=list(structured.get("entity_keys", [])),
        source_refs=list(src.source_refs),
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.65,
        review_status=ReviewStatus.needs_review,
        review_flags=["declared_scope"],
        parser_version=src.parser_version,
    )


def declared_scope_questions(
    *, project_id: str, atoms: Sequence[EvidenceAtom]
) -> list[EvidenceAtom]:
    """The pass. Returns ONLY new open_question atoms (possibly none)."""
    out: list[EvidenceAtom] = []
    declared = _declared_site_count(atoms)

    if declared is not None:
        n, src, confirmed = declared
        found = _found_site_count(atoms)
        # Sites a later line took back out (000132: "Need Troy and Wilmington
        # sites removed.") leave the declaration AND the found list: the gap
        # read "declare 6 locations; 5 identified" against four live sites.
        removed_names, removed_keys, removers = _removed_sites(
            atoms, _listed_sites(atoms), after=src,
        )
        removed = min(len(removed_names), n)
        live_declared = n - removed
        found = max(0, found - len(removed_keys))
        if found < live_declared:
            qualifier = (
                "" if confirmed
                else " The only source is a quoted email in a forwarded"
                     " thread - confirm the count with the customer."
            )
            declared_txt = (
                f"{n} locations; {removed} later removed"
                f" ({', '.join(sorted(n_.title() for n_ in removed_names))}), {live_declared} remain;"
                if removed else f"{n} locations;"
            )
            text = (
                f"Customer documents declare {declared_txt} {found}"
                f" identified in the parsed files. Request the site list"
                f" (names and addresses) before SOW work.{qualifier}"
                f' Declared in: "{(src.raw_text or "").strip()[:160]}"'
            )
            out.append(_question(
                project_id=project_id, kind="site_count_gap", text=text, src=src,
                structured={
                    "kind": "site_count_gap",
                    "declared_count": n,
                    "removed_count": removed,
                    "removed_sites": sorted(removed_names),
                    "removing_atom_ids": [a.id for a in removers],
                    "live_declared_count": live_declared,
                    "found_count": found,
                    "declaration_confirmed": confirmed,
                    "declaring_atom_id": src.id,
                    "entity_keys": [f"quantity:{n}"],
                },
            ))

    # Per-site documents referenced but absent from the intake.
    filenames = {
        (ref.filename or "")
        for atom in atoms
        for ref in atom.source_refs
    }
    has_sow_file = any(_SOW_FILENAME_RE.search(f) for f in filenames)
    if not has_sow_file:
        for atom in atoms:
            m = _PER_SITE_DOCS_RE.search(atom.raw_text or "")
            if not m:
                continue
            text = (
                f"Documents reference per-site statements of work"
                f' ("{(atom.raw_text or "").strip()[:140]}") but no SOW file'
                f" is present in the intake. Request the per-site SOWs."
            )
            out.append(_question(
                project_id=project_id, kind="referenced_sows_missing",
                text=text, src=atom,
                structured={
                    "kind": "referenced_sows_missing",
                    "referencing_atom_id": atom.id,
                    "entity_keys": [],
                },
            ))
            break  # one question, not one per mention

    return out
