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
    r"(" + _NUM + r")\s+(?:" + _SITE_NOUNS + r")\b",
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
            confirmed = 1 if atom.authority_class in _CONFIRMED_AUTHORITIES else 0
            cand = (confirmed, n, -idx, atom)
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best is None:
        return None
    confirmed, n, _, atom = best
    return n, atom, bool(confirmed)


def _found_site_count(atoms: Sequence[EvidenceAtom]) -> int:
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
        if found < n:
            qualifier = (
                "" if confirmed
                else " The only source is a quoted email in a forwarded"
                     " thread - confirm the count with the customer."
            )
            text = (
                f"Customer documents declare {n} locations; {found}"
                f" identified in the parsed files. Request the site list"
                f" (names and addresses) before SOW work.{qualifier}"
                f' Declared in: "{(src.raw_text or "").strip()[:160]}"'
            )
            out.append(_question(
                project_id=project_id, kind="site_count_gap", text=text, src=src,
                structured={
                    "kind": "site_count_gap",
                    "declared_count": n,
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
