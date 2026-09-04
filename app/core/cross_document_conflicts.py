"""Cross-document conflicts: the same clause in two documents, different numbers.

A deal folder often holds two versions of one contract template (live 010300:
NewBold's signed network PSOW and NewBold's draft phone PSOW). They agree on
the wording of a clause and disagree on its value -- a $500 versus a $300
cancellation fee, two (2) weeks versus five (5) business days of notice, ZIP
30641 versus 30341 for the same street -- and a PM building a SOW from the
folder needs to know which one applies. Semantic dedup rightly keeps both
atoms (they are not duplicates); nothing raised the disagreement.

This pass is shape-based: normalise every atom's text by replacing numbers,
money and number-words with a placeholder, group atoms from DIFFERENT
documents that share the normalised text, and where the original numeric
tokens differ emit one ``open_question`` naming both values and both sources.
No vocabulary, no domain knowledge -- a template is recognised by being the
same words with different figures.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus

_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand"
)
#: A figure: money, a percentage, a decimal, an integer (with thousands
#: separators), or a spelled-out number optionally followed by "(n)".
_FIGURE_RE = re.compile(
    r"\$\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*%|\b\d[\d,]*(?:\.\d+)?\b"
    r"|\b(?:" + _NUMBER_WORDS + r")\b(?:\s*\(\s*\d+\s*\))?",
    re.I,
)
_SPACE_RE = re.compile(r"\s+")
_STRIP_RE = re.compile(r"[^a-z0-9#$%. ]+")

#: Text kinds that are provenance, not claims.
_SKIP_KINDS = ("_marker", "_header", "hubspot_note_meta", "signature_chrome")
_SKIP_TYPES = frozenset({"open_question", "stakeholder", "signatory", "physical_site", "raw_utterance", "deal_metadata"})

_MIN_WORDS = 6


def _atom_type(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def _text(atom: Any) -> str:
    return str(getattr(atom, "raw_text", None) or getattr(atom, "normalized_text", None) or "")


def _figures(text: str) -> list[str]:
    out = []
    for m in _FIGURE_RE.finditer(text):
        tok = _SPACE_RE.sub(" ", m.group(0).strip().lower())
        # "two (2)" and "2" are one figure; keep the digits when present.
        digits = re.search(r"\d[\d,]*(?:\.\d+)?", tok)
        out.append(("$" if "$" in tok else "") + (digits.group(0).replace(",", "") if digits else tok) + ("%" if "%" in tok else ""))
    return out


def _template(text: str) -> str:
    low = _SPACE_RE.sub(" ", text.lower()).strip()
    low = _FIGURE_RE.sub(" # ", low)
    low = _STRIP_RE.sub(" ", low)
    low = _SPACE_RE.sub(" ", low).strip(" .")
    return low


def _word_count(template: str) -> int:
    return len([w for w in template.split() if w != "#" and len(w) > 1])


def find_cross_document_conflicts(atoms: list[Any], *, project_id: str) -> list[EvidenceAtom]:
    """Return NEW open_question atoms, one per conflicting clause (possibly none)."""
    groups: dict[str, list[Any]] = {}
    for atom in atoms:
        at = _atom_type(atom)
        if at in _SKIP_TYPES:
            continue
        v = getattr(atom, "value", None)
        kind = str((v or {}).get("kind") or "") if isinstance(v, dict) else ""
        if kind.endswith(_SKIP_KINDS[:2]) or kind in _SKIP_KINDS:
            continue
        text = _text(atom)
        figs = _figures(text)
        if not figs:
            continue
        tpl = _template(text)
        if "#" not in tpl or _word_count(tpl) < _MIN_WORDS:
            continue
        groups.setdefault(tpl, []).append(atom)

    out: list[EvidenceAtom] = []
    for tpl, members in groups.items():
        by_doc: dict[str, Any] = {}
        for a in members:
            by_doc.setdefault(str(getattr(a, "artifact_id", "")), a)
        if len(by_doc) < 2:
            continue
        variants: dict[str, Any] = {}
        for a in by_doc.values():
            variants.setdefault(" ".join(_figures(_text(a))), a)
        if len(variants) < 2:
            continue  # same figures in every document: agreement, not conflict
        reps = list(variants.values())
        first = reps[0]
        values = [
            {
                "figures": key,
                "text": _text(a)[:300],
                "atom_id": str(getattr(a, "id", "")),
                "artifact_id": str(getattr(a, "artifact_id", "")),
            }
            for key, a in variants.items()
        ]
        summary = " vs ".join(v["figures"] for v in values)
        text = (
            f"Documents disagree on one clause: {summary}. "
            f"\"{_text(first)[:160].strip()}\" -- confirm which version governs before SOW work."
        )
        refs: list[Any] = []
        for a in reps:
            for r in (getattr(a, "source_refs", None) or []):
                if r not in refs:
                    refs.append(r)
        if not refs:
            continue
        out.append(
            EvidenceAtom(
                id=stable_id("xdoc_conflict", f"{project_id}:{tpl[:120]}"),
                project_id=project_id,
                artifact_id=str(getattr(first, "artifact_id", "")),
                atom_type=AtomType.open_question,
                raw_text=text,
                normalized_text=text.lower(),
                value={
                    "kind": "cross_document_conflict",
                    "text": text,
                    "template": tpl[:200],
                    "values": values,
                    "atom_ids": [v["atom_id"] for v in values],
                    "artifact_ids": sorted({v["artifact_id"] for v in values}),
                },
                entity_keys=[],
                source_refs=refs,
                authority_class=AuthorityClass.machine_extractor,
                confidence=0.7,
                review_status=ReviewStatus.needs_review,
                review_flags=["cross_document_conflict"],
                parser_version=str(getattr(first, "parser_version", "") or "cross_document_conflicts_v1"),
            )
        )
    return out
