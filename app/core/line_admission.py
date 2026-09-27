"""Admit a line the parser made no atom of.

Every other gate in the compile takes atoms away. This one is the only thing
that can put a fact back, and it is the only seam where a labeler's judgement
can reach the question "should this have been an atom at all?".

WHY IT IS A SEPARATE STAGE
--------------------------
`span_admission` already serves the trained admission heads, but it operates on
atoms that EXIST: it re-types a weak `scope_item` into a `requirement`. A line
the parser skipped has no atom to re-type, so nothing in the pipeline could
ever recover it. Measured on 010180 (2026-09-27): 2,634 unclaimed lines across
38 documents, 910 of which nothing read at all.

ADDITIVE ONLY, DELIBERATELY
---------------------------
This stage may only ADD. It reads lines that became nothing and mints atoms
from the ones it is confident about; it never removes or re-types an atom the
parser admitted. That asymmetry is the whole safety argument, and it is not
theoretical: every defect found on 010180 in the week to 2026-09-27 was
something being taken away -- eight rooms deleted by a substance gate, a
drawing set aside as another job, a paragraph glued into one atom, invite
fields admitted as scope, wreckage minted from a floor plan's walls. A head
that can only add cannot repeat any of them.

GUESS-FREE
----------
A taught correction decides first; the trained head decides only what the store
has no opinion on; below threshold nothing happens. No LLM. Same contract as
`span_admission`, which is where the heads and their precision targets live.

OFF BY DEFAULT
--------------
`SOWSMITH_LINE_ADMISSION`. It stays off until the corpus has positives: the
`keep` class comes from a labeler promoting a missed line, and on 2026-09-27
there were 30 `drop` rows and 1 `keep` across every labelled deal. A head
trained on that admits nothing, which is the correct behaviour for a head that
has been taught nothing, and is also the reason not to switch it on yet.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional

#: The atom a recovered line becomes before the typing stages see it. Weak on
#: purpose: the line was admitted on the evidence that it says something, not
#: on evidence about WHAT it says, and `typed_atom_classification` runs after
#: this and is the thing that decides.
RECOVERED_ATOM_TYPE = "scope_item"

#: Floor on a line's length. Below this a "line" is a table cell, a page
#: number or a fragment of a word split by a wall -- the shapes that made 18 of
#: 010180's 40 drawing atoms wreckage.
MIN_CHARS = 24

#: `unclaimed` states worth considering. "chrome" is a header or a signature
#: the coverage pass already recognised as furniture, and an "image" region is
#: not text at all.
CONSIDERED_STATES = ("unread", "suppressed")

RELATION = "admission"
CANDIDATES: tuple[str, str] = ("keep", "drop")
_INSTRUCTION = (
    "This line of the document became no atom. Is it a fact this deal needs "
    "(keep), or furniture -- a header, a signature, boilerplate (drop)?"
)


def enabled() -> bool:
    return os.environ.get("SOWSMITH_LINE_ADMISSION", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def unclaimed_lines(coverage: Any) -> list[dict[str, Any]]:
    """Every line no atom claims, once, with the count of how often it repeats.

    A quoted thread carries the same signature block and meeting invite down
    every reply, so the raw list is mostly one line repeated: 910 unclaimed
    lines on 010180 are 34 distinct texts, and "Bell Works | 101 Crawfords
    Corner Road" is 174 of them. Judging the same words 174 times is 174
    chances to be inconsistent about one decision.
    """
    rows = getattr(coverage, "text", None)
    if rows is None and isinstance(coverage, dict):
        rows = coverage.get("text")
    seen: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        row = row if isinstance(row, dict) else {}
        artifact_id = str(row.get("artifact_id") or "")
        for line in row.get("unclaimed") or []:
            line = line if isinstance(line, dict) else {}
            if str(line.get("state") or "") not in CONSIDERED_STATES:
                continue
            text = " ".join(str(line.get("text") or "").split())
            if len(text) < MIN_CHARS:
                continue
            key = text.lower()
            hit = seen.get(key)
            if hit is not None:
                hit["repeats"] += 1
                continue
            seen[key] = {
                "artifact_id": artifact_id,
                "line": int(line.get("line") or 0),
                "text": text,
                "state": str(line.get("state") or ""),
                "repeats": 1,
            }
    return list(seen.values())


def _verdict(text: str, scope: Any) -> tuple[Optional[str], str, float]:
    """(verdict, source, confidence). A taught correction outranks the head."""
    try:
        from app.core.decide import decide
    except Exception:  # pragma: no cover - decide is always present in prod
        return None, "unavailable", 0.0
    try:
        d = decide(RELATION, text[:600], list(CANDIDATES), instruction=_INSTRUCTION,
                   scope=scope, model=None)
    except Exception:
        return None, "error", 0.0
    verdict = getattr(d, "verdict", None)
    return verdict, getattr(d, "source", "fallback"), float(getattr(d, "confidence", 0.0) or 0.0)


def admit_missed_lines(
    coverage: Any,
    *,
    project_id: str = "",
    make_atom: Any = None,
    lines: Optional[Iterable[dict[str, Any]]] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Mint an atom for each unclaimed line a teacher or head says to keep.

    Returns ``(atoms, verdicts)``. ``verdicts`` carries every line considered,
    kept or not, so a compile can report what it looked at and did not take --
    a stage that only reports what it added is a stage nobody can audit.

    ``make_atom(line)`` builds the EvidenceAtom; the caller owns atom identity,
    source refs and receipts, which is where they belong.
    """
    if not enabled() or make_atom is None:
        return [], []
    try:
        from app.core.decide import DecisionScope
        scope = DecisionScope(deal_id=str(project_id or ""))
    except Exception:  # pragma: no cover
        scope = None

    out: list[Any] = []
    verdicts: list[dict[str, Any]] = []
    for line in lines if lines is not None else unclaimed_lines(coverage):
        verdict, source, confidence = _verdict(line["text"], scope)
        record = {**line, "verdict": verdict, "source": source,
                  "confidence": round(confidence, 3), "admitted": False}
        # Guess-free: only an explicit keep admits. An abstain, a drop, or a
        # head that has never been taught all leave the line exactly where the
        # parser left it.
        if verdict == "keep":
            try:
                atom = make_atom(line)
            except Exception:
                atom = None
            if atom is not None:
                out.append(atom)
                record["admitted"] = True
        verdicts.append(record)
    return out, verdicts


__all__ = ["RELATION", "CANDIDATES", "RECOVERED_ATOM_TYPE", "MIN_CHARS",
           "CONSIDERED_STATES", "enabled", "unclaimed_lines",
           "admit_missed_lines"]
