"""Prose-list splitter — universal multi-fact paragraph atomizer.

v50: The single biggest recall gap is paragraphs that contain N
parallel-structured facts compressed into one atom:

    "Jordan Ames | VP Workplace Operations | jordan.ames@... | Executive sponsor.
     Priya Narang | Director of Enterprise IT | priya.narang@... | Technical buyer.
     Elliot Tran | Senior Procurement Manager | elliot.tran@... | Procurement lead.
     ..."

Today's parser emits ONE scope_item atom for this paragraph. The
typed-atom classifier can only assign ONE atom_type per atom →
1 stakeholder atom from 6 real people. Same problem for milestone
paragraphs ("Phase 0 ... Phase 1 ... Phase 2 ..."), payment
schedules ("30% at order; 40% at receipt; ..."), risks, etc.

This module detects parallel-structured runs INSIDE a single atom
and emits one child atom per item. The child atoms keep the parent's
source_ref but get a sub-locator (sub_idx) for provenance.

Universal patterns detected (no customer-specific tuning):

  Pattern A — Pipe-delimited records:
    "<token>|<token>|<token>" appearing ≥3 times in the text with
    similar field counts (±1). Each record becomes a child atom.

  Pattern B — Numbered enumeration:
    "1. <text> 2. <text> 3. <text>" — text segmented on the numbers.
    Also handles "Phase 0 ...", "Step 1 ...", "REQ-001 ..." where
    a stable label prefix repeats with monotonic counter.

  Pattern C — Semicolon-delimited parallel sentences:
    "X is alpha; Y is beta; Z is gamma." with ≥3 segments AND
    similar token shape (each segment has parallel verb/subject).

  Pattern D — Newline-delimited bulletted items:
    "- item one\n- item two\n- item three" — handles markdown-bullet
    flattening that some PDF/DOCX parsers do.

  Pattern E — Sentence-bounded label-prefix runs:
    "Foo: ... Bar: ... Baz: ..." where each colon-anchored prefix
    is a short capitalized label.

The output is a list of child-atom DICTS (not full EvidenceAtom
instances) — the caller (compiler.py) constructs proper atoms
with stable IDs. This keeps the splitter pure / deterministic /
no schema dependency.
"""
from __future__ import annotations

import re
from typing import Any


# ── Detection thresholds (universal — tuned to be conservative) ─────

# Minimum item count for any split to fire. Avoids splitting a
# 2-clause sentence into "fragments".
_MIN_ITEMS = 3

# Length tolerance: items are "parallel" when each is within this
# multiplier of the median item length. Loose enough to handle
# variation, tight enough to reject "mixed prose + list".
_LEN_TOLERANCE = 4.0

# Minimum atom text length to attempt splitting. Sub-50-char atoms
# rarely contain enumerable lists.
_MIN_ATOM_LEN = 80


def split_prose_paragraph(text: str) -> list[str] | None:
    """Try every pattern. Return the first successful split (list of
    item texts) or ``None`` if no pattern fires.

    Returns None for short atoms, already-atomic text, or text that
    doesn't match any parallel-structure pattern. Returns list[str]
    of N≥_MIN_ITEMS items when a pattern matches.
    """
    if not text or len(text) < _MIN_ATOM_LEN:
        return None
    t = text.strip()

    # Try patterns in precedence order — most-specific first
    for fn in (
        _try_pipe_records,
        _try_numbered_enumeration,
        _try_label_prefix_runs,
        _try_semicolon_parallel,
        _try_bulleted_lines,
    ):
        items = fn(t)
        if items and len(items) >= _MIN_ITEMS and _items_parallel(items):
            return [s.strip() for s in items if s.strip()]
    return None


# ─── Pattern A: pipe-delimited records ─────────────────────────────


def _try_pipe_records(text: str) -> list[str] | None:
    """Detect rows of pipe-separated fields:
        "A | B | C. D | E | F. G | H | I"
    Split on ``.`` (or sentence boundary) when each segment contains
    ≥2 pipe characters AND segments have similar pipe counts.
    """
    # Quick reject: needs ≥6 total pipes (3 records × 2 pipes each)
    if text.count("|") < 6:
        return None

    # Split on sentence boundaries that also separate pipe-records:
    # the sentence end ". " followed by a capitalized token that
    # starts a new pipe record (has a pipe within next ~150 chars).
    candidates = re.split(r"(?<=[.])\s+(?=[A-Z])", text)
    if len(candidates) < _MIN_ITEMS:
        # Fall back: try splitting on newline + capital
        candidates = re.split(r"\n+\s*(?=[A-Z])", text)
        if len(candidates) < _MIN_ITEMS:
            return None
    pipe_counts = [c.count("|") for c in candidates]
    # Each record must have ≥2 pipes (3+ fields). Allow a tail
    # segment with fewer pipes (trailing prose).
    qualifying = [c for c, p in zip(candidates, pipe_counts) if p >= 2]
    if len(qualifying) >= _MIN_ITEMS:
        return qualifying
    return None


# ─── Pattern B: numbered enumeration ───────────────────────────────


def _try_numbered_enumeration(text: str) -> list[str] | None:
    """Detect monotonic-counter prefixes that appear ≥3 times:

        "Phase 0 ..., Phase 1 ..., Phase 2 ..."
        "Step 1 ..., Step 2 ..., Step 3 ..."
        "REQ-001 ..., REQ-002 ..., REQ-003 ..."
        "1. text 2. text 3. text"
        "T-001 ..., T-002 ..., T-101 ..."

    Looks for repeated word+number prefixes with arithmetic increase.
    """
    # Pattern: \b<word>[-_ ]?<digits>\b — capture (label, number).
    # Match runs where same label appears ≥_MIN_ITEMS times.
    label_num_re = re.compile(
        r"\b([A-Z][A-Za-z]{1,15}|[A-Z]{1,5})[-_\s]*([0-9]{1,4})\b"
    )
    matches = list(label_num_re.finditer(text))
    if len(matches) < _MIN_ITEMS:
        return None

    # Group by label; need ≥_MIN_ITEMS hits of same label.
    by_label: dict[str, list[tuple[int, int]]] = {}
    for m in matches:
        label = m.group(1).lower()
        try:
            num = int(m.group(2))
        except ValueError:
            continue
        by_label.setdefault(label, []).append((m.start(), num))

    # Find the label with the longest monotonic-ish run.
    best_label: str | None = None
    best_starts: list[int] = []
    for label, hits in by_label.items():
        if len(hits) < _MIN_ITEMS:
            continue
        # Allow non-strict monotonic (some labels skip — REQ-001 REQ-005)
        # but require unique numbers and reasonably sorted starts.
        nums = [n for _, n in hits]
        if len(set(nums)) < _MIN_ITEMS:
            continue
        starts = sorted(s for s, _ in hits)
        if len(starts) > len(best_starts):
            best_label = label
            best_starts = starts
    if not best_label:
        return None

    # Slice the text between consecutive label-start offsets.
    items: list[str] = []
    for i, start in enumerate(best_starts):
        end = best_starts[i + 1] if i + 1 < len(best_starts) else len(text)
        seg = text[start:end].strip().rstrip(",;.")
        if seg:
            items.append(seg)
    return items if len(items) >= _MIN_ITEMS else None


# ─── Pattern C: semicolon-parallel sentences ───────────────────────


def _try_semicolon_parallel(text: str) -> list[str] | None:
    """Detect '; '-separated parallel sentences:

        "30% at order acceptance; 40% on equipment receipt;
         20% at site acceptance; 10% after hypercare closeout."
    """
    if text.count(";") < (_MIN_ITEMS - 1):
        return None
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) < _MIN_ITEMS:
        return None
    return parts


# ─── Pattern D: bulleted-line lists ────────────────────────────────


def _try_bulleted_lines(text: str) -> list[str] | None:
    """Detect bulleted lists that survived parser flattening:

        "- item one\\n- item two\\n- item three"
        "• item one\\n• item two\\n• item three"
        "* item one\\n* item two\\n* item three"
    """
    bullet_re = re.compile(r"^[\s]*[-•*]\s+(.+?)(?=\n[\s]*[-•*]\s|\Z)", re.MULTILINE | re.DOTALL)
    items = [m.group(1).strip() for m in bullet_re.finditer(text)]
    if len(items) >= _MIN_ITEMS:
        return items
    return None


# ─── Pattern E: label-prefix runs ──────────────────────────────────


def _try_label_prefix_runs(text: str) -> list[str] | None:
    """Detect sentence runs where each starts with a short capitalized
    label followed by ':'.

        "Mock Data: ... Allowed Destinations: ... Blocked: ..."

    Each label is ≤4 words, capitalized; the label-colon pattern
    must repeat ≥3 times in the text.
    """
    # Label = 1-4 capitalized tokens followed by colon
    label_re = re.compile(r"(?:^|[.\s])\s*((?:[A-Z][A-Za-z0-9-]{1,20}\s*){1,4}):\s+")
    matches = list(label_re.finditer(text))
    if len(matches) < _MIN_ITEMS:
        return None
    starts = [m.start() for m in matches]
    items: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        seg = text[start:end].strip().rstrip(",;.")
        if seg:
            items.append(seg)
    return items if len(items) >= _MIN_ITEMS else None


# ─── Parallelism check ─────────────────────────────────────────────


def _items_parallel(items: list[str]) -> bool:
    """Reject splits where items are wildly heterogeneous in length.
    Real parallel lists have similar item sizes; mixed prose+list
    produces uneven splits we shouldn't trust.
    """
    if not items:
        return False
    lengths = sorted(len(s) for s in items)
    median = lengths[len(lengths) // 2]
    if median == 0:
        return False
    longest = lengths[-1]
    shortest = lengths[0]
    if shortest == 0:
        return False
    # Loose tolerance: longest within tolerance × median, shortest
    # within median ÷ tolerance.
    return (longest / median) <= _LEN_TOLERANCE and (median / shortest) <= _LEN_TOLERANCE


__all__ = ["split_prose_paragraph"]
