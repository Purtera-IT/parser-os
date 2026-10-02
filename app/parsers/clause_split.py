"""Clause-level splitting of SOW / contract paragraphs, shared by the PDF and
DOCX parsers so a draft (.docx) and its signed copy (.pdf) produce the same
atoms.

A paragraph that states several facts or obligations ("Acceptance occurs on
... . Customer has five business days ... . Silence is acceptance ...") is one
atom per fact. A list lead-in ("Provider will perform the following:") glued to
the end of the paragraph before it is its own atom. A lead-in line followed by
manual-line-break rows (a rate table typed as lines) is the lead-in plus one
atom per row.

Conservative where it matters:

* a split point is sentence-ending punctuation followed by a capital, so a
  decimal ("$93,583.25"), an abbreviation ("a.m.", "No.") or a lowercase
  continuation never splits, and a sentence wrapped across lines (the text is
  whitespace-joined first) stays whole;
* a sentence that leans on the one before ("It is ...", "This includes ...")
  stays attached to it;
* a short paragraph (under ``min_chars``) stays whole unless its last sentence
  is a list lead-in.
"""

from __future__ import annotations

import re

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[A-Z(“\"])")
# "Ms." is the honorific only as written, title case: upper-case "MS." is
# Mississippi. Matched case-insensitively, live 000132's "... and Tupelo, MS.
# Customer requires ..." never split while the same paragraph ending "...
# Wilmington, DE. Customer requires ..." did, so one SOW paragraph was two
# atoms in one version and one atom in the next.
ABBREV_TAIL_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|No|St|Ave|Rd|Dr|Mr|Mrs|(?-i:Ms)|Inc|Ltd|Corp|a\.m|p\.m|approx|Sec|Fig|U\.S)\.$",
    re.I,
)
# A sentence that refers back to the previous one rather than stating a fact
# of its own.
_ANAPHORIC_RE = re.compile(
    r"^(?:it|they|these|those|such|this|that|which)\s+"
    r"(?:is|are|was|were|will|shall|may|can|could|would|should|must|includes?|"
    r"means|applies|apply|requires?|covers?|consists?|refers?)\b",
    re.I,
)
_BULLET_GLYPH_RE = re.compile(r"^(?:[•▪●◦‣\-–—*]|\d{1,2}[.)]|[a-z][.)])\s+\S")
_MIN_SENTENCE = 25


def sentences(text: str) -> list[str]:
    """Sentences of ``text`` (whitespace-joined first)."""
    t = " ".join((text or "").split())
    if not t:
        return []
    parts: list[str] = []
    buf = ""
    for piece in SENTENCE_SPLIT_RE.split(t):
        piece = piece.strip()
        if not piece:
            continue
        if buf and ABBREV_TAIL_RE.search(buf):
            buf = f"{buf} {piece}"
            continue
        if buf:
            parts.append(buf)
        buf = piece
    if buf:
        parts.append(buf)
    return parts


def _merge_dependent(parts: list[str]) -> list[str]:
    out: list[str] = []
    for p in parts:
        if out and (_ANAPHORIC_RE.match(p) or len(out[-1]) < _MIN_SENTENCE):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    # A trailing fragment too short to stand alone rides with the one before,
    # unless it is a list lead-in.
    if len(out) >= 2 and len(out[-1]) < _MIN_SENTENCE and not out[-1].endswith(":"):
        tail = out.pop()
        out[-1] = f"{out[-1]} {tail}"
    return out


def _split_semicolon_rules(sentence: str) -> list[str]:
    """A sentence that is a run of ';'-separated rules ("Business Hours are
    8-5 M-F; after-hours work is billed at 150%; Sundays and holidays at 200%")
    is one atom per rule. Only when there are at least two ';' and every part
    is substantial."""
    if sentence.count(";") < 2 or len(sentence) < 120:
        return [sentence]
    parts = [p.strip() for p in sentence.split(";")]
    if any(len(p) < 15 for p in parts):
        return [sentence]
    return parts


def split_clauses(text: str, *, min_chars: int = 160) -> list[str]:
    """The paragraph's clauses, or ``[]`` when it should stay one atom."""
    raw = text or ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # A lead-in line ("Business hours are as follows:") over rows typed as
    # separate lines (a rate table, a parts list). The lead-in may wrap over
    # several lines and close a longer paragraph; the rows are short lines
    # that do not continue a sentence.
    colon = next((i for i, ln in enumerate(lines) if ln.endswith(":")), None)
    rows = lines[colon + 1:] if colon is not None else []
    widest = max((len(ln) for ln in lines), default=0)

    def _wraps(r: str) -> bool:
        # A line that runs the full measure without closing punctuation is a
        # sentence wrapping onto the next line, not a row.
        return widest >= 60 and len(r) >= 0.85 * widest and not re.search(r"[.!?:;%)\d]$", r)

    if colon is not None and len(rows) >= 2 and all(
        len(r) <= 100 and not r[:1].islower() for r in rows
    ) and not any(_wraps(r) for r in rows[:-1]):
        head_text = " ".join(lines[: colon + 1])
        head = sentences(head_text)
        units: list[str] = []
        if len(head) >= 2:
            units.extend(_merge_dependent(head[:-1]))
            units.append(head[-1])
        else:
            units.append(head_text)
        for ln in rows:
            units.extend(split_clauses(ln, min_chars=min_chars) or [ln])
        return units
    # Rows that each carry a bullet glyph / enumerator.
    if len(lines) >= 2 and sum(1 for ln in lines if _BULLET_GLYPH_RE.match(ln)) >= len(lines) - 1:
        units = []
        for ln in lines:
            units.extend(split_clauses(ln, min_chars=min_chars) or [ln])
        return units

    joined = " ".join(raw.split())
    parts = sentences(joined)
    if len(parts) < 2 and joined.count(";") < 2:
        return []
    lead_in = parts[-1] if len(parts) >= 2 and parts[-1].endswith(":") else None
    if lead_in is not None and len(joined) < min_chars:
        body = " ".join(parts[:-1])
        return [body, lead_in] if body else []
    if len(joined) < min_chars:
        return []
    merged = _merge_dependent(parts)
    out: list[str] = []
    for s in merged:
        out.extend(_split_semicolon_rules(s))
    return out if len(out) >= 2 else []
