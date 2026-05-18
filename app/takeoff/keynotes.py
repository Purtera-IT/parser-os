"""Parse the keynote table on a page + resolve per-device keynote refs.

Construction drawings carry a small numbered legend (commonly titled
``KEYED NOTES`` or ``GENERAL NOTES``) on each plan page. The notes
describe install details for the symbols on the plan, and each symbol
that needs one of those details is annotated on the plan with a small
circled number pointing at the matching note.

This module does three things:

1. Find the keynote block on a page (anywhere — top-left, top-right,
   bottom — by anchoring on the title text like ``KEYED NOTES``).
2. Parse the numbered entries inside the block. Supports both
   ``1. text...`` and ``1) text...`` and ``1 text...`` forms.
3. For each device candidate, find adjacent circled-number callouts (the
   small isolated digit tokens within ~30 pt of the candidate center)
   and return (number, description) pairs.

Nothing here uses OCR or vision — purely native PDF text via PyMuPDF.

Returns are deterministic given identical inputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.takeoff.pdf_native import PdfWord
from app.takeoff.schemas import BBox


# A keynote line looks like one of:
#   "1. INSTALL CAT6 CABLE..."
#   "1) INSTALL CAT6 CABLE..."
#   "1   INSTALL CAT6 CABLE..."   (no separator — common in some PDFs)
# We allow up to 3 leading digits to support "10.", "100.", etc.
_KEYNOTE_LINE_RE = re.compile(
    r"^\s*(\d{1,3})\s*[\.\)\:]?\s+([A-Z][A-Z0-9 ,\-/&\(\)\.\"#'\+\*=]{6,})\s*$"
)

# Block-start markers — the parser only treats lines AFTER one of these
# as keynote entries. That prevents random "1. RFP CLAUSE..." paragraphs
# elsewhere on the page from polluting the keynote table.
_KEYNOTE_HEADERS = (
    "KEYED NOTES",
    "KEY NOTES",
    "PLAN NOTES",
    "GENERAL NOTES",
    "DRAWING NOTES",
    "CABLE ZONING NOTES",  # commonly appears alongside HOMERUN zones
)

# A "block-end" sentinel — stop parsing keynote entries when we hit a
# header that looks like another section.
_BLOCK_END_HEADERS = (
    "SHEET NUMBER",
    "DRAWING TITLE",
    "REVISIONS",
    "COOPER CARRY",
    "CONSULTANTS",
)


@dataclass
class Keynote:
    """A single parsed keynote entry."""

    number: str  # text form, e.g. "1" or "10"
    description: str  # text after the number


@dataclass
class KeynoteTable:
    """All keynote blocks on one page, merged into a single number→note map.

    ``blocks`` records the source: which header line each entry came from.
    """

    page_index: int
    notes: dict[str, str] = field(default_factory=dict)
    blocks: list[str] = field(default_factory=list)

    def get(self, number: str) -> str | None:
        return self.notes.get(number)

    def __bool__(self) -> bool:
        return bool(self.notes)


def parse_keynote_table(page_index: int, page_text: str) -> KeynoteTable:
    """Extract a :class:`KeynoteTable` from a page's raw text.

    Walks line-by-line, anchoring on a header like ``KEYED NOTES``. We
    support two PDF layouts:

    1. **Inline** — each entry is ``"1. body..."`` on one line.
    2. **Column-split** — numbers and descriptions live in separate
       PDF text blocks, so the text extraction returns the numbers
       in order followed by the descriptions in order. We pair them
       1:1 by sequence after the header.
    """
    table = KeynoteTable(page_index=page_index)
    if not page_text:
        return table

    in_block = False
    last_number_for_continuation: str | None = None
    # Column-split buffer — numbers and bodies collected separately
    # between block-start and block-end markers.
    pending_numbers: list[str] = []
    pending_bodies: list[str] = []

    def _flush_columns() -> None:
        """Pair queued numbers with queued bodies 1:1."""
        for num, body in zip(pending_numbers, pending_bodies):
            if num not in table.notes:  # don't clobber inline matches
                table.notes[num] = body
        pending_numbers.clear()
        pending_bodies.clear()

    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        upper = line.upper()

        # Block-end first — common headers that mean we're past the notes.
        if in_block and any(end in upper for end in _BLOCK_END_HEADERS):
            _flush_columns()
            in_block = False
            last_number_for_continuation = None
            continue

        # Block-start
        if any(hdr in upper for hdr in _KEYNOTE_HEADERS):
            _flush_columns()
            in_block = True
            if upper not in table.blocks:
                table.blocks.append(upper)
            last_number_for_continuation = None
            continue

        if not in_block:
            continue

        # Inline "1. body" form.
        m = _KEYNOTE_LINE_RE.match(line)
        if m:
            number, body = m.group(1), m.group(2).strip()
            body = re.sub(r"\s+", " ", body).strip(" .")
            table.notes[number] = body
            last_number_for_continuation = number
            continue

        # Column-split: lone digit on its own line is a number.
        if line.isdigit() and 1 <= len(line) <= 3:
            pending_numbers.append(line)
            last_number_for_continuation = None
            continue

        # Column-split: description-shaped line.
        # Heuristic: at least 8 chars, doesn't start with a digit, mostly
        # uppercase-or-mixed prose, and isn't a one-token short noise.
        if len(line) > 8 and not line[0].isdigit():
            looks_like_body = (
                any(c.isalpha() for c in line)
                and not all(c.isupper() and not c.isalpha() for c in line)
                and " " in line  # multi-word
            )
            if looks_like_body:
                pending_bodies.append(re.sub(r"\s+", " ", line).strip(" ."))
                # Also act as continuation for any inline last entry.
                if last_number_for_continuation is not None:
                    existing = table.notes.get(last_number_for_continuation, "")
                    joined = (existing + " " + line).strip()
                    table.notes[last_number_for_continuation] = re.sub(r"\s+", " ", joined)
                continue

    # Flush any unpaired column-split entries at end of document.
    _flush_columns()
    return table


def find_keynote_refs_near(
    *,
    bbox: BBox,
    page_words: list[PdfWord],
    radius_pt: float = 40.0,
) -> list[str]:
    """Return isolated digit tokens within ``radius_pt`` of ``bbox`` center.

    A "circled keynote reference" on the plan is just a small number
    drawn near the symbol. The number itself is plain text in the PDF
    layer; the surrounding circle is vector geometry. We only need the
    number.

    Filters:
    * Token must be a digit (1–3 characters).
    * Token must not be inside the candidate's own bbox.
    * Returned in ascending distance order.
    """
    cx = (bbox.x0 + bbox.x1) / 2.0
    cy = (bbox.y0 + bbox.y1) / 2.0
    own_x0, own_y0, own_x1, own_y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

    candidates: list[tuple[float, str]] = []
    for w in page_words:
        text = (w.text or "").strip().strip(".,;:")
        if not text.isdigit() or not (1 <= len(text) <= 3):
            continue
        wx = (w.x0 + w.x1) / 2.0
        wy = (w.y0 + w.y1) / 2.0
        # Skip if this word overlaps the candidate's bbox (own_*).
        if own_x0 <= wx <= own_x1 and own_y0 <= wy <= own_y1:
            continue
        d = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
        if d > radius_pt:
            continue
        candidates.append((d, text))

    candidates.sort(key=lambda row: row[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, text in candidates:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def resolve_keynote(
    *,
    refs: list[str],
    table: KeynoteTable,
) -> tuple[str | None, str | None]:
    """Pick the most-likely keynote ref from a list + look up its text.

    Returns ``(number, description)`` for the nearest ref that has a match
    in the table, or ``(refs[0], None)`` if there's a nearby ref but no
    matching entry, or ``(None, None)`` if no refs at all.
    """
    if not refs:
        return (None, None)
    for ref in refs:
        if ref in table.notes:
            return (ref, table.notes[ref])
    return (refs[0], None)


__all__ = [
    "Keynote",
    "KeynoteTable",
    "parse_keynote_table",
    "find_keynote_refs_near",
    "resolve_keynote",
]
