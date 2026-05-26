"""Parse keyed-notes blocks and resolve body callouts to note text.

Construction drawings rely on keyed notes for the bulk of their
non-symbol requirements.  A header like ``KEYED NOTES`` or
``GENERAL NOTES`` introduces a numbered list (``1. Provide P/N…``,
``2. Coordinate w/ owner``), and the drawing body refers to those
numbers via callout bubbles or simple integers near devices.

Parsing produces:

- ``KeyedNote(number, text, bbox)`` records — one per numbered row.
- Callout bbox lists keyed to each note number so source_replay
  can verify the body markers.

Detection is text-rule only.  Numbered rows are split on the
common ``\\d+\\.`` / ``\\d+\\)`` row prefixes; body callouts are any
small TextBlock whose entire text is a bare 1-3 digit integer
appearing in or adjacent to a circle/parenthesis pattern.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.ids import stable_id
from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock


_NOTES_HEADER_RE = re.compile(
    r"\b(general\s+notes|keyed\s+notes|sheet\s+notes|note\s+legend|"
    r"installation\s+notes|construction\s+notes|fire\s+alarm\s+notes|"
    r"low\s+voltage\s+notes)\b",
    re.IGNORECASE,
)
_NOTES_ROW_RE = re.compile(
    r"^\s*(\d{1,3})[.\)]\s+(.+?)\s*$"
)
_CALLOUT_INTEGER_RE = re.compile(r"^\s*\(?(\d{1,3})\)?\s*$")


@dataclass(frozen=True)
class KeyedNote:
    note_id: str
    page_index: int
    sheet_number: str | None
    number: str
    text: str
    bbox: tuple[float, float, float, float]
    callout_bboxes: tuple[tuple[float, float, float, float], ...] = ()
    confidence: float = 0.85


def _find_notes_block(
    blocks: Sequence[TextBlock],
) -> list[TextBlock]:
    """Return the contiguous TextBlocks that make up the keyed-notes
    block, starting at the header and growing downward until a
    36pt gap or unrelated content.
    """
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    header: TextBlock | None = None
    for blk in sorted_blocks:
        if _NOTES_HEADER_RE.search(blk.text):
            header = blk
            break
    if header is None:
        return []
    out: list[TextBlock] = [header]
    last_bottom = header.bbox[3]
    for blk in sorted_blocks:
        if blk is header:
            continue
        if blk.bbox[1] < header.bbox[1]:
            continue
        gap = blk.bbox[1] - last_bottom
        if gap > 36.0:
            break
        # Headed-note rows are typically left-aligned with the header
        # (or slightly indented). Accept if its x is within 24 pt.
        if abs(blk.bbox[0] - header.bbox[0]) > 24.0:
            continue
        out.append(blk)
        last_bottom = max(last_bottom, blk.bbox[3])
    return out


def detect_keyed_notes(
    *,
    page_index: int,
    sheet_number: str | None,
    blocks: Sequence[TextBlock],
) -> list[KeyedNote]:
    """Find the numbered keyed-note rows on this page.

    Returns ``[]`` when no ``KEYED NOTES`` / ``GENERAL NOTES`` header
    is present. The caller's exclusion-zone logic already keeps the
    region out of symbol detection; this parser turns its contents
    into ``schematic_keyed_note`` atoms with body callouts attached
    when present.
    """
    notes_block = _find_notes_block(blocks)
    if not notes_block:
        return []
    parsed: list[KeyedNote] = []
    notes_region = (
        min(b.bbox[0] for b in notes_block),
        min(b.bbox[1] for b in notes_block),
        max(b.bbox[2] for b in notes_block),
        max(b.bbox[3] for b in notes_block),
    )
    for blk in notes_block:
        m = _NOTES_ROW_RE.match(blk.text)
        if not m:
            continue
        number = m.group(1).strip()
        text = m.group(2).strip()
        note_id = stable_id("keyed_note", page_index, sheet_number or "", number)
        # Find body callouts: bare integer blocks OUTSIDE the notes
        # region that match this note's number.
        callouts: list[tuple[float, float, float, float]] = []
        for body in blocks:
            if body is blk:
                continue
            if (
                body.bbox[0] >= notes_region[0] - 1
                and body.bbox[2] <= notes_region[2] + 1
                and body.bbox[1] >= notes_region[1] - 1
                and body.bbox[3] <= notes_region[3] + 1
            ):
                continue
            m2 = _CALLOUT_INTEGER_RE.match(body.text.strip())
            if m2 and m2.group(1) == number:
                callouts.append(body.bbox)
        parsed.append(
            KeyedNote(
                note_id=note_id,
                page_index=page_index,
                sheet_number=sheet_number,
                number=number,
                text=text,
                bbox=blk.bbox,
                callout_bboxes=tuple(callouts),
                confidence=0.85,
            )
        )
    parsed.sort(key=lambda n: (round(n.bbox[1], 2), n.number))
    return parsed
