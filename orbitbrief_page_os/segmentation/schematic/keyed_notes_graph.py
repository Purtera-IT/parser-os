"""Keyed-notes graph: link numbered callouts on the drawing to the
notes-list table in the corner.

DD/CD drawings constantly reference "see Keyed Note 3" — those notes
live in a corner box like:

    KEYED NOTES
    ───────────
    1. PROVIDE 1-1/4" CONDUIT FROM MDF TO IDF, MIN 36" RADIUS BENDS.
    2. COORDINATE MOUNTING HEIGHT WITH ARCHITECTURAL DRAWINGS.
    3. PROVIDE 1" CONDUIT FOR HORN/STROBE, MIN 36" RADIUS BENDS.
    4. CARD READER TO BE COMPATIBLE WITH EXISTING SYSTEM.
    5. ...

And on the drawing body, numbered callout markers (often inside a
triangle or hexagon) reference the note. Today's parser captures
the notes text but doesn't link callouts to notes — so a PM loses
critical scope-clarification context.

This module:

1. Detects KEYED-NOTES boxes by header pattern + tabular row layout
2. Parses numbered rows into a {number → note_text} map
3. Detects callout markers on the drawing (small numbered shapes
   inside triangles / hexagons / circles)
4. Emits a typed atom per (callout location → note text) pair

Deterministic. No LLM. The pipeline is text-blocks-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence


# Header patterns that indicate a keyed-notes box.
_KEYED_NOTES_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bKEYED\s+NOTES?\b", re.IGNORECASE),
    re.compile(r"\bDRAWING\s+NOTES?\b", re.IGNORECASE),
    re.compile(r"\bSHEET\s+NOTES?\b", re.IGNORECASE),
    re.compile(r"\bGENERAL\s+NOTES?\b", re.IGNORECASE),
)

# Pattern for numbered note rows.
_NUMBERED_NOTE_PATTERN = re.compile(
    r"^\s*\(?(\d{1,2})[\.\)]\s*(.+)$"
)


@dataclass(frozen=True)
class KeyedNote:
    """One numbered note from a keyed-notes box."""

    page_index: int
    note_number: int
    note_text: str
    box_bbox_pdf: tuple[float, float, float, float] | None = None
    header_text: str = ""


@dataclass(frozen=True)
class KeyedNotesBox:
    """The full keyed-notes box on a page."""

    page_index: int
    header_text: str
    bbox_pdf: tuple[float, float, float, float]
    notes: tuple[KeyedNote, ...]


@dataclass(frozen=True)
class CalloutMatch:
    """A callout-marker → keyed-note link."""

    page_index: int
    note_number: int
    note_text: str
    callout_bbox_pdf: tuple[float, float, float, float]
    confidence: float


# ── Helpers ──────────────────────────────────────────────────────


def _block_text(b: Any) -> str:
    text = getattr(b, "text", None)
    if text is None and isinstance(b, dict):
        text = b.get("text")
    return str(text or "").strip()


def _block_bbox(b: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(b, "bbox", None)
    if bbox is None and isinstance(b, dict):
        bbox = b.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None


def _is_keyed_notes_header(text: str) -> bool:
    for p in _KEYED_NOTES_HEADER_PATTERNS:
        if p.search(text):
            return True
    return False


# ── Box detection ────────────────────────────────────────────────


def locate_keyed_notes_boxes(
    *,
    page_index: int,
    blocks: Sequence[Any],
) -> list[KeyedNotesBox]:
    """Find every keyed-notes box on a page.

    A box is the union bbox of the header + all numbered rows below
    that header (until a clear gap or a different header).
    """
    if not blocks:
        return []

    # Sort blocks top-to-bottom, left-to-right
    sorted_blocks = sorted(
        blocks,
        key=lambda b: ((_block_bbox(b) or (0.0,) * 4)[1], (_block_bbox(b) or (0.0,) * 4)[0]),
    )

    boxes: list[KeyedNotesBox] = []
    header_indices: list[int] = []
    for i, b in enumerate(sorted_blocks):
        text = _block_text(b)
        if not text or len(text) > 80:
            continue
        if _is_keyed_notes_header(text):
            header_indices.append(i)

    for header_idx in header_indices:
        header_block = sorted_blocks[header_idx]
        header_bbox = _block_bbox(header_block)
        header_text = _block_text(header_block)
        if header_bbox is None:
            continue

        # Look downward for numbered rows within a reasonable column band
        header_x0, header_y0, header_x1, header_y1 = header_bbox
        column_band_x0 = header_x0 - 12.0
        column_band_x1 = header_x1 + 200.0            # notes may extend right of header

        collected_notes: list[KeyedNote] = []
        box_x0, box_y0 = header_x0, header_y0
        box_x1, box_y1 = header_x1, header_y1
        last_y1 = header_y1
        gap_threshold = 80.0                           # > this px gap ends the box

        for j in range(header_idx + 1, len(sorted_blocks)):
            other = sorted_blocks[j]
            other_text = _block_text(other)
            other_bbox = _block_bbox(other)
            if not other_text or other_bbox is None:
                continue
            ox0, oy0, ox1, oy1 = other_bbox

            # Below the header, in roughly the same column?
            if oy0 < last_y1 - 4.0:
                continue
            if ox0 < column_band_x0 or ox0 > column_band_x1:
                continue
            if oy0 - last_y1 > gap_threshold:
                break
            # Another notes header ends this box
            if _is_keyed_notes_header(other_text):
                break

            m = _NUMBERED_NOTE_PATTERN.match(other_text)
            if m:
                number = int(m.group(1))
                note_text = m.group(2).strip()
                collected_notes.append(
                    KeyedNote(
                        page_index=page_index,
                        note_number=number,
                        note_text=note_text,
                        header_text=header_text,
                    )
                )
                last_y1 = oy1
                box_x0 = min(box_x0, ox0)
                box_y1 = max(box_y1, oy1)
                box_x1 = max(box_x1, ox1)
            else:
                # Some legend lines continue from the previous note —
                # attach to last note if we have one.
                if collected_notes and oy0 < last_y1 + 24.0:
                    last = collected_notes[-1]
                    new_text = (last.note_text + " " + other_text).strip()
                    collected_notes[-1] = KeyedNote(
                        page_index=last.page_index,
                        note_number=last.note_number,
                        note_text=new_text,
                        header_text=last.header_text,
                    )
                    last_y1 = oy1
                    box_y1 = max(box_y1, oy1)

        if collected_notes:
            boxes.append(
                KeyedNotesBox(
                    page_index=page_index,
                    header_text=header_text,
                    bbox_pdf=(box_x0, box_y0, box_x1, box_y1),
                    notes=tuple(collected_notes),
                )
            )

    return boxes


# ── Callout-marker detection ─────────────────────────────────────


def _is_callout_marker_text(text: str) -> int | None:
    """Standalone "1" / "2" / "(3)" / "<4>" markers used as callouts
    on the drawing. Returns the number if it looks like a callout."""
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) > 6:
        return None
    m = re.match(r"^[\(\<\[\{]?\s*(\d{1,2})\s*[\)\>\]\}]?$", stripped)
    if not m:
        return None
    return int(m.group(1))


def link_callouts_to_notes(
    *,
    page_index: int,
    blocks: Sequence[Any],
    notes_box: KeyedNotesBox,
) -> list[CalloutMatch]:
    """For every numbered-marker block on the page, link it to a
    keyed note by matching the number.

    Markers INSIDE the notes-box itself are ignored (those are the
    row labels in the notes table).
    """
    out: list[CalloutMatch] = []
    note_map = {n.note_number: n for n in notes_box.notes}
    box_bbox = notes_box.bbox_pdf
    for b in blocks:
        bbox = _block_bbox(b)
        text = _block_text(b)
        if bbox is None or not text:
            continue
        # Skip if the marker is inside the notes box (it's a row label)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if box_bbox[0] <= cx <= box_bbox[2] and box_bbox[1] <= cy <= box_bbox[3]:
            continue
        n = _is_callout_marker_text(text)
        if n is None:
            continue
        note = note_map.get(n)
        if note is None:
            continue
        out.append(
            CalloutMatch(
                page_index=page_index,
                note_number=n,
                note_text=note.note_text,
                callout_bbox_pdf=bbox,
                confidence=0.75,
            )
        )
    return out


__all__ = [
    "CalloutMatch",
    "KeyedNote",
    "KeyedNotesBox",
    "link_callouts_to_notes",
    "locate_keyed_notes_boxes",
]
