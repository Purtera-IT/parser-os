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
# elsewhere on the page from polluting the keynote table. Each of these
# is a SIBLING block — KEYED NOTES and CABLE ZONING NOTES can coexist
# on one page without one ending the other.
_KEYNOTE_HEADERS = (
    "KEYED NOTES",
    "KEY NOTES",
    "PLAN NOTES",
    "GENERAL NOTES",
    "DRAWING NOTES",
    "CABLE ZONING NOTES",
)

# A "block-end" sentinel — stop parsing keynote entries when we hit a
# header that looks like a different section entirely.
_BLOCK_END_HEADERS = (
    "SHEET NUMBER",
    "DRAWING TITLE",
    "REVISIONS",
    "COOPER CARRY",
    "CONSULTANTS",
    "SCOPE DOCUMENTS",
    "THIS DRAWING IS AN INSTRUMENT",
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


def _find_header_bbox(
    page_words: list[PdfWord], header_text: str
) -> tuple[float, float, float, float] | None:
    """Locate ``header_text`` in ``page_words`` and return its bbox.

    Matches when the words of ``header_text`` appear consecutively on the
    same baseline. Returns the union bbox of those words, or ``None``.
    """
    target = [t for t in header_text.upper().split() if t]
    if not target:
        return None
    # Sort by y-center then x for consecutive scanning.
    by_yx = sorted(page_words, key=lambda w: ((w.y0 + w.y1) / 2, w.x0))
    n = len(target)
    for i in range(len(by_yx) - n + 1):
        seg = by_yx[i : i + n]
        if any((w.text or "").strip().upper() != target[k] for k, w in enumerate(seg)):
            continue
        # Same-baseline check.
        ys = [(w.y0 + w.y1) / 2 for w in seg]
        if max(ys) - min(ys) > 6:
            continue
        return (
            min(w.x0 for w in seg),
            min(w.y0 for w in seg),
            max(w.x1 for w in seg),
            max(w.y1 for w in seg),
        )
    return None


def _phrases_in_region(
    page_words: list[PdfWord],
    region: tuple[float, float, float, float],
    y_tolerance: float = 5.0,
    max_gap: float | None = None,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return ``(text, bbox)`` phrases inside ``region`` (x0,y0,x1,y1).

    Words are grouped into phrases by same-baseline + adjacent-x.
    """
    rx0, ry0, rx1, ry1 = region
    inside = [
        w for w in page_words
        if w.x0 >= rx0 - 1 and w.x1 <= rx1 + 1 and w.y0 >= ry0 - 1 and w.y1 <= ry1 + 1
    ]
    if not inside:
        return []
    inside.sort(key=lambda w: ((w.y0 + w.y1) / 2, w.x0))
    if max_gap is None:
        # Default gap = 2× median word height.
        heights = sorted(w.y1 - w.y0 for w in inside)
        med_h = heights[len(heights) // 2] if heights else 10.0
        max_gap = max(6.0, 2.0 * med_h)
    phrases: list[tuple[str, tuple[float, float, float, float]]] = []
    cur_text: list[str] = [inside[0].text]
    cur_box = (inside[0].x0, inside[0].y0, inside[0].x1, inside[0].y1)
    cur_yc = (inside[0].y0 + inside[0].y1) / 2
    prev_right = inside[0].x1
    for w in inside[1:]:
        wyc = (w.y0 + w.y1) / 2
        if abs(wyc - cur_yc) <= y_tolerance and (w.x0 - prev_right) <= max_gap and w.x0 >= cur_box[0]:
            cur_text.append(w.text)
            cur_box = (cur_box[0], min(cur_box[1], w.y0), w.x1, max(cur_box[3], w.y1))
            prev_right = w.x1
            continue
        phrases.append((" ".join(cur_text).strip(), cur_box))
        cur_text = [w.text]
        cur_box = (w.x0, w.y0, w.x1, w.y1)
        cur_yc = wyc
        prev_right = w.x1
    phrases.append((" ".join(cur_text).strip(), cur_box))
    return phrases


def _looks_like_keynote_body(text: str) -> bool:
    """True when a phrase looks like a keynote body (multi-word, alpha)."""
    t = text.strip()
    if len(t) < 8 or t[0].isdigit():
        return False
    if " " not in t:
        return False
    if not any(c.isalpha() for c in t):
        return False
    return True


def parse_keynote_table_spatial(
    *,
    page_index: int,
    page_text: str,
    page_words: list[PdfWord],
) -> KeynoteTable:
    """Spatial-pairing keynote parser.

    Unlike the sequence-based :func:`parse_keynote_table`, this version
    uses bbox geometry to pair numbers with descriptions:

    1. Find each ``KEYED NOTES`` / ``CABLE ZONING NOTES`` header bbox.
    2. For each header, define a "block region" extending downward
       until the next header or a hard block-end header is hit.
    3. Within the block region, identify isolated digit words AND
       multi-word body-shaped phrases.
    4. For each digit, pair it with the description-phrase on the same
       baseline whose left edge is closest to the digit's right edge.
       Numbers without a matching same-row phrase are skipped.

    Handles multi-sibling keynote sections (KEYED NOTES AND CABLE ZONING
    NOTES) — both blocks contribute entries to the same KeynoteTable.

    Falls back to the sequence-based parser when no header bbox can be
    located in ``page_words`` (e.g. when the header sits in a graphical
    element rather than native text).
    """
    table = KeynoteTable(page_index=page_index)
    if not page_words:
        return parse_keynote_table(page_index=page_index, page_text=page_text)

    # 1. Locate each known header bbox.
    found_headers: list[tuple[str, tuple[float, float, float, float]]] = []
    for hdr in _KEYNOTE_HEADERS:
        bb = _find_header_bbox(page_words, hdr)
        if bb is not None:
            found_headers.append((hdr, bb))
            if hdr.upper() not in table.blocks:
                table.blocks.append(hdr.upper())
    if not found_headers:
        # No header found — fall back to text-based parse so we still
        # capture inline-numbered "1. body" style.
        return parse_keynote_table(page_index=page_index, page_text=page_text)

    # 2. Hard block-end Y positions (where the keynote region must end).
    block_end_ys: list[float] = []
    for end_hdr in _BLOCK_END_HEADERS:
        bb = _find_header_bbox(page_words, end_hdr)
        if bb is not None:
            block_end_ys.append(bb[1])  # y0 of the end header

    # 3. For each header, parse its region.
    sorted_headers = sorted(found_headers, key=lambda h: h[1][1])  # by y0
    for idx, (header_text, hbbox) in enumerate(sorted_headers):
        hx0, hy0, hx1, hy1 = hbbox
        # Region: from below the header to the next header / block-end.
        next_starts = [h[1][1] for h in sorted_headers[idx + 1 :]]
        possible_ends = [y for y in next_starts + block_end_ys if y > hy1 + 2]
        region_bottom = min(possible_ends) if possible_ends else hy1 + 1500.0
        # Extend the region horizontally to accommodate columns to the
        # right of the number column. We use a generous X-window relative
        # to the header position.
        region = (max(0.0, hx0 - 30.0), hy1 + 1.0, hx1 + 1500.0, region_bottom)

        # Collect words in the region.
        region_words = [
            w for w in page_words
            if w.x0 >= region[0] and w.x1 <= region[2]
            and w.y0 >= region[1] and w.y1 <= region[3]
        ]
        if not region_words:
            continue

        # Isolated digit "number" words — 1–3 chars, pure digits.
        digit_words = []
        for w in region_words:
            t = (w.text or "").strip().rstrip(".:)")
            if t.isdigit() and 1 <= len(t) <= 3:
                digit_words.append((t, w))

        # Body phrases — multi-word descriptions inside the region.
        phrases = _phrases_in_region(page_words, region)
        body_phrases = [(t, b) for (t, b) in phrases if _looks_like_keynote_body(t)]

        # 4. Pair each digit with the nearest body phrase on the same row.
        for number, dw in digit_words:
            d_yc = (dw.y0 + dw.y1) / 2
            d_h = max(1.0, dw.y1 - dw.y0)
            # Row tolerance proportional to digit height (digits ~6-12 pt).
            row_tol = max(8.0, d_h * 1.2)
            d_right = dw.x1
            best: tuple[float, str] | None = None
            for ph_text, ph_bbox in body_phrases:
                p_yc = (ph_bbox[1] + ph_bbox[3]) / 2
                if abs(p_yc - d_yc) > row_tol:
                    continue
                if ph_bbox[0] < d_right - 1.0:
                    continue
                dist = ph_bbox[0] - d_right
                # Plus a y-distance penalty so a precisely-aligned row wins.
                cost = dist + abs(p_yc - d_yc) * 1.5
                if best is None or cost < best[0]:
                    best = (cost, ph_text)
            if best is not None and number not in table.notes:
                # Clean trailing punctuation/whitespace.
                clean = re.sub(r"\s+", " ", best[1]).strip(" .,;:")
                table.notes[number] = clean

    # If the spatial pass left the table empty (e.g. degenerate layout),
    # fall back to the sequence-based parser as a safety net.
    if not table.notes:
        fallback = parse_keynote_table(page_index=page_index, page_text=page_text)
        table.notes.update(fallback.notes)

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


def find_keynote_ref_bbox(
    *,
    bbox: BBox,
    page_words: list[PdfWord],
    keynote_number: str,
    radius_pt: float = 40.0,
) -> BBox | None:
    """Locate the bbox of the nearest digit word matching ``keynote_number``.

    Used by the QA overlay to draw a colored marker around the *callout*
    on the plan (the small "4" sitting next to a WN symbol) — distinct
    from the device symbol itself.
    """
    cx = (bbox.x0 + bbox.x1) / 2.0
    cy = (bbox.y0 + bbox.y1) / 2.0
    target = keynote_number.strip()
    best: tuple[float, PdfWord] | None = None
    for w in page_words:
        text = (w.text or "").strip().strip(".,;:")
        if text != target:
            continue
        wx = (w.x0 + w.x1) / 2.0
        wy = (w.y0 + w.y1) / 2.0
        # Skip if inside the candidate's own bbox.
        if bbox.x0 <= wx <= bbox.x1 and bbox.y0 <= wy <= bbox.y1:
            continue
        d = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
        if d > radius_pt:
            continue
        if best is None or d < best[0]:
            best = (d, w)
    if best is None:
        return None
    w = best[1]
    return BBox(x0=w.x0, y0=w.y0, x1=w.x1, y1=w.y1, coord_space="pdf_pt")


__all__ = [
    "Keynote",
    "KeynoteTable",
    "parse_keynote_table",
    "parse_keynote_table_spatial",
    "find_keynote_refs_near",
    "find_keynote_ref_bbox",
    "resolve_keynote",
]
