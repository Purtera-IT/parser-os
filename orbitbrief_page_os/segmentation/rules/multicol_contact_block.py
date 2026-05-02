"""Universal rule: multi-column contact/team block detection.

Symptom that motivates this rule
---------------------------------
Cover sheets and general-notes sheets often contain a "PROJECT TEAM" or
"CONSULTANTS" section laid out as::

    PROJECT TEAM                        ← wide single-line heading, larger font
    ARCHITECTURAL  CIVIL  STRUCTURAL … ← N ≥ 3 short label columns, same font
    Verne Reimer   WSP    WSP         ← body paragraphs under each label
    109-374 River… 237-4th Ave…

There are NO drawn rectangle borders anywhere in this area — it is pure
free-floating text.  The raster contour detector finds nothing; the
text-section detector does not emit boxes here because the section lacks
a border or underline.

Two-stage fingerprint (universal, geometry-only)
-------------------------------------------------
Stage 1 — column-header row detection
    Find groups of N ≥ 3 single-line text blocks that share the same y0
    (within one line-height), carry the same font size, are spread across
    ≥ 35% of the usable page width, and whose adjacent x-gaps are all
    ≥ 8% of page width.  Each block's text must be a SHORT LABEL (≤ 35
    chars, ≤ 3 words, no sentence-ending punctuation — this excludes
    prose sentences that happen to share a y-coordinate).

Stage 2 — heading + body verification
    For each candidate column-header row:
    • Find the heading: a single-line block just above (within 4× the
      column-header line-height), larger font (≥ 1.25×), starting near
      the left edge of the column group.
    • Find body blocks: multi-line blocks below the column headers whose
      x0 aligns with one of the column header x0 values (within 15 pts).

Emitted boxes
-------------
One ORANGE ``mccol_group_*`` wrapper covering the full group (heading top
→ body bottom, full column span).

One CYAN ``mccol_hdr_*`` box per column header label.

One BLUE ``mccol_heading_*`` box for the wide heading line.

One ORANGE ``mccol_body_*`` box per column body block cluster.

These are synthetic boxes — no raster contour needed.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

# ── tuneable thresholds ──────────────────────────────────────────────────────

# Minimum number of label columns to qualify as a multi-col contact block
MIN_COLS: int = 3

# Column header label constraints
LABEL_MAX_CHARS: int = 35
LABEL_MAX_WORDS: int = 4  # e.g. "INTERIOR DESIGN" = 2, "MECHANICAL" = 1
LABEL_MIN_CHARS: int = 3  # exclude single letters/numbers used as section bullets
LABEL_MIN_FONT_SIZE: float = 6.0  # exclude tiny legend table headers (e.g. 2.4pt CMP/COAX)
# Characters that indicate prose rather than a label.
# Note: trailing colon is a common discipline-header convention (ARCHITECTURAL:)
# and must NOT be treated as prose punctuation.
_PROSE_PUNCT: re.Pattern = re.compile(r"[.,;!?]")

# Column layout constraints (fraction of usable page width)
MIN_CONTENT_SPAN_FRAC: float = 0.35   # columns must span ≥ 35% of page
MIN_COL_GAP_FRAC: float = 0.07        # adjacent columns ≥ 7% of page apart

# Heading constraints
HEADING_MIN_RATIO: float = 1.20       # heading font ≥ 1.2× col-label font
HEADING_MAX_Y_ABOVE: float = 4.0      # heading ≤ 4× col-label line-height above

# Body block alignment tolerance (PDF pts)
BODY_X_TOL: float = 20.0             # body block x0 within 20 pts of col header x0
BODY_MAX_Y_BELOW: float = 700.0      # body blocks within 700 pts below col header row (covers long drawing indexes)

# Y-bucket size for grouping blocks at the same row (PDF pts)
Y_BUCKET: float = 6.0

# Exclude right-margin title block (fraction of page width)
TITLE_BLOCK_X_FRAC: float = 0.88


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class _ColHeaderBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    font_size: float


@dataclass
class _MultiColGroup:
    heading_bbox: Optional[tuple[float, float, float, float]]  # (x0,y0,x1,y1) PDF pts
    heading_text: str
    col_headers: list[_ColHeaderBlock]
    body_bboxes: list[tuple[float, float, float, float]]  # one per col, body only, PDF pts
    col_bboxes: list[tuple[float, float, float, float]]   # one per col, hdr+body combined, PDF pts
    group_bbox: tuple[float, float, float, float]          # full group, PDF pts
    title_bbox: Optional[tuple[float, float, float, float]]  # big page title above heading, PDF pts
    title_text: str


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_size(block: dict) -> float:
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            return float(span.get("size", 0))
    return 0.0


def _get_text(block: dict) -> str:
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return " ".join(parts).strip()


def _is_label(text: str) -> bool:
    """True if text looks like a short discipline/category label, not prose."""
    t = text.strip()
    if not t:
        return False
    if len(t) < LABEL_MIN_CHARS:
        return False
    if len(t) > LABEL_MAX_CHARS:
        return False
    if _PROSE_PUNCT.search(t):
        return False
    words = t.split()
    if len(words) > LABEL_MAX_WORDS:
        return False
    return True


def _get_size_from_bbox(
    bbox: tuple[float, float, float, float],
    blocks: list[dict],
) -> float:
    """Look up font size of the block matching bbox (for title comparison)."""
    bx0, by0, bx1, by1 = bbox
    for b in blocks:
        if (abs(b["bbox"][0] - bx0) < 2 and abs(b["bbox"][1] - by0) < 2):
            return _get_size(b)
    return 0.0


# ── main detector ────────────────────────────────────────────────────────────

def find_multicol_contact_blocks(
    pdf_path: str | Path,
    page_index: int = 0,
) -> list[_MultiColGroup]:
    """Return all multi-column contact/team groups found on the page.

    Each result contains the full geometry in PDF-space points ready for
    the caller to scale to pixel coordinates.
    """
    if fitz is None:
        return []

    doc = fitz.open(str(pdf_path))
    if page_index >= len(doc):
        return []
    page = doc[page_index]
    W: float = page.rect.width

    # Usable content area: exclude right-margin title block
    max_x: float = W * TITLE_BLOCK_X_FRAC

    raw_blocks = page.get_text("dict").get("blocks", [])
    # Keep only text blocks inside the usable area
    text_blocks = [
        b for b in raw_blocks
        if b.get("type") == 0 and b["bbox"][2] < max_x
    ]

    # ── Stage 1: find col-header rows ────────────────────────────────────────
    # Build candidate "label items" from two sources:
    # A) Whole single-line blocks (PROJECT TEAM / ELECTRICAL: style)
    # B) Individual spans inside multi-line blocks at distinct x-positions
    #    (ARCHITECTURAL: / STRUCTURAL: / MECHANICAL: merged into one block)
    # Strip trailing colon — discipline headers often end with ':'.

    @dataclass
    class _LabelItem:
        bbox: tuple
        text: str
        size: float
        source_block: dict

    label_items: list[_LabelItem] = []

    for b in text_blocks:
        lines = b.get("lines", [])
        if len(lines) == 1:
            t = _get_text(b).rstrip(":")
            s = _get_size(b)
            if t and s > 0:
                label_items.append(_LabelItem(
                    bbox=tuple(b["bbox"]), text=t, size=s, source_block=b,
                ))
        else:
            # Multi-line block: treat each span as a potential col-header item
            for line in lines:
                for span in line.get("spans", []):
                    t = span.get("text", "").strip().rstrip(":")
                    s = float(span.get("size", 0))
                    if not t or s <= 0:
                        continue
                    sb = span["bbox"]
                    label_items.append(_LabelItem(
                        bbox=(sb[0], sb[1], sb[2], sb[3]),
                        text=t, size=s, source_block=b,
                    ))

    # Bucket items by y0
    y_buckets: dict[int, list[_LabelItem]] = defaultdict(list)
    for item in label_items:
        y_key = int(item.bbox[1] / Y_BUCKET)
        y_buckets[y_key].append(item)

    groups: list[_MultiColGroup] = []

    for y_key, row_items in y_buckets.items():
        if len(row_items) < MIN_COLS:
            continue

        # All items must share the same font size (within 1 pt)
        sizes = [item.size for item in row_items]
        if not sizes or max(sizes) - min(sizes) > 1.0:
            continue
        col_size = sizes[0]
        if col_size <= 0:
            continue

        # Reject tiny font sizes (legend table micro-headers)
        if col_size < LABEL_MIN_FONT_SIZE:
            continue

        # All texts must be short labels
        texts = [item.text for item in row_items]
        if not all(_is_label(t) for t in texts):
            continue

        # Require all label texts to be unique — repeated labels (e.g. WSP×4)
        # indicate body text rows, not a multi-discipline column header row
        if len(set(texts)) < len(texts):
            continue

        # Sort by x0
        row_items.sort(key=lambda item: item.bbox[0])
        xs = [item.bbox[0] for item in row_items]
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        content_span = xs[-1] - xs[0]

        # Content must span enough of the page
        if content_span < W * MIN_CONTENT_SPAN_FRAC:
            continue

        # Adjacent columns must be far enough apart
        if min(gaps) < W * MIN_COL_GAP_FRAC:
            continue

        # ── Stage 2: find heading above col-header row ───────────────────────
        row_y0 = row_items[0].bbox[1]
        row_x0 = row_items[0].bbox[0]
        row_x1 = row_items[-1].bbox[2]

        heading_block: Optional[dict] = None
        for b in text_blocks:
            if len(b.get("lines", [])) != 1:
                continue
            size = _get_size(b)
            if size < col_size * HEADING_MIN_RATIO:
                continue
            bx0, by0, bx1, by1 = b["bbox"]
            if by1 > row_y0:
                continue
            if row_y0 - by1 > col_size * HEADING_MAX_Y_ABOVE:
                continue
            if bx0 > row_x0 + col_size * 3:
                continue
            heading_block = b
            break

        # ── Find body blocks per column ──────────────────────────────────────
        col_headers: list[_ColHeaderBlock] = []
        body_bboxes: list[tuple[float, float, float, float]] = []
        col_bboxes: list[tuple[float, float, float, float]] = []

        # Cap body search at next section heading
        row_y_bottom = row_items[0].bbox[3]
        next_section_y: float = row_y_bottom + BODY_MAX_Y_BELOW
        for b in text_blocks:
            bx0, by0, bx1, by1 = b["bbox"]
            if by0 < row_y_bottom + col_size * 2:
                continue
            if bx0 > row_x0 + col_size * 3:
                continue
            if len(b.get("lines", [])) == 1 and _get_size(b) >= col_size * 1.1:
                next_section_y = min(next_section_y, by0)

        for item in row_items:
            cx0, cy0, cx1, cy1 = item.bbox
            ch = _ColHeaderBlock(
                x0=cx0, y0=cy0, x1=cx1, y1=cy1,
                text=item.text,
                font_size=col_size,
            )
            col_headers.append(ch)

            # Body blocks: x0-aligned, below header, capped at next section
            col_body = [
                b for b in text_blocks
                if abs(b["bbox"][0] - cx0) <= BODY_X_TOL
                and b["bbox"][1] > cy1
                and b["bbox"][1] < next_section_y
            ]
            if col_body:
                col_body.sort(key=lambda b: b["bbox"][1])
                bx0_all = min(b["bbox"][0] for b in col_body)
                by0_all = col_body[0]["bbox"][1]
                bx1_all = max(b["bbox"][2] for b in col_body)
                by1_all = max(b["bbox"][3] for b in col_body)
                body_bboxes.append((bx0_all, by0_all, bx1_all, by1_all))
                # Per-column combined box: from header top to body bottom
                col_bboxes.append((
                    min(cx0, bx0_all),
                    cy0,
                    max(cx1, bx1_all),
                    by1_all,
                ))
            else:
                body_bboxes.append((cx0, cy1, cx1, cy1))
                col_bboxes.append((cx0, cy0, cx1, cy1))

        # ── Compute group bbox ───────────────────────────────────────────────
        group_x0 = min(ch.x0 for ch in col_headers)
        group_x1 = max(
            max(ch.x1 for ch in col_headers),
            max((bb[2] for bb in col_bboxes), default=0.0),
        )
        if heading_block:
            group_y0 = heading_block["bbox"][1]
            heading_bbox: Optional[tuple] = tuple(heading_block["bbox"])
            heading_text = _get_text(heading_block)
        else:
            group_y0 = col_headers[0].y0
            heading_bbox = None
            heading_text = ""

        group_y1 = max(
            max(ch.y1 for ch in col_headers),
            max((bb[3] for bb in col_bboxes), default=0.0),
        )

        # ── Find big page title above the heading ────────────────────────────
        # A single-line or two-line block with a significantly larger font
        # (≥ 2× heading font or ≥ 1.5× if heading is absent), sitting above
        # the heading (or col-header row), spanning a wide x-range.
        title_bbox: Optional[tuple] = None
        title_text: str = ""
        heading_size = _get_size(heading_block) if heading_block else col_size
        search_y_top = group_y0  # look above the group
        for b in text_blocks:
            bx0, by0, bx1, by1 = b["bbox"]
            if by1 > search_y_top:
                continue  # must be above group
            size = _get_size(b)
            if size < heading_size * 1.5:
                continue  # must be noticeably larger
            bw = bx1 - bx0
            if bw < W * 0.25:
                continue  # must span a decent width
            # Take the largest-font block above the group
            if title_bbox is None or size > _get_size_from_bbox(title_bbox, text_blocks):
                title_bbox = (bx0, by0, bx1, by1)
                title_text = _get_text(b)

        groups.append(_MultiColGroup(
            heading_bbox=heading_bbox,
            heading_text=heading_text,
            col_headers=col_headers,
            body_bboxes=body_bboxes,
            col_bboxes=col_bboxes,
            group_bbox=(group_x0, group_y0, group_x1, group_y1),
            title_bbox=title_bbox,
            title_text=title_text,
        ))

    doc.close()
    return groups
