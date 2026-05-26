"""Identify page regions where symbol detection should be suppressed.

A token like ``PTZ`` appearing in a body region is a real symbol
detection.  The same token appearing inside the title block ("PTZ
ROOM"), a keyed-notes block ("3. PTZ AND DOME CAMERAS PER…"), or a
schedule cell ("PTZ-101 / 7' AFF / WALL-MOUNT") is part of drawing
furniture, not a device on the floor.  This module returns
``(label, bbox)`` regions the symbol detector should exclude.

All detection is text-rule based and deterministic.  No LLM, no
classifier.  The detector returns regions sorted by (y0, x0, label).

Region kinds:

- ``title_block``   — bottom-right or right-edge region containing
  the sheet number, project name, scale, revision, designer.
- ``drawing_index`` — page-level index of sheets (``DRAWING INDEX``).
- ``keyed_notes``   — numbered notes block ("GENERAL NOTES", "KEYED
  NOTES", "SHEET NOTES").
- ``schedule``      — equipment/door/camera/fixture schedule table.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock


_NOTES_HEADER_RE = re.compile(
    r"\b(general\s+notes|keyed\s+notes|sheet\s+notes|note\s+legend|legend\s+notes|"
    r"installation\s+notes|construction\s+notes|fire\s+alarm\s+notes|"
    r"low\s+voltage\s+notes)\b",
    re.IGNORECASE,
)
_DRAWING_INDEX_HEADER_RE = re.compile(
    r"\b(drawing\s+index|sheet\s+index|sheet\s+list|index\s+of\s+drawings)\b",
    re.IGNORECASE,
)
_TITLE_BLOCK_TOKENS_RE = re.compile(
    r"\b(project(?:\s+name)?\s*[:#]?|sheet\s+title|sheet\s+number|drawn\s+by|"
    r"checked\s+by|drafter|checker|approved\s+by|scale|date|rev(?:ision)?\s*\d|"
    r"drawing\s+number|client|owner|architect|engineer)\b",
    re.IGNORECASE,
)
_SCHEDULE_HEADER_RE = re.compile(
    r"\b(\w+\s+)?schedule\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExclusionRegion:
    label: str
    bbox: tuple[float, float, float, float]


def _union_bbox(
    boxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _grow_block_region(
    seed: TextBlock,
    blocks: Sequence[TextBlock],
    *,
    max_y_gap: float = 36.0,
    max_x_drift: float = 240.0,
) -> tuple[float, float, float, float]:
    """Cluster blocks vertically below ``seed`` until a vertical gap or
    a horizontal drift exceeds the bound.  Used for notes / schedule
    regions where the header is followed by a column of rows.
    """
    region = seed.bbox
    last_bottom = seed.bbox[3]
    seed_left = seed.bbox[0]
    candidates = sorted(blocks, key=lambda b: b.bbox[1])
    for blk in candidates:
        if blk is seed:
            continue
        if blk.bbox[1] < seed.bbox[1] - 1.0:
            continue
        gap = blk.bbox[1] - last_bottom
        if gap > max_y_gap:
            break
        if abs(blk.bbox[0] - seed_left) > max_x_drift:
            continue
        region = (
            min(region[0], blk.bbox[0]),
            min(region[1], blk.bbox[1]),
            max(region[2], blk.bbox[2]),
            max(region[3], blk.bbox[3]),
        )
        last_bottom = max(last_bottom, blk.bbox[3])
    return region


def _title_block_region(
    blocks: Sequence[TextBlock],
    page_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Estimate the title-block region on a typical construction sheet.

    Heuristic: the title block is the rectangle in the lower 25% (or
    rightmost 25%) of the page that contains the bottom-right sheet
    number, project name, scale, etc.  We collect blocks that:

      - sit in the bottom band (y0 >= page_height * 0.75) OR the
        right edge (x0 >= page_width * 0.70), AND
      - either match one of the canonical title-block tokens or
        contain a sheet-number-shaped string (``[A-Z]+\\d+(?:\\.\\d+)?``).

    Returns ``None`` when no such region is found.
    """
    if not blocks:
        return None
    if page_bbox is None:
        # No page rect: fall back to the union of the lowest 25% of blocks.
        ys = sorted(b.bbox[1] for b in blocks)
        if not ys:
            return None
        cutoff = ys[int(len(ys) * 0.75)]
        bottom = [b for b in blocks if b.bbox[1] >= cutoff]
        if not bottom:
            return None
        return _union_bbox(b.bbox for b in bottom)
    _, py0, px1, py1 = page_bbox
    page_h = py1 - py0
    page_w = px1 - page_bbox[0]
    # Bottom 15% strip and rightmost 10% strip is where title-block
    # furniture really lives. Earlier the cutoffs were 0.75 and 0.70
    # and we OR'd them, which led one stray sheet-number match in the
    # top-right corner to drag the entire title-block region across
    # the drawing body. Use tighter cutoffs AND require corroborating
    # signals (token match or two strips overlapping) to keep blocks.
    band_y = py0 + 0.85 * page_h
    band_x = page_bbox[0] + 0.90 * page_w
    keep: list[tuple[float, float, float, float]] = []
    for blk in blocks:
        in_bottom_band = blk.bbox[1] >= band_y
        in_right_band = blk.bbox[0] >= band_x
        rotated = bool(getattr(blk, "rotation_deg", 0))
        # Rotated text in any region is almost always title-block /
        # border text in construction drawings.  Even when it lives
        # outside the bottom-right band, treat it as title-block
        # furniture so symbol detection doesn't consume it.
        if rotated:
            keep.append(blk.bbox)
            continue
        if not (in_bottom_band or in_right_band):
            continue
        text = (blk.text or "").strip()
        # Right-margin disclaimer prose ("Owner and Architect ...") is
        # often LONG (>40 chars) and contains title-block keywords by
        # accident. Restrict token-hit qualification to SHORT blocks
        # — real title-block labels are concise ("Date", "Drawn by",
        # "Project Name").
        token_hit = bool(_TITLE_BLOCK_TOKENS_RE.search(text)) and len(text) <= 40
        sheet_hit = bool(re.search(r"\b[A-Z]{1,3}\d+(?:\.\d+)?\b", text))
        if token_hit and (in_bottom_band or in_right_band):
            keep.append(blk.bbox)
            continue
        if in_bottom_band and sheet_hit:
            keep.append(blk.bbox)
            continue
        if in_bottom_band and in_right_band:
            # Inner intersection of bottom + right is the title block
            # corner by convention.
            keep.append(blk.bbox)
    if not keep:
        return None
    # Clamp the title-block region's top to band_y. Any block above
    # band_y that landed in `keep` (e.g. a "Date" column header on a
    # tall right-margin stamp) is a STRUCTURAL marker, but unioning
    # its bbox with the bottom-strip blocks creates a title-block
    # region that swallows the drawing body. The drawing body is
    # ALWAYS above band_y, so clamping the region top there is safe:
    # title-block furniture above band_y is rare, and when it
    # exists it's covered by separate right-strip handling in
    # detect_exclusion_zones (drawing_index / keyed_notes).
    union = _union_bbox(keep)
    if union is None:
        return None
    x0, y0, x1, y1 = union
    if y0 < band_y:
        y0 = band_y
    if x0 < page_bbox[0]:
        x0 = page_bbox[0]
    if y1 > py1:
        y1 = py1
    if x1 > px1:
        x1 = px1
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def detect_exclusion_zones(
    blocks: Sequence[TextBlock],
    page_bbox: tuple[float, float, float, float] | None = None,
) -> list[ExclusionRegion]:
    """Return the deterministic list of regions where symbol detection
    should be suppressed.
    """
    regions: list[ExclusionRegion] = []

    tb = _title_block_region(blocks, page_bbox)
    if tb is not None:
        regions.append(ExclusionRegion(label="title_block", bbox=tb))

    for blk in blocks:
        if _DRAWING_INDEX_HEADER_RE.search(blk.text):
            region = _grow_block_region(blk, blocks, max_y_gap=72.0, max_x_drift=320.0)
            regions.append(ExclusionRegion(label="drawing_index", bbox=region))
            break  # one index per page

    for blk in blocks:
        if _NOTES_HEADER_RE.search(blk.text):
            region = _grow_block_region(blk, blocks, max_y_gap=36.0, max_x_drift=300.0)
            regions.append(ExclusionRegion(label="keyed_notes", bbox=region))

    seen_schedules: set[tuple[float, float, float, float]] = set()
    for blk in blocks:
        if _SCHEDULE_HEADER_RE.search(blk.text):
            # Don't collide with the legend/notes/index headers.
            t = blk.text.lower()
            if "drawing index" in t or "sheet index" in t:
                continue
            if "notes" in t or "legend" in t:
                continue
            region = _grow_block_region(blk, blocks, max_y_gap=24.0, max_x_drift=360.0)
            if region in seen_schedules:
                continue
            seen_schedules.add(region)
            regions.append(ExclusionRegion(label="schedule", bbox=region))

    regions.sort(key=lambda r: (round(r.bbox[1], 2), round(r.bbox[0], 2), r.label))
    return regions
