"""Parse construction schedule tables and join rows to detections.

Real construction sheets often carry per-device schedules — Camera
Schedule, Door Schedule, Equipment Schedule, Fixture Schedule, Panel
Schedule — that list every device by tag with model / manufacturer /
mounting / power / NIC marker / remarks.  Joining a schedule row to
its corresponding symbol detection multiplies what the parser knows
about the device: a ``CR`` detection becomes a CR identified as
``CR-101 / HID Signo 20 / 48" AFF / wall mount``.

Parsing is text-rule + bbox-clustering and entirely deterministic.

The detector:

1. Finds a TextBlock whose text matches a known schedule header
   (``CAMERA SCHEDULE``, ``DOOR SCHEDULE``, etc.).
2. Grows the schedule region downward until a 36 pt vertical gap.
3. Detects the column-header row inside that region and builds a
   typed column map (reusing the legend_parser's
   ``_classify_header_cell`` for header tokens).
4. Parses every subsequent row into a ``ScheduleRow`` record whose
   ``tag`` is the value in the leftmost (typically TAG / ID / MARK)
   column.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.core.ids import stable_id
from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock
from orbitbrief_page_os.segmentation.schematic.legend_parser import (
    _build_column_map,
    _classify_header_cell,
    _cluster_rows,
    _row_to_column_cells,
)


_SCHEDULE_HEADER_RE = re.compile(
    r"\b(?P<kind>camera|door|equipment|fixture|panel|device|access\s+control|wire|circuit|"
    r"speaker|av|low\s+voltage|cable|jack|fire\s+alarm|reader)\s+schedule\b",
    re.IGNORECASE,
)
_TAG_HEADER_TOKENS = ("tag", "id", "mark", "device id", "device tag", "fixture id", "device", "no", "no.")


@dataclass(frozen=True)
class ScheduleRow:
    row_id: str
    page_index: int
    sheet_number: str | None
    schedule_kind: str
    tag: str
    bbox: tuple[float, float, float, float]
    fields: tuple[tuple[str, str], ...]
    confidence: float = 0.85

    def fields_dict(self) -> dict[str, str]:
        return dict(self.fields)


def _is_tag_header(text: str) -> bool:
    n = (text or "").strip().lower()
    return any(n == t or n.startswith(t + " ") or n.endswith(" " + t) for t in _TAG_HEADER_TOKENS)


def _find_schedule_region(
    blocks: Sequence[TextBlock],
    header_blk: TextBlock,
) -> tuple[list[TextBlock], str]:
    """Cluster downward from a schedule header. Returns (region_blocks, kind)."""
    m = _SCHEDULE_HEADER_RE.search(header_blk.text)
    kind = (m.group("kind") if m else "").strip().lower() if m else ""
    out: list[TextBlock] = [header_blk]
    last_bottom = header_blk.bbox[3]
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    for blk in sorted_blocks:
        if blk is header_blk or blk.bbox[1] < header_blk.bbox[1]:
            continue
        gap = blk.bbox[1] - last_bottom
        if gap > 36.0:
            break
        out.append(blk)
        last_bottom = max(last_bottom, blk.bbox[3])
    return out, kind


def detect_schedules(
    *,
    page_index: int,
    sheet_number: str | None,
    blocks: Sequence[TextBlock],
) -> list[ScheduleRow]:
    """Parse every schedule table on this page into rows."""
    headers = [
        blk for blk in blocks if _SCHEDULE_HEADER_RE.search(blk.text)
    ]
    if not headers:
        return []
    out: list[ScheduleRow] = []
    seen_regions: set[tuple[float, float, float, float]] = set()
    for header in headers:
        region_blocks, kind = _find_schedule_region(blocks, header)
        if len(region_blocks) < 3:
            continue  # need header + col-header + at least 1 row
        bbox = (
            min(b.bbox[0] for b in region_blocks),
            min(b.bbox[1] for b in region_blocks),
            max(b.bbox[2] for b in region_blocks),
            max(b.bbox[3] for b in region_blocks),
        )
        if bbox in seen_regions:
            continue
        seen_regions.add(bbox)

        rows = _cluster_rows(region_blocks)
        if len(rows) < 2:
            continue
        # First row is the schedule header itself; second is the
        # column header. Treat any row whose first cell is a tag
        # header as the column header row.
        column_map: list[tuple[float, float, str | None]] = []
        col_header_index = None
        for i, row in enumerate(rows[1:], start=1):
            sorted_row = sorted(row, key=lambda b: b.bbox[0])
            if not sorted_row:
                continue
            first_cell = sorted_row[0].text.strip()
            if _is_tag_header(first_cell):
                column_map = _build_column_map(sorted_row)
                # Override the first column key to '__tag__' so the
                # tag column is always identifiable.
                if column_map:
                    column_map[0] = (column_map[0][0], column_map[0][1], "__tag__")
                col_header_index = i
                break
            # Fallback: treat the first row after the header as
            # column header when it looks header-like (multiple cells,
            # at least one classifiable cell).
            if len(sorted_row) >= 2 and any(
                _classify_header_cell(c.text) is not None for c in sorted_row
            ):
                column_map = _build_column_map(sorted_row)
                column_map[0] = (column_map[0][0], column_map[0][1], "__tag__")
                col_header_index = i
                break
        if not column_map or col_header_index is None:
            continue

        for row in rows[col_header_index + 1:]:
            if not row:
                continue
            bucketed = _row_to_column_cells(row, column_map)
            tag = (bucketed.get("__tag__") or "").strip()
            if not tag or len(tag) > 30:
                continue
            # Skip obvious header repeats.
            if _is_tag_header(tag):
                continue
            fields = {
                k: v
                for k, v in bucketed.items()
                if k not in (None, "__tag__") and v
            }
            row_bbox = (
                min(b.bbox[0] for b in row),
                min(b.bbox[1] for b in row),
                max(b.bbox[2] for b in row),
                max(b.bbox[3] for b in row),
            )
            row_id = stable_id(
                "schedule_row",
                page_index,
                sheet_number or "",
                kind,
                tag,
            )
            out.append(
                ScheduleRow(
                    row_id=row_id,
                    page_index=page_index,
                    sheet_number=sheet_number,
                    schedule_kind=kind or "unknown",
                    tag=tag,
                    bbox=row_bbox,
                    fields=tuple(sorted((str(k), str(v)) for k, v in fields.items())),
                    confidence=0.85,
                )
            )
    out.sort(
        key=lambda r: (
            round(r.bbox[1], 2),
            round(r.bbox[0], 2),
            r.schedule_kind,
            r.tag,
        )
    )
    return out


def join_schedule_rows_to_detections(
    schedule_rows: Sequence[ScheduleRow],
    detections: Sequence[Any],
) -> dict[str, ScheduleRow]:
    """Map ``{detection_id: ScheduleRow}`` by tag match.

    A schedule row joins to a detection when the detection's
    ``nearby_text`` contains the row tag as a whole token (e.g.
    ``CR-101`` in the detection's nearby_text matches schedule
    row tag ``CR-101``).  Ties resolve by lexicographically smallest
    row_id.
    """
    if not schedule_rows or not detections:
        return {}
    out: dict[str, ScheduleRow] = {}
    rows_by_tag: dict[str, list[ScheduleRow]] = {}
    for row in schedule_rows:
        rows_by_tag.setdefault(row.tag.upper(), []).append(row)
    for det in detections:
        nearby = (getattr(det, "nearby_text", None) or "").upper()
        if not nearby:
            continue
        best: ScheduleRow | None = None
        for tag, rows in rows_by_tag.items():
            if not re.search(rf"\b{re.escape(tag)}\b", nearby):
                continue
            for row in rows:
                if best is None or row.row_id < best.row_id:
                    best = row
        if best is not None:
            out[det.detection_id] = best
    return out
