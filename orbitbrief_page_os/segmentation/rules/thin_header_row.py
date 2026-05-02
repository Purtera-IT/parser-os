"""Universal rule: a TABLE'S TOP ROW that's much shorter than the data
rows below is the column-header row.  Its cells should be CYAN
(synthesized as ``colhdr_*`` boxes).

Symptom that motivates this rule
--------------------------------
On test7 the EXTERIOR WALL ASSEMBLIES, INTERIOR PARTITIONS, FLOOR
ASSEMBLIES, and ROOF ASSEMBLIES tables each have a short row at the
top with cells like ``TYPE | WALL | HEIGHT | FRR | STC | LB``.  These
are clearly column headers — they label what each column below
contains.  But without an explicit "SCHEDULE" keyword in the table
title, the existing column-header detection (gated on schedule
keywords) doesn't fire.  The cells render as plain ORANGE same as
data cells, so the user can't tell what's a header vs what's data.

Inputs
------
A list of detected boxes.  Looks for non-synthetic BLUE ``v*``
contour wrappers that contain ≥4 ORANGE leaf cells (≥4 rules out
sidebars and revisions blocks that have only 2–3 cells).

Outputs
-------
A list of (cell_box, table_wrapper) pairs identifying which ORANGE
leaf cells should be turned into CYAN column headers.  The caller
decides how to act on this — typically by emitting a synthetic
``colhdr_*`` box at the cell's bbox so the renderer paints the
cyan ring.

Why universal
-------------
Tables of this shape have a consistent visual signature that
distinguishes them from schedule-style tables (where the colhdr
detector already fires) and from sidebars / title-blocks (where
there's no header-row pattern):

    - Wrapper has many child cells (≥6) — a real table.
    - First row of cells is a regular row (≥3 cells) at the TOP
      of the wrapper.
    - First row's max cell height is small (≤50 px image) — a
      single-line header.
    - The first row of equal-or-greater cell count BELOW it has
      max cell height ≥2× the header row's height — substantial
      data rows.

These four gates discriminate cleanly between:

    - Real column-header rows (test7 tables): all gates pass.
    - Schedule title strips (test5): hdr_row has 1 cell; size
      ratio also fails.
    - Title-blocks/sidebars (test7 v1): hdr_row has 2 cells, OR
      the structure doesn't have proper data rows below.

Tunable parameters
------------------
``min_table_kids`` (4): minimum ORANGE leaves under a wrapper to
                       call it a table (4 catches compact GC/work-order grids).
``min_header_cells`` (3): minimum cells in the candidate header row.
``max_header_cell_h_px`` (50): header row's max cell height ceiling.
``min_below_to_header_ratio`` (2.0): the next substantive row's
                       max cell height must be at least this many
                       times the header row's max cell height.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Tuple


def find_thin_header_rows(
    boxes: List[Any],
    *,
    min_table_kids: int = 4,
    min_header_cells: int = 3,
    max_header_cell_h_px: float = 50.0,
    min_below_to_header_ratio: float = 2.0,
) -> List[Tuple[Any, List[Any]]]:
    """Return ``[(table_wrapper, header_cells), …]`` for tables whose
    top row is a thin column-header row.

    Both ``table_wrapper`` (a non-synthetic BLUE ``v*`` box) and the
    cells in ``header_cells`` (non-synthetic ORANGE ``v*`` boxes) come
    directly from the input list — no copies, no mutations.
    """
    # Bucket non-synthetic ORANGE leaf cells by their parent wrapper.
    orange_leaves = [
        b for b in boxes
        if (not getattr(b, "synthetic", False)
            and getattr(b, "color", None) == "ORANGE"
            and (getattr(b, "box_id", "") or "").startswith("v"))
    ]
    blue_wrappers = [
        b for b in boxes
        if (not getattr(b, "synthetic", False)
            and getattr(b, "color", None) == "BLUE"
            and (getattr(b, "box_id", "") or "").startswith("v"))
    ]

    results: List[Tuple[Any, List[Any]]] = []
    for wrap in blue_wrappers:
        wx0, wy0, wx1, wy1 = wrap.px_bbox
        # Children of this wrapper by centroid containment.  Use
        # centroid because contour cells share borders with their
        # wrapper — strict edge containment misses them.
        kids = []
        for ol in orange_leaves:
            ox0, oy0, ox1, oy1 = ol.px_bbox
            ocx = (ox0 + ox1) / 2.0
            ocy = (oy0 + oy1) / 2.0
            if (wx0 + 6) <= ocx <= (wx1 - 6) and (wy0 + 4) <= ocy <= (wy1 - 4):
                kids.append(ol)
        if len(kids) < min_table_kids:
            continue

        # Bucket by y (8 px buckets).
        by_row: dict[int, list] = defaultdict(list)
        for c in kids:
            cy = (c.px_bbox[1] + c.px_bbox[3]) // 2
            yk = (cy // 8) * 8
            by_row[int(yk)].append(c)
        rows = sorted(by_row.items())
        if len(rows) < 2:
            continue

        hdr_row = rows[0][1]
        if len(hdr_row) < min_header_cells:
            continue
        hdr_max_h = max(c.px_bbox[3] - c.px_bbox[1] for c in hdr_row)
        if hdr_max_h > max_header_cell_h_px:
            continue

        # Find the first row below with at least (header_count - 1)
        # cells.  -1 tolerance handles tables where one column lacks a
        # cell on the first data row.
        cand_rows = [
            (yk, cells) for (yk, cells) in rows[1:]
            if len(cells) >= len(hdr_row) - 1
        ]
        if not cand_rows:
            continue
        next_yk, next_cells = cand_rows[0]
        next_max_h = max(c.px_bbox[3] - c.px_bbox[1] for c in next_cells)
        if hdr_max_h <= 0:
            continue
        ratio = next_max_h / hdr_max_h
        if ratio < min_below_to_header_ratio:
            continue

        results.append((wrap, hdr_row))

    return results
