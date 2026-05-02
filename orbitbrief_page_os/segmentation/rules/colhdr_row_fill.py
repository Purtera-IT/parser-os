"""Universal rule: a cell beside >=2 cyan peers in the same row gets
cyan too — purely structural row-fill, no text rule.

Symptom that motivates this rule
--------------------------------
``DIMENSIONS / LENGTH x WIDTH x HEIGHT`` is a column header on the AHU
and range-hood schedules.  After newline replacement the legacy text-cap
gate counted 6 word-tokens (``LENGTH x WIDTH x HEIGHT DIMENSIONS``,
where the ``x`` separators are tokens too) and rejected the cell as
"not a column header" because its rule is "<=5 words."  The other
columns in the same row got cyan rings; this one didn't.  Visually:
one missing cyan ring in an otherwise complete row.

The user's principle (verbatim): "if it's beside other cyan and not
green, it should be cyan too."  This rule encodes that — purely by
spatial neighbour-count, no per-text rule.

Inputs
------
The full set of detected boxes after the upstream colhdr emitters have
finished.  An optional helper for emitting tight word-bounded bboxes
(falls back to the cell bbox if absent).

Outputs
-------
A list of new colhdr_ ``VisibleBox`` entries to append to the page.

Why universal
-------------
The discriminator is purely structural: row-neighbours are cyan, the
candidate cell sits inside that row's x/y band, no green ring covers it
already.  Any column header missed by upstream text gates is recovered
purely by spatial context — no per-table tuning, no schedule-specific
logic, no text classification.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple

BBox = Tuple[int, int, int, int]


def _has_overlap(bb1: BBox, bb2_list: List[BBox], min_frac: float = 0.5) -> bool:
    """True when ``bb1`` substantially overlaps any bbox in ``bb2_list``."""
    x0, y0, x1, y1 = bb1
    a = max(1, (x1 - x0) * (y1 - y0))
    for (ox0, oy0, ox1, oy1) in bb2_list:
        xa, ya = max(x0, ox0), max(y0, oy0)
        xb, yb = min(x1, ox1), min(y1, oy1)
        if xb <= xa or yb <= ya:
            continue
        if (xb - xa) * (yb - ya) / a >= min_frac:
            return True
    return False


def row_fill_missing_colhdrs(
    boxes: List[Any],
    *,
    make_box: Callable[[BBox, str, int], Any],
    starting_id: int,
    tight_bbox: Optional[Callable[[int, int, int, int], BBox]] = None,
    y_band_slack_px: int = 12,
    cell_max_h_px: int = 50,
    overlap_dedup_frac: float = 0.20,
) -> List[Any]:
    """Walk colhdr siblings grouped by ``(parent_box_id, y-band)`` and emit
    cyan rings for non-synthetic ORANGE peers that are missing them.

    Parameters
    ----------
    boxes : list of VisibleBox-like
        The detected boxes after upstream colhdr emission.  Used to
        identify both the existing colhdr siblings and the candidate
        ORANGE peers to gap-fill.
    make_box : callable
        Factory ``(px_bbox, parent_id, new_id_int) -> VisibleBox``.
        Lets the caller construct boxes with whatever schema/imports it
        already uses (kept here in the rules layer to stay independent).
    starting_id : int
        The integer to start numbering new ``colhdr_<N>`` ids from.
    tight_bbox : callable, optional
        ``(x0, y0, x1, y1) -> (tx0, ty0, tx1, ty1)`` returning a
        word-tight inner bbox.  Falls back to the cell bbox if absent
        or if the returned box is too small.
    y_band_slack_px : int
        Vertical slack added above/below the existing row's y-extent
        when looking for ORANGE peers.  Compensates for the size
        difference between word-tight rings and full cell extents.
    cell_max_h_px : int
        Reject candidate ORANGE cells taller than this (likely a
        multi-row span, not a header).
    overlap_dedup_frac : float
        Minimum overlap fraction with an existing colhdr/subhdr that
        causes the candidate to be skipped.

    Returns
    -------
    A list of new boxes ready to append to the page.
    """
    existing_colhdr = [b for b in boxes if b.box_id.startswith("colhdr_")]
    existing_subhdr_bboxes = [
        b.px_bbox for b in boxes if b.box_id.startswith("subhdr_")
    ]
    existing_colhdr_bboxes = [b.px_bbox for b in existing_colhdr]

    # Group existing colhdr by (parent_box_id, y-band) — gives one group
    # per column-header row per parent.
    row_groups: dict[tuple[str, int], list] = defaultdict(list)
    for h in existing_colhdr:
        pid = getattr(h, "parent_box_id", None) or ""
        ymid = int(round(0.5 * (h.px_bbox[1] + h.px_bbox[3])))
        yk = (ymid // 12) * 12
        row_groups[(pid, yk)].append(h)

    orange_cells = [
        b for b in boxes
        if getattr(b, "color", None) == "ORANGE"
        and not getattr(b, "synthetic", False)
    ]

    emitted: List[Any] = []
    next_id = starting_id

    for (pid, _yk), siblings in row_groups.items():
        sib_y0 = min(s.px_bbox[1] for s in siblings)
        sib_y1 = max(s.px_bbox[3] for s in siblings)
        band_y0 = sib_y0 - y_band_slack_px
        band_y1 = sib_y1 + y_band_slack_px
        band_orange = [
            c for c in orange_cells
            if (getattr(c, "parent_box_id", None) or "") == pid
            and not (c.px_bbox[3] < band_y0 or c.px_bbox[1] > band_y1)
            and (c.px_bbox[3] - c.px_bbox[1]) <= cell_max_h_px
        ]
        if len(siblings) >= 2:
            sib_x0 = min(s.px_bbox[0] for s in siblings)
            sib_x1 = max(s.px_bbox[2] for s in siblings)
        elif len(siblings) == 1 and len(band_orange) >= 4:
            # One cyan + many orange peers in the same header band → fill the row.
            sib_x0 = min(c.px_bbox[0] for c in band_orange)
            sib_x1 = max(c.px_bbox[2] for c in band_orange)
        else:
            continue

        for c in orange_cells:
            cx0, cy0, cx1, cy1 = c.px_bbox
            if cx1 <= sib_x0 - 4 or cx0 >= sib_x1 + 4:
                continue
            if cy0 < band_y0 or cy1 > band_y1:
                continue
            if cy1 - cy0 > cell_max_h_px:
                continue
            if _has_overlap((cx0, cy0, cx1, cy1),
                            existing_colhdr_bboxes,
                            min_frac=overlap_dedup_frac):
                continue
            if _has_overlap((cx0, cy0, cx1, cy1),
                            existing_subhdr_bboxes,
                            min_frac=overlap_dedup_frac):
                continue   # green takes precedence

            # Pick a tight inner bbox if possible; fall back to the cell.
            tx0, ty0, tx1, ty1 = cx0, cy0, cx1, cy1
            if tight_bbox is not None:
                try:
                    tb = tight_bbox(cx0, cy0, cx1, cy1)
                    if tb and (tb[2] - tb[0]) >= 4 and (tb[3] - tb[1]) >= 4:
                        tx0, ty0, tx1, ty1 = tb
                except Exception:
                    pass

            next_id += 1
            new_box = make_box((tx0, ty0, tx1, ty1), pid, next_id)
            if new_box is None:
                continue
            emitted.append(new_box)
            existing_colhdr_bboxes.append((tx0, ty0, tx1, ty1))

    return emitted
