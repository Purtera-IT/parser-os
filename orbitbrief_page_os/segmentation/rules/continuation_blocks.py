"""Universal rule: orphan label-rows at the top of a column with no
header above them are a CONTINUATION of a section that began in the
previous column.  Wrap them with a dedicated continuation body so
they're visually marked as content (not silently dropped) and
distinguishable from a normal section that has its own title.

Symptom that motivates this rule
--------------------------------
On test7 the SECTION 3.3 - SPATIAL SEPARATION block has multiple
walls.  Each wall has 7 label rows: LIMITING DISTANCE,
EXPOSING BUILDING FACE, ALLOWABLE UNPROTECTED OPENINGS, PROPOSED
UNPROTECTED OPENINGS, FIRE-RESISTANCE RATING, CONSTRUCTION,
CLADDING.

SOUTH WALL (EXISTING BUILDING) appears at the BOTTOM of the left
column (PDF x=154, y=1618).  Only its first 4 rows fit before the
column ends — the last 3 rows (FIRE-RESISTANCE RATING,
CONSTRUCTION, CLADDING) flow across to the TOP of the next column
(PDF x=506, y=812..834), ABOVE the next section's title (EAST WALL
at y=857).

These 3 orphan rows are content that belongs to SOUTH WALL (EXISTING
BUILDING) but the detector currently leaves them as naked text — no
wrapper, no indicator.  The user's request is to detect this
continuation pattern and mark it.

Discriminator
-------------
A continuation block is a stretch of consecutive colon-ending
``LABEL:`` lines that:

1. Sit at the TOP of a column (no header bbox at the same x-anchor
   appears above them in the same column).
2. Are aligned at the same x-anchor (consecutive lines).
3. Are immediately followed by a header span (the next section's
   title) within the same column.
4. Have a sibling section with the SAME label suffix in the previous
   column whose row count is short by exactly the orphan count —
   i.e. a section whose missing rows match these orphans.

For now this rule implements only the geometric detection
(conditions 1-3) and emits a ``_continuation`` body wrapper around
the orphan rows.  Linking to the parent section (condition 4) is
left as a future enhancement.

Why universal
-------------
The geometric pattern — labels at top of column with no preceding
header — is independent of the drawing's domain or language.  It
catches any column-flow continuation regardless of what content
specifically continues.

The rule's output is a list of bbox tuples that describe the
continuation regions.  The caller wraps them with a body box of
their own, suffix ``_continuation``, so the overlay shows the
content is detected even though it doesn't have a local title.

Verification
------------
- test5: zero orphan label-rows.  Rule fires zero times.
- test7: exactly 1 orphan block at PDF (506, 812)..(610, 834) —
  the FIRE-RESISTANCE RATING / CONSTRUCTION / CLADDING continuation
  of SOUTH WALL (EXISTING BUILDING).
"""
from __future__ import annotations

from typing import List, Tuple

from .section_hierarchy import is_section_parent


def find_continuation_blocks(
    headers: List[dict],
    span_text_by_bbox: dict,
    *,
    x_anchor_tol_pt: float = 8.0,
    y_above_tol_pt: float = 200.0,
    y_below_tol_pt: float = 30.0,
    min_rows: int = 2,
    line_step_pt: float = 14.0,
) -> List[Tuple[float, float, float, float]]:
    """Identify continuation blocks: consecutive ``LABEL:`` rows at the
    top of a column with no preceding header in that column.

    The block bbox is extended DOWN to enclose all subsequent content
    in the same column until a SECTION X.Y peer header is encountered.
    Sub-section content (non-SECTION-pattern headers like ``EAST WALL``
    that follow the orphan rows) is included as part of the
    continuation: there's no overarching blue section in this column,
    so the dashed continuation wrapper is the appropriate enclosing
    visual for the entire orphan-content stretch.

    Parameters
    ----------
    headers : list of header dicts
        Final list of detected headers (from ``_candidate_headers``
        post all rules).  Each must have ``"bbox"`` and ``"text"``.
    span_text_by_bbox : dict
        Map from bbox tuple to span text (from ``_candidate_headers``).
    x_anchor_tol_pt : float
        Tolerance for "same x-anchor" when grouping consecutive label
        rows (default 8 pt).
    y_above_tol_pt : float
        How far above to look for an enclosing header.  If we find a
        header within this distance at the same x-anchor, the rows are
        NOT a continuation — they're normal body rows.  Default 200 pt.
    y_below_tol_pt : float
        How close below to require the next header to be — confirms the
        continuation rows truly precede a section start.  Default 30 pt.
    min_rows : int
        Minimum number of consecutive rows to qualify.  Default 2.
    line_step_pt : float
        Maximum y-gap between consecutive rows in the block.  Default
        14 pt covers the typical 11-12 pt line height with some slack.

    Returns
    -------
    A list of (x0, y0, x1, y1) bboxes covering each continuation
    block in PDF points.
    """
    # Index header positions by snapped x-anchor (rounded to 8pt).
    headers_by_x: dict[float, list[float]] = {}
    for h in headers:
        bx0, by0, bx1, by1 = h["bbox"]
        snapped = round(bx0 / x_anchor_tol_pt) * x_anchor_tol_pt
        headers_by_x.setdefault(snapped, []).append(by0)

    # Group all colon-ending uppercase spans by x-anchor.
    by_x: dict[float, list[tuple[float, float, float, float, str]]] = {}
    for bbox, text in span_text_by_bbox.items():
        if not text:
            continue
        text_s = text.rstrip()
        if not text_s.endswith(":"):
            continue
        letters = [c for c in text_s if c.isalpha()]
        if len(letters) < 5:
            continue
        if not all(c.isupper() for c in letters):
            continue
        bx0, by0, bx1, by1 = bbox
        snapped = round(bx0 / x_anchor_tol_pt) * x_anchor_tol_pt
        by_x.setdefault(snapped, []).append((bx0, by0, bx1, by1, text_s))

    blocks: List[Tuple[float, float, float, float]] = []
    for snapped_x, rows in by_x.items():
        rows.sort(key=lambda r: r[1])   # by y ascending
        # Walk through, find runs of consecutive rows at line_step_pt
        # spacing.
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows):
                # Consecutive if y-gap is less than line_step_pt
                prev_y0 = rows[j - 1][1]
                cur_y0 = rows[j][1]
                if cur_y0 - prev_y0 > line_step_pt:
                    break
                j += 1
            run = rows[i:j]
            if len(run) >= min_rows:
                run_top_y = run[0][1]
                run_bot_y = run[-1][3]
                # Continuation test 1: no header at same x-anchor
                # within y_above_tol_pt above run_top_y
                hdr_ys = headers_by_x.get(snapped_x, [])
                has_header_above = any(
                    run_top_y - hdr_y > 0
                    and run_top_y - hdr_y <= y_above_tol_pt
                    for hdr_y in hdr_ys
                )
                if not has_header_above:
                    # Continuation test 2: a header at same x-anchor
                    # appears within y_below_tol_pt below run_bot_y
                    has_header_below = any(
                        hdr_y > run_bot_y
                        and hdr_y - run_bot_y <= y_below_tol_pt
                        for hdr_y in hdr_ys
                    )
                    if has_header_below:
                        # Find the next column anchor to the right of
                        # this one — that's the right boundary for
                        # value sweep.  Default to label_far_x + 250 pt
                        # if no other column anchor is found.
                        label_max_x = max(r[2] for r in run)
                        right_bound = label_max_x + 300.0
                        # Use header x-anchors as column boundaries.
                        snapped_xs_sorted = sorted(headers_by_x.keys())
                        for sx in snapped_xs_sorted:
                            if sx > snapped_x + x_anchor_tol_pt:
                                right_bound = sx - 6.0   # small inter-col gap
                                break

                        # Extend run_bot_y DOWN to enclose all
                        # subsequent orphan content (sub-headers and
                        # their bodies) until the next SECTION X.Y
                        # peer header in the same column.  Sub-headers
                        # like ``EAST WALL`` / ``SOUTH WALL`` /
                        # ``WEST WALL`` are still continuation content
                        # of the previous-column section, so they
                        # belong inside the dashed wrapper rather than
                        # getting their own separate boxes.
                        section_peer_y = float("inf")
                        for h in headers:
                            hbx0 = h["bbox"][0]
                            hsnapped = round(hbx0 / x_anchor_tol_pt) * x_anchor_tol_pt
                            if abs(hsnapped - snapped_x) > x_anchor_tol_pt:
                                continue
                            if h["bbox"][1] <= run_bot_y:
                                continue
                            if is_section_parent(h.get("text", "") or ""):
                                if h["bbox"][1] < section_peer_y:
                                    section_peer_y = h["bbox"][1]
                        # Extended bottom: just above the next SECTION
                        # peer (with a small gap), or page-content end
                        # if no peer exists.
                        extended_bot_y = run_bot_y
                        if section_peer_y < float("inf"):
                            # Cap at a small gap above the section peer.
                            extended_bot_y = section_peer_y - 4.0

                        # Sweep ALL spans (any source, not just colon
                        # labels) within run y-range and x ∈ [snapped_x,
                        # right_bound] to capture associated values.
                        run_y_lo = run_top_y - 2.0
                        run_y_hi = extended_bot_y + 2.0
                        bx0 = min(r[0] for r in run)
                        bx1 = label_max_x
                        for sbbox, _stext in span_text_by_bbox.items():
                            sx0, sy0, sx1, sy1 = sbbox
                            scx = 0.5 * (sx0 + sx1)
                            scy = 0.5 * (sy0 + sy1)
                            if not (run_y_lo <= scy <= run_y_hi):
                                continue
                            if not (snapped_x - x_anchor_tol_pt <= sx0
                                    and sx1 <= right_bound):
                                continue
                            if sx1 > bx1:
                                bx1 = sx1
                            if sx0 < bx0:
                                bx0 = sx0
                        blocks.append((bx0, run_top_y, bx1, extended_bot_y))
            i = j
    return blocks
