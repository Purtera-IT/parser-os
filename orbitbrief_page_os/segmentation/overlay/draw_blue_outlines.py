"""Structural blue: regular ``color == BLUE`` wrappers + synthetic ``*_body`` outlines.

Title-band **fills** live in ``draw_title_highlights`` — this module is outline /
geometry only (closed rectangles, no orphan strokes).

Wide shells with a single ``*_sec*_title`` child lower the drawn rectangle so the
left/right strokes do not bisect the ``colhdr_*`` row (MATV / DESCRIPTION …).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import cv2

from ..overlay_layers import OverlayLayer
from .colors_bgr import BLUE, LABEL_BG, ORANGE
from .style import DEFAULT_STYLE


def _draw_dashed_rect(canvas: Any, x0: int, y0: int, x1: int, y1: int,
                      color, thickness: int,
                      *, dash_px: int = 10, gap_px: int = 6) -> None:
    """Draw a rectangle whose perimeter is rendered as dashed segments.
    Used to mark continuation blocks (orphan content from cross-column
    section flow) so they read distinctly from regular section bodies.
    """
    period = dash_px + gap_px
    if period <= 0:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)
        return
    # Top and bottom edges
    x = x0
    while x < x1:
        x_end = min(x + dash_px, x1)
        cv2.line(canvas, (x, y0), (x_end, y0), color, thickness)
        cv2.line(canvas, (x, y1), (x_end, y1), color, thickness)
        x += period
    # Left and right edges
    y = y0
    while y < y1:
        y_end = min(y + dash_px, y1)
        cv2.line(canvas, (x0, y), (x0, y_end), color, thickness)
        cv2.line(canvas, (x1, y), (x1, y_end), color, thickness)
        y += period


def is_spec_section_box(box: Any) -> bool:
    """True for outer spec-section wrappers (~250 px wide, not synthetic)."""
    bw = box.px_bbox[2] - box.px_bbox[0]
    return 245 <= bw <= 260 and not getattr(box, "synthetic", False)


def _blue_wrapper_overlaps_titleblk_logo(
        box: Any,
        titleblk_px_bboxes: Sequence[tuple[int, int, int, int]],
        *,
        pad: int = 10,
) -> bool:
    """True when this BLUE shell is really framing a ``titleblk*`` logo strip.

    Title-block logos get a purple ring elsewhere; a contour BLUE wrapper that
    tightly hugs the same panel (CLIENT / consultant logo + address) should not
    also read as a structural table shell.
    """
    if not titleblk_px_bboxes:
        return False
    if is_spec_section_box(box):
        return False
    x0, y0, x1, y1 = box.px_bbox
    bw, bh = x1 - x0, y1 - y0
    ba = max(1, bw * bh)
    # Huge wrappers are real schedule shells even if a tiny logo sits in a corner.
    if ba >= 520_000:
        return False
    p = int(pad)
    for L0, L1, L2, L3 in titleblk_px_bboxes:
        L0, L1, L2, L3 = L0 - p, L1 - p, L2 + p, L3 + p
        ix0, iy0 = max(x0, L0), max(y0, L1)
        ix1, iy1 = min(x1, L2), min(y1, L3)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        ia = (ix1 - ix0) * (iy1 - iy0)
        if ia / float(ba) >= 0.10:
            return True
    return False


def _is_interior_table_separator_strip(box: Any, page_w: int) -> bool:
    """Thin full-width BLUE strips between header/data rows — not outer shells."""
    if getattr(box, "synthetic", False) or page_w < 200:
        return False
    if is_spec_section_box(box):
        return False
    x0, y0, x1, y1 = box.px_bbox
    w, h = x1 - x0, y1 - y0
    if h > 22 or w < int(0.48 * page_w):
        return False
    if int(getattr(box, "nested_depth", 0) or 0) < 2:
        return False
    asp = w / float(max(1, h))
    return asp >= 16.0


def _colhdr_belongs_to_wrapper(
        h: Any, wrapper_id: str, synth_by_id: dict[str, Any],
) -> bool:
    """True when ``colhdr_*`` is harvested for this non-synthetic BLUE shell.

    Parents are either the wrapper (TEST_3 / v8) or a synthetic ``*_title`` child
    of the wrapper (``v2_sec0_title`` → ``v2``).
    """
    pid = getattr(h, "parent_box_id", None) or ""
    if pid == wrapper_id:
        return True
    pt = synth_by_id.get(pid)
    if pt is None:
        return False
    return getattr(pt, "parent_box_id", None) == wrapper_id


def _wide_single_title_shell_top_inset_px(
        box: Any,
        synth_by_id: dict[str, Any],
        colhdr_boxes: Sequence[Any],
) -> int | None:
    """Lower the top of the structural rectangle so it clears the column-header row.

    Narrow shells already use ``{wrapper}_title``; wide MATV-style shells only have
    ``{wrapper}_sec0_title`` so the old branch never ran and the full outline cut
    through DESCRIPTION / MANUFACTURER / ….

    Restrict to **exactly one** non-``textsec_*`` ``*_title`` child so multi-band
    columns (``v2`` with eight sections) keep a single outer frame.
    """
    wid = getattr(box, "box_id", "") or ""
    if not wid or not colhdr_boxes:
        return None
    x0, y0, x1, y1 = box.px_bbox
    bw, bh = x1 - x0, y1 - y0
    if bw <= 220 or bh < 72:
        return None
    main_ts = [
        b for b in synth_by_id.values()
        if getattr(b, "synthetic", False)
        and str(getattr(b, "box_id", "")).endswith("_title")
        and not str(getattr(b, "box_id", "")).startswith("textsec_")
        and getattr(b, "parent_box_id", None) == wid
    ]
    if len(main_ts) != 1:
        return None
    belonging = [
        h for h in colhdr_boxes
        if str(getattr(h, "box_id", "")).startswith("colhdr_")
        and _colhdr_belongs_to_wrapper(h, wid, synth_by_id)]
    if len(belonging) < 1:
        return None
    min_y0 = min(int(h.px_bbox[1]) for h in belonging)
    row_tol = 14
    first_row = [h for h in belonging if int(h.px_bbox[1]) <= min_y0 + row_tol]
    hdr_bot = max(int(h.px_bbox[3]) for h in first_row)
    h_wr = max(1, bh)
    if hdr_bot > y0 + int(0.55 * h_wr):
        return None
    ty1 = int(main_ts[0].px_bbox[3])
    tcut = max(ty1 + 1, hdr_bot + 2)
    if tcut >= y1 - 3:
        return None
    return tcut


def draw_blue_wrapper_box(
        canvas: Any,
        box: Any,
        *,
        page_w: int,
        spec_orange_inset: bool,
        synth_by_id: dict[str, Any],
        titleblk_px_bboxes: Sequence[tuple[int, int, int, int]],
        draw_labels: bool,
        colhdr_boxes: Sequence[Any] | None = None,
) -> None:
    """Draw one non-synthetic BLUE wrapper (Pass 1).

    ``spec_orange_inset`` is owned by the orchestrator (typically: draw orange
    halo when the orange overlay layer is on).  This module does not read
    ``OverlayLayer.ORANGE`` directly.
    """
    if _is_interior_table_separator_strip(box, page_w):
        return
    if _blue_wrapper_overlaps_titleblk_logo(box, titleblk_px_bboxes):
        return
    x0, y0, x1, y1 = box.px_bbox
    # Universal outset (see OverlayStyle.blue_wrapper_outset_px).  Drawing
    # the wrapper a few px outside its bbox ensures the BLUE line is visible
    # outside any inner ORANGE cell that hugs the original bbox; without it,
    # the later ORANGE pass overpaints the BLUE line on coincident edges
    # and the wrapper disappears.  Clip to canvas bounds so we never read
    # past the image.
    page_h = int(canvas.shape[0])
    outset = int(DEFAULT_STYLE.blue_wrapper_outset_px)
    x0 = max(0, x0 - outset)
    y0 = max(0, y0 - outset)
    x1 = min(page_w - 1, x1 + outset)
    y1 = min(page_h - 1, y1 + outset)
    # Universal blue line width — every BLUE wrapper renders at the same
    # stroke regardless of whether it's a core contour wrapper, a
    # title-block cell (tbstruct_*), or a tabular-cluster outer frame
    # (tbgroup_*).  Per-type stroke bumps were brittle: every new wrapper
    # type needed its own line-width rule, and the inconsistent widths
    # made the overlay read as if some boxes were "more important" than
    # others.  Visual separation between wrappers and inner cells is now
    # achieved via geometric outset (blue_wrapper_outset_px) rather
    # than stroke thickness.
    is_tbstruct = box.box_id.startswith("tbstruct_")
    line_px = DEFAULT_STYLE.blue_wrapper_px
    if is_spec_section_box(box):
        if spec_orange_inset:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), ORANGE, DEFAULT_STYLE.orange_cell_px)
            cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), BLUE, line_px)
        else:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), BLUE, line_px)
    elif str(getattr(box, "box_id", "")).endswith("_continuation"):
        # Continuation blocks (orphan content from a section that began
        # in the previous column) render as a DASHED blue outline so
        # they're visually distinguishable from regular ``_body`` solid
        # outlines.  The dashed pattern signals "this content has no
        # local title — it continues from somewhere else."
        _draw_dashed_rect(canvas, x0, y0, x1, y1, BLUE, line_px,
                          dash_px=10, gap_px=6)
    else:
        bw = x1 - x0
        bh = y1 - y0
        # Previous behaviour clipped the top off narrow titled wrappers (bw ≤ 220,
        # bh ≥ 72) by starting the BLUE rectangle at the title's bottom edge —
        # the original intent was to avoid a stroke "bisecting" content under
        # the title.  But the wrapper's own top edge sits ABOVE the title, not
        # inside the body, so the clip just removed the BLUE outline around
        # the title cell, leaving an orange-only outline that violated the
        # legend (BLUE = structural wrapper, ORANGE = inner cell).  Draw the
        # full wrapper unconditionally for narrow titled shells too.
        tcut_wide: int | None = None
        if colhdr_boxes is not None and not is_tbstruct:
            tcut_wide = _wide_single_title_shell_top_inset_px(
                box, synth_by_id, colhdr_boxes)
        if tcut_wide is not None:
            cv2.rectangle(canvas, (x0, tcut_wide), (x1, y1), BLUE, line_px)
        else:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), BLUE, line_px)

    if draw_labels and not box.box_id.endswith(("_title", "_sublabel")):
        lbl = f"{box.box_id} d={box.nested_depth} ch={box.children_count}"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, DEFAULT_STYLE.label_font_scale, DEFAULT_STYLE.label_thickness)
        if (y1 - y0) >= lh + 6:
            ly0 = y0
            tb_c = synth_by_id.get(f"{box.box_id}_title")
            if (tb_c is not None and (x1 - x0) <= 220
                    and getattr(tb_c, "synthetic", False)):
                ly0 = max(y0, tb_c.px_bbox[3] + 2)
            if ly0 + lh + 6 > y1:
                ly0 = y0
            cv2.rectangle(canvas, (x0, ly0), (x0 + lw + 4, ly0 + lh + 6),
                          LABEL_BG, cv2.FILLED)
            cv2.putText(canvas, lbl, (x0 + 2, ly0 + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, DEFAULT_STYLE.label_font_scale, (255, 255, 255), DEFAULT_STYLE.label_thickness,
                        cv2.LINE_AA)


def draw_regular_blue_wrappers(
        canvas: Any,
        regular_sorted: list[Any],
        *,
        layers: OverlayLayer,
        spec_orange_inset: bool,
        synth_by_id: dict[str, Any],
        titleblk_px_bboxes: Sequence[tuple[int, int, int, int]],
        draw_labels: bool,
        colhdr_boxes: Sequence[Any] | None = None,
) -> None:
    """Pass 1: largest-first regular boxes with ``color == \"BLUE\"``."""
    if not (layers & OverlayLayer.BLUE_WRAPPERS):
        return
    page_w = int(canvas.shape[1])
    for box in regular_sorted:
        if box.color == "BLUE":
            draw_blue_wrapper_box(
                canvas, box,
                page_w=page_w,
                spec_orange_inset=spec_orange_inset,
                synth_by_id=synth_by_id,
                titleblk_px_bboxes=titleblk_px_bboxes,
                draw_labels=draw_labels,
                colhdr_boxes=colhdr_boxes,
            )


def draw_synthetic_blue_bodies(
        canvas: Any,
        body_wrappers: list[Any],
        *,
        skip_if_titleblk_band: Callable[[tuple[int, int, int, int]], bool],
        skip_if_splinter: Callable[[Any], bool],
) -> None:
    """Pass 2.5: ``*_body`` outlined rectangles (no fill).

    Boxes whose id ends in ``_continuation`` render as a DASHED outline
    instead of solid — they're orphan content from another column,
    visually distinct from a regular ``_body`` whose section has its
    own local title.
    """
    for bw in body_wrappers:
        if skip_if_titleblk_band(bw.px_bbox):
            continue
        if skip_if_splinter(bw):
            continue
        x0, y0, x1, y1 = bw.px_bbox
        if str(getattr(bw, "box_id", "")).endswith("_continuation"):
            _draw_dashed_rect(canvas, x0, y0, x1, y1, BLUE,
                              DEFAULT_STYLE.blue_wrapper_px,
                              dash_px=10, gap_px=6)
        else:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), BLUE,
                          DEFAULT_STYLE.blue_wrapper_px)
