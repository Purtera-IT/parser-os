"""Multi-column contact/team block detection pass.

Detects borderless PROJECT TEAM / CONSULTANTS sections made of pure
free-floating text columns (no drawn rectangles, no borders).

Emitted boxes:
    mccol_group_N    BLUE   — full group wrapper (heading → body bottom)
    mccol_heading_N  BLUE   — the wide heading line ("PROJECT TEAM")
    mccol_hdr_N_C    CYAN   — each column-label cell ("ARCHITECTURAL" etc.)
    mccol_body_N_C   ORANGE — each column's body paragraph cluster
"""
from __future__ import annotations

import re

from ..core.models import Rect, VisibleBox, VisibleBoxResult
from .base import PageContext, PassInfo, PipelineState
from ..rules.multicol_contact_block import find_multicol_contact_blocks


def _is_v_box(b: VisibleBox) -> bool:
    return (
        not getattr(b, "synthetic", False)
        and bool(re.fullmatch(r"v\d+", b.box_id or ""))
    )


def _bbox_overlaps(a: tuple[int, int, int, int],
                   b: tuple[int, int, int, int]) -> bool:
    if a[2] <= b[0] or b[2] <= a[0]:
        return False
    if a[3] <= b[1] or b[3] <= a[1]:
        return False
    return True


def _make_box(
    box_id: str,
    pdf_bbox: tuple[float, float, float, float],
    color: str,
    scale: float,
    depth: int = 3,
    parent: str | None = None,
) -> VisibleBox:
    x0, y0, x1, y1 = pdf_bbox
    # clamp
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    px = (int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale))
    return VisibleBox(
        box_id=box_id,
        rect=Rect(x0, y0, x1, y1),
        area_pt2=max(1.0, (x1 - x0) * (y1 - y0)),
        fill_ratio=1.0,
        nested_depth=depth,
        is_outer_wrapper=False,
        parent_box_id=parent,
        color=color,
        px_bbox=px,
        children_count=0,
        synthetic=True,
    )


class MultiColContactPass:
    """Detects borderless multi-column team/consultant contact blocks."""

    info: PassInfo = PassInfo(
        order=238,
        name="multicol_contact_block",
        stage="semantic",
        layer_flag=None,
        description=(
            "Detects borderless PROJECT TEAM / CONSULTANTS multi-column blocks "
            "using text geometry: N≥3 evenly-spaced short label columns with "
            "body paragraphs below, optionally preceded by a larger heading."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None:
            return state

        groups = find_multicol_contact_blocks(ctx.pdf_path, ctx.page_index)
        if not groups:
            return state

        # Render scale: pixels per PDF point
        scale: float = ctx.cfg.render_scale if ctx.cfg else 2.5

        # Suppress mccol detections inside any real ``vN`` table region.
        # The contour pass already owns those rectangles and our cell
        # detection mis-fires when it sees evenly spaced column-header
        # text inside table cells, drawing tiny blue/cyan boxes over
        # cell content.  Universal — any document with detected tables.
        v_table_bboxes = [
            tuple(b.px_bbox) for b in state.result.boxes if _is_v_box(b)
        ]

        new_boxes: list[VisibleBox] = []

        for gi, group in enumerate(groups):
            gx0, gy0, gx1, gy1 = group.group_bbox
            group_px = (
                int(gx0 * scale), int(gy0 * scale),
                int(gx1 * scale), int(gy1 * scale),
            )
            if any(_bbox_overlaps(group_px, vb) for vb in v_table_bboxes):
                continue
            prefix = f"mccol_{gi}"

            # 1. Group wrapper (BLUE)
            group_id = f"{prefix}_group"
            new_boxes.append(_make_box(
                box_id=group_id,
                pdf_bbox=group.group_bbox,
                color="BLUE",
                scale=scale,
                depth=1,
                parent=None,
            ))

            # 2. Big page title above heading (BLUE, bold wash)
            if group.title_bbox:
                new_boxes.append(_make_box(
                    box_id=f"{prefix}_title",
                    pdf_bbox=group.title_bbox,
                    color="BLUE",
                    scale=scale,
                    depth=2,
                    parent=group_id,
                ))

            # 3. Heading ("PROJECT TEAM") — BLUE
            if group.heading_bbox:
                new_boxes.append(_make_box(
                    box_id=f"{prefix}_heading",
                    pdf_bbox=group.heading_bbox,
                    color="BLUE",
                    scale=scale,
                    depth=2,
                    parent=group_id,
                ))

            # 4. Per-column tight box: hdr label + body combined (BLUE outline)
            for ci, cb in enumerate(group.col_bboxes):
                bx0, by0, bx1, by1 = cb
                if by1 > by0:
                    new_boxes.append(_make_box(
                        box_id=f"{prefix}_col_{ci}",
                        pdf_bbox=cb,
                        color="BLUE",
                        scale=scale,
                        depth=3,
                        parent=group_id,
                    ))

            # 5. Column header labels (CYAN)
            for ci, ch in enumerate(group.col_headers):
                new_boxes.append(_make_box(
                    box_id=f"{prefix}_hdr_{ci}",
                    pdf_bbox=(ch.x0, ch.y0, ch.x1, ch.y1),
                    color="CYAN",
                    scale=scale,
                    depth=4,
                    parent=f"{prefix}_col_{ci}",
                ))

            # 6. Column body: NO separate orange box — the body text is the
            # interior of the BLUE column box. Orange = data cell, which is
            # wrong for free-floating contact paragraphs.

        if not new_boxes:
            return state

        # Merge into existing box list
        all_boxes = list(state.result.boxes) + new_boxes
        new_result = VisibleBoxResult(
            boxes=all_boxes,
            image_width=state.result.image_width,
            image_height=state.result.image_height,
            debug_stats=state.result.debug_stats,
        )
        return PipelineState(result=new_result, rgb=state.rgb)
