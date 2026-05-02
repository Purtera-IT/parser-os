"""Text/geometry-aware cleanup passes for v2 overlay semantics.

The core detector is intentionally kept as a compatibility pass, but the
behavioural fixes now live here as small, testable filters.  Each function has
one concern:

* purple is only for real title-block graphics/logos, not whole text panels;
* cyan column headers must be word/row-header tight, never body paragraphs;
* blue title washes must be title-band tight, never broad body regions;
* mini-table extensions must not paint across the main spec body.

All rules are geometry-only by default so they are fast and deterministic.  They
operate on structured boxes before painting, so JSON and overlay pixels agree.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from ..core.models import VisibleBox, VisibleBoxResult
from .base import PageContext, PassInfo, PipelineState


def _wh(b: VisibleBox) -> tuple[int, int]:
    x0, y0, x1, y1 = b.px_bbox
    return max(0, int(x1 - x0)), max(0, int(y1 - y0))


def _area(b: VisibleBox) -> int:
    w, h = _wh(b)
    return w * h


def _centroid_in(inner: VisibleBox, outer: VisibleBox, pad: int = 0) -> bool:
    ix0, iy0, ix1, iy1 = inner.px_bbox
    ox0, oy0, ox1, oy1 = outer.px_bbox
    cx = 0.5 * (ix0 + ix1)
    cy = 0.5 * (iy0 + iy1)
    return (ox0 - pad) <= cx <= (ox1 + pad) and (oy0 - pad) <= cy <= (oy1 + pad)


def _contained(child: VisibleBox, parent: VisibleBox, tol: int = 2) -> bool:
    x0, y0, x1, y1 = child.px_bbox
    px0, py0, px1, py1 = parent.px_bbox
    return (
        px0 - tol <= x0 <= px1 + tol
        and py0 - tol <= y0 <= py1 + tol
        and px0 - tol <= x1 <= px1 + tol
        and py0 - tol <= y1 <= py1 + tol
    )


def _drop_false_purple(b: VisibleBox, W: int, H: int) -> bool:
    """Return True when a PURPLE synthetic box is text chrome, not a logo.

    The v1 title-block image pass sometimes receives one giant PDF XObject for
    the entire right title strip.  Ringing that whole strip purple mislabels
    normal notes, stamps, revision rows, and project text as artwork.  Keep
    compact image/vector marks; remove panel/text/group expansions.
    """
    if b.color != "PURPLE" or not b.synthetic or not b.box_id.startswith("titleblk"):
        return False
    bid = b.box_id
    w, h = _wh(b)
    page_area = max(1, W * H)
    a = w * h

    if bid.startswith("titleblkimg"):
        # Whole-sidebar raster placements are not logo semantics.  They can be
        # reparsed by title-block/grid passes instead of one purple slab.
        if h >= int(0.34 * H) and w <= int(0.18 * W):
            return True
        if a >= int(0.035 * page_area) and h >= int(0.16 * H):
            return True
        return False

    if bid.startswith("titleblkvect"):
        if h >= int(0.22 * H) or a >= int(0.025 * page_area):
            return True
        return False

    # These are text/panel heuristics in v1.  They are useful signals for a
    # future LOGO_WORDMARK / TITLEBLOCK_PANEL layer, but they should not be
    # painted as purple image/logo rings today.
    if bid.startswith((
        "titleblkgrp_",
        "titleblkpanel_",
        "titleblkslice_",
        "titleblkword_",
        "titleblktext_",
    )):
        return True

    return False


def _drop_false_colhdr(b: VisibleBox, W: int, H: int) -> bool:
    """Cyan rings must be header-tight, not paragraph/tall-column hulls."""
    if not b.synthetic or not b.box_id.startswith("colhdr_"):
        return False
    w, h = _wh(b)
    if h <= 0 or w <= 0:
        return True
    # At ordinary sheet scales, word/column-label rings are short.  Paragraph
    # hulls and body columns show up as tall rectangles like 45x145 px.
    if h > max(28, int(0.035 * H)):
        return True
    if h > max(22, int(2.4 * max(1, w))):
        return True
    if (w * h) > int(0.006 * W * H):
        return True
    return False


def _drop_overbroad_title_band(
    b: VisibleBox,
    by_id: dict[str, VisibleBox],
    W: int,
    H: int,
) -> bool:
    """Blue title wash is for caption bands, not the first body block.

    Core synthesis can emit ``vN_title`` covering the top quarter of a large
    spec block.  That creates a blue wash over paragraphs/cells.  Text-section
    titles are already word-tight, so this rule only touches contour-derived
    title bands.
    """
    if not b.synthetic or b.color != "BLUE" or not b.box_id.endswith("_title"):
        return False
    if b.box_id.startswith("textsec_"):
        return False
    w, h = _wh(b)
    if h <= 0:
        return True
    parent = by_id.get(b.parent_box_id or "")
    if parent is not None:
        _, ph = _wh(parent)
        if ph > 0 and h >= max(34, int(0.18 * ph)):
            return True
    # Absolute page guard for unparented synthetic title strips.
    if h >= max(38, int(0.075 * H)) and w >= int(0.22 * W):
        return True
    return False


def _drop_false_minitable_extension(b: VisibleBox, W: int, H: int) -> bool:
    """Remove mini-table extension rows that actually span the main body."""
    bid = b.box_id
    if "_ext" not in bid:
        return False
    if not (bid.endswith("_mtrow") or "_mtcell" in bid):
        return False
    w, h = _wh(b)
    if w >= max(150, int(0.42 * W)):
        return True
    if h <= 2 and w >= int(0.22 * W):
        return True
    return False


def cleanup_boxes(boxes: Iterable[VisibleBox], W: int, H: int) -> tuple[list[VisibleBox], dict[str, int]]:
    """Return cleaned boxes and a compact drop-count report."""
    src = list(boxes)
    by_id = {b.box_id: b for b in src}
    dropped: dict[str, int] = {
        "false_purple": 0,
        "false_colhdr": 0,
        "overbroad_title": 0,
        "false_minitable_ext": 0,
    }
    drop_ids: set[str] = set()
    for b in src:
        if _drop_false_purple(b, W, H):
            drop_ids.add(b.box_id)
            dropped["false_purple"] += 1
            continue
        if _drop_false_colhdr(b, W, H):
            drop_ids.add(b.box_id)
            dropped["false_colhdr"] += 1
            continue
        if _drop_overbroad_title_band(b, by_id, W, H):
            drop_ids.add(b.box_id)
            dropped["overbroad_title"] += 1
            continue
        if _drop_false_minitable_extension(b, W, H):
            drop_ids.add(b.box_id)
            dropped["false_minitable_ext"] += 1
            continue

    cleaned = [b for b in src if b.box_id not in drop_ids]

    # Repair dangling parent pointers after synthetic cleanup.  The box remains
    # geometrically useful, but the deleted parent must not imply a hidden layer.
    if drop_ids:
        cleaned = [
            replace(b, parent_box_id=None) if b.parent_box_id in drop_ids else b
            for b in cleaned
        ]
    return cleaned, dropped


@dataclass
class SemanticCleanupPass:
    """Post-detection cleanup that makes overlay colors match legend semantics."""

    info: PassInfo = PassInfo(
        name="semantic_cleanup",
        stage="postprocess",
        layer_flag=None,
        order=200,
        description=(
            "Drop false purple slabs, overbroad cyan/header hulls, body-sized "
            "blue title washes, and mini-table extensions that bleed into main text."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None:
            return state
        result = state.result
        cleaned, dropped = cleanup_boxes(result.boxes, result.image_width, result.image_height)
        stats = dict(result.debug_stats or {})
        stats["semantic_cleanup"] = dropped
        stats["total_before_cleanup"] = len(result.boxes)
        stats["total_after_cleanup"] = len(cleaned)
        stats["purple_after_cleanup"] = sum(1 for b in cleaned if b.color == "PURPLE")
        stats["colhdr_after_cleanup"] = sum(1 for b in cleaned if b.box_id.startswith("colhdr_"))
        state.result = VisibleBoxResult(
            boxes=cleaned,
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats=stats,
        )
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        state.artifacts["semantic_cleanup"] = dropped
        return state
