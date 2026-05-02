"""High-confidence line repair for title-block/sidebar grids.

This pass is deliberately narrow: it does not try to rediscover every table in
an engineering sheet.  It looks for long, axis-aligned black rules in the
right-hand title-block/sidebar strip that the core contour classifier often
suppresses, and emits thin synthetic ORANGE line boxes.  That gives QA a visible
"there is structure here" signal without misclassifying the whole strip as a
purple logo.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.models import Rect, VisibleBox, VisibleBoxResult
from .base import PageContext, PassInfo, PipelineState


def _line_box(
    box_id: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    scale: float,
) -> VisibleBox:
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return VisibleBox(
        box_id=box_id,
        rect=Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
        area_pt2=max(1.0, (x1 - x0) * (y1 - y0) / max(scale * scale, 1e-6)),
        fill_ratio=1.0,
        nested_depth=3,
        is_outer_wrapper=False,
        parent_box_id=None,
        color="ORANGE",
        px_bbox=(x0, y0, x1, y1),
        children_count=0,
        synthetic=True,
    )


def _candidate_sidebar_roi(boxes: list[VisibleBox], W: int, H: int) -> tuple[int, int, int, int] | None:
    """Find the right-hand title block / sidebar strip from existing boxes."""
    candidates: list[tuple[int, int, int, int]] = []
    for b in boxes:
        x0, y0, x1, y1 = b.px_bbox
        w = x1 - x0
        h = y1 - y0
        if x0 < int(0.82 * W):
            continue
        if w > int(0.20 * W) or w < 12:
            continue
        if h < int(0.25 * H):
            continue
        if b.color not in ("BLUE", "ORANGE", "PURPLE"):
            continue
        candidates.append((x0, y0, x1, y1))
    if not candidates:
        return None
    x0 = max(0, min(c[0] for c in candidates) - 2)
    y0 = max(0, min(c[1] for c in candidates) - 2)
    x1 = min(W - 1, max(c[2] for c in candidates) + 2)
    y1 = min(H - 1, max(c[3] for c in candidates) + 2)
    if x1 <= x0 + 10 or y1 <= y0 + 20:
        return None
    return (x0, y0, x1, y1)


def _dedupe_segments(segments: list[tuple[int, int, int, int]], tol: int = 2) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for seg in sorted(segments, key=lambda s: (s[1], s[0], s[3], s[2])):
        x0, y0, x1, y1 = seg
        dup = False
        for k in kept:
            if (
                abs(x0 - k[0]) <= tol and abs(y0 - k[1]) <= tol
                and abs(x1 - k[2]) <= tol and abs(y1 - k[3]) <= tol
            ):
                dup = True
                break
        if not dup:
            kept.append(seg)
    return kept


def extract_sidebar_line_repairs(rgb: np.ndarray, boxes: list[VisibleBox], scale: float) -> list[VisibleBox]:
    H, W = rgb.shape[:2]
    roi = _candidate_sidebar_roi(boxes, W, H)
    if roi is None:
        return []
    rx0, ry0, rx1, ry1 = roi
    crop = rgb[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Dark PDF rules.  Text is later suppressed by morphology length filters.
    dark = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    rh, rw = dark.shape[:2]
    if rh < 20 or rw < 10:
        return []

    h_kernel = max(7, int(0.30 * rw))
    v_kernel = max(12, int(0.035 * rh))
    hmask = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1)),
        iterations=1,
    )
    vmask = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel)),
        iterations=1,
    )

    segments: list[tuple[int, int, int, int]] = []
    cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < max(10, int(0.42 * rw)):
            continue
        if h > 5:
            continue
        y_mid = ry0 + y + max(0, h // 2)
        segments.append((rx0 + x, y_mid, rx0 + x + w, y_mid + 1))

    cnts, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < max(14, int(0.045 * rh)):
            continue
        if w > 5:
            continue
        x_mid = rx0 + x + max(0, w // 2)
        segments.append((x_mid, ry0 + y, x_mid + 1, ry0 + y + h))

    segments = _dedupe_segments(segments)
    # Keep the pass high-confidence and bounded.
    if len(segments) > 180:
        segments = segments[:180]
    return [_line_box(f"line_repair_{i:03d}", *seg, scale) for i, seg in enumerate(segments)]


@dataclass
class RasterLineRepairPass:
    info: PassInfo = PassInfo(
        name="raster_line_repair",
        stage="postprocess",
        layer_flag="ORANGE",
        order=240,
        description="Add thin ORANGE repair strokes for missed sidebar/title-block grid rules.",
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None or state.rgb is None:
            return state
        result = state.result
        scale = float(result.debug_stats.get("render_scale_used") or ctx.cfg.render_scale or 1.0)
        repairs = extract_sidebar_line_repairs(state.rgb, result.boxes, scale)
        if not repairs:
            state.artifacts.setdefault("stage_order", []).append(self.info.name)
            state.artifacts["raster_line_repairs"] = 0
            return state
        stats = dict(result.debug_stats or {})
        stats["raster_line_repairs"] = len(repairs)
        state.result = VisibleBoxResult(
            boxes=[*result.boxes, *repairs],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats=stats,
        )
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        state.artifacts["raster_line_repairs"] = len(repairs)
        return state
