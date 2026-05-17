"""OpenCV template-matching pass for low-voltage symbol icons.

The text-only candidate detector misses icons that aren't accompanied
by a nearby native-text label — common on Cooper-Carry T-sets for
ballroom levels (T1.03) and the lobby (T1.01) where the architects
relied on the legend's icon shape to identify each device.

This module rasterizes each legend "cell" into a small template image
on first parse, then runs ``cv2.matchTemplate`` across every
floor_plan / typical_plan page. Surviving matches become
``SymbolCandidate`` objects tagged ``source_methods=["shape_template"]``
that fuse with the native-text candidates downstream.

OpenCV is optional in the same sense that PyMuPDF is in the rest of
the takeoff pipeline. The module imports ``cv2`` lazily; if the
import fails, the public ``shape_candidates_for_page`` function returns
an empty list and the pipeline records a one-time warning.

Threshold note (from spec):
    Start at ``MATCH_THRESHOLD = 0.85`` and iterate. WN cross-validated
    count should be very close to text-only count (335) — large
    divergence means the threshold needs tuning or the template is
    wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.ids import stable_id
from app.takeoff.plan_regions import is_excluded, is_inside
from app.takeoff.schemas import BBox, LegendRule, SheetRecord, SymbolCandidate

# ─── Tuning constants ───
# Template rendering scale — the legend page is rendered at LEGEND_DPI
# / 72 (~3x).
LEGEND_RENDER_SCALE = 3.0
# Plan rendering scale — same factor for templates to match.
PLAN_RENDER_SCALE = 3.0
# Template-matching score threshold (NCC, ``TM_CCOEFF_NORMED``). 0.85
# was the spec's starting point.
MATCH_THRESHOLD = 0.85
# Non-max suppression radius in pixels (at PLAN_RENDER_SCALE).
NMS_RADIUS_PX = 18.0
# A template smaller than this is too tiny to match reliably — skip.
MIN_TEMPLATE_PX = 12
# Pad each legend-cell crop by this many pixels in every direction.
CELL_PAD_PX = 4


@dataclass
class ShapeTemplate:
    """A single legend-cell raster template."""

    raw_symbol: str
    image: Any  # numpy.ndarray
    height: int
    width: int
    legend_bbox_pt: BBox  # the bbox we cropped from, in PDF points


def _try_import_cv2() -> tuple[Any, Any, str | None]:
    """Return ``(cv2, np, reason)`` — reason is non-None on failure."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - env-specific
        return (None, None, f"opencv_import_failed: {exc}")
    return (cv2, np, None)


def _render_page_grayscale(page: Any, scale: float) -> tuple[Any, float] | None:
    """Render a PyMuPDF page to a grayscale numpy array at ``scale``."""
    cv2, np, reason = _try_import_cv2()
    if reason is not None:
        return None
    try:
        import fitz  # type: ignore
    except Exception:  # pragma: no cover - env-specific
        return None
    try:
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape((pix.height, pix.width))
        return (arr, float(scale))
    except Exception:
        return None


def extract_templates_from_legend(
    legend_page: Any,
    legend_rules: list[LegendRule],
) -> list[ShapeTemplate]:
    """Crop one template image per legend rule from the legend page.

    Strategy: locate each rule's ``raw_symbol`` text on the legend
    page, then crop a square region centered on a point slightly to
    the LEFT of the text — that's where the icon sits in the
    Cooper-Carry legend cells. Empty / blank cells produce no template.

    Side effect: each matched rule has its ``source_bbox`` populated
    with the legend cell location in PDF points.
    """
    cv2, np, reason = _try_import_cv2()
    if reason is not None:
        return []
    if legend_page is None:
        return []
    rendered = _render_page_grayscale(legend_page, LEGEND_RENDER_SCALE)
    if rendered is None:
        return []
    image, scale = rendered

    try:
        words = legend_page.get_text("words")
    except Exception:
        return []

    word_index: dict[str, list[tuple[float, float, float, float]]] = {}
    for w in words:
        word_index.setdefault(w[4], []).append((w[0], w[1], w[2], w[3]))

    templates: list[ShapeTemplate] = []
    seen_symbols: set[str] = set()
    img_h, img_w = image.shape[:2]

    for rule in legend_rules:
        sym = rule.raw_symbol
        if sym in seen_symbols:
            continue
        positions = word_index.get(sym, [])
        if not positions:
            continue
        # Cooper-Carry T0.01 legend cells put the icon CENTERED
        # below the label text (a small glyph — triangle, circle,
        # square, etc.). Try EVERY occurrence of the symbol on the
        # legend page; for each, search a small grid around the
        # label for the offset with the highest ink density × centrality
        # score. Pick the best globally.
        best: tuple[float, float, float, float] | None = None
        best_score = 0.0
        for (x0, y0, x1, y1) in positions:
            text_h = y1 - y0
            label_cx = (x0 + x1) / 2.0
            label_cy = (y0 + y1) / 2.0
            side = max(text_h * 2.2, 18.0)
            # Sweep BELOW the label first (most cells); also try a
            # smaller sweep above and to the side.
            dy_steps = (
                text_h * 1.0,
                text_h * 1.5,
                text_h * 2.0,
                text_h * 2.5,
                text_h * 3.0,
                -text_h * 1.5,
                -text_h * 2.5,
            )
            dx_steps = (0.0, -side * 0.8, side * 0.8)
            for dy in dy_steps:
                for dx in dx_steps:
                    cx_pt = label_cx + dx
                    cy_pt = label_cy + dy
                    cell_x0 = max(0.0, cx_pt - side / 2.0)
                    cell_y0 = max(0.0, cy_pt - side / 2.0)
                    cell_x1 = cx_pt + side / 2.0
                    cell_y1 = cy_pt + side / 2.0
                    px0 = int(round(cell_x0 * scale)) - CELL_PAD_PX
                    py0 = int(round(cell_y0 * scale)) - CELL_PAD_PX
                    px1 = int(round(cell_x1 * scale)) + CELL_PAD_PX
                    py1 = int(round(cell_y1 * scale)) + CELL_PAD_PX
                    px0 = max(0, px0)
                    py0 = max(0, py0)
                    px1 = min(img_w, px1)
                    py1 = min(img_h, py1)
                    if px1 - px0 < MIN_TEMPLATE_PX or py1 - py0 < MIN_TEMPLATE_PX:
                        continue
                    crop = image[py0:py1, px0:px1]
                    ink = (crop < 200).astype("uint8")
                    ink_total = int(ink.sum())
                    ink_h, ink_w = ink.shape
                    ink_density = ink_total / max(1, ink_h * ink_w)
                    # Reject cells with too much ink — they're probably
                    # touching adjacent label rows or grid lines (>40%
                    # ink fraction).
                    if ink_density > 0.40 or ink_density < 0.02:
                        continue
                    # Prefer crops where ink concentrates centrally.
                    # The centroid distance from the geometric center
                    # gives a centrality penalty.
                    ys_ink, xs_ink = ink.nonzero()
                    if ys_ink.size == 0:
                        continue
                    centroid_x = xs_ink.mean()
                    centroid_y = ys_ink.mean()
                    cx_dist = abs(centroid_x - ink_w / 2.0) / ink_w
                    cy_dist = abs(centroid_y - ink_h / 2.0) / ink_h
                    centrality = 1.0 - min(1.0, cx_dist + cy_dist)
                    score = ink_density * centrality
                    if score > best_score:
                        best_score = score
                        best = (cell_x0, cell_y0, cell_x1, cell_y1)
        if best is None:
            continue
        cell_x0, cell_y0, cell_x1, cell_y1 = best
        px0 = int(round(cell_x0 * scale)) - CELL_PAD_PX
        py0 = int(round(cell_y0 * scale)) - CELL_PAD_PX
        px1 = int(round(cell_x1 * scale)) + CELL_PAD_PX
        py1 = int(round(cell_y1 * scale)) + CELL_PAD_PX
        px0 = max(0, px0)
        py0 = max(0, py0)
        px1 = min(img_w, px1)
        py1 = min(img_h, py1)
        crop = image[py0:py1, px0:px1]
        templates.append(
            ShapeTemplate(
                raw_symbol=sym,
                image=crop,
                height=crop.shape[0],
                width=crop.shape[1],
                legend_bbox_pt=BBox(
                    x0=cell_x0,
                    y0=cell_y0,
                    x1=cell_x1,
                    y1=cell_y1,
                    coord_space="pdf_pt",
                ),
            )
        )
        rule.source_bbox = BBox(
            x0=cell_x0,
            y0=cell_y0,
            x1=cell_x1,
            y1=cell_y1,
            coord_space="pdf_pt",
        )
        seen_symbols.add(sym)
    return templates


# ─── Match runner ───────


def _nms(points: list[tuple[int, int, float]], radius: float) -> list[tuple[int, int, float]]:
    """Greedy non-max suppression on (x, y, score) tuples.

    Sorted by score descending; keep a point unless there's already a
    surviving point within ``radius`` pixels.
    """
    points = sorted(points, key=lambda p: p[2], reverse=True)
    kept: list[tuple[int, int, float]] = []
    for x, y, s in points:
        ok = True
        for kx, ky, _ in kept:
            if (kx - x) ** 2 + (ky - y) ** 2 < radius ** 2:
                ok = False
                break
        if ok:
            kept.append((x, y, s))
    return kept


def shape_candidates_for_page(
    *,
    page: Any,
    sheet: SheetRecord,
    templates: list[ShapeTemplate],
    rules_by_symbol: dict[str, LegendRule],
    threshold: float = MATCH_THRESHOLD,
) -> list[SymbolCandidate]:
    """Run template matching on a single page; return shape candidates.

    ``templates`` is the list returned by ``extract_templates_from_legend``.
    Pages that aren't device-bearing (page_type not in
    {floor_plan, typical_plan}) get scanned only to emit rejection
    candidates with ``rejection_reason="non_floor_plan"``.

    Each surviving (post-NMS) match becomes a ``SymbolCandidate``
    with ``source_methods=["shape_template"]``. Confidence is mapped
    from the NCC score linearly into ``[0.70, 0.99]`` so 0.85 ≈ 0.70
    and 1.00 ≈ 0.99.
    """
    if not templates:
        return []
    cv2, np, reason = _try_import_cv2()
    if reason is not None:
        return []

    rendered = _render_page_grayscale(page, PLAN_RENDER_SCALE)
    if rendered is None:
        return []
    image, scale = rendered

    viewport = sheet.plan_viewport
    excluded = sheet.excluded_regions

    candidates: list[SymbolCandidate] = []
    page_type = sheet.page_type
    is_device_bearing = page_type in {"floor_plan", "typical_plan"} and sheet.in_scope

    for template in templates:
        tpl = template.image
        th, tw = tpl.shape[:2]
        if th < MIN_TEMPLATE_PX or tw < MIN_TEMPLATE_PX:
            continue
        if image.shape[0] < th + 1 or image.shape[1] < tw + 1:
            continue
        try:
            res = cv2.matchTemplate(image, tpl, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        # Pull all match coords above threshold.
        ys, xs = np.where(res >= threshold)
        points: list[tuple[int, int, float]] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            score = float(res[y, x])
            # Centroid coords (top-left of match → add half template).
            cx = int(x + tw / 2.0)
            cy = int(y + th / 2.0)
            points.append((cx, cy, score))
        kept = _nms(points, NMS_RADIUS_PX)
        rule = rules_by_symbol.get(template.raw_symbol)
        normalized = rule.normalized_class if rule else None
        for cx_px, cy_px, score in kept:
            # Convert image pixels → PDF points.
            cx_pt = cx_px / scale
            cy_pt = cy_px / scale
            half_pt = (max(th, tw) / 2.0) / scale
            bbox = BBox(
                x0=cx_pt - half_pt,
                y0=cy_pt - half_pt,
                x1=cx_pt + half_pt,
                y1=cy_pt + half_pt,
                coord_space="pdf_pt",
            )
            rejection_reason: str | None = None
            if not is_device_bearing:
                rejection_reason = f"page_type={page_type} is not device-bearing"
            elif viewport is not None and not is_inside(bbox, viewport):
                rejection_reason = "outside plan_viewport"
            elif excluded and is_excluded(bbox, excluded):
                rejection_reason = "inside excluded_region (titleblock)"
            # Map NCC score [threshold .. 1.0] → confidence [0.70 .. 0.99]
            t = max(0.0, min(1.0, (score - threshold) / max(1e-6, 1.0 - threshold)))
            confidence = 0.70 + t * 0.29
            candidates.append(
                SymbolCandidate(
                    id=stable_id(
                        "shapecand",
                        sheet.page_index,
                        template.raw_symbol,
                        round(cx_pt, 1),
                        round(cy_pt, 1),
                    ),
                    page_index=sheet.page_index,
                    raw_symbol=template.raw_symbol,
                    normalized_class=normalized,
                    bbox=bbox,
                    source_methods=["shape_template"],
                    confidence=confidence if rejection_reason is None else 0.5,
                    rejection_reason=rejection_reason,
                    needs_review=True,  # shape-only candidates ALWAYS need review
                    nearby_text=[],
                )
            )
    return candidates


__all__ = [
    "MATCH_THRESHOLD",
    "ShapeTemplate",
    "extract_templates_from_legend",
    "shape_candidates_for_page",
]
