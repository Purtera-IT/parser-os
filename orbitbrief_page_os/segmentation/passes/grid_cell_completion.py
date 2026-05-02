"""Grid-cell completion pass for schedule tables.

Root cause addressed
--------------------
The core contour detector uses a morphological closing kernel whose width
exceeds the narrowest column cells in dense engineering schedules.  This merges
adjacent cells before thresholding, so the narrow columns are never generated
as standalone candidates.  The resulting gap is systematic: leftmost TAG-column
cells and any column narrower than ~line_kernel_px are absent from the orange
overlay even though strong black grid rules are visible in the raster image.

This pass fixes the gap by a different strategy: instead of re-running contour
detection, it directly extracts horizontal and vertical raster-line positions
inside each qualifying BLUE structural wrapper, forms a grid of cell rectangles
from adjacent line pairs, and emits thin synthetic ORANGE boxes for every cell
not already covered by an existing detection.

What this pass does NOT do
--------------------------
- It does not edit core chunks.
- It does not replace the contour detector.
- It does not emit boxes over title-block artwork or PURPLE regions.
- It does not emit wide title-band or paragraph-area boxes.
- It does not run outside BLUE schedule wrappers.
- All thresholds are exposed through Cfg (new fields added in config.py /
  Cfg subclass) so nothing is hardcoded at call sites.

Universal rule
--------------
Inside any BLUE wrapper with >= min_orange_children existing ORANGE detections,
if the raster image shows repeated H-lines and V-lines forming a grid, emit thin
ORANGE cell boxes for every grid cell not already covered.

Dedup metric: coverage-of-candidate (intersection / candidate_area).  A
candidate is skipped when any existing ORANGE box covers >= gcc_dedup_coverage_ratio
of its area.  This correctly passes full-row candidates (11 px tall) even when
a thin 8 px thin strip already exists in the same row — the strip covers only
~73 % of the candidate area, below the default 0.85 threshold.  Pure IoU would
suppress these candidates because the strip's area is similar to the candidate's,
pushing IoU above a low threshold like 0.35.

Reject candidate cells that are full-wrapper-width (divider/title strip),
taller than max_cell_h_px, wider than max_cell_w_fraction of wrapper,
or overlapping an existing BLUE title/PURPLE box with IoU >= title_guard_iou.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from ..core.models import Rect, VisibleBox, VisibleBoxResult
from ..rules import candidate_bisects_any_word
from .base import PageContext, PassInfo, PipelineState

# ---------------------------------------------------------------------------
# Threshold defaults (overridable via Cfg attributes of the same name)
# ---------------------------------------------------------------------------
_D = {
    # Minimum existing ORANGE children a BLUE wrapper must have to be treated
    # as a schedule table worth completing.
    "gcc_min_orange_children": 4,
    # Minimum number of distinct H-line positions required before we attempt
    # grid synthesis (avoids running on plain content blocks).
    "gcc_min_hlines": 2,
    # Minimum number of distinct V-line positions (columns need at least 2
    # vertical edges to form one cell column).
    "gcc_min_vlines": 2,
    # Pixel tolerance for clustering nearby line positions into one canonical
    # position (handles antialiasing / sub-pixel jitter in PDF rules).
    "gcc_line_cluster_tol": 3,
    # Minimum fraction of wrapper width a horizontal line must cover to count
    # as a grid rule (avoids short text underlines triggering column detection).
    "gcc_hline_min_coverage": 0.30,
    # Minimum fraction of wrapper height a vertical line must cover to count
    # as a column separator.  0.10 was too low: text strokes in label cells
    # (TAG column, ~15-20 px tall) triggered false V-lines that produced 7-8 px
    # wide slivers across the whole left margin.  Real column separators span
    # ≥ 50-90 % of the wrapper height; 0.30 filters text strokes while keeping
    # all genuine full-height column rules.
    "gcc_vline_min_coverage": 0.30,
    # Morphological kernel lengths for line extraction inside the wrapper crop.
    # Chosen to be shorter than line_kernel_px so narrow cells are not merged.
    "gcc_h_morph_px": 12,
    "gcc_v_morph_px": 8,
    # Cell candidate rejection: a cell wider than this fraction of wrapper
    # width is likely a divider strip / title band, not a data cell.
    "gcc_max_cell_w_fraction": 0.97,
    # Cell candidate rejection: a cell taller than this many pixels is a
    # paragraph / notes region, not a schedule row.
    "gcc_max_cell_h_px": 60,
    # Cell candidate rejection: minimum cell height to avoid sub-pixel noise.
    "gcc_min_cell_h_px": 4,
    # Cell candidate rejection: minimum cell width.  Raised from 6 to 12 as a
    # belt-and-suspenders guard against any text-stroke slivers that survive the
    # vline_min_coverage filter.  No real schedule column is narrower than 12 px
    # at scale 2.0.
    "gcc_min_cell_w_px": 12,
    # Coverage-of-candidate threshold for deduplication against existing ORANGE
    # boxes.  A candidate is skipped when any existing ORANGE box covers >=
    # this fraction of the candidate's own area (inter / candidate_area).
    # Using coverage-of-candidate instead of IoU lets full-row candidates
    # (e.g. 11 px tall) survive even when a thin 8 px thin strip already
    # occupies the same row: the strip covers ~73 % of the candidate, which is
    # below the 0.85 threshold, so the candidate is kept and the complete row
    # gets a proper orange box.  A genuine duplicate (same or larger box)
    # achieves coverage ≈ 1.0 and is correctly suppressed.
    "gcc_dedup_coverage_ratio": 0.85,
    # IoU threshold against BLUE title and PURPLE boxes.  A candidate
    # overlapping a title/logo by >= this is rejected.
    "gcc_title_guard_iou": 0.10,
    # Hard cap on synthetic cells emitted per wrapper (safety valve).
    "gcc_max_cells_per_wrapper": 600,
}


def _cfg_get(cfg, key: str):
    """Read a threshold from Cfg if it has the attribute, else use default."""
    return getattr(cfg, key, _D[key])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(union, 1)


def _cluster_positions(positions: list[int], tol: int) -> list[int]:
    """Merge nearby line positions into one canonical value (median of group)."""
    if not positions:
        return []
    positions = sorted(set(positions))
    groups: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(np.median(g)) for g in groups]


# ---------------------------------------------------------------------------
# Raster line extraction
# ---------------------------------------------------------------------------

def _extract_grid_lines(
    crop: np.ndarray,
    crop_x0: int,
    crop_y0: int,
    wrapper_w: int,
    wrapper_h: int,
    cfg,
) -> tuple[list[int], list[int]]:
    """Return lists of absolute pixel y-positions (H-lines) and x-positions (V-lines)."""
    h_morph = _cfg_get(cfg, "gcc_h_morph_px")
    v_morph = _cfg_get(cfg, "gcc_v_morph_px")
    hline_min_cov = _cfg_get(cfg, "gcc_hline_min_coverage")
    vline_min_cov = _cfg_get(cfg, "gcc_vline_min_coverage")
    cluster_tol = _cfg_get(cfg, "gcc_line_cluster_tol")

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Binarise: dark pixels = potential rules.
    _, dark = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    rh, rw = dark.shape[:2]
    if rh < 4 or rw < 4:
        return [], []

    # --- Horizontal lines ---
    h_kern = max(h_morph, int(hline_min_cov * rw))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kern, 1))
    hmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)

    h_positions: list[int] = []
    cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < int(hline_min_cov * rw):
            continue
        if h > 6:
            continue
        y_mid = crop_y0 + y + h // 2
        h_positions.append(y_mid)

    # --- Vertical lines ---
    v_kern = max(v_morph, int(vline_min_cov * rh))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kern))
    vmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)

    v_positions: list[int] = []
    cnts, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < int(vline_min_cov * rh):
            continue
        if w > 6:
            continue
        x_mid = crop_x0 + x + w // 2
        v_positions.append(x_mid)

    h_positions = _cluster_positions(h_positions, cluster_tol)
    v_positions = _cluster_positions(v_positions, cluster_tol)
    return h_positions, v_positions


# ---------------------------------------------------------------------------
# Cell synthesis from grid intersections
# ---------------------------------------------------------------------------

def _coverage_of_candidate(
    cand: tuple[int, int, int, int],
    ex: tuple[int, int, int, int],
) -> float:
    """Fraction of candidate area that is covered by existing box `ex`.

    Returns inter(cand, ex) / area(cand).  Used for deduplication so that
    thin thin strips do not suppress taller full-row candidates.
    """
    ax0, ay0, ax1, ay1 = cand
    bx0, by0, bx1, by1 = ex
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    cand_area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / cand_area


def _synthesize_cells(
    h_lines: list[int],
    v_lines: list[int],
    wrapper: VisibleBox,
    existing_orange: list[tuple[int, int, int, int]],
    guard_boxes: list[tuple[int, int, int, int]],
    cfg,
    scale: float,
    counter_start: int,
    pdf_word_image_xspans: list[tuple[int, int, int, int]] | None = None,
) -> list[VisibleBox]:
    """Build ORANGE cell boxes from adjacent line pairs and dedup against existing."""
    dedup_coverage = _cfg_get(cfg, "gcc_dedup_coverage_ratio")
    title_guard_iou  = _cfg_get(cfg, "gcc_title_guard_iou")
    max_cell_h = _cfg_get(cfg, "gcc_max_cell_h_px")
    min_cell_h = _cfg_get(cfg, "gcc_min_cell_h_px")
    max_cell_w_frac = _cfg_get(cfg, "gcc_max_cell_w_fraction")
    min_cell_w = _cfg_get(cfg, "gcc_min_cell_w_px")
    max_per_wrapper = _cfg_get(cfg, "gcc_max_cells_per_wrapper")

    wx0, wy0, wx1, wy1 = wrapper.px_bbox
    wrapper_w = max(1, wx1 - wx0)

    emitted: list[VisibleBox] = []
    idx = counter_start

    for (y0, y1), (x0, x1) in itertools.product(
        zip(h_lines, h_lines[1:]), zip(v_lines, v_lines[1:])
    ):
        if len(emitted) >= max_per_wrapper:
            break

        cell_h = y1 - y0
        cell_w = x1 - x0

        # Reject out-of-wrapper or degenerate
        if cell_h < min_cell_h or cell_w < min_cell_w:
            continue
        if x0 < wx0 - 4 or x1 > wx1 + 4 or y0 < wy0 - 4 or y1 > wy1 + 4:
            continue

        # Reject title-band / full-width dividers
        if cell_h > max_cell_h:
            continue
        if cell_w >= int(max_cell_w_frac * wrapper_w):
            continue

        candidate = (int(x0), int(y0), int(x1), int(y1))

        # Skip if the candidate area is already substantially covered by an
        # existing orange box.  We use coverage-of-candidate (inter /
        # candidate_area) rather than IoU: thin 8 px thin strips share high
        # IoU with an 11 px full-row candidate because both areas are similar,
        # but the strip only covers ~73 % of the candidate — below the 0.85
        # threshold — so the full-row box is kept and painted correctly.
        if any(_coverage_of_candidate(candidate, ex) >= dedup_coverage
               for ex in existing_orange):
            continue

        # Skip if it overlaps a title/logo guard box
        if any(_iou(candidate, g) >= title_guard_iou for g in guard_boxes):
            continue

        # Universal text-bisection rule — inline impl mirrors rules/text_bisection.py
        # exactly.  The rule is the contract; this is the call site.
        if pdf_word_image_xspans:
            cand_x0, cand_y0, cand_x1, cand_y1 = candidate
            bisects = False
            for (wx0, wy0, wx1, wy1) in pdf_word_image_xspans:
                if wy1 <= cand_y0 or wy0 >= cand_y1:
                    continue
                tol = 1
                if (wx0 + tol) < cand_x0 < (wx1 - tol):
                    bisects = True
                    break
                if (wx0 + tol) < cand_x1 < (wx1 - tol):
                    bisects = True
                    break
            if bisects:
                continue

        bid = f"gridcell_{idx:04d}"
        px = (int(x0), int(y0), int(x1), int(y1))
        emitted.append(VisibleBox(
            box_id=bid,
            rect=Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
            area_pt2=float(cell_w * cell_h) / max(scale * scale, 1e-6),
            fill_ratio=1.0,
            nested_depth=2,
            is_outer_wrapper=False,
            parent_box_id=wrapper.box_id,
            color="ORANGE",
            px_bbox=px,
            children_count=0,
            # synthetic=False so the core renderer's Pass 3 draws these as
            # regular orange cells.  The core renderer splits boxes into
            # `regular` (synthetic=False) and `synth` (synthetic=True); only
            # `regular` ORANGE boxes reach the orange paint pass.  The earlier
            # `synth` orange paths only handle named patterns (gapcell_,
            # gapsep_, gapcell_hdr_) and would silently drop gridcell_ boxes.
            # Semantically correct too: these ARE real data cells, just
            # detected by raster line geometry rather than contour+NMS.
            synthetic=False,
        ))
        idx += 1

    return emitted


# ---------------------------------------------------------------------------
# Per-wrapper orchestration
# ---------------------------------------------------------------------------

def _orange_children(wrapper: VisibleBox, all_boxes: Sequence[VisibleBox]) -> list[VisibleBox]:
    wid = wrapper.box_id
    return [b for b in all_boxes if b.color == "ORANGE" and b.parent_box_id == wid]


def _has_blue_children(wrapper: VisibleBox, all_boxes: Sequence[VisibleBox]) -> bool:
    """True when any real (non-synthetic) BLUE box is a direct child of this wrapper.

    A BLUE wrapper that contains other real BLUE wrappers is a structural
    container (the outer page frame or a section group), not a leaf schedule
    table.  Running grid synthesis on a structural container would synthesize
    cells across the entire page and pollute the dedup set for inner tables.

    Synthetic BLUE boxes (e.g. ``_title`` boxes from CellularTitlePass) are
    excluded so this gate stays stable even after those passes have run.
    """
    wid = wrapper.box_id
    return any(
        b.color == "BLUE"
        and b.parent_box_id == wid
        and not getattr(b, "synthetic", False)
        for b in all_boxes
    )


def _is_schedule_wrapper(
    wrapper: VisibleBox,
    orange_children: list[VisibleBox],
    all_boxes: Sequence[VisibleBox],
    cfg,
) -> bool:
    """True when the wrapper is a leaf schedule table worth completing.

    Gates:
    - must be BLUE (is_outer_wrapper=True on all BLUE boxes in the core engine
      detector — it is a synonym for 'this box is structural/BLUE', not a
      meaningful frame-depth signal, so we do NOT gate on it here)
    - must not contain other BLUE boxes as direct children: a wrapper that
      has BLUE children is a structural container (e.g. the page frame v0),
      not a leaf schedule table
    - must have >= gcc_min_orange_children existing orange cells: confirms
      the core detector found a table here (avoids empty content blocks)
    """
    min_ch = _cfg_get(cfg, "gcc_min_orange_children")
    if wrapper.color != "BLUE":
        return False
    if _has_blue_children(wrapper, all_boxes):
        return False
    return len(orange_children) >= min_ch


def complete_grid_cells(
    rgb: np.ndarray,
    boxes: list[VisibleBox],
    scale: float,
    cfg,
    pdf_word_image_xspans: list[tuple[int, int, int, int]] | None = None,
) -> list[VisibleBox]:
    """Main entry: return a list of new synthetic ORANGE cell boxes to append.

    ``pdf_word_image_xspans`` is an optional list of ``(wx0, wy0, wx1, wy1)``
    image-pixel bboxes, one per PDF text word.  When provided, candidate
    cells whose left or right edge would bisect a word are rejected.  This
    enforces the universal rule that an orange line never cuts through text
    — relevant for merged column headers like ``ELECTRICAL DATA`` that span
    multiple sub-columns whose vertical separator extends up into the
    parent header band.
    """
    H, W = rgb.shape[:2]
    min_hlines = _cfg_get(cfg, "gcc_min_hlines")
    min_vlines = _cfg_get(cfg, "gcc_min_vlines")

    # Build lookup structures
    all_orange_bboxes = [b.px_bbox for b in boxes if b.color == "ORANGE"]
    guard_bboxes = [
        b.px_bbox for b in boxes
        if b.color in ("BLUE", "PURPLE") and (
            b.box_id.endswith("_title") or b.box_id.startswith("titleblk")
        )
    ]

    new_cells: list[VisibleBox] = []
    cell_counter = 0

    for wrapper in boxes:
        if wrapper.color != "BLUE":
            continue

        orange_ch = _orange_children(wrapper, boxes)
        if not _is_schedule_wrapper(wrapper, orange_ch, boxes, cfg):
            continue

        wx0, wy0, wx1, wy1 = (int(v) for v in wrapper.px_bbox)
        # Clamp to image bounds
        cx0 = max(0, wx0)
        cy0 = max(0, wy0)
        cx1 = min(W, wx1)
        cy1 = min(H, wy1)
        if cx1 <= cx0 + 10 or cy1 <= cy0 + 10:
            continue

        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue

        h_lines, v_lines = _extract_grid_lines(
            crop, cx0, cy0,
            wrapper_w=cx1 - cx0,
            wrapper_h=cy1 - cy0,
            cfg=cfg,
        )

        # Anchor lines to wrapper edges so cells span the full grid
        if v_lines:
            if abs(v_lines[0] - cx0) > 6:
                v_lines = [cx0] + v_lines
            if abs(v_lines[-1] - cx1) > 6:
                v_lines = v_lines + [cx1]
        if h_lines:
            if abs(h_lines[0] - cy0) > 6:
                h_lines = [cy0] + h_lines
            if abs(h_lines[-1] - cy1) > 6:
                h_lines = h_lines + [cy1]

        if len(h_lines) < min_hlines + 1 or len(v_lines) < min_vlines + 1:
            continue

        cells = _synthesize_cells(
            h_lines=h_lines,
            v_lines=v_lines,
            wrapper=wrapper,
            existing_orange=all_orange_bboxes,
            guard_boxes=guard_bboxes,
            cfg=cfg,
            scale=scale,
            counter_start=cell_counter,
            pdf_word_image_xspans=pdf_word_image_xspans,
        )
        new_cells.extend(cells)
        # Register newly emitted cells into the dedup set for subsequent wrappers
        all_orange_bboxes.extend(c.px_bbox for c in cells)
        cell_counter += len(cells)

    return new_cells


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------

@dataclass
class GridCellCompletionPass:
    """Synthesize missing ORANGE cell boxes from raster grid lines.

    This pass is designed to run after SemanticCleanupPass (order 200) and
    before RasterLineRepairPass (order 240).  It targets BLUE structural
    wrappers that already have some ORANGE children (confirming a schedule
    table was detected) and fills in cells missed by the contour detector.

    The fix is universal: it depends only on raster line geometry and
    containment relationships, not on PDF filenames or coordinates.
    """

    info: PassInfo = PassInfo(
        name="grid_cell_completion",
        stage="synthesize",
        layer_flag="ORANGE",
        order=230,
        description=(
            "Synthesize missing ORANGE cell boxes from repeated H/V raster grid "
            "lines inside BLUE schedule wrappers.  Fills gaps left by the contour "
            "detector when cells are narrower than the morphological close kernel."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None or state.rgb is None:
            return state

        result = state.result
        scale = float(
            result.debug_stats.get("render_scale_used")
            or ctx.cfg.render_scale
            or 1.0
        )

        # Build PDF-word image-coord bboxes for the text-bisection guard.
        # Failure modes (no fitz, page can't be opened, rotation lookup fails)
        # are non-fatal: the guard becomes a no-op and emission falls back
        # to the raster-only behaviour.
        word_xspans: list[tuple[int, int, int, int]] = []
        try:
            import fitz as _fitz
            page_rotation_qt = int(
                result.debug_stats.get("rotated_cw_quarter_turns", 0) or 0)
            with _fitz.open(str(ctx.pdf_path)) as _doc:
                _page = _doc[ctx.page_index]
                pw = _page.rect.width
                ph = _page.rect.height
                qt = page_rotation_qt % 4

                def _pdf_to_img_xy(px, py):
                    if qt == 0:
                        return px * scale, py * scale
                    if qt == 3:
                        return py * scale, (pw - px) * scale
                    if qt == 1:
                        return (ph - py) * scale, px * scale
                    return (pw - px) * scale, (ph - py) * scale

                for w in _page.get_text("words") or []:
                    wx0, wy0, wx1, wy1 = (
                        float(w[0]), float(w[1]), float(w[2]), float(w[3]))
                    corners = [
                        _pdf_to_img_xy(wx0, wy0),
                        _pdf_to_img_xy(wx1, wy0),
                        _pdf_to_img_xy(wx0, wy1),
                        _pdf_to_img_xy(wx1, wy1),
                    ]
                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    word_xspans.append(
                        (int(min(xs)), int(min(ys)),
                         int(max(xs)), int(max(ys))))
        except Exception:
            word_xspans = []

        new_cells = complete_grid_cells(
            rgb=state.rgb,
            boxes=result.boxes,
            scale=scale,
            cfg=ctx.cfg,
            pdf_word_image_xspans=word_xspans,
        )

        stats = dict(result.debug_stats or {})
        stats["grid_cell_completion"] = len(new_cells)

        state.result = VisibleBoxResult(
            boxes=[*result.boxes, *new_cells],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats=stats,
        )
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        state.artifacts["grid_cell_completion"] = len(new_cells)
        return state


__all__ = [
    "GridCellCompletionPass",
    "complete_grid_cells",
    "extract_grid_lines_for_wrapper",
]


# Convenience alias used by tests
def extract_grid_lines_for_wrapper(rgb, wrapper_bbox, cfg=None):
    """Test helper: return (h_lines, v_lines) for a given wrapper bbox tuple."""
    from ..core.config import Cfg
    cfg = cfg or Cfg()
    x0, y0, x1, y1 = wrapper_bbox
    H, W = rgb.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    crop = rgb[cy0:cy1, cx0:cx1]
    return _extract_grid_lines(crop, cx0, cy0, cx1 - cx0, cy1 - cy0, cfg)
