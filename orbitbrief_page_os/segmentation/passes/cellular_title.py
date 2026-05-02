"""Cellular title detection pass — schedule title band highlighter.

The earlier ``_title`` synthesis identifies schedule title bands (e.g. "PUMP
SCHEDULE") by scanning the PDF text layer for title strings, then emitting a
``{wrapper_id}_title`` synthetic BLUE box so the renderer's Pass 2 draws a
semi-transparent blue fill.  That path requires fitz/PyMuPDF, which may be
unavailable.

This pass replicates the same output using purely geometric signals:

    The topmost ORANGE band inside a BLUE schedule wrapper that spans
    ≥ ct_title_min_width_frac of the wrapper width is the title band.

Title-block cell branch (``tbstruct_*``)
----------------------------------------
Title-block right-margin cells from ``title_block_detection`` carry the same
"named, important text" semantics as schedule title rows but live in their own
cell grid.  Project-name cells (tall multi-line rotated text), abbreviation
keys, and drawing-index cells are the title-class content authors meant to
emphasise.  The universal raster signal is height-relative: a ``tbstruct_*``
cell whose band height is ``ct_tbstruct_title_height_mult`` × the median
tbstruct height earns a title highlight.  This catches the right cells across
drawing types (architectural, electrical, mechanical) without per-template
text matching, because every drawing's title block reserves its tallest cells
for project metadata.

Gate rules
----------
- Schedule branch: BLUE wrapper with no non-synthetic BLUE children (leaf table).
- Title-block branch: ``tbstruct_*`` cell whose height ≥ multiple × median
  ``tbstruct_*`` height (and ≥ ``ct_tbstruct_title_min_h_px`` absolute floor).
- The candidate cell top must be within ct_title_max_top_offset_px of the
  wrapper top (schedule branch only).
- Already-synthesised ``_title`` boxes are skipped to avoid duplication.

Output
------
``{wrapper_id}_title`` synthetic BLUE boxes consumed by the core renderer's
``draw_title_band_highlights`` in Pass 2.  No frozen code is touched.

All thresholds are exposed through Cfg (``ct_*`` prefix).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.models import Rect, VisibleBox, VisibleBoxResult
from .base import PageContext, PassInfo, PipelineState

# ---------------------------------------------------------------------------
# Threshold defaults
# ---------------------------------------------------------------------------
_D: dict[str, object] = {
    # An ORANGE cell must span at least this fraction of wrapper width to be
    # classified as the schedule title band.
    "ct_title_min_width_frac": 0.80,
    # The candidate cell's top edge must be within this many px of the wrapper
    # top edge.  Keeps wide-but-mid-table divider bands from misclassifying.
    "ct_title_max_top_offset_px": 80,
    # Title-block branch: a tbstruct_* cell qualifies as a title cell when its
    # band height is ≥ this multiple of the median tbstruct_* cell height.
    # 2.5× catches the project-name block (tall multi-line rotated text) and
    # any equivalently-tall section like an abbreviations panel, while leaving
    # the small CODE/REVISION/SCALE rows alone.  Lowering to 1.5× would start
    # tagging mid-size cells; raising to 4× would miss the abbreviation keys
    # that some sheets pack tighter than full project blocks.
    "ct_tbstruct_title_height_mult": 2.5,
    # Absolute floor in px so we don't tag a tall cell on a sheet where every
    # tbstruct is small (the multiple alone would still fire).  At render
    # scale 2.5, 100 px ≈ 0.4" of paper — comfortably above any small label
    # row (≤ 50 px) but below real title cells (≥ 150 px).
    "ct_tbstruct_title_min_h_px": 100,
    # Minimum dark-pixel density a tall tbstruct cell must have to qualify
    # as a title.  Empty data areas (e.g. blank revision-schedule body that
    # sits below its column-header strip) can be tall enough to pass the
    # height gate but contain no labelling text — their density is ≤ 0.06
    # because they're mostly white.  Real title cells always carry a
    # prominent label whose dark pixels push the density above 0.10 even
    # when the label is short relative to cell height (rotated text in
    # tall cells, label-with-padding in normal cells).  0.08 keeps a small
    # safety margin under the 0.10 minimum observed across labelled cells.
    "ct_tbstruct_title_min_density": 0.08,
    # Tabular-cluster (tbgroup_*) table-title detection: the topmost cell of
    # a cluster qualifies for a title wash when the cell directly below it
    # is a thin column-header strip.  ct_tbgroup_colhdr_max_h_px sets the
    # maximum height of that strip — 10 px catches the 6-7 px revision-
    # schedule header band on test5 while excluding normal label-value
    # rows (≥ 22 px).  Plain stacked label-value clusters (no thin band
    # below the top row) won't trigger this, which is the correct behaviour:
    # a JOB NO / DATE / DRAWN stack has no "table title."
    "ct_tbgroup_colhdr_max_h_px": 10,
}


def _cfg_get(cfg, key: str):
    return getattr(cfg, key, _D[key])


def _has_nonsynth_blue_children(
    wrapper: VisibleBox,
    all_boxes: Sequence[VisibleBox],
) -> bool:
    """True when any real (non-synthetic) BLUE box is a direct child.

    We exclude synthetic BLUE boxes (e.g. later-added _title boxes from this
    very pass) so the check remains stable across pipeline runs.
    """
    wid = wrapper.box_id
    return any(
        b.color == "BLUE"
        and b.parent_box_id == wid
        and not getattr(b, "synthetic", False)
        for b in all_boxes
    )


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_cellular_titles(
    boxes: list[VisibleBox],
    scale: float,
    cfg,
    rgb=None,
    pdf_path=None,
    page_index: int = 0,
    page_rotation_qt: int = 0,
) -> list[VisibleBox]:
    """Return ``{wrapper_id}_title`` synthetic BLUE boxes for each schedule wrapper.

    When ``rgb`` is provided, the title-block branch additionally checks the
    candidate cell's dark-pixel density before tagging it as a title.  An
    empty data area can be tall enough to pass the height gate but should
    not be treated as a labelled section — dense text, not size, is what
    makes a cell a title.  When ``rgb`` is None (e.g. unit-test calls), the
    density gate is skipped and the height-only rule applies.

    When ``pdf_path`` is provided, the tabular-cluster branch additionally
    emits ``colhdr_*`` CYAN rings for each column-key word inside the thin
    column-header strip below a cluster's title row, using PyMuPDF's text
    layer.  Without ``pdf_path`` (or without fitz), the rings simply do not
    fire — no false positives, just missing rings on the late-emitted
    tbgroup table titles.
    """
    min_w_frac = float(_cfg_get(cfg, "ct_title_min_width_frac"))
    max_top_off = int(_cfg_get(cfg, "ct_title_max_top_offset_px"))

    existing_title_ids = {
        b.box_id for b in boxes if b.box_id.endswith("_title")
    }

    new_boxes: list[VisibleBox] = []

    for wrapper in boxes:
        if wrapper.color != "BLUE":
            continue
        # Title-block structural cells are handled by the dedicated branch
        # below using a height-relative signal.  Skip them here to avoid
        # firing the schedule rule (every tbstruct ORANGE child fills its
        # parent and would always pass the 80% width gate).
        if wrapper.box_id.startswith("tbstruct_"):
            continue
        # Title-block tabular clusters (tbgroup_*) are pre-collapsed mini-tables
        # whose internal rows are intentionally ORANGE-only.  Painting a title
        # wash on the topmost row would defeat the very point of the collapse
        # by re-introducing blue inside the table on an internal cellular line.
        if wrapper.box_id.startswith("tbgroup_"):
            continue
        if _has_nonsynth_blue_children(wrapper, boxes):
            continue  # structural container, not a leaf schedule table

        wx0, wy0, wx1, wy1 = wrapper.px_bbox
        wrapper_w = max(1, wx1 - wx0)

        bid = f"{wrapper.box_id}_title"
        if bid in existing_title_ids:
            continue  # already synthesised by core text-extraction path

        # Collect non-synthetic, non-gridcell ORANGE children, top-to-bottom
        orange_ch = sorted(
            [
                b for b in boxes
                if b.parent_box_id == wrapper.box_id
                and b.color == "ORANGE"
                and not getattr(b, "synthetic", False)
                and not b.box_id.startswith("gridcell_")
            ],
            key=lambda b: b.px_bbox[1],
        )
        if not orange_ch:
            continue

        # Check the topmost few candidates for the title-band signature:
        # wide span + near the wrapper top
        for ob in orange_ch[:6]:
            ox0, oy0, ox1, oy1 = ob.px_bbox
            cell_w = ox1 - ox0
            top_offset = oy0 - wy0
            if cell_w >= int(min_w_frac * wrapper_w) and top_offset <= max_top_off:
                new_boxes.append(VisibleBox(
                    box_id=bid,
                    rect=Rect(
                        ox0 / scale, oy0 / scale,
                        ox1 / scale, oy1 / scale,
                    ),
                    area_pt2=float(cell_w * (oy1 - oy0)) / max(scale * scale, 1e-6),
                    fill_ratio=1.0,
                    nested_depth=2,
                    is_outer_wrapper=False,
                    parent_box_id=wrapper.box_id,
                    color="BLUE",
                    px_bbox=(ox0, oy0, ox1, oy1),
                    children_count=0,
                    synthetic=True,
                ))
                break  # only the topmost wide band per wrapper

    # ── Title-block branch ─────────────────────────────────────────────────
    # Cells from title_block_detection.py carry "named, important text"
    # semantics like schedule title rows.  The universal raster signal is
    # height-relative: a tbstruct_* whose band height is meaningfully larger
    # than its peers contains title-class content (project name, abbreviation
    # key, drawing index).  This generalises across drawing types because
    # every title block reserves its tallest cells for project metadata.
    #
    # Height alone is insufficient: a tall-but-empty cell (e.g. the data
    # area of a revision-schedule mini-table that sits below its column
    # headers) passes the height gate but is not a title — it's a blank
    # tabular zone waiting for entries.  The density gate filters those out:
    # title cells contain a prominent label, so their dark-pixel ratio is
    # comparable to short-but-labelled cells (≥ 0.10 across the test set).
    # An empty data area sits well below this floor (≤ 0.06).
    #
    # Only solo (post-collapse) tbstruct_* cells qualify.  Cells inside a
    # tabular cluster were collapsed and renamed to tbcell_*, with one outer
    # tbgroup_* BLUE wrapper around them — those should not be candidates
    # because the cluster as a whole is a table, not a single label cell.
    # Only solo (post-collapse) tbstruct_* cells qualify.  Cells inside a
    # tabular cluster were collapsed and renamed to tbcell_*, with one outer
    # tbgroup_* BLUE wrapper around them — those should not be candidates
    # because the cluster as a whole is a table, not a single label cell.
    tb_cells = [
        b for b in boxes
        if b.color == "BLUE"
        and b.box_id.startswith("tbstruct_")
    ]
    # Median over the full title-block cell population (solo cells + demoted
    # cluster members) so the "peers" reference includes every cell, not just
    # the few solo cells.  Without this, after collapse the survivor set could
    # be just the project-name cell + one tiny code-bar cell, and the median
    # would inflate to the project cell's own height — making it impossible
    # for any cell to exceed 2.5 × its own height and the title rule never fires.
    pop_heights = [
        int(b.px_bbox[3] - b.px_bbox[1])
        for b in boxes
        if b.box_id.startswith("tbstruct_") or b.box_id.startswith("tbcell_")
    ]
    if tb_cells and pop_heights:
        h_mult = float(_cfg_get(cfg, "ct_tbstruct_title_height_mult"))
        h_min = int(_cfg_get(cfg, "ct_tbstruct_title_min_h_px"))
        density_min = float(_cfg_get(cfg, "ct_tbstruct_title_min_density"))
        # Sorted heights → median; population includes cluster members too.
        heights_sorted = sorted(pop_heights)
        median_h = heights_sorted[len(heights_sorted) // 2]
        threshold = max(h_min, int(h_mult * median_h))

        # Compute density for cells passing the height gate (only when rgb
        # is available — keeps unit tests that pass boxes-only working).
        if rgb is not None:
            import cv2
            dark_thr = int(getattr(cfg, "tbd_dark_threshold", 210))

        for cell in tb_cells:
            cx0, cy0, cx1, cy1 = cell.px_bbox
            cell_h = cy1 - cy0
            if cell_h < threshold:
                continue
            bid = f"{cell.box_id}_title"
            if bid in existing_title_ids:
                continue
            # Density gate: if rgb available, reject empty-but-tall cells.
            if rgb is not None:
                crop = rgb[cy0:cy1, cx0:cx1]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
                _, dark = cv2.threshold(gray, dark_thr, 255, cv2.THRESH_BINARY_INV)
                area = max(1, gray.shape[0] * gray.shape[1])
                density = float(cv2.countNonZero(dark)) / area
                if density < density_min:
                    continue
            new_boxes.append(VisibleBox(
                box_id=bid,
                rect=Rect(cx0 / scale, cy0 / scale, cx1 / scale, cy1 / scale),
                area_pt2=float((cx1 - cx0) * cell_h) / max(scale * scale, 1e-6),
                fill_ratio=1.0,
                nested_depth=2,
                is_outer_wrapper=False,
                parent_box_id=cell.box_id,
                color="BLUE",
                px_bbox=(cx0, cy0, cx1, cy1),
                children_count=0,
                synthetic=True,
            ))

    # ── Tabular-cluster table-title branch ────────────────────────────────
    # Inside a tbgroup_* mini-table, the topmost cell is a labelled header
    # row (e.g. "REVISION SCHEDULE") when it sits directly above a thin
    # column-header strip (the row carrying #/DESCRIPTION/DATE column keys).
    # That structural pattern — labelled header + thin column-header band +
    # data area — is the canonical "table with a title" shape, and the
    # labelled header deserves the same title wash as a schedule's name row.
    #
    # Universal because the discriminator is the thin column-header band
    # below the candidate, not text content.  A cluster of plain stacked
    # label-value rows (JOB NO / DATE / DRAWN, SHEET NAME / SHEET NO) has
    # no thin band and won't trigger — none of those rows is a "table title."
    tbg_thin_max_h = int(_cfg_get(cfg, "ct_tbgroup_colhdr_max_h_px"))
    # Group tbcells by parent tbgroup_*.
    by_group: dict[str, list[VisibleBox]] = {}
    for b in boxes:
        if b.box_id.startswith("tbcell_"):
            pid = b.parent_box_id or ""
            if pid.startswith("tbgroup_"):
                by_group.setdefault(pid, []).append(b)
    # Counter for synthetic colhdr_ rings emitted by this pass.  Use a high
    # offset so we don't collide with existing colhdr_NN ids.
    tbg_colhdr_id = 9000
    for group_id, members in by_group.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda b: b.px_bbox[1])
        top = members_sorted[0]
        next_cell = members_sorted[1]
        # The next cell must be a thin band immediately below the top cell,
        # i.e. the column-header strip carrying per-column keys.
        next_h = next_cell.px_bbox[3] - next_cell.px_bbox[1]
        if next_h > tbg_thin_max_h:
            continue
        # Top cell qualifies — emit the title wash on it.
        bid = f"{top.box_id}_title"
        if bid in existing_title_ids:
            continue
        tx0, ty0, tx1, ty1 = top.px_bbox
        new_boxes.append(VisibleBox(
            box_id=bid,
            rect=Rect(tx0 / scale, ty0 / scale, tx1 / scale, ty1 / scale),
            area_pt2=float((tx1 - tx0) * (ty1 - ty0)) / max(scale * scale, 1e-6),
            fill_ratio=1.0,
            nested_depth=2,
            is_outer_wrapper=False,
            parent_box_id=top.box_id,
            color="BLUE",
            px_bbox=(tx0, ty0, tx1, ty1),
            children_count=0,
            synthetic=True,
        ))

        # ── Cyan column-header rings inside the strip ─────────────────────
        # The thin strip directly below a tabular-cluster title row IS the
        # column-header row.  Each word in it (e.g. #, DESCRIPTION, DATE) is
        # a column key and earns a CYAN ring per the legend.  The earlier
        # colhdr emitter ran in visible_boxes (order 100) BEFORE this
        # pass synthesised the title row, so it never saw this candidate.
        # Reproduce the semantics here using the PDF text layer when fitz
        # is available; otherwise fall back to leaving no rings (no false
        # positives — the core detector covers all schedules with their
        # own _title bands, this branch only handles the late-emitted ones).
        cx0_, cy0_, cx1_, cy1_ = next_cell.px_bbox
        cell_words = _word_rects_in_box(pdf_path, page_index, scale,
                                         cx0_, cy0_, cx1_, cy1_,
                                         page_rotation_qt)
        for (wx0, wy0, wx1, wy1) in cell_words:
            ww = wx1 - wx0
            wh = wy1 - wy0
            # Permissive size floor: single-character column keys (e.g. "#"
            # in a revision schedule) can be 2-3 px wide and 5-6 px tall.
            # Filter only obviously degenerate boxes.
            if ww < 1 or wh < 3:
                continue
            new_boxes.append(VisibleBox(
                box_id=f"colhdr_{tbg_colhdr_id}",
                rect=Rect(wx0 / scale, wy0 / scale, wx1 / scale, wy1 / scale),
                area_pt2=float(ww * wh) / max(scale * scale, 1e-6),
                fill_ratio=1.0,
                nested_depth=3,
                is_outer_wrapper=False,
                parent_box_id=bid,  # parent the ring to the title band
                color="CYAN",
                px_bbox=(wx0, wy0, wx1, wy1),
                children_count=0,
                synthetic=True,
            ))
            tbg_colhdr_id += 1

    return new_boxes


def _word_rects_in_box(
    pdf_path,
    page_index: int,
    scale: float,
    ix0: int, iy0: int, ix1: int, iy1: int,
    page_rotation_qt: int,
) -> list[tuple[int, int, int, int]]:
    """Return image-space bboxes for each PDF text word inside the given image rect.

    Maps the image rectangle back into PDF coordinates (accounting for the
    detector's CW quarter-turn rotation), asks fitz for words clipped to
    that rect, then maps each word's PDF bbox forward to image pixels.

    Returns empty list when fitz is unavailable or no words fall in the rect.
    """
    try:
        import fitz as _fitz
    except Exception:
        return []
    try:
        doc = _fitz.open(str(pdf_path))
        page = doc[page_index]
        pw = page.rect.width
        ph = page.rect.height
        # Coord transforms must mirror the core ``_img_to_pdf_rect``
        # convention used by the rest of the codebase (legacy_003.pyfrag
        # lines 122+).  The detector rotates the rendered bitmap CW by
        # ``page_rotation_qt`` quarter-turns to make it upright; the
        # inverse mapping that goes from image-space pixels back to PDF
        # native coordinates therefore takes the OPPOSITE direction.
        # Following the core code exactly so any future page-rotation
        # cases match its (already-tested) behaviour.
        qt = page_rotation_qt % 4

        def _img_to_pdf_rect(_x0, _y0, _x1, _y1):
            if qt == 0:
                return _fitz.Rect(_x0/scale, _y0/scale, _x1/scale, _y1/scale)
            if qt == 3:  # CCW 90 inverse
                px0 = pw - _y1/scale; px1 = pw - _y0/scale
                py0 = _x0/scale; py1 = _x1/scale
                return _fitz.Rect(min(px0, px1), min(py0, py1),
                                  max(px0, px1), max(py0, py1))
            if qt == 1:  # CW 90 inverse
                py0 = ph - _x1/scale; py1 = ph - _x0/scale
                px0 = _y0/scale; px1 = _y1/scale
                return _fitz.Rect(min(px0, px1), min(py0, py1),
                                  max(px0, px1), max(py0, py1))
            # qt == 2: 180
            return _fitz.Rect(pw - _x1/scale, ph - _y1/scale,
                              pw - _x0/scale, ph - _y0/scale)

        def _pdf_to_img_xy(px, py):
            """Forward transform: PDF (px, py) → image (ix, iy)."""
            if qt == 0:
                return px * scale, py * scale
            if qt == 3:
                return py * scale, (pw - px) * scale
            if qt == 1:
                return (ph - py) * scale, px * scale
            # qt == 2
            return (pw - px) * scale, (ph - py) * scale

        clip = _img_to_pdf_rect(ix0, iy0, ix1, iy1)
        words = page.get_text("words", clip=clip) or []
        out: list[tuple[int, int, int, int]] = []
        for w in words:
            wx0, wy0, wx1, wy1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
            corners = [
                _pdf_to_img_xy(wx0, wy0),
                _pdf_to_img_xy(wx1, wy0),
                _pdf_to_img_xy(wx0, wy1),
                _pdf_to_img_xy(wx1, wy1),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            out.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        doc.close()
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------

@dataclass
class CellularTitlePass:
    """Identify schedule title bands from raster geometry alone.

    Runs after GridCellCompletionPass (order 230) so the newly synthesised
    gridcell_ boxes don't interfere with the wrapper-gating check, and so
    the title band exclusion in GridCellCompletionPass doesn't need updating.

    The emitted ``{wrapper_id}_title`` boxes are synthetic=True BLUE boxes
    consumed by the core renderer's Pass 2 (draw_title_band_highlights),
    which draws the standard semi-transparent blue fill over the title cell.
    """

    info: PassInfo = PassInfo(
        name="cellular_title",
        stage="synthesize",
        layer_flag="BLUE",
        order=235,
        description=(
            "Emit {wrapper_id}_title synthetic BLUE boxes for the topmost "
            "wide ORANGE band in each BLUE schedule wrapper, so the renderer "
            "draws the standard blue-fill title highlight without requiring "
            "PDF text extraction."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None:
            return state

        result = state.result
        scale = float(
            result.debug_stats.get("render_scale_used")
            or ctx.cfg.render_scale
            or 1.0
        )

        new_boxes = detect_cellular_titles(
            boxes=result.boxes,
            scale=scale,
            cfg=ctx.cfg,
            rgb=state.rgb,
            pdf_path=ctx.pdf_path,
            page_index=ctx.page_index,
            page_rotation_qt=int(
                result.debug_stats.get("rotated_cw_quarter_turns", 0) or 0),
        )

        stats = dict(result.debug_stats or {})
        stats["cellular_title"] = len(new_boxes)

        state.result = VisibleBoxResult(
            boxes=[*result.boxes, *new_boxes],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats=stats,
        )
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        state.artifacts["cellular_title"] = len(new_boxes)
        return state


__all__ = ["CellularTitlePass", "detect_cellular_titles"]
