"""Title-block detection pass — right-margin logos and title cells.

Engineering drawings carry a title block in the right (or bottom) margin
containing stamps, company logos, project info, and certification text.
The core contour detector misses this zone because:

  1. The content is mixed raster (circular seals, icon logos) + text rather
     than a clean tabular grid.
  2. No closed rectangular border contour produces a BLUE structural wrapper
     for the region.
  3. Circular/irregular logo shapes fragment into noise at the fill-ratio
     thresholds the core detector uses.

Strategy
--------
  * Scan x > tbd_right_margin_frac of the image width.
  * Extract horizontal rules (H-lines) to segment the strip into title-block
    cells (same morphological technique as grid_cell_completion).
  * For each non-empty cell band:
      - Find the bounding rect of all significant dark content.
      - Classify as logo-like or text-like using contour compactness and size.
      - Emit a ``titleblkimg_*`` (logo) or ``titleblkpanel_*`` (text/panel)
        synthetic PURPLE box.
  * The core renderer's Pass 3.5 draws all ``titleblk*`` boxes as PURPLE
    rings with no further changes to frozen code.

What this pass does NOT do
--------------------------
- Does not edit any core chunk.
- Does not fire when a non-synthetic BLUE box already covers the right margin.
- Does not emit boxes that overlap existing PURPLE detections.

All thresholds are exposed through Cfg (``tbd_*`` prefix).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from ..core.models import Rect, VisibleBox, VisibleBoxResult
from .base import PageContext, PassInfo, PipelineState

# ---------------------------------------------------------------------------
# Threshold defaults
# ---------------------------------------------------------------------------
_D: dict[str, object] = {
    # Fraction of image width: columns beyond this belong to the title block.
    "tbd_right_margin_frac": 0.80,
    # Minimum dark-pixel count in a band before we bother processing it.
    "tbd_min_band_dark_px": 40,
    # Minimum contour area (px²) to consider a region significant.
    "tbd_min_region_area_px": 150,
    # Morphological close kernel to merge nearby content within a band.
    "tbd_close_px": 10,
    # Pixel value below which a pixel is "dark" (inverted threshold).
    "tbd_dark_threshold": 210,
    # H-line extraction: minimum span fraction of strip width.
    # Lowered from 0.55 to 0.30: title block cell borders often don't span the
    # full strip width (they're scoped to sub-groups), so a lower threshold is
    # needed.  Rotated text inside title cells won't create false H-lines because
    # rotated text runs VERTICALLY, not horizontally.
    "tbd_hline_min_coverage": 0.30,
    # H-line morphological kernel length (px).
    "tbd_hline_morph_px": 15,
    # Fallback: if fewer than this many H-lines are found, use horizontal
    # projection (dark-pixel row density) to locate content gaps instead.
    "tbd_min_hlines_before_fallback": 2,
    # Projection fallback: a row with <= this many dark pixels is a "gap" row
    # (empty space between title cells).
    "tbd_projection_gap_max_dark_px": 5,
    # Minimum consecutive gap rows to count as a true cell boundary.
    "tbd_projection_min_gap_rows": 3,
    # A contour whose bounding-rect aspect ratio is below this is "compact"
    # (logo-like).  Text banners are typically wide and short → aspect > 2.5.
    "tbd_logo_compact_aspect": 2.5,
    # Minimum area of the single largest contour to qualify as a logo region.
    "tbd_logo_min_large_contour_px": 800,
    # Skip the whole pass if a real (non-synthetic) BLUE box already covers
    # the right margin — earlier detection already handled it.
    "tbd_skip_if_blue_exists": True,
    # After per-band detection, merge adjacent content rects whose vertical gap
    # is <= this many pixels.  8 px merges within-section text-row gaps (2-5 px)
    # while keeping genuine inter-section whitespace (10+ px) as separate regions.
    "tbd_merge_gap_px": 8,
    # Sub-band refinement: any band taller than this (px) is additionally
    # searched for projection-gap split points.  Title-block sections like the
    # project-owner block and revision schedule often lack full-width H-lines
    # but are separated from each other by a few rows of near-zero dark pixels.
    # Setting to 0 disables refinement.
    "tbd_subband_split_min_h": 260,
    # ── BLUE structural left-edge snapping (cell-border alignment) ──────────
    # When a band's content is dominated by rotated/vertical text, the content
    # bounding rect's left edge lands on the leftmost text stroke instead of
    # on the cell's actual vertical border line.  These thresholds drive the
    # band-local vertical-rule detector that snaps BLUE structural boxes onto
    # the true cell border.
    #
    # tbd_vline_min_h_frac: a vertical run must span at least this fraction of
    # the band height to qualify as a cell border (vs. a text stroke).
    # 0.55 picks up genuine cell borders even when the band has tall capitals
    # at the same column, while filtering noise.
    "tbd_vline_min_h_frac": 0.55,
    # tbd_vline_max_thickness_px: cell border lines are thin (1-3 px); a wide
    # vertical blob is most likely text or a logo body, not a border.
    "tbd_vline_max_thickness_px": 4,
    # tbd_vline_search_px: how far LEFT of the detected content edge to look
    # for the cell border.  Rotated text leaves a few px of whitespace between
    # the leftmost stroke and the cell border; 30 px covers normal whitespace
    # plus modest scale variation while not crossing into adjacent cells.
    "tbd_vline_search_px": 30,
    # ── Sub-band split corroboration ───────────────────────────────────────
    # The sub-band projection-gap fallback splits big bands at any whitespace
    # gap — fine when cells are separated by real horizontal rules that the
    # main detector missed, but it also fires inside a single cell whenever
    # that cell holds two paragraphs of text separated by whitespace (rotated
    # project title above location text, abbreviation list above legend, etc.).
    # Before honouring a candidate split y, we now require corroborating
    # evidence: a thin horizontal run of dark pixels of at least this width
    # within ± `tbd_subband_rule_y_slack_px` rows of the candidate.  No rule
    # → the gap is whitespace inside one cell, reject the split.
    #
    # tbd_subband_rule_min_w_px: minimum horizontal-run length to count as a
    # real (partial) cell divider.  Set to ~25-30 % of typical strip widths so
    # it accepts rules shorter than the global H-line threshold (which demands
    # full strip-width rules) but still excludes incidental text horizontals
    # like the bottom of a "T" or "I" letter (≤ 15 px).  35 px works across
    # render scales 1.0-2.5.
    "tbd_subband_rule_min_w_px": 35,
    # tbd_subband_rule_y_slack_px: ± y window around the candidate gap
    # midpoint.  Projection gaps span several rows; the rule, if real, sits
    # somewhere inside the gap.  ±5 px covers normal printing/anti-alias
    # variation without bleeding into adjacent text rows above or below.
    "tbd_subband_rule_y_slack_px": 5,
    # ── Thin-band content discrimination ───────────────────────────────────
    # The Phase-3 filter `rh_box < 8 → reject` was too coarse: it rejected
    # legitimate column-header strips (the 6-7 px tall #/DESCRIPTION/DATE row
    # of a revision-schedule mini-table) along with the rule slivers it was
    # meant to drop.  These knobs replace the hard floor with a content
    # discriminator that counts rows of real text inside the candidate band.
    #
    # tbd_thin_band_row_dark_floor: a band row counts as "content" when it
    # has at least this many dark pixels across the strip width.  20 catches
    # even sparse character rows (a single 'I' or '.' has ≥ 20 dark px at
    # render scale 2.5) while ignoring anti-alias noise around printed rules.
    "tbd_thin_band_row_dark_floor": 20,
    # tbd_thin_band_min_content_rows: a band must have at least this many
    # content-rows to be accepted.  Rule slivers concentrate in 1-3 rows;
    # real text spans 4+ rows even at small font sizes.  4 is a safe floor.
    "tbd_thin_band_min_content_rows": 4,
    # ── Tabular-cluster collapsing ─────────────────────────────────────────
    # Multiple tbstruct cells stacked with shared rules form one mini-table
    # (header label + column-header strip + data area, or stacked label-value
    # rows like JOB NO / DATE / DRAWN).  The legend says BLUE = outer table
    # frame, ORANGE = inner cell.  Without collapsing, every internal row gets
    # its own BLUE outline and the table reads as "table of tables".
    #
    # tbd_table_max_y_gap_px: cells separated by ≤ this many pixels share a
    # printed rule and belong to the same cluster.  3 px allows for the rule
    # itself plus minor anti-alias slop while still separating genuinely
    # independent label rows that have a real whitespace gap (≥ 5 px).
    "tbd_table_max_y_gap_px": 3,
    # tbd_table_solo_title_min_h_px: a cell taller than this is treated as a
    # solo prominent block (project-name title, abbreviations) regardless of
    # whether it sits adjacent to other cells.  Such cells are not table rows
    # and never get demoted to ORANGE.
    "tbd_table_solo_title_min_h_px": 200,
    # tbd_table_xext_tol_px: maximum px difference in left/right x between
    # consecutive cluster members.  Two cells with substantially different
    # x-extents are structurally distinct elements (e.g. a normal title-block
    # cell at width 120 next to a wider page-footer cell at width 151).
    # Grouping them produces a step-shape that renders as a bounding rectangle
    # with dead corners; breaking on the discontinuity keeps each table a
    # clean rectangle.  10 px tolerates anti-alias / vline-snap variance
    # while still flagging the 31-px footer-vs-cell jump on test5.
    "tbd_table_xext_tol_px": 10,
    # tbd_table_wrapper_pad_px: extends the tabular-cluster outer wrapper a
    # few px beyond the union of its inner cells so the BLUE outline stays
    # visible after the ORANGE cell-outline pass paints over coincident edges.
    # 3 px gives the blue ring visible separation from inner cells without
    # overlapping adjacent clusters (the smallest inter-cluster gap on test5
    # is 7 px between tbgroup_0000 and tbgroup_0001).  Combined with a
    # thicker line in the renderer (3 px stroke for tbgroup_*), the outer
    # frame reads clearly at normal zoom.
    "tbd_table_wrapper_pad_px": 3,
}


def _cfg_get(cfg, key: str):
    return getattr(cfg, key, _D[key])


# ---------------------------------------------------------------------------
# H-line extraction (strip-local coordinates)
# ---------------------------------------------------------------------------

def _extract_hlines(gray_strip: np.ndarray, cfg) -> list[int]:
    """Return y-positions of horizontal rules inside the strip (morphological)."""
    rh, rw = gray_strip.shape[:2]
    h_morph = _cfg_get(cfg, "tbd_hline_morph_px")
    hline_cov = _cfg_get(cfg, "tbd_hline_min_coverage")
    dark_thr = _cfg_get(cfg, "tbd_dark_threshold")

    _, dark = cv2.threshold(gray_strip, dark_thr, 255, cv2.THRESH_BINARY_INV)
    h_kern = max(h_morph, int(hline_cov * rw))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kern, 1))
    hmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    positions: list[int] = []
    cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= int(hline_cov * rw) and h <= 6:
            positions.append(y + h // 2)

    if not positions:
        return []
    positions = sorted(set(positions))
    grouped: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - grouped[-1][-1] <= 5:
            grouped[-1].append(p)
        else:
            grouped.append([p])
    return [int(np.median(g)) for g in grouped]


def _projection_gaps(gray_strip: np.ndarray, cfg) -> list[int]:
    """Fallback: find cell boundaries via horizontal dark-pixel density gaps.

    When horizontal rules don't span enough of the strip width for morphological
    H-line detection, the dark-pixel row sum still drops to near-zero in the
    white space between title-block cells.  This finds those drop-off points.
    """
    dark_thr = _cfg_get(cfg, "tbd_dark_threshold")
    gap_max = int(_cfg_get(cfg, "tbd_projection_gap_max_dark_px"))
    min_gap_rows = int(_cfg_get(cfg, "tbd_projection_min_gap_rows"))

    _, dark = cv2.threshold(gray_strip, dark_thr, 255, cv2.THRESH_BINARY_INV)
    row_proj = (dark > 0).sum(axis=1)  # dark px count per row

    H = len(row_proj)
    positions: list[int] = []
    gap_start: int | None = None

    for y in range(H):
        is_gap = int(row_proj[y]) <= gap_max
        if is_gap and gap_start is None:
            gap_start = y
        elif not is_gap and gap_start is not None:
            gap_len = y - gap_start
            if gap_len >= min_gap_rows:
                positions.append(gap_start + gap_len // 2)
            gap_start = None

    if gap_start is not None:
        gap_len = H - gap_start
        if gap_len >= min_gap_rows:
            positions.append(gap_start + gap_len // 2)

    return positions


def _leftmost_vline_x(
    gray_band: np.ndarray,
    content_x0: int,
    cfg,
) -> int | None:
    """Find the x-coordinate of the leftmost vertical cell-border rule in a band.

    Title-block cells have thin (1-3 px) vertical border lines on their left
    edge.  When the cell content is rotated text (vertical strokes), the
    bounding-rect's left edge lands on the leftmost text stroke rather than
    on the cell border itself, leaving the BLUE structural box visually
    floating inside the cell.

    This finder isolates *tall, thin* vertical runs of dark pixels via a
    vertical morphological open, then returns the rightmost border x within
    `tbd_vline_search_px` to the LEFT of `content_x0`.  Returning the rightmost
    border (when several exist) avoids snapping to an unrelated cell border
    further out in the title-block strip.

    Returns None when no qualifying rule is found.
    """
    bh, bw = gray_band.shape[:2]
    if bh < 4 or bw < 4:
        return None

    dark_thr = int(_cfg_get(cfg, "tbd_dark_threshold"))
    vline_min_h_frac = float(_cfg_get(cfg, "tbd_vline_min_h_frac"))
    vline_max_thick = int(_cfg_get(cfg, "tbd_vline_max_thickness_px"))
    search_px = int(_cfg_get(cfg, "tbd_vline_search_px"))

    _, dark = cv2.threshold(gray_band, dark_thr, 255, cv2.THRESH_BINARY_INV)

    # Vertical morphological open: keep only runs of dark pixels at least
    # vline_min_h_frac of the band height tall.  Text strokes (even rotated
    # ones) rarely span this much continuous height per column.
    #
    # For very thin bands (column-header rows ≤ 8 px) the fractional kernel
    # collapses to 3-4 px; we instead require a full-height vertical run.
    # A border line that genuinely spans a thin band is unambiguous evidence
    # of a cell edge, while text columns in such bands rarely make full-height
    # contiguous runs (letters have horizontal serifs/baselines/holes).
    if bh <= 8:
        v_kern_h = bh
    else:
        v_kern_h = max(6, int(vline_min_h_frac * bh))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kern_h))
    vmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[int] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Cell border lines are thin (≤ vline_max_thick px) and tall.
        # Filter out wide blobs (logo bodies, bold text columns).
        if h < v_kern_h or w > vline_max_thick:
            continue
        cx = x + w // 2
        # Must sit just left of (or at) content_x0
        if cx <= content_x0 and cx >= content_x0 - search_px:
            candidates.append(cx)

    if not candidates:
        return None
    # Cluster nearby candidates: a "double-stroke" cell frame consists of two
    # parallel rules within ~6 px of each other.  Within such a cluster, the
    # LEFTMOST line is the true outer cell edge.  Across clusters we want the
    # RIGHTMOST cluster (closest to the content) to avoid snapping past
    # adjacent-cell borders that happen to live in the search window.
    candidates.sort()
    clusters: list[list[int]] = [[candidates[0]]]
    for cx in candidates[1:]:
        if cx - clusters[-1][-1] <= 6:
            clusters[-1].append(cx)
        else:
            clusters.append([cx])
    return min(clusters[-1])


def _has_horizontal_rule_near(
    gray_strip: np.ndarray,
    y_candidate: int,
    cfg,
) -> bool:
    """Return True when a thin dark horizontal run exists near `y_candidate`.

    Sub-band projection gaps fire on any whitespace ≥ a few rows tall, which
    triggers false splits *within* a single cell whenever the cell holds two
    text paragraphs separated by white space (e.g. rotated project title above
    horizontal location text in the same cell).

    Real cell dividers always have a horizontal rule.  The global H-line
    detector (``_extract_hlines``) demands the rule span ≥ 30 % of the strip
    width — too strict for partial sub-cell rules — but a real divider still
    has *some* contiguous dark horizontal run.  This relaxed check looks for
    any horizontal rule at least ``tbd_subband_rule_min_w_px`` wide within a
    small ± y-window of the candidate.

    Use:
        not _has_horizontal_rule_near(...)   →   reject the split.
    """
    rh, rw = gray_strip.shape[:2]
    if rh == 0 or rw < 20:
        return False
    dark_thr = int(_cfg_get(cfg, "tbd_dark_threshold"))
    rule_min_w = int(_cfg_get(cfg, "tbd_subband_rule_min_w_px"))
    y_slack = int(_cfg_get(cfg, "tbd_subband_rule_y_slack_px"))

    y0 = max(0, y_candidate - y_slack)
    y1 = min(rh, y_candidate + y_slack + 1)
    band = gray_strip[y0:y1, :]
    if band.shape[0] < 1 or band.shape[1] < rule_min_w:
        return False

    _, dark = cv2.threshold(band, dark_thr, 255, cv2.THRESH_BINARY_INV)
    # Look for a thin horizontal rule via morphological open with a wide
    # horizontal kernel — preserves only contiguous horizontal runs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rule_min_w, 1))
    hmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    return bool(cv2.countNonZero(hmask) > 0)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_title_block(
    rgb: np.ndarray,
    boxes: list[VisibleBox],
    scale: float,
    cfg,
) -> tuple[list[VisibleBox], int]:
    """Return (new_boxes, effective_x0) for title block content in the right margin.

    effective_x0 is the pixel x-coordinate of the true title block left edge.
    Callers can use it to strip core boxes from that zone.
    """
    H, W = rgb.shape[:2]
    right_frac = _cfg_get(cfg, "tbd_right_margin_frac")
    x_cut = int(right_frac * W)

    # Skip if a real BLUE wrapper already covers this zone.
    if _cfg_get(cfg, "tbd_skip_if_blue_exists"):
        if any(
            b.color == "BLUE"
            and not getattr(b, "synthetic", False)
            and b.px_bbox[0] >= x_cut - 30
            for b in boxes
        ):
            return [], x_cut

    # ------------------------------------------------------------------
    # Compute the TRUE title block left edge.
    #
    # Schedule BLUE wrappers extend from the drawing body into the right
    # margin.  Their row-separator H-lines appear in the full strip and
    # flood the H-line detector with ~40-50 false entries — one per
    # schedule row — instead of the ~3-5 genuine title-block section
    # dividers.  By starting the strip at the rightmost right-edge of any
    # non-outer-wrapper BLUE box, we restrict detection to the column
    # that only the title block occupies (seal, logos, project text).
    #
    # Outer-wrapper guard: the page BLUE wrapper spans nearly the full
    # width (> 70 % of W).  We skip it so it doesn't push effective_x0
    # past the title block itself.
    # ------------------------------------------------------------------
    effective_x0 = x_cut
    for b in boxes:
        if b.color == "BLUE" and not getattr(b, "synthetic", False):
            bx0_, by0_, bx1_, by1_ = b.px_bbox
            box_w = bx1_ - bx0_
            if box_w < 0.70 * W:  # skip full-page outer wrappers
                effective_x0 = max(effective_x0, bx1_)

    strip_rgb = rgb[:, effective_x0:]
    gray_strip = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    strip_h, strip_w = gray_strip.shape[:2]

    if strip_w < 20:
        return [], effective_x0  # title block column too narrow to be meaningful

    # Segment by horizontal rules in the TRUE title block column.
    # With schedule rows excluded, only genuine section dividers remain.
    min_hlines = int(_cfg_get(cfg, "tbd_min_hlines_before_fallback"))
    h_lines_raw = _extract_hlines(gray_strip, cfg)
    using_hlines = len(h_lines_raw) >= min_hlines
    if using_hlines:
        h_lines = h_lines_raw
        # H-lines are structural cell boundaries — honour them as authority.
        # merge_gap=0 keeps each cell as its own box; we only allow 2 px of
        # slack to handle contour-adjacency artefacts within the same band.
        effective_merge_gap = 2
    else:
        h_lines = _projection_gaps(gray_strip, cfg)
        # Projection fallback creates one gap per text row; use the full
        # merge_gap to re-collapse those into logical sections.
        effective_merge_gap = int(_cfg_get(cfg, "tbd_merge_gap_px"))

    # Sub-band refinement: for any band taller than tbd_subband_split_min_h,
    # run projection-gap detection within that sub-strip and inject the gap
    # midpoints as additional split points — but ONLY when there is real
    # horizontal-rule evidence near the candidate y.  Without the rule check,
    # a single tall cell holding two text paragraphs (e.g. a rotated project
    # title above horizontal location text) gets falsely split at the
    # whitespace between the paragraphs.  Real cell dividers are always
    # backed by at least a partial horizontal rule that is too short for the
    # global H-line detector but visible to the relaxed local check.
    subband_min_h = int(_cfg_get(cfg, "tbd_subband_split_min_h"))
    if subband_min_h > 0:
        # Build band boundaries from current h_lines plus implicit 0 and strip_h
        breaks = sorted(set(h_lines))
        band_pairs = list(zip([0] + breaks, breaks + [strip_h]))
        extra: list[int] = []
        for by0, by1 in band_pairs:
            if by1 - by0 > subband_min_h:
                sub_gaps = _projection_gaps(gray_strip[by0:by1], cfg)
                for g in sub_gaps:
                    abs_y = by0 + g
                    if _has_horizontal_rule_near(gray_strip, abs_y, cfg):
                        extra.append(abs_y)
        if extra:
            h_lines = sorted(set(h_lines + extra))

    if not h_lines or h_lines[0] > 15:
        h_lines = [0] + h_lines
    if not h_lines or h_lines[-1] < strip_h - 15:
        h_lines = h_lines + [strip_h]

    dark_thr = int(_cfg_get(cfg, "tbd_dark_threshold"))
    min_dark = int(_cfg_get(cfg, "tbd_min_band_dark_px"))
    min_area = int(_cfg_get(cfg, "tbd_min_region_area_px"))
    close_px = int(_cfg_get(cfg, "tbd_close_px"))
    compact_asp = float(_cfg_get(cfg, "tbd_logo_compact_aspect"))
    logo_min_cnt = int(_cfg_get(cfg, "tbd_logo_min_large_contour_px"))
    merge_gap = int(_cfg_get(cfg, "tbd_merge_gap_px"))

    # Existing PURPLE bboxes for dedup
    existing_purple = [b.px_bbox for b in boxes if b.color == "PURPLE"]

    # -----------------------------------------------------------------------
    # Phase 1: collect one content rect per non-empty band.
    # Each entry: (abs_x0, abs_y0, abs_x1, abs_y1, largest_cnt_area,
    #              colored_frac, max_compactness)
    #
    # colored_frac    — fraction of band pixels with HSV saturation > 40.
    #                   Logos (seals, diamond icons) have colour; text doesn't.
    # max_compactness — 4π·area/perimeter² for the largest contour.
    #                   A perfect circle = 1.0; text strokes are near 0.
    # -----------------------------------------------------------------------
    band_rects: list[tuple] = []

    for y0, y1 in zip(h_lines, h_lines[1:]):
        band_h = y1 - y0
        if band_h < 5:
            continue

        band_gray = gray_strip[y0:y1, :]
        _, dark = cv2.threshold(band_gray, dark_thr, 255, cv2.THRESH_BINARY_INV)
        if cv2.countNonZero(dark) < min_dark:
            continue

        # Close to merge content within the band
        ck = max(3, min(close_px, band_h // 2))
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
        closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kern)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) >= min_area]
        if not valid:
            continue

        # Exclude thin vertical border lines from the content bounding rect.
        # The cell border stroke (1-3 px wide) sits at the strip left edge and
        # would otherwise pull content_x0 all the way to the column edge.
        #
        # min_cnt_w threshold drops border lines from the CLOSED mask.  However,
        # the close kernel (ck px) thickens the border to ~2*ck px, which can
        # exceed min_cnt_w and merge the border into the content blob — pulling
        # bx0 back to 0.  Fix: compute bx0 from the RAW (pre-close) dark mask
        # contours instead.  On the raw mask, the border line is 1-3 px wide and
        # is reliably filtered by min_cnt_w (>=6 px).  bx1/by0/by1 still come
        # from the closed-mask union (better spatial coverage for those edges).
        min_cnt_w = max(6, strip_w // 15)  # ~6-15 px: wider than a border line

        # bx1/by0/by1: from CLOSED mask — merged blobs give fuller extent
        wide_closed = [c for c in valid if cv2.boundingRect(c)[2] >= min_cnt_w]
        bounds_src = wide_closed if wide_closed else valid
        all_pts = np.concatenate([c.reshape(-1, 2) for c in bounds_src])
        bx0_close = int(np.min(all_pts[:, 0]))

        # bx0: use the RAW mask only when the closed mask has already been pulled
        # to 0 by the close kernel expanding the left border into content.
        # When bx0_close > 5, the closed mask already filtered the border and gives
        # better spatial coverage (rotated-text strokes are narrow in the raw mask
        # and get cut off by min_cnt_w, producing a bx0 that is too far right).
        if bx0_close <= 5:
            cnts_raw, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            wide_raw = [c for c in cnts_raw
                        if cv2.boundingRect(c)[2] >= min_cnt_w
                        and cv2.boundingRect(c)[3] >= 3]
            bx0 = int(np.min(
                np.concatenate([c.reshape(-1, 2) for c in wide_raw])[:, 0]
            )) if wide_raw else bx0_close
        else:
            bx0 = bx0_close
        by0 = int(np.min(all_pts[:, 1]))
        bx1 = int(np.max(all_pts[:, 0]))
        by1 = int(np.max(all_pts[:, 1]))

        abs_x0 = effective_x0 + bx0
        abs_y0 = y0 + by0
        abs_x1 = effective_x0 + bx1
        abs_y1 = y0 + by1

        if (abs_x1 - abs_x0) < 4 or (abs_y1 - abs_y0) < 4:
            continue

        # ── Cell-border snap candidate (band-local x in strip coords) ─────
        # Look for a thin tall vertical rule just LEFT of bx0.  If found, this
        # is the cell's true left border; Phase 3 will use it for BLUE.
        # ORANGE/PURPLE keep using content_x0 — only the BLUE structural box
        # snaps to the border, since BLUE represents structure (cell frame).
        border_strip_x = _leftmost_vline_x(band_gray, bx0, cfg)
        border_abs_x = effective_x0 + border_strip_x if border_strip_x is not None else None

        largest_cnt_area = float(max(cv2.contourArea(c) for c in valid))

        # ── Signal 1: colour — logos have saturated pixels, text is B&W ─────
        band_rgb_local = strip_rgb[y0:y1, :]
        hsv_band = cv2.cvtColor(band_rgb_local, cv2.COLOR_RGB2HSV)
        sat = hsv_band[:, :, 1].astype(np.float32)
        colored_frac = float(np.sum(sat > 40)) / max(1, sat.size)

        # ── Signal 2: Hough circles — seals/stamps have a clear circular ring ─
        # minRadius scales to band height so it catches real seal circles (~50 px
        # radius in a 115 px band) but ignores letter 'O' (~7 px radius).
        # We also store the circle's strip-local (cx, cy, cr) so Phase 3 can
        # compute a pixel-perfect tight bounding box without corner whitespace.
        min_r = max(8, band_h // 3)
        max_r = max(min_r + 1, band_h // 2 + 10)
        circles = cv2.HoughCircles(
            band_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_r,
            param1=50,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r,
        )
        has_circle = circles is not None
        # best_circle: (cx, cy, cr) in strip-local coords, or None
        if has_circle:
            # pick the largest detected circle
            best = max(circles[0], key=lambda c: c[2])
            best_circle: tuple | None = (float(best[0]), float(best[1]), float(best[2]))
        else:
            best_circle = None

        band_rects.append((abs_x0, abs_y0, abs_x1, abs_y1,
                           largest_cnt_area, colored_frac, has_circle,
                           best_circle, y0, border_abs_x))

    # -----------------------------------------------------------------------
    # Phase 2: merge adjacent band rects (projection fallback only).
    #
    # When H-lines were found: each band already represents one structural cell
    # of the title block grid.  Content fills the band edge-to-edge, so any
    # merge threshold > 0 collapses adjacent cells.  Skip Phase 2 entirely —
    # treat band_rects as the final set of regions.
    #
    # When projection fallback was used: the gaps are between individual text
    # rows inside one logical section.  Merge with tbd_merge_gap_px (8 px) to
    # re-group those sub-rows into coherent sections.
    # -----------------------------------------------------------------------
    if not band_rects:
        return [], effective_x0

    if using_hlines:
        merged = [list(r) for r in band_rects]
    else:
        band_rects.sort(key=lambda r: r[1])
        merge_gap = int(_cfg_get(cfg, "tbd_merge_gap_px"))
        merged = [list(band_rects[0])]
        for r in band_rects[1:]:
            bx0, by0, bx1, by1, lca, cf, hc, bc, band_y0, border_x = r
            prev = merged[-1]
            if by0 - prev[3] <= merge_gap:
                prev[0] = min(prev[0], bx0)
                prev[2] = max(prev[2], bx1)
                prev[3] = max(prev[3], by1)
                prev[4] = max(prev[4], lca)
                prev[5] = max(prev[5], cf)
                prev[6] = prev[6] or hc
                if prev[7] is None:
                    prev[7] = bc
                    prev[8] = band_y0
                # border_abs_x: keep the leftmost (innermost) detected border
                # across the merged bands so BLUE still hugs the cell frame.
                if prev[9] is None:
                    prev[9] = border_x
                elif border_x is not None:
                    prev[9] = min(prev[9], border_x)
            else:
                merged.append(list(r))

    # Portrait memo/RFP covers: centered body copy bleeds serifs into the
    # right-margin strip as a few ultra-narrow "cells".  Real drawing title
    # blocks use materially wider text panels (typically ≥35 % of strip width).
    # Skip emission entirely so downstream cover-title synthesis can own the page.
    if merged and H > W:
        has_v_wrapper = any(
            b.color == "BLUE"
            and not getattr(b, "synthetic", False)
            and re.fullmatch(r"v\d+", b.box_id or "")
            for b in boxes
        )
        if not has_v_wrapper:
            strip_w = max(1, W - int(effective_x0))
            cont_widths = [max(0, int(m[2]) - int(m[0])) for m in merged]
            mw = max(cont_widths) if cont_widths else 0
            if mw < max(100, int(0.35 * strip_w)) and len(merged) <= 6:
                return [], effective_x0

    # -----------------------------------------------------------------------
    # Phase 3: classify and emit boxes.
    #
    # LOGO detection — two reliable signals:
    #   colored_frac  > 0.015 → band has saturated colour pixels (seal, diamond)
    #   max_compactness > 0.45 → largest contour is roughly circular (seal)
    # Either signal → PURPLE titleblkimg_* (renderer draws purple via box_id)
    #
    # TEXT cells → two boxes with NON-titleblk IDs so renderer uses normal paths:
    #   BLUE  tbstruct_*  synthetic=False → normal BLUE structural outline
    #   ORANGE tbtext_*   synthetic=False → normal ORANGE content fill
    # -----------------------------------------------------------------------
    new_boxes: list[VisibleBox] = []
    counter = 0

    col_x0 = effective_x0
    col_x1 = W - 2

    # Compute global right border from the max content_x1 across non-logo bands.
    # Per-band contour detection can under-detect thin strokes at band boundaries
    # (e.g. WASILLA thin rightmost characters land at x=1860 vs true border 1907).
    # Logo bands are excluded since their tight x extent is intentional.
    non_logo_x1s = [
        m[2] for m in merged
        if not (float(m[5]) > 0.015 or m[6])  # not colored / not has_circle
    ]
    right_border_x = max(non_logo_x1s) if non_logo_x1s else col_x1

    for content_x0, abs_y0, content_x1, abs_y1, largest_cnt_area, colored_frac, has_circle, best_circle, band_y0, border_abs_x in merged:
        rh_box = abs_y1 - abs_y0
        col_w = col_x1 - col_x0
        content_w = content_x1 - content_x0
        # Skip bands that are too narrow overall, or whose dark content is just
        # a border sliver (content_w < 12 px).
        #
        # Thin-band discrimination — rule slivers vs real text rows:
        # The the previous filter was `rh_box < 8 → reject`, which dropped legitimate
        # column-header strips (e.g. the 6-7 px tall #/DESCRIPTION/DATE row in
        # the revision-schedule mini-table) along with the rule slivers it was
        # actually trying to filter.  Better discriminator: a real text band
        # has dark pixels distributed across multiple rows of the band — every
        # row contributes something because letterforms have varied vertical
        # extent.  A rule sliver concentrates its dark pixels in 1-3 contiguous
        # rows.  Count rows above a small dark-pixel floor; require at least
        # `tbd_thin_band_min_content_rows` such rows for the band to count.
        if col_w < 10 or content_w < 12:
            continue
        if rh_box < 8:
            # Look at the actual raster to decide: count rows in [abs_y0, abs_y1]
            # that have ≥ tbd_thin_band_row_dark_floor dark pixels across the
            # full strip width.  Real text bands clear this gate easily; rule
            # slivers do not.
            thin_floor = int(_cfg_get(cfg, "tbd_thin_band_row_dark_floor"))
            thin_min_rows = int(_cfg_get(cfg, "tbd_thin_band_min_content_rows"))
            band_strip = gray_strip[abs_y0:abs_y1 + 1, :]
            if band_strip.shape[0] < thin_min_rows:
                continue
            _, band_dark = cv2.threshold(band_strip, dark_thr, 255, cv2.THRESH_BINARY_INV)
            row_counts = (band_dark > 0).sum(axis=1)
            content_rows = int((row_counts >= thin_floor).sum())
            if content_rows < thin_min_rows:
                continue
            # Real thin-text band: pad rh_box up to a minimum render height
            # so the resulting cell is visually meaningful.  Use the band-local
            # h-line spacing if available, else a small floor.
            if rh_box < 6:
                rh_box = 6
                abs_y1 = abs_y0 + rh_box

        # Dedup against existing PURPLE
        skip = False
        for px0, py0, px1, py1 in existing_purple:
            iw = max(0, min(col_x1, px1) - max(col_x0, px0))
            ih = max(0, min(abs_y1, py1) - max(abs_y0, py0))
            if iw * ih > 0.5 * col_w * rh_box:
                skip = True
                break
        if skip:
            continue

        # Logo = has colour (Spark Design diamond) OR Hough circle (Alaska seal).
        # Text = neither → BLUE/ORANGE cell.
        is_logo = (colored_frac > 0.015) or bool(has_circle)

        if is_logo:
            # ── PURPLE: tight around the actual logo ──────────────────────────
            # If a Hough circle was found, use its center+radius directly for
            # the y-bounds (pixel-perfect vertical extent of the seal ring).
            # For lx0, Hough is used only when it TIGHTENS the left edge —
            # a false-positive circle (e.g. from a large coloured text block)
            # can have cx-cr < 0, extending left of the strip and pulling
            # lx0 back to col_x0.  Guarding with max(content_x0, hough_lx0)
            # means Hough can only shrink the box from the left, never expand.
            if best_circle is not None:
                cx, cy, cr = best_circle
                hough_lx0 = effective_x0 + int(cx - cr)
                lx0 = max(col_x0, max(content_x0, hough_lx0))
                lx1 = min(col_x1, effective_x0 + int(cx + cr))
                ly0 = max(0, band_y0 + int(cy - cr))
                ly1 = min(H, band_y0 + int(cy + cr))
                abs_y0, abs_y1 = ly0, ly1
            else:
                lx0 = max(col_x0, content_x0)
                lx1 = min(col_x1, content_x1)
            lw = max(1, lx1 - lx0)
            new_boxes.append(VisibleBox(
                box_id=f"titleblkimg_{counter:04d}",
                rect=Rect(lx0 / scale, abs_y0 / scale, lx1 / scale, abs_y1 / scale),
                area_pt2=float(lw * rh_box) / max(scale * scale, 1e-6),
                fill_ratio=0.5,
                nested_depth=1,
                is_outer_wrapper=False,
                parent_box_id=None,
                color="PURPLE",
                px_bbox=(lx0, abs_y0, lx1, abs_y1),
                children_count=0,
                synthetic=True,
            ))
        else:
            struct_id = f"tbstruct_{counter:04d}"
            # ── BLUE: structural cell from column left-border to content right-edge ──
            # BLUE structural box spans from col_x0 to the global right border.
            # Using right_border_x (max content_x1 across non-logo bands) instead
            # of per-band content_x1 — thin strokes at band edges can under-detect
            # by 40+ px (e.g. WASILLA thin "A" at band boundary lands at 1860
            # while the true PDF right border is 1907).
            # ORANGE uses per-band content_x1 to stay tight to actual content.
            #
            # Cell-border snap (universal): when a band-local thin tall vertical
            # rule was detected just left of content_x0, that's the cell's true
            # left border.  Snap BLUE there so it hugs the cell frame instead
            # of floating inside the text.  Bands without a snap (most cells)
            # fall back to content_x0 — same behaviour as before.
            base_sx0 = max(col_x0, content_x0)
            if border_abs_x is not None:
                sx0 = max(col_x0, min(base_sx0, border_abs_x))
            else:
                sx0 = base_sx0
            sx1 = min(col_x1, right_border_x)
            sw = max(1, sx1 - sx0)
            new_boxes.append(VisibleBox(
                box_id=struct_id,
                rect=Rect(sx0 / scale, abs_y0 / scale, sx1 / scale, abs_y1 / scale),
                area_pt2=float(sw * rh_box) / max(scale * scale, 1e-6),
                fill_ratio=0.5,
                nested_depth=1,
                is_outer_wrapper=False,
                parent_box_id=None,
                color="BLUE",
                px_bbox=(sx0, abs_y0, sx1, abs_y1),
                children_count=1,
                synthetic=False,
            ))
            # ── ORANGE: text-content box, cell-aligned ────────────────────────
            # Mirror the BLUE wrapper's cell extent so the ORANGE indicates
            # "this whole cell is text content" rather than hugging individual
            # text strokes.  When BLUE was border-snapped (rotated text cells),
            # ORANGE gets the same border alignment so the two boxes nest cleanly.
            cx0 = sx0
            cx1 = sx1
            if cx1 - cx0 >= 4:
                new_boxes.append(VisibleBox(
                    box_id=f"tbtext_{counter:04d}",
                    rect=Rect(cx0 / scale, abs_y0 / scale, cx1 / scale, abs_y1 / scale),
                    area_pt2=float((cx1 - cx0) * rh_box) / max(scale * scale, 1e-6),
                    fill_ratio=0.5,
                    nested_depth=2,
                    is_outer_wrapper=False,
                    parent_box_id=struct_id,
                    color="ORANGE",
                    px_bbox=(cx0, abs_y0, cx1, abs_y1),
                    children_count=0,
                    synthetic=False,
                ))

        counter += 1

    # -----------------------------------------------------------------------
    # Phase 4: tabular-cluster collapse.
    #
    # When several tbstruct cells stack vertically with shared horizontal
    # rules (≤ tbd_table_max_y_gap_px between consecutive cells), the cluster
    # is one logical mini-table — a header row, optional column-header strip,
    # and one or more data rows.  The legend treats BLUE as the *outer* table
    # frame and ORANGE as inner cells, so emitting BLUE on every internal row
    # of a table reads as "this table has many tables inside it" — wrong.
    #
    # Collapse rule:
    #   1. Group consecutive tbstruct cells with y-gap ≤ table_gap_px.
    #   2. Singleton groups stay as-is (single labelled cells are not tables).
    #   3. Groups containing a "title-class" cell (height ≥ title_min_h_px)
    #      stay as-is — the project-name block is a solo prominent cell, not
    #      a row of any table.
    #   4. For multi-cell groups: emit one outer ``tbgroup_*`` BLUE wrapper
    #      spanning the cluster; demote internal tbstruct cells by renaming
    #      to ``tbcell_*`` and recolouring to ORANGE so the renderer draws
    #      them as inner cells, not outer frames.  Drop the redundant
    #      tbtext_ children of demoted cells (the demoted cell IS the
    #      orange content marker now).
    # -----------------------------------------------------------------------
    table_gap_px = int(_cfg_get(cfg, "tbd_table_max_y_gap_px"))
    title_min_h = int(_cfg_get(cfg, "tbd_table_solo_title_min_h_px"))

    # Index by id for easy parent lookup
    struct_cells = sorted(
        [b for b in new_boxes if b.box_id.startswith("tbstruct_")],
        key=lambda b: b.px_bbox[1],
    )
    if len(struct_cells) >= 2:
        # Group adjacent cells.  Two conditions must hold to extend a cluster:
        #
        #  1. y-gap between cells ≤ table_gap_px (they share a printed rule).
        #
        #  2. x-extents match within `tbd_table_xext_tol_px` on each side.
        #     A step-change in x-extent (e.g. a normal cell at x=1789..1909
        #     followed by a wider footer cell at x=1758..1909) is a structural
        #     break — those cells belong to different tables.  Without this
        #     check the cluster becomes a step-shape but renders as a
        #     bounding rectangle, leaving dead white space at the corner
        #     where the narrower cells don't reach the wrapper edge.
        xext_tol = int(_cfg_get(cfg, "tbd_table_xext_tol_px"))
        groups: list[list[VisibleBox]] = [[struct_cells[0]]]
        for cell in struct_cells[1:]:
            prev = groups[-1][-1]
            y_close = (cell.px_bbox[1] - prev.px_bbox[3]) <= table_gap_px
            x_match = (
                abs(cell.px_bbox[0] - prev.px_bbox[0]) <= xext_tol
                and abs(cell.px_bbox[2] - prev.px_bbox[2]) <= xext_tol
            )
            if y_close and x_match:
                groups[-1].append(cell)
            else:
                groups.append([cell])

        import dataclasses as _dc
        rebuilt: list[VisibleBox] = []
        # Carry through non-tbstruct, non-tbtext boxes (logos, etc.) unchanged
        non_struct = [
            b for b in new_boxes
            if not b.box_id.startswith("tbstruct_")
            and not b.box_id.startswith("tbtext_")
        ]
        original_tbtext = [b for b in new_boxes if b.box_id.startswith("tbtext_")]

        # Track which struct cells got demoted so we can drop their tbtext.
        demoted_ids: set[str] = set()

        group_counter = 0
        for grp in groups:
            # Skip if singleton OR contains a tall solo title cell
            has_title_cell = any(
                (c.px_bbox[3] - c.px_bbox[1]) >= title_min_h
                for c in grp
            )
            if len(grp) < 2 or has_title_cell:
                # Keep cells as-is, including their tbtext children
                for cell in grp:
                    rebuilt.append(cell)
                continue

            # Multi-cell tabular cluster: emit outer wrapper + demoted inner cells.
            gx0 = min(c.px_bbox[0] for c in grp)
            gy0 = min(c.px_bbox[1] for c in grp)
            gx1 = max(c.px_bbox[2] for c in grp)
            gy1 = max(c.px_bbox[3] for c in grp)
            # Visual separation from inner cells is handled universally at
            # draw time via OverlayStyle.blue_wrapper_outset_px — every BLUE
            # wrapper draws a few px outside its bbox.  No per-pass padding
            # needed here.
            gw = max(1, gx1 - gx0)
            gh = max(1, gy1 - gy0)
            wrapper_id = f"tbgroup_{group_counter:04d}"
            group_counter += 1
            rebuilt.append(VisibleBox(
                box_id=wrapper_id,
                rect=Rect(gx0 / scale, gy0 / scale, gx1 / scale, gy1 / scale),
                area_pt2=float(gw * gh) / max(scale * scale, 1e-6),
                fill_ratio=0.5,
                nested_depth=1,
                is_outer_wrapper=False,
                parent_box_id=None,
                color="BLUE",
                px_bbox=(gx0, gy0, gx1, gy1),
                children_count=len(grp),
                # synthetic=False: the core renderer's blue-wrapper pass
                # draws only NON-synthetic BLUE boxes (synthetic=True is
                # reserved for *_title alpha washes and *_body outlines).
                # Tabular cluster wrappers ARE the structural outer frame
                # and need to render as a regular blue rectangle.
                synthetic=False,
            ))
            # Demote each cell: tbstruct_* → tbcell_*, BLUE → ORANGE.
            # Reparent under the new wrapper so structure stays clean.
            for cell in grp:
                demoted_ids.add(cell.box_id)
                new_id = cell.box_id.replace("tbstruct_", "tbcell_", 1)
                rebuilt.append(_dc.replace(
                    cell,
                    box_id=new_id,
                    color="ORANGE",
                    parent_box_id=wrapper_id,
                ))

        # Carry through tbtext_ boxes whose parent was NOT demoted.  Dropped
        # tbtext children of demoted cells: the demoted cell itself is now
        # the orange content marker.
        kept_tbtext = [b for b in original_tbtext if b.parent_box_id not in demoted_ids]

        new_boxes = non_struct + rebuilt + kept_tbtext

    return new_boxes, effective_x0


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------

@dataclass
class TitleBlockDetectionPass:
    """Synthesize PURPLE title-block boxes from right-margin raster content.

    Fires when no real BLUE wrapper covers x > tbd_right_margin_frac of
    the image width.  Segments the strip by horizontal rules and emits
    ``titleblkimg_*`` (logo-like) or ``titleblkpanel_*`` (text-panel) PURPLE
    boxes consumed by the core renderer's Pass 3.5 unchanged.
    """

    info: PassInfo = PassInfo(
        name="title_block_detection",
        stage="detect",
        layer_flag="PURPLE",
        order=205,
        description=(
            "Detect right-margin title block content (logos, stamps, project "
            "info) via raster H-line segmentation. Emits titleblk* PURPLE "
            "boxes for all content bands. Works without PDF text layer."
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

        new_boxes, effective_x0 = detect_title_block(
            rgb=state.rgb,
            boxes=result.boxes,
            scale=scale,
            cfg=ctx.cfg,
        )

        # Strip/clip core boxes that conflict with the title block column.
        #
        # ORANGE eviction: the core detector had no knowledge of the title
        # block zone and produced v* ORANGE boxes overlapping our tbtext_* /
        # titleblkimg_* boxes.  Remove any non-synthetic ORANGE whose left
        # edge sits within the title block column — our detection replaces them.
        #
        # BLUE clipping: large earlier BLUE wrappers (e.g. the outer schedule
        # wrapper v0: x=128-1911) bleed their right border into the title block
        # column.  Clip their right edge to effective_x0 so the renderer only
        # draws the wrapper inside the schedule zone, not overlapping our title
        # block cells.  Only wrappers whose right edge is ≥ effective_x0 + 5
        # (i.e. meaningfully inside the title block) are clipped; boxes that
        # legitimately end just before the border are left alone.
        import dataclasses as _dc

        if new_boxes and effective_x0 > 0:
            kept_boxes_raw = [
                b for b in result.boxes
                if not (
                    b.color == "ORANGE"
                    and not getattr(b, "synthetic", False)
                    and b.px_bbox[0] >= effective_x0 - 5
                )
            ]
            kept_boxes = []
            for b in kept_boxes_raw:
                if (b.color == "BLUE"
                        and not getattr(b, "synthetic", False)
                        and not b.box_id.startswith("tbstruct_")
                        and b.px_bbox[2] >= effective_x0 + 5):
                    # Clip right edge to the title block boundary
                    new_x1 = effective_x0 - 1
                    new_rect = Rect(
                        b.rect.x0, b.rect.y0,
                        new_x1 / scale, b.rect.y1,
                    )
                    kept_boxes.append(_dc.replace(
                        b,
                        rect=new_rect,
                        px_bbox=(b.px_bbox[0], b.px_bbox[1], new_x1, b.px_bbox[3]),
                    ))
                else:
                    kept_boxes.append(b)
        else:
            kept_boxes = list(result.boxes)

        stats = dict(result.debug_stats or {})
        stats["title_block_detection"] = len(new_boxes)
        stats["title_block_legacy_orange_removed"] = len(result.boxes) - len(kept_boxes)

        state.result = VisibleBoxResult(
            boxes=[*kept_boxes, *new_boxes],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats=stats,
        )
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        state.artifacts["title_block_detection"] = len(new_boxes)
        return state


__all__ = ["TitleBlockDetectionPass", "detect_title_block"]
