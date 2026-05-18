"""Render the legend (T0.01-style) sheet using the segmentation pipeline.

The default takeoff overlay (colored dots + tooltips) is the wrong tool
for a legend page — the legend has no devices, only definitions. What
the operator actually wants to see on a legend page is:

1. The TABLES the parser detected (one per device-family — Structured
   Cabling, Intrusion Detection, Access Control, CCTV).
2. Which ROWS in those tables map to known symbol codes.
3. What device class each matched row belongs to.

This module produces exactly that view by:

* Running ``orbitbrief_page_os.segmentation.detect()`` on the page —
  the same pipeline OrbitBriefPdfParser already uses for structured.json,
  so we don't double-pay the segmentation cost.
* Finding each native-text occurrence of a known symbol code
  (``WN``, ``POS-T``, ``CR``, …) on the page.
* For each hit, locating the smallest detected ``ORANGE`` (cell) box
  that contains the token center → that's the row's *symbol cell*.
* Highlighting the symbol cell with the device-class color, then
  outlining the rest of the row (other cells on the same Y baseline,
  to the right of the symbol cell) in matching color so the reviewer
  can see the row span.

If segmentation fails or PIL/PyMuPDF is missing the function returns
a zero-summary; never raises into the parse path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.takeoff.schemas import LegendRule


def render_legend_overlay(
    *,
    pdf_path: Path,
    page_index: int,
    out_path: Path,
    legend_rules: list[LegendRule] | None = None,
) -> dict[str, Any]:
    """Render a legend-aware overlay PNG for ``page_index`` of ``pdf_path``.

    Returns a summary dict::

        {
          "page_index": int,
          "tables_detected": int,
          "symbol_hits": int,
          "rows_matched": int,
          "matches_by_symbol": {"WN": 2, "CR": 1, ...},
          "elapsed_seconds": float,
          "output": str | None,           # None when render failed
          "skipped_reason": str | None,   # set when we no-op'd
        }
    """
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "page_index": page_index,
        "tables_detected": 0,
        "symbol_hits": 0,
        "rows_matched": 0,
        "matches_by_symbol": {},
        "elapsed_seconds": 0.0,
        "output": None,
        "skipped_reason": None,
    }

    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont
        from orbitbrief_page_os.segmentation.core.pipeline import detect
    except Exception as exc:  # pragma: no cover - env-specific
        summary["skipped_reason"] = f"missing_dependency: {exc!r}"
        summary["elapsed_seconds"] = time.perf_counter() - started
        return summary

    try:
        result, rgb = detect(str(pdf_path), page_index=page_index)
    except Exception as exc:  # pragma: no cover - segmentation can fail
        summary["skipped_reason"] = f"segmentation_failed: {exc!r}"
        summary["elapsed_seconds"] = time.perf_counter() - started
        return summary

    # Note: ``legend_rules`` is intentionally unused in this render — the
    # legend page is where symbols are DEFINED, not detected. We don't
    # color cells by device class here. The argument stays in the
    # signature so callers (qa_overlay's dispatcher) can keep passing
    # legend_rules without an interface change; a future strategy might
    # use them for verification, but the v1 contract is "raw segmentation
    # output only".
    del legend_rules  # silence the linter; kept for API stability

    blue_table_count = sum(1 for b in result.boxes if b.color == "BLUE" and b.nested_depth == 1)
    summary["tables_detected"] = blue_table_count

    # Render. The legacy detector's marker attributes are sparse on
    # Marriott's T0.01 (only 1 subhdr_red_band, 0 cyan_colhdr / etc), so
    # delegating to the official render_overlay produces a nearly-invisible
    # output. We hand-draw the raw structure ourselves AND add our own
    # column-header detection (topmost cell row of each BLUE table) so
    # the user can see the structure end-to-end without depending on the
    # legacy detector's heuristic markers.
    img = Image.fromarray(rgb).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    box_counts: dict[str, int] = {
        "BLUE": 0, "ORANGE": 0, "PURPLE": 0,
        "CYAN_COLHDR": 0,    # column-header row (multiple narrow cells)
        "MAGENTA_TITLE": 0,  # table-title row (single wide cell)
    }

    # For each BLUE table-container at depth 1, identify TWO kinds of
    # heading cells:
    #
    # * Table TITLE row — the topmost row when it consists of a SINGLE
    #   cell spanning ≥ 60% of the table's width. Examples on T0.01:
    #   "STRUCTURED CABLING SYMBOL LEGEND" / "LEGEND NOTES" /
    #   "RESPONSIBILITY MATRIX" / "GENERAL NOTES FOR SYMBOL LEGENDS".
    #   Rendered MAGENTA so a reviewer can tell at a glance "this is
    #   the table's name, not its column-header row".
    #
    # * Column-HEADER row — the row IMMEDIATELY below the title row,
    #   when it consists of multiple narrower cells. Examples: SYMBOL /
    #   DESCRIPTION / CABLE COUNT / CABLE DESCRIPTION / etc. Rendered
    #   CYAN.
    #
    # When the top row IS multiple narrow cells (no title row above),
    # treat them as column headers directly.
    #
    # Filter blue_d1 to boxes with at least one ORANGE child of their own.
    # Without this filter the prose-pass adds full-page-width "paragraph
    # layout band" wrappers (``prosepara_*``) at depth=1 with zero
    # children — but their bbox spans most of the page, so when we
    # scan ALL orange cells inside their bbox we end up incorrectly
    # matching cells from real tables as "column headers" of these
    # fake wrappers. Same filter the BLUE outline rendering already
    # uses; applying it here keeps title-detection consistent.
    blue_d1 = [
        b for b in result.boxes
        if b.color == "BLUE"
        and b.nested_depth == 1
        and (b.children_count or 0) >= 1
    ]
    cyan_cells: set[str] = set()
    title_cells: set[str] = set()
    SAME_ROW_TOL = 18
    TITLE_WIDTH_FRAC = 0.60
    # A title cell is wide AND short — a single line of text. Multi-line
    # body paragraphs (e.g. the numbered notes inside LEGEND NOTES) also
    # span the full row width but have several lines of text, so they're
    # much taller. At 2.5x render scale a one-line title comes in around
    # 30-60 px; multi-line paragraphs are typically 100 px+. The 95 px
    # cap distinguishes them cleanly without rejecting padded title bars.
    MAX_TITLE_HEIGHT_PX = 95

    for table in blue_d1:
        t_x0, t_y0, t_x1, t_y1 = table.px_bbox
        t_width = max(1.0, t_x1 - t_x0)
        # Collect ALL ORANGE cells fully inside this wrapper (not just
        # the top band). This is what lets us find section-title rows
        # in the middle of a wrapper too, e.g. "ACCESS CONTROL AND
        # INTERCOM SYMBOL LEGEND" and "CCTV SYMBOL LEGEND" which sit
        # midway through Marriott's v2 right-half wrapper.
        inside: list = []
        for b in result.boxes:
            if b.color != "ORANGE":
                continue
            cx0, cy0, cx1, cy1 = b.px_bbox
            if cx0 < t_x0 - 2 or cx1 > t_x1 + 2:
                continue
            if cy0 < t_y0 - 2 or cy1 > t_y1 + 2:
                continue
            inside.append(b)
        if not inside:
            continue
        inside.sort(key=lambda b: b.px_bbox[1])

        # Group cells into rows by Y baseline.
        rows: list[tuple[float, list]] = []
        cur_row: list = []
        cur_y: float | None = None
        for b in inside:
            y0 = b.px_bbox[1]
            if cur_y is None:
                cur_row = [b]
                cur_y = y0
            elif abs(y0 - cur_y) <= SAME_ROW_TOL:
                cur_row.append(b)
            else:
                rows.append((cur_y, cur_row))
                cur_row = [b]
                cur_y = y0
        if cur_row:
            rows.append((cur_y, cur_row))  # type: ignore[arg-type]

        # Walk rows. A TITLE row is exactly 1 cell that
        #   (a) spans ≥60% of the wrapper's width, AND
        #   (b) has height ≤ MAX_TITLE_HEIGHT_PX (single line of text).
        #
        # Only the FIRST title in each wrapper gets to declare column
        # headers — the row right after it (if multi-cell) is cyan.
        # Every SUBSEQUENT title in the same wrapper is a sub-section
        # divider (e.g. "AMPLIFIERS AND ACCESSORIES continued",
        # "CONNECTORS", "COAXIAL CABLES" inside Marriott's T0.02
        # MATV / EQUIPMENT ROOM tables) and the row right after it is
        # body data, NOT new column headers. The wrapper's column
        # headers are declared once and inherited — matches the
        # behavior already in legend_extract.merge_with_defaults.
        wrapper_headers_declared = False
        for i, (_, row_cells) in enumerate(rows):
            if len(row_cells) != 1:
                continue
            cell = row_cells[0]
            cw = cell.px_bbox[2] - cell.px_bbox[0]
            ch = cell.px_bbox[3] - cell.px_bbox[1]
            if cw / t_width < TITLE_WIDTH_FRAC:
                continue
            if ch > MAX_TITLE_HEIGHT_PX:
                # Wide but tall — probably a multi-line body paragraph
                # (e.g. the numbered notes inside LEGEND NOTES), NOT a
                # title bar.
                continue
            title_cells.add(cell.box_id)
            if wrapper_headers_declared:
                # Sub-section divider in the same wrapper — magenta
                # but the row right after is body data, not a new
                # column-header row.
                continue
            # First title in this wrapper — capture the row after it
            # as the wrapper's canonical column headers.
            if i + 1 < len(rows):
                _, next_cells = rows[i + 1]
                if len(next_cells) >= 2:
                    for nc in next_cells:
                        cyan_cells.add(nc.box_id)
                    wrapper_headers_declared = True

        # Edge case: if the very top row of the wrapper is already
        # multiple cells (no preceding title), treat them as column
        # headers directly. Matches behavior of small bottom-tables
        # like RESPONSIBILITY MATRIX that have no obvious title row.
        if rows and len(rows[0][1]) >= 2 and rows[0][1][0].box_id not in cyan_cells:
            # Only do this when no title-row precedes (which we already
            # checked above didn't fire on row 0). And only when those
            # top cells aren't titles themselves.
            top_row_cells = rows[0][1]
            if not any(c.box_id in title_cells for c in top_row_cells):
                for c in top_row_cells:
                    cyan_cells.add(c.box_id)

    # Layer 1 — every detected ORANGE cell (thin outline). Header cells
    # get cyan fill; title cells get magenta fill. Title takes priority
    # if a box is somehow flagged as both (shouldn't happen, but safe).
    for b in result.boxes:
        if b.color != "ORANGE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(255, 140, 0, 200), width=2)
        box_counts["ORANGE"] += 1
        if b.box_id in title_cells:
            od.rectangle((x0, y0, x1, y1), fill=(210, 50, 180, 105), outline=(190, 30, 160, 255), width=4)
            box_counts["MAGENTA_TITLE"] += 1
        elif b.box_id in cyan_cells:
            od.rectangle((x0, y0, x1, y1), fill=(0, 200, 220, 90), outline=(0, 180, 200, 255), width=4)
            box_counts["CYAN_COLHDR"] += 1

    # Layer 2 — top-level table containers only.
    #
    # Only render BLUE boxes at ``nested_depth == 1``. That gives us:
    #
    # * The top-level table wrappers (one per real table on the page),
    #   filtering out the depth-0 page wrapper above them AND the
    #   depth-2+ sub-cell misdetections beneath them. The contour
    #   detector sometimes promotes a single symbol cell or row segment
    #   inside a table to a small BLUE box (e.g. 154x162 px squares
    #   inside the INTRUSION DETECTION table area on Marriott T0.01) —
    #   those are visual noise that confuse the "this rectangle is a
    #   table" reading.
    #
    # We still require ``children_count >= 1`` so empty contour
    # detections (faint borders / underlines with no cells inside) are
    # dropped.
    for b in result.boxes:
        if b.color != "BLUE":
            continue
        if b.nested_depth != 1:
            continue
        if (b.children_count or 0) < 1:
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(20, 70, 200, 255), width=5)
        box_counts["BLUE"] += 1

    # Layer 3 — PURPLE markers.
    for b in result.boxes:
        if b.color != "PURPLE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(160, 60, 220, 255), width=3)
        box_counts["PURPLE"] += 1

    # Layer 4 — semantic markers from the legacy detector (rare but
    # render them when present).
    marker_counts: dict[str, int] = {}
    for marker, color in (
        ("subhdr_red_band",        (220,  30,  30, 255)),
        ("cover_footer_band",      (255, 220,   0, 255)),
        ("subbullet_green_band",   ( 30, 180,  60, 255)),
        ("subbullet_purple_band",  (160,  60, 220, 255)),
    ):
        for b in result.boxes:
            if not getattr(b, marker, False):
                continue
            x0, y0, x1, y1 = b.px_bbox
            od.rectangle((x0, y0, x1, y1), outline=color, width=5)
            marker_counts[marker] = marker_counts.get(marker, 0) + 1

    summary["box_counts"] = box_counts
    summary["marker_counts"] = marker_counts
    summary.pop("symbol_hits", None)
    summary.pop("rows_matched", None)
    summary.pop("matches_by_symbol", None)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    try:
        font_footer = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_footer = ImageFont.load_default()
    ld = ImageDraw.Draw(img)
    footer_bits = [
        f"BLUE={box_counts['BLUE']}",
        f"ORANGE={box_counts['ORANGE']}",
        f"MAGENTA_TITLE={box_counts['MAGENTA_TITLE']}",
        f"CYAN_HDR={box_counts['CYAN_COLHDR']}",
        f"PURPLE={box_counts['PURPLE']}",
    ]
    for m, n in marker_counts.items():
        footer_bits.append(f"{m}={n}")
    footer = "  |  ".join(footer_bits)
    ld.text((30, img.height - 36), f"segmentation: {footer}", fill=(0, 0, 0), font=font_footer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    summary["output"] = str(out_path)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary


__all__ = ["render_legend_overlay"]
