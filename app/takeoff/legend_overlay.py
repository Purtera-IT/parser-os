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

    box_counts: dict[str, int] = {"BLUE": 0, "ORANGE": 0, "PURPLE": 0, "CYAN_COLHDR": 0}

    # Find each BLUE table-container at depth 1, then identify the
    # ORANGE cells whose top edge is within ~20 px of the container's
    # top — those are the column-header row. Color them cyan.
    blue_d1 = [b for b in result.boxes if b.color == "BLUE" and b.nested_depth == 1]
    cyan_cells: set[str] = set()
    HEADER_BAND_PX = 90  # ~ first row height at 2.5x scale (≈ 36 pt × 2.5)
    for table in blue_d1:
        t_x0, t_y0, t_x1, t_y1 = table.px_bbox
        for b in result.boxes:
            if b.color != "ORANGE":
                continue
            cx0, cy0, cx1, cy1 = b.px_bbox
            # Must be horizontally inside the table.
            if cx0 < t_x0 - 2 or cx1 > t_x1 + 2:
                continue
            # Must sit in the top header band of the table.
            if cy0 < t_y0 - 2 or cy0 > t_y0 + HEADER_BAND_PX:
                continue
            cyan_cells.add(b.box_id)

    # Layer 1 — every detected ORANGE cell (thin outline). Cells flagged
    # as header rows get a cyan fill on top of the outline so the
    # header band reads obviously.
    for b in result.boxes:
        if b.color != "ORANGE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(255, 140, 0, 200), width=2)
        box_counts["ORANGE"] += 1
        if b.box_id in cyan_cells:
            od.rectangle((x0, y0, x1, y1), fill=(0, 200, 220, 90), outline=(0, 180, 200, 255), width=4)
            box_counts["CYAN_COLHDR"] += 1

    # Layer 2 — every BLUE box (nesting hierarchy via line width).
    for b in result.boxes:
        if b.color != "BLUE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        depth = max(0, min(b.nested_depth, 4))
        width = max(2, 9 - 2 * depth)
        od.rectangle((x0, y0, x1, y1), outline=(20, 70, 200, 255), width=width)
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
