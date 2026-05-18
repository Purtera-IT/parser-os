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

    # Render — pure raw segmentation output, no device-class matching.
    # The legend page is where symbols are DEFINED; we don't try to detect
    # devices on it. We just visualize what the structural pipeline saw:
    #
    #   * BLUE  outlines  = every BLUE box (table wrappers + sub-wrappers)
    #                        line width grows with nesting depth so the
    #                        hierarchy is visible
    #   * ORANGE outlines = every ORANGE box (cell-level detection)
    #   * PURPLE outlines = every PURPLE box (semantic-cleanup markers)
    #
    # No filling. No device-class colors. No row matching. Pure structure.
    img = Image.fromarray(rgb).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    box_counts: dict[str, int] = {"BLUE": 0, "ORANGE": 0, "PURPLE": 0}

    # Orange first (thin, beneath the blue hierarchy).
    for b in result.boxes:
        if b.color != "ORANGE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(255, 140, 0, 200), width=2)
        box_counts["ORANGE"] += 1

    # Blue next, every depth. Outline width scales inversely with depth so
    # outer tables read as thick frames and inner sub-blocks as thinner
    # ones. Caps at depth 4.
    for b in result.boxes:
        if b.color != "BLUE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        depth = max(0, min(b.nested_depth, 4))
        width = max(2, 9 - 2 * depth)  # depth 0 → 9px, depth 4 → 1px
        od.rectangle((x0, y0, x1, y1), outline=(20, 70, 200, 255), width=width)
        box_counts["BLUE"] += 1

    # Purple last (semantic markers — rare but worth showing).
    for b in result.boxes:
        if b.color != "PURPLE":
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(160, 60, 220, 255), width=3)
        box_counts["PURPLE"] += 1

    summary["box_counts"] = box_counts
    # Drop the symbol-matching summary fields entirely from the legend
    # render — they're a floor-plan concept, not a legend-page concept.
    summary.pop("symbol_hits", None)
    summary.pop("rows_matched", None)
    summary.pop("matches_by_symbol", None)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Small footer line with just the box counts. Doesn't sit on the
    # table content. No device-class swatches — those are floor-plan
    # concerns.
    try:
        font_footer = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_footer = ImageFont.load_default()
    ld = ImageDraw.Draw(img)
    footer = (
        f"segmentation (raw): BLUE={box_counts['BLUE']}  ORANGE={box_counts['ORANGE']}"
        f"  PURPLE={box_counts['PURPLE']}"
    )
    ld.text((30, img.height - 36), footer, fill=(0, 0, 0), font=font_footer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    summary["output"] = str(out_path)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary


__all__ = ["render_legend_overlay"]
