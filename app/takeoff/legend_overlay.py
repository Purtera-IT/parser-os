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

from app.takeoff.legend_extractor import load_default_legend_rules
from app.takeoff.pdf_native import extract_page_words
from app.takeoff.schemas import LegendRule


# Symbol color palette — mirrors the takeoff QA overlay so a reviewer
# can correlate "WN cyan on T0.01" with "WN cyan on T1.03". Kept local
# here (not imported from qa_overlay) so the legend overlay has no
# coupling to the QA overlay module — they share a convention, not code.
_COLORS: dict[str, tuple[int, int, int]] = {
    "WN":    (  0, 200, 255),
    "POS-T": ( 80, 200,  80),
    "POS-P": ( 30, 150,  30),
    "TV":    (255, 140,   0),
    "CR":    (220,   0, 180),
    "DA":    (220,  30,  30),
    "H":     (255, 200,   0),
}


def _color_for_symbol(symbol: str) -> tuple[int, int, int]:
    return _COLORS.get(symbol.upper(), (90, 90, 90))


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

    if legend_rules is None:
        legend_rules = load_default_legend_rules()
    symbol_set = {r.raw_symbol.upper() for r in legend_rules}

    # Compute pdf-pt → image-px scale (segmentation renders at 2.5x by default
    # but we derive it from the actual image so any cfg overrides work).
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        sx = result.image_width / page.rect.width
        sy = result.image_height / page.rect.height
        words = extract_page_words(page)
    finally:
        doc.close()

    # Find symbol-token hits on the page.
    symbol_hits: list[tuple[str, float, float, Any]] = []
    for w in words:
        text = (w.text or "").strip().upper().rstrip(".,;:")
        if text in symbol_set:
            cx = (w.x0 + w.x1) / 2 * sx
            cy = (w.y0 + w.y1) / 2 * sy
            symbol_hits.append((text, cx, cy, w))
    summary["symbol_hits"] = len(symbol_hits)

    blue_table_count = sum(1 for b in result.boxes if b.color == "BLUE" and b.nested_depth == 1)
    summary["tables_detected"] = blue_table_count

    # Render base image.
    img = Image.fromarray(rgb).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # Layer 1 — outline each detected table container in blue.
    for b in result.boxes:
        if b.color != "BLUE" or b.nested_depth != 1:
            continue
        x0, y0, x1, y1 = b.px_bbox
        od.rectangle((x0, y0, x1, y1), outline=(20, 70, 200, 255), width=8)

    # Layer 2 — for each symbol hit, find its containing orange cell + color
    # the cell + outline same-baseline neighbour cells in the matching color.
    matches_by_symbol: dict[str, int] = {}
    for sym, cx, cy, _word in symbol_hits:
        # Smallest orange box containing the hit center is the symbol cell.
        cell = None
        cell_area = float("inf")
        for b in result.boxes:
            if b.color != "ORANGE":
                continue
            x0, y0, x1, y1 = b.px_bbox
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                area = (x1 - x0) * (y1 - y0)
                if area < cell_area:
                    cell_area = area
                    cell = b
        if cell is None:
            continue
        color = _color_for_symbol(sym)
        cx0, cy0, cx1, cy1 = cell.px_bbox
        od.rectangle((cx0, cy0, cx1, cy1), fill=color + (120,), outline=color + (255,), width=6)
        # Outline neighbour cells on same baseline, to the right.
        row_y0, row_y1 = cy0 - 6, cy1 + 6
        for b in result.boxes:
            if b.color != "ORANGE" or b is cell:
                continue
            bx0, by0, bx1, by1 = b.px_bbox
            if by0 >= row_y0 and by1 <= row_y1 and bx0 > cx1:
                od.rectangle((bx0, by0, bx1, by1), outline=color + (220,), width=3)
        matches_by_symbol[sym] = matches_by_symbol.get(sym, 0) + 1

    summary["matches_by_symbol"] = matches_by_symbol
    summary["rows_matched"] = sum(matches_by_symbol.values())

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Legend annotation box.
    try:
        font_lg = ImageFont.truetype("arial.ttf", 44)
        font_md = ImageFont.truetype("arial.ttf", 30)
        font_sm = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    ld = ImageDraw.Draw(img)
    LBOX_W = 740
    LBOX_H = 240 + 42 * len(_COLORS)
    ld.rectangle((30, 30, 30 + LBOX_W, 30 + LBOX_H), fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    ld.text((45, 42), "Legend page — segmentation-aware overlay", fill=(0, 0, 0), font=font_lg)
    ld.text((45, 100), "Tables detected by segmentation pipeline, rows", fill=(50, 50, 50), font=font_md)
    ld.text((45, 138), "colored by matched device class.", fill=(50, 50, 50), font=font_md)
    y = 200
    ld.rectangle((50, y + 4, 76, y + 30), fill=(255, 255, 255), outline=(20, 70, 200), width=4)
    ld.text((90, y + 4), f"BLUE: {blue_table_count} table container(s)", fill=(0, 0, 0), font=font_sm)
    y += 44
    ld.text(
        (45, y),
        f"{summary['rows_matched']} row(s) matched / {summary['symbol_hits']} symbol token(s) found",
        fill=(0, 80, 0),
        font=font_md,
    )
    y += 44
    for sym, color in _COLORS.items():
        n = matches_by_symbol.get(sym, 0)
        ld.rectangle((50, y + 4, 76, y + 30), fill=color, outline=(0, 0, 0))
        text_color = (0, 0, 0) if n else (140, 140, 140)
        ld.text((90, y + 4), f"{sym}: {n} matched row(s)", fill=text_color, font=font_sm)
        y += 42

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    summary["output"] = str(out_path)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary


__all__ = ["render_legend_overlay"]
