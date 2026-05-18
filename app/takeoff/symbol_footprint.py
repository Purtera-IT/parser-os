"""Per-symbol footprint inflation factors, derived from the legend.

When the detector finds a symbol on a plan, it matches the *text token*
("WN", "CR", "POS-T"). The text bbox alone is just the letters — the
device's visual extent on the drawing is the FULL legend symbol: text
PLUS the icon strokes (antenna for WN, hexagon for CR, etc.).

This module reads the legend page once per PDF and computes, for each
symbol code, the inflation ratios needed to expand a text bbox to cover
the entire legend-defined shape. It's intentionally read-only and side
effect free — the output is a flat dict that the QA overlay consumes.

The pipeline never overwrites :attr:`SymbolCandidate.bbox` with the
inflated geometry, because every downstream layer (zone routing, room
labeling, candidate fusion) reasons about the text's coordinate to
stay deterministic. Inflation is purely a render-time concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SymbolFootprint:
    """How to inflate a text bbox into the full symbol footprint.

    All four offsets are *signed multiples of the text's width/height*,
    measured from the text's center. Given a text bbox with width
    ``tw`` / height ``th`` centered at ``(tcx, tcy)``::

        new_x0 = tcx + left  * tw
        new_x1 = tcx + right * tw
        new_y0 = tcy + top   * th
        new_y1 = tcy + bot   * th

    A no-op footprint is ``left=-0.5, right=+0.5, top=-0.5, bot=+0.5``
    (i.e. "the symbol IS just the text"). For WN on the Marriott legend
    the values look like ``left=-0.5, right=+0.5, top=-0.5, bot=+1.93``
    — the antenna icon extends roughly two text-heights *below* the
    text, mirroring how the symbol is drawn on every plan.
    """

    symbol_code: str
    left: float    # negative — extension left of text center
    right: float   # positive — extension right of text center
    top: float     # negative — extension above text center
    bot: float     # positive — extension below text center


NO_OP_FOOTPRINT = SymbolFootprint(
    symbol_code="",
    left=-0.5,
    right=0.5,
    top=-0.5,
    bot=0.5,
)


def _resolve_symbol_col(headers: list[dict[str, Any]] | None) -> int | None:
    """Return the column index of the SYMBOL column, or None if no columns."""
    if not headers:
        return None
    for i, c in enumerate(headers):
        if "SYMBOL" in (c.get("text") or "").upper():
            return i
    return 0


def _iter_legend_pages(legend_doc: Any) -> list[dict[str, Any]]:
    """Normalize the legend doc into a list of single-page docs."""
    if isinstance(legend_doc, list):
        return [d for d in legend_doc if isinstance(d, dict)]
    if isinstance(legend_doc, dict):
        return [legend_doc]
    return []


def build_symbol_footprints(
    *,
    pdf_path: Path,
    legend_doc: Any,
    known_codes: set[str],
) -> dict[str, SymbolFootprint]:
    """Build a footprint map covering every code in ``known_codes`` that
    appears as a text token inside a legend SYMBOL cell.

    Codes missing from the legend (or whose cell has no extractable
    drawings) are absent from the result — the caller falls back to
    :data:`NO_OP_FOOTPRINT`.

    Silently no-ops (returns ``{}``) when PyMuPDF is unavailable or
    the PDF can't be opened.
    """
    out: dict[str, SymbolFootprint] = {}
    try:
        import fitz
    except Exception:  # pragma: no cover - env-specific
        return out

    pages = _iter_legend_pages(legend_doc)
    if not pages:
        return out

    try:
        pdf_doc = fitz.open(str(pdf_path))
    except Exception:  # pragma: no cover - env-specific
        return out

    try:
        # Per-page scan — one legend doc per PDF page captured.
        for legend_page in pages:
            page_index = legend_page.get("page_index")
            if page_index is None:
                continue
            try:
                page_index = int(page_index)
            except (TypeError, ValueError):
                continue
            if not (0 <= page_index < pdf_doc.page_count):
                continue
            page = pdf_doc[page_index]
            drawings = page.get_drawings() or []
            words = page.get_text("words") or []

            for table in legend_page.get("tables", []) or []:
                for section in table.get("sections", []) or []:
                    cols = section.get("column_headers") or []
                    sym_col = _resolve_symbol_col(cols)
                    if sym_col is None:
                        continue
                    for row in section.get("rows", []) or []:
                        _scan_row_for_footprint(
                            row=row,
                            sym_col=sym_col,
                            words=words,
                            drawings=drawings,
                            known_codes=known_codes,
                            out=out,
                        )
    finally:
        pdf_doc.close()

    return out


def _scan_row_for_footprint(
    *,
    row: dict[str, Any],
    sym_col: int,
    words: list[Any],
    drawings: list[Any],
    known_codes: set[str],
    out: dict[str, SymbolFootprint],
) -> None:
    """Find the text token + drawing union inside a single legend row's
    SYMBOL cell, and record it as a footprint in ``out``.

    Skips when the row has already been captured (first occurrence wins —
    if the same symbol code appears in two rows, e.g. WN ceiling vs WN
    wall, the geometry is consistent enough that the first sample is
    representative).
    """
    cells = row.get("cells") or []
    if sym_col >= len(cells):
        return
    cell_bbox = cells[sym_col].get("bbox_pt")
    if not cell_bbox or len(cell_bbox) != 4:
        return
    cx0, cy0, cx1, cy1 = cell_bbox
    cell_w = cx1 - cx0
    cell_h = cy1 - cy0
    if cell_w <= 0 or cell_h <= 0:
        return

    # Find the FIRST text token inside the cell that matches a known code.
    text_box: tuple[float, float, float, float] | None = None
    matched_code: str | None = None
    for w in words:
        if len(w) < 5:
            continue
        wx0, wy0, wx1, wy1, wt = w[0], w[1], w[2], w[3], w[4]
        if not (cx0 <= wx0 and wx1 <= cx1 and cy0 <= wy0 and wy1 <= cy1):
            continue
        cleaned = str(wt).strip().upper()
        if cleaned in known_codes:
            text_box = (float(wx0), float(wy0), float(wx1), float(wy1))
            matched_code = cleaned
            break

    if text_box is None or matched_code is None:
        return
    if matched_code in out:
        return  # first row wins

    # Union the text bbox with every drawing primitive that lives
    # entirely inside the cell (excluding the cell-border rectangle
    # and 0-area markers).
    ux0, uy0, ux1, uy1 = text_box
    for d in drawings:
        r = d.get("rect") if isinstance(d, dict) else None
        if r is None:
            continue
        # PyMuPDF Rect — read via attributes (also works with namedtuple-likes).
        try:
            rx0, ry0, rx1, ry1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
        except AttributeError:
            continue
        if not (rx0 >= cx0 and rx1 <= cx1 and ry0 >= cy0 and ry1 <= cy1):
            continue
        rw, rh = rx1 - rx0, ry1 - ry0
        # Skip cell-border-shaped rectangles.
        if rw > 0.8 * cell_w and rh > 0.8 * cell_h:
            continue
        # Skip 0-area markers (degenerate horizontal/vertical lines collapse
        # to one dimension being 0 — we keep those because they may be the
        # antenna line; only skip when BOTH are 0).
        if rw == 0 and rh == 0:
            continue
        ux0 = min(ux0, rx0)
        uy0 = min(uy0, ry0)
        ux1 = max(ux1, rx1)
        uy1 = max(uy1, ry1)

    tw = max(0.5, text_box[2] - text_box[0])
    th = max(0.5, text_box[3] - text_box[1])
    tcx = (text_box[0] + text_box[2]) / 2.0
    tcy = (text_box[1] + text_box[3]) / 2.0
    out[matched_code] = SymbolFootprint(
        symbol_code=matched_code,
        left=(ux0 - tcx) / tw,
        right=(ux1 - tcx) / tw,
        top=(uy0 - tcy) / th,
        bot=(uy1 - tcy) / th,
    )


def inflate_bbox(
    *,
    text_x0: float,
    text_y0: float,
    text_x1: float,
    text_y1: float,
    footprint: SymbolFootprint,
) -> tuple[float, float, float, float]:
    """Return ``(x0, y0, x1, y1)`` of the inflated bbox for given text geometry."""
    tw = max(0.5, text_x1 - text_x0)
    th = max(0.5, text_y1 - text_y0)
    tcx = (text_x0 + text_x1) / 2.0
    tcy = (text_y0 + text_y1) / 2.0
    return (
        tcx + footprint.left * tw,
        tcy + footprint.top * th,
        tcx + footprint.right * tw,
        tcy + footprint.bot * th,
    )


__all__ = [
    "NO_OP_FOOTPRINT",
    "SymbolFootprint",
    "build_symbol_footprints",
    "inflate_bbox",
]
