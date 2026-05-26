"""Crop the symbol icon from each legend entry.

Real DD legends look like:

    ┌──────────────────────────────────────────────┐
    │  SYMBOL  │  DESCRIPTION             │ CABLE  │
    │  ──────  │  ─────────────────────   │ ─────  │
    │  [icon]  │  CARD READER             │ CAT6   │
    │  [icon]  │  PTZ CAMERA              │ FIBER  │
    │  [icon]  │  WIRELESS ACCESS POINT   │ CAT6   │
    └──────────────────────────────────────────────┘

The icon is a small graphical glyph (often a vector shape or
custom font character) sitting in the left column. We need PNG
crops of those icons because they become the visual templates the
vision-LLM uses to locate the same icon on drawing pages.

This module extracts:

* the **symbol bbox** for each ``ParsedLegendEntry`` (from
  ``symbol_bbox_pdf``) by inflating slightly to capture the full
  glyph including any thin strokes
* a PNG render of that bbox at fixed DPI (300) for downstream
  visual matching
* a manifest mapping ``entry_id → png_path`` so the orchestrator
  can find them deterministically
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Render the icon at high DPI so the vision model has detail.
# 300 DPI matches the parser-os SCHEMATIC_REPLAY_DPI default and
# keeps file size reasonable (~5-20 KB per crop).
ICON_DPI = 300

# Inflate the symbol bbox by this many points to capture full glyph.
ICON_PAD_PT = 4.0


@dataclass(frozen=True)
class LegendSymbolCrop:
    """One symbol icon crop ready for visual matching."""

    entry_id: str
    legend_id: str
    page_index: int
    symbol_text: str                                  # e.g., "CR" / "PTZ"
    label_text: str                                   # e.g., "CARD READER"
    bbox_pdf: tuple[float, float, float, float]
    png_relative_path: str                            # "schematic_legend_crops/<entry_id>.png"
    png_absolute_path: Path
    png_bytes_sha256: str
    width_px: int
    height_px: int


def _inflate_bbox(
    bbox: tuple[float, float, float, float],
    page_rect: tuple[float, float, float, float],
    *,
    pad: float = ICON_PAD_PT,
) -> tuple[float, float, float, float]:
    """Pad bbox by ``pad`` points, clamped to the page rectangle."""
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page_rect
    return (
        max(px0, x0 - pad),
        max(py0, y0 - pad),
        min(px1, x1 + pad),
        min(py1, y1 + pad),
    )


def extract_legend_symbol_crops(
    *,
    legends: Iterable[Any] | None = None,
    legend_entries: Iterable[Any] | None = None,
    pdf_path: Path,
    out_dir: Path,
    relative_prefix: str = "schematic_legend_crops",
    dpi: int = ICON_DPI,
) -> list[LegendSymbolCrop]:
    """Render each ParsedLegendEntry's ``symbol_bbox_pdf`` to a PNG.

    Pass either ``legends`` (preferred — each ParsedLegend carries
    ``page_index`` and ``legend_id``) or ``legend_entries`` (backwards
    compat; entries must have ``page_index`` / ``legend_id`` attributes
    on them, otherwise they're skipped).

    The orchestrator writes:
      <out_dir>/<relative_prefix>/<entry_id>.png

    and the result is added to ``derived_files`` alongside the existing
    schematic_legends.json / schematic_targets.json / schematic_detections.json.
    """
    try:
        import fitz                                   # type: ignore[import-not-found]
    except Exception:                                 # pragma: no cover
        return []

    # Normalize input to a list of (entry, page_index, legend_id) tuples.
    entry_records: list[tuple[Any, int, str]] = []
    if legends is not None:
        for legend in legends:
            lid = getattr(legend, "legend_id", "") or ""
            pix = getattr(legend, "page_index", None)
            if pix is None:
                continue
            for e in getattr(legend, "entries", ()) or ():
                entry_records.append((e, int(pix), lid))
    if legend_entries is not None:
        for e in legend_entries:
            pix = getattr(e, "page_index", None)
            if pix is None:
                continue
            entry_records.append(
                (e, int(pix), getattr(e, "legend_id", "") or "")
            )

    if not entry_records:
        return []

    crops: list[LegendSymbolCrop] = []
    crops_dir = out_dir / relative_prefix
    crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:                                 # pragma: no cover
        return []

    seen_entry_ids: set[str] = set()                  # legends may share entry_id (de-dupe)
    try:
        for entry, page_index, legend_id in entry_records:
            entry_id = getattr(entry, "entry_id", None) or ""
            if not entry_id or entry_id in seen_entry_ids:
                continue
            seen_entry_ids.add(entry_id)

            try:
                page = doc.load_page(int(page_index))
            except Exception:                         # pragma: no cover
                continue
            r = page.rect
            page_rect = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))

            symbol_bbox = getattr(entry, "symbol_bbox_pdf", None)
            if not symbol_bbox or len(symbol_bbox) != 4:
                continue

            try:
                bbox = (
                    float(symbol_bbox[0]),
                    float(symbol_bbox[1]),
                    float(symbol_bbox[2]),
                    float(symbol_bbox[3]),
                )
            except (TypeError, ValueError):
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue                              # degenerate

            inflated = _inflate_bbox(bbox, page_rect)
            try:
                clip = fitz.Rect(*inflated)
                # Matrix scales from 72 DPI (PDF unit) to target DPI.
                scale = dpi / 72.0
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    clip=clip,
                    alpha=False,
                )
            except Exception:                         # pragma: no cover
                continue

            png_relative = f"{relative_prefix}/{entry_id}.png"
            png_absolute = out_dir / relative_prefix / f"{entry_id}.png"
            try:
                pix.save(str(png_absolute))
                bytes_ = png_absolute.read_bytes()
            except Exception:                         # pragma: no cover
                continue
            sha = hashlib.sha256(bytes_).hexdigest()

            crops.append(
                LegendSymbolCrop(
                    entry_id=entry_id,
                    legend_id=legend_id,
                    page_index=int(page_index),
                    symbol_text=str(getattr(entry, "raw_symbol_text", "") or ""),
                    label_text=str(getattr(entry, "label_text", "") or ""),
                    bbox_pdf=inflated,
                    png_relative_path=png_relative,
                    png_absolute_path=png_absolute,
                    png_bytes_sha256=sha,
                    width_px=int(pix.width),
                    height_px=int(pix.height),
                )
            )
    finally:
        try:
            doc.close()
        except Exception:                             # pragma: no cover
            pass
    return crops


__all__ = [
    "ICON_DPI",
    "ICON_PAD_PT",
    "LegendSymbolCrop",
    "extract_legend_symbol_crops",
]
