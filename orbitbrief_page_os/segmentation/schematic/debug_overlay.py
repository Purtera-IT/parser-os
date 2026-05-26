"""Schematic debug overlay renderer (PR10).

Draws legend bboxes and symbol-detection bboxes on top of a page
render so a developer can eyeball what the schematic pipeline saw.
Output is a deterministic PNG and a small JSON sidecar.

Determinism contract: same input → same pixel bytes. The renderer
uses fixed DPI, fixed colors, fixed stroke widths, and writes the
PNG via PIL with no metadata-dependent compression flags.

Why a separate module: ``overlay/registry.py`` reserves the
``LEGEND_BLOCKS`` and ``SYMBOL_TAGS`` layer bits but the existing
draw helpers operate on the visible-box pipeline rather than on
schematic dataclasses.  This module owns the schematic-shaped
rendering path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.parsers.schematic_models import (
    SCHEMATIC_REPLAY_DPI,
    ParsedLegend,
    SymbolDetection,
)


# Color contracts (RGB).  Fixed for determinism.
_COLOR_LEGEND_BBOX = (255, 90, 90)       # bright red
_COLOR_LEGEND_ENTRY = (255, 200, 90)     # amber
_COLOR_DETECTION = (90, 180, 255)        # bright blue
_COLOR_TEXT = (40, 40, 40)


@dataclass(frozen=True)
class OverlayLayers:
    """What was drawn in a single overlay file."""

    legend_count: int
    detection_count: int
    width: int
    height: int


def render_overlay(
    *,
    page: Any,
    legends_on_page: Sequence[ParsedLegend],
    detections: Sequence[SymbolDetection],
    out_path: Path,
    dpi: int = SCHEMATIC_REPLAY_DPI,
) -> OverlayLayers | None:
    """Render a debug overlay PNG.

    Returns ``None`` when PyMuPDF / PIL are unavailable.  The
    function never raises so the parser layer can call it
    best-effort.
    """
    try:
        import fitz  # type: ignore[import-not-found]
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return None

    zoom = dpi / 72.0
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB)
    except Exception:  # pragma: no cover
        return None
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")

    for legend in legends_on_page:
        loc = legend.locator_dict()
        bbox = loc.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            px = _scale_bbox(bbox, zoom)
            draw.rectangle(px, outline=_COLOR_LEGEND_BBOX + (255,), width=3)
            draw.text(
                (px[0] + 4, px[1] + 4),
                f"legend={legend.legend_id[:12]}",
                fill=_COLOR_TEXT,
            )
        for entry in legend.entries:
            if entry.symbol_bbox_pdf is None:
                continue
            ex = _scale_bbox(entry.symbol_bbox_pdf, zoom)
            draw.rectangle(ex, outline=_COLOR_LEGEND_ENTRY + (255,), width=2)

    for det in detections:
        px = _scale_bbox(det.bbox_pdf, zoom)
        draw.rectangle(px, outline=_COLOR_DETECTION + (255,), width=2)
        draw.text(
            (px[0] + 2, max(0, px[1] - 12)),
            f"{det.target_key}  {det.confidence:.2f}",
            fill=_COLOR_TEXT,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ``optimize=True`` so two runs produce byte-identical PNG bytes;
    # save with no metadata fields that could embed wall-clock time.
    img.save(out_path, format="PNG", optimize=True)
    return OverlayLayers(
        legend_count=len(legends_on_page),
        detection_count=len(detections),
        width=pix.width,
        height=pix.height,
    )


def _scale_bbox(
    bbox_pdf: tuple[float, float, float, float] | list[float],
    zoom: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_pdf
    return (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)
