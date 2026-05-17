"""OCR fallback for embedded raster text in low-voltage drawings.

Most low-voltage drawing sets ship as native-PDF text — the symbol
labels (WN, POS-T, TV, …) are extractable as words. A small number of
PDFs (legacy renderings, scanned plotter output, or annotated
photocopies) embed the labels as raster pixels inside an image XObject;
the text pass misses these, and ``shape_signals`` only finds them when
the icon glyph is also distinctive.

This module wraps optional OCR engines (EasyOCR / Tesseract) behind a
lazy import so neither becomes a hard dependency. When neither engine
is installed, the public ``ocr_candidates_for_page`` function returns
``[]`` and the pipeline emits a one-time warning. When an engine IS
available, the function rasterizes the page at 2x scale, runs OCR,
and produces a ``SymbolCandidate`` for every recognized symbol token.

Fusion rules (applied in candidate_fusion.py — TODO):
- OCR + native_text near same center → confidence 0.96 (high — both
  text channels agree).
- OCR + shape near same center → confidence 0.97.
- OCR only → confidence 0.65, needs_review=True.

For now the OCR path is OFF by default — even when ``easyocr`` is
installed, the operator must opt in via
``PARSER_OS_ENABLE_OCR_SIGNALS=1`` because OCR runs are slow (a few
seconds per page).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.ids import stable_id
from app.takeoff.plan_regions import is_excluded, is_inside
from app.takeoff.schemas import BBox, LegendRule, SheetRecord, SymbolCandidate

# Page render scale (smaller than shape-signals because OCR doesn't
# need high resolution — saves time).
PLAN_RENDER_SCALE = 2.0

# OCR confidence cutoff. Anything below this is dropped.
OCR_CONFIDENCE_FLOOR = 0.4


@dataclass
class OCREngineHandle:
    """Wrap whichever OCR engine is in use."""

    name: str  # 'easyocr' | 'pytesseract' | 'none'
    detector: Any | None = None


def _load_engine() -> OCREngineHandle:
    """Probe for an installed OCR engine — return a handle.

    Order: EasyOCR (preferred, GPU-capable) → Tesseract → none.
    """
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return OCREngineHandle(name="easyocr", detector=reader)
    except Exception:  # pragma: no cover - env-specific
        pass
    try:
        import pytesseract  # type: ignore
        return OCREngineHandle(name="pytesseract", detector=pytesseract)
    except Exception:  # pragma: no cover - env-specific
        pass
    return OCREngineHandle(name="none", detector=None)


def _render_page_grayscale(page: Any, scale: float) -> Any | None:
    """Render to a numpy array; return None if the import fails."""
    try:
        import fitz  # type: ignore
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        return None
    try:
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width))
        return arr
    except Exception:
        return None


def _easyocr_extract(
    handle: OCREngineHandle,
    image: Any,
) -> list[tuple[str, BBox, float]]:
    """EasyOCR -> [(text, image_bbox, conf), …]."""
    try:
        results = handle.detector.readtext(image)
    except Exception:
        return []
    out: list[tuple[str, BBox, float]] = []
    for entry in results:
        # entry == ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 'text', conf)
        try:
            poly, text, conf = entry
        except Exception:
            continue
        if conf < OCR_CONFIDENCE_FLOOR:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = BBox(
            x0=float(min(xs)),
            y0=float(min(ys)),
            x1=float(max(xs)),
            y1=float(max(ys)),
            coord_space="image_px",
        )
        out.append((str(text), bbox, float(conf)))
    return out


def _pytesseract_extract(
    handle: OCREngineHandle,
    image: Any,
) -> list[tuple[str, BBox, float]]:
    """Tesseract -> [(text, image_bbox, conf), …]."""
    try:
        import pandas as pd  # noqa: F401 (only used inside try)
        df = handle.detector.image_to_data(
            image, output_type=handle.detector.Output.DICT
        )
    except Exception:
        return []
    out: list[tuple[str, BBox, float]] = []
    n = len(df.get("text", []) or [])
    for i in range(n):
        text = (df["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(df["conf"][i]) / 100.0
        except Exception:
            conf = 0.0
        if conf < OCR_CONFIDENCE_FLOOR:
            continue
        x = float(df["left"][i])
        y = float(df["top"][i])
        w = float(df["width"][i])
        h = float(df["height"][i])
        bbox = BBox(x0=x, y0=y, x1=x + w, y1=y + h, coord_space="image_px")
        out.append((text, bbox, conf))
    return out


def ocr_candidates_for_page(
    *,
    page: Any,
    sheet: SheetRecord,
    legend_rules: list[LegendRule],
    engine: OCREngineHandle | None = None,
) -> tuple[list[SymbolCandidate], str | None]:
    """Run OCR on a single page; return (candidates, warning_reason).

    ``warning_reason`` is non-None when the pass was skipped — e.g.
    "ocr_engine_unavailable".
    """
    eng = engine if engine is not None else _load_engine()
    if eng.name == "none":
        return ([], "ocr_engine_unavailable: install easyocr or pytesseract")
    image = _render_page_grayscale(page, PLAN_RENDER_SCALE)
    if image is None:
        return ([], "ocr_render_failed")
    raw_symbols = {r.raw_symbol for r in legend_rules}
    rule_index = {r.raw_symbol: r for r in legend_rules}

    if eng.name == "easyocr":
        recognized = _easyocr_extract(eng, image)
    else:
        recognized = _pytesseract_extract(eng, image)

    viewport = sheet.plan_viewport
    excluded = sheet.excluded_regions
    page_type = sheet.page_type
    is_device_bearing = page_type in {"floor_plan", "typical_plan"} and sheet.in_scope

    candidates: list[SymbolCandidate] = []
    for text, img_bbox, conf in recognized:
        token = text.strip()
        if token not in raw_symbols:
            continue
        # Convert image bbox → PDF points.
        pdf_bbox = BBox(
            x0=img_bbox.x0 / PLAN_RENDER_SCALE,
            y0=img_bbox.y0 / PLAN_RENDER_SCALE,
            x1=img_bbox.x1 / PLAN_RENDER_SCALE,
            y1=img_bbox.y1 / PLAN_RENDER_SCALE,
            coord_space="pdf_pt",
        )
        rule = rule_index.get(token)
        normalized = rule.normalized_class if rule else None
        rejection_reason: str | None = None
        if not is_device_bearing:
            rejection_reason = f"page_type={page_type} is not device-bearing"
        elif viewport is not None and not is_inside(pdf_bbox, viewport):
            rejection_reason = "outside plan_viewport"
        elif excluded and is_excluded(pdf_bbox, excluded):
            rejection_reason = "inside excluded_region (titleblock)"
        cx, cy = pdf_bbox.center()
        candidates.append(
            SymbolCandidate(
                id=stable_id("ocrcand", sheet.page_index, token, round(cx, 1), round(cy, 1)),
                page_index=sheet.page_index,
                raw_symbol=token,
                normalized_class=normalized,
                bbox=pdf_bbox,
                source_methods=["ocr"],
                confidence=max(0.6, min(0.99, conf if rejection_reason is None else 0.5)),
                rejection_reason=rejection_reason,
                needs_review=True,  # OCR-only candidates always need review
                nearby_text=[],
            )
        )
    return (candidates, None)


__all__ = [
    "OCREngineHandle",
    "OCR_CONFIDENCE_FLOOR",
    "ocr_candidates_for_page",
]
