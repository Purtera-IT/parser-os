"""Raster / scanned-drawing fallback for schematic pages (PR8).

PyMuPDF's text layer is empty (or close to empty) on a meaningful
fraction of construction drawings — scans, image-only PDF exports
from raster CAD plotters, or PDFs where the text was flattened into
the page raster. The text-extractable pipeline can do nothing with
those.

This module provides three primitives — all deterministic, all
optional-dependency aware:

- ``render_page_to_ndarray(page, dpi=200)`` — render a PyMuPDF page
  into a grayscale ``np.ndarray`` at the schematic replay DPI. Used
  both by the legend candidate ranker and by the OCR adapter.
- ``deskew_grayscale(arr)`` — small-angle deskew via OpenCV affine
  transform; uses the rotated minAreaRect of the connected components
  to estimate skew. Idempotent on already-aligned images.
- ``is_text_poor_page(page, min_chars=120)`` — quick test the parser
  pre-pass can use to decide whether to take the raster fallback
  branch for a page.

The actual symbol detection path on a raster page is the same OpenCV
``matchTemplate`` flow already in
``orbitbrief_page_os.segmentation.schematic.symbol_detector`` —
nothing new is needed there because PyMuPDF can render any page
(text or raster) into pixels. The fallback's only job is to:

1. Identify that we are on a raster page.
2. Emit a structured ``ocr_unavailable`` warning if OCR is needed
   but Tesseract is not installed.
3. Hand a deterministic raster off to the existing detector.

All public functions handle missing OpenCV / NumPy / PyMuPDF
gracefully by returning ``None`` or empty results rather than
raising — the parser layer is wrapped in a try/except already, but
fail-closed behavior here keeps the failure mode discoverable.
"""
from __future__ import annotations

from typing import Any

from app.parsers.schematic_models import SCHEMATIC_REPLAY_DPI


def is_text_poor_page(page: Any, min_chars: int = 120) -> bool:
    """Return ``True`` when the page's text layer is too thin for vector parsing.

    The threshold is the same heuristic the existing PDF parser uses
    (``app/parsers/orbitbrief_pdf.py``) for its low-text marker
    path. Pages that fall under it should be routed through the
    raster pipeline.
    """
    try:
        text = page.get_text("text") or ""
    except Exception:  # pragma: no cover
        return True
    return len(text.strip()) < min_chars


def render_page_to_ndarray(page: Any, *, dpi: int = SCHEMATIC_REPLAY_DPI) -> Any | None:
    """Render a PyMuPDF page to a ``np.ndarray`` (grayscale, uint8).

    Returns ``None`` when NumPy / PyMuPDF are unavailable or the
    page cannot be rasterized.  The render parameters are fixed so
    two runs of the same input PDF produce byte-identical output.
    """
    try:
        import fitz  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return None
    try:
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csGRAY)
    except Exception:  # pragma: no cover
        return None
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if arr.size != pix.height * pix.width * pix.n:
        return None
    return arr.reshape(pix.height, pix.width)


def deskew_grayscale(arr: Any, *, max_angle_deg: float = 10.0) -> Any | None:
    """Estimate a small rotational skew and undo it.

    Returns the deskewed image or ``None`` when OpenCV / NumPy are
    unavailable. The rotation angle is rounded to 0.1° to keep the
    transform byte-stable.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return None
    if arr is None:
        return None
    # Threshold to binary so connected components ignore page noise.
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None:
        return arr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45.0:
        angle = -(90.0 + angle)
    else:
        angle = -angle
    if abs(angle) > max_angle_deg or abs(angle) < 0.1:
        return arr
    angle_round = round(angle, 1)
    h, w = arr.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_round, 1.0)
    return cv2.warpAffine(
        arr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
