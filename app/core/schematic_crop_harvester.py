"""Universal symbol-crop harvester — clean, detection-sourced crops from ANY
drawing page (vector or raster).

Replaces the naive whole-page contour scan, which failed on dense VECTOR sheets
(everything is one connected stroke network -> no symbol-sized isolated contours
-> 0 crops) and produced noisy blobs everywhere else.

Two routes, picked per page automatically:

* **Vector pages** -> :func:`region_proposals.propose_regions` clusters
  symbol-sized vector strokes into candidate regions. This is the detection-
  sourced path: clean symbol clusters, and it works regardless of how dense the
  sheet is (it reads the vector layer, not the rasterized pixels).
* **Raster / scanned pages** (no vector strokes) -> a scale-aware contour
  fallback: downscale huge sheets so symbols land in a sane pixel band, then
  take connected components in a symbol-sized range.

Both yield (bbox_pdf, png_bytes) so the same crops can feed the contrastive
embedder's training AND the teacher-capture store (legend<->canvas pairs).
"""
from __future__ import annotations

import io
from typing import Iterator

import numpy as np

_RENDER_DPI = 200
_PT_PER_IN = 72.0


def _proposals(page, page_index: int):
    try:
        from orbitbrief_page_os.segmentation.schematic.region_proposals import (
            propose_regions,
        )
    except Exception:
        return []
    try:
        return propose_regions(page=page, page_index=page_index, max_proposals=400)
    except Exception:
        return []


def _crop_png(img, x0: int, y0: int, x1: int, y1: int, pad: int = 4) -> bytes | None:
    from PIL import Image

    W, H = img.size
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    buf = io.BytesIO()
    img.crop((x0, y0, x1, y1)).save(buf, format="PNG")
    return buf.getvalue()


def _vector_crops(page, page_index: int, dpi: int) -> list[tuple[tuple, bytes]]:
    props = _proposals(page, page_index)
    if not props:
        return []
    import fitz  # noqa: F401  (page already a fitz page)
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    scale = dpi / _PT_PER_IN
    out = []
    for p in props:
        x0, y0, x1, y1 = p.bbox_pdf
        png = _crop_png(img, int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale))
        if png:
            out.append((p.bbox_pdf, png))
    return out


def _raster_crops(page, dpi: int, max_side: int = 4000,
                  min_px: int = 14, max_px: int = 160) -> list[tuple[tuple, bytes]]:
    """Contour fallback for raster pages. Downscale huge sheets first so symbol
    glyphs fall into [min_px, max_px] instead of exceeding the cap."""
    try:
        import cv2
    except Exception:
        return []
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
    H, W = gray.shape
    ds = 1.0
    if max(H, W) > max_side:
        ds = max_side / float(max(H, W))
        gray = cv2.resize(gray, (int(W * ds), int(H * ds)))
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pil = Image.fromarray(gray)
    inv_scale = _PT_PER_IN / dpi / (ds if ds else 1.0)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if min_px <= w <= max_px and min_px <= h <= max_px and 0.33 <= w / h <= 3.0:
            png = _crop_png(pil, x, y, x + w, y + h)
            if png:
                bbox_pt = (x * inv_scale, y * inv_scale, (x + w) * inv_scale, (y + h) * inv_scale)
                out.append((bbox_pt, png))
    return out


def harvest_page(page, page_index: int, *, dpi: int = _RENDER_DPI) -> list[tuple[tuple, bytes]]:
    """Clean symbol crops for one page. Prefers vector region proposals; falls
    back to the scale-aware contour route on raster pages."""
    crops = _vector_crops(page, page_index, dpi)
    if crops:
        return crops
    return _raster_crops(page, dpi)


def harvest_pdf(pdf_path: str, *, dpi: int = _RENDER_DPI, max_pages: int = 40,
                max_crops: int = 5000) -> list[bytes]:
    """All symbol crops (PNG bytes) for a PDF, across both routes."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
    except Exception:
        return []
    out: list[bytes] = []
    for pi in range(min(doc.page_count, max_pages)):
        try:
            for _bbox, png in harvest_page(doc[pi], pi, dpi=dpi):
                out.append(png)
                if len(out) >= max_crops:
                    return out
        except Exception:
            continue
    return out
