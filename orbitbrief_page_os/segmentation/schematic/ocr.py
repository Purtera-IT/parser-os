"""Optional local OCR adapter for raster schematic pages (PR8).

Pure-Python wrapper over the Tesseract binary via ``pytesseract``.
Fails CLOSED — if the binary is not installed, ``ocr_words`` returns
an empty list and ``status_warning()`` returns a structured
``ocr_unavailable`` warning instead of raising. Callers can decide
whether the missing OCR is a hard failure (per gold-grid policy) or
acceptable (text-extractable PDFs don't need OCR at all).

Determinism: Tesseract is deterministic given identical inputs and
config. We pin the OEM (3 = LSTM-only) and PSM (6 = single block)
so two runs on the same raster produce the same word stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.parsers.schematic_models import SchematicWarning


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float  # 0..1 (Tesseract reports 0..100; we normalize)
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coordinates


def is_available() -> bool:
    """Return True only if pytesseract + the Tesseract binary are usable."""

    try:
        import pytesseract  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        version = pytesseract.get_tesseract_version()
    except Exception:
        return False
    return bool(version)


def ocr_words(image: Any, *, lang: str = "eng", min_confidence: float = 0.6) -> list[OcrWord]:
    """Run Tesseract on a NumPy grayscale image and return word boxes.

    Returns an empty list when Tesseract is not installed or the
    image cannot be OCR'd. Words below ``min_confidence`` are
    dropped.  This function never raises — the parser layer should
    instead inspect ``is_available()`` and emit ``status_warning``
    when OCR coverage is required but unavailable.
    """
    if not is_available() or image is None:
        return []
    try:
        import pytesseract  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return []
    try:
        config = "--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:  # pragma: no cover
        return []
    n = len(data.get("text", []) or [])
    out: list[OcrWord] = []
    for i in range(n):
        text = str(data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        conf_norm = max(0.0, min(1.0, conf / 100.0))
        if conf_norm < min_confidence:
            continue
        try:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
        except (TypeError, ValueError):
            continue
        out.append(
            OcrWord(
                text=text,
                confidence=conf_norm,
                bbox=(x, y, x + w, y + h),
            )
        )
    out.sort(key=lambda w: (w.bbox[1], w.bbox[0], w.text))
    return out


def status_warning(*, page_index: int, sheet_number: str | None) -> SchematicWarning:
    """Build the structured ``ocr_unavailable`` warning."""

    return SchematicWarning.make(
        warning_type="ocr_unavailable",
        page_index=page_index,
        sheet_number=sheet_number,
        detail="Local OCR (Tesseract) is not installed; raster page parsed without OCR.",
    )
