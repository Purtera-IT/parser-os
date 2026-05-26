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


def words_to_textblocks(
    words: list["OcrWord"],
    *,
    page_dpi: int,
) -> list:
    """Project ``OcrWord`` records (pixel coords) onto ``TextBlock`` rows
    in PDF points.

    Reverses the schematic replay DPI used by ``render_page_to_ndarray``
    so the resulting TextBlock bbox lines up with the same coordinate
    space the legend locator and symbol detector use. Words on the
    same y-band are grouped into a single line-shaped TextBlock so
    the locator's row clustering can find tabular legend rows.

    Returns an empty list if ``words`` is empty.  Determinism: same
    word list -> same TextBlock list (the grouping uses a sorted
    y-band and concatenates words by ascending x0).
    """
    from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock

    if not words:
        return []
    scale = 72.0 / float(page_dpi)
    rows: list[list[OcrWord]] = []
    sorted_words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    Y_TOL_PT = 6.0
    for w in sorted_words:
        y_center_pt = ((w.bbox[1] + w.bbox[3]) / 2.0) * scale
        placed = False
        for row in rows:
            ref_y = sum(((rw.bbox[1] + rw.bbox[3]) / 2.0) * scale for rw in row) / len(row)
            if abs(y_center_pt - ref_y) <= Y_TOL_PT:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    out: list = []
    for row_idx, row in enumerate(rows):
        row.sort(key=lambda w: w.bbox[0])
        text = " ".join(w.text for w in row)
        x0 = min(w.bbox[0] for w in row) * scale
        y0 = min(w.bbox[1] for w in row) * scale
        x1 = max(w.bbox[2] for w in row) * scale
        y1 = max(w.bbox[3] for w in row) * scale
        out.append(
            TextBlock(
                text=text,
                bbox=(x0, y0, x1, y1),
                block_index=row_idx,
                line_index=0,
            )
        )
    out.sort(key=lambda b: (round(b.bbox[1], 2), round(b.bbox[0], 2), b.block_index, b.line_index))
    return out
