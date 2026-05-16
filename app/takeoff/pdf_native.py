"""Native PyMuPDF text extraction helpers for the takeoff pipeline.

The takeoff layer never opens the file via OCR — it relies entirely on
PyMuPDF's native text layer. Adobe-style PDF generators frequently
emit duplicate words at the same coordinate (typically two copies of a
symbol token rendered for outline + fill), so the helpers here include
a coordinate-tolerant de-duplicator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PdfWord:
    """A single native-text word with its bounding box and source slot."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_no: int
    line_no: int
    word_no: int

    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


def extract_page_words(page: Any) -> list[PdfWord]:
    """Return the native words on a PyMuPDF :class:`fitz.Page`.

    ``get_text("words")`` yields tuples of
    ``(x0, y0, x1, y1, text, block_no, line_no, word_no)``. We normalize
    them into :class:`PdfWord` records and strip pure-whitespace text.
    """
    raw = page.get_text("words") or []
    out: list[PdfWord] = []
    for row in raw:
        if len(row) < 8:
            continue
        x0, y0, x1, y1, text, block_no, line_no, word_no = row[:8]
        if text is None:
            continue
        stripped = str(text).strip()
        if not stripped:
            continue
        out.append(
            PdfWord(
                text=stripped,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                block_no=int(block_no),
                line_no=int(line_no),
                word_no=int(word_no),
            )
        )
    return out


def dedupe_words(words: list[PdfWord], tolerance_pt: float = 0.5) -> list[PdfWord]:
    """De-duplicate words by (text, rounded center) at a coordinate tolerance.

    Adobe-style PDF generators frequently render a single label twice at
    the exact same coordinate (once for outline, once for fill). The
    coordinates are byte-identical in practice but we round to
    ``tolerance_pt`` (default 0.5 PDF points) for robustness.

    Returns a list preserving the order of first occurrence so that
    downstream IDs remain deterministic.
    """
    if tolerance_pt <= 0:
        tolerance_pt = 0.5
    seen: set[tuple[str, int, int]] = set()
    out: list[PdfWord] = []
    for w in words:
        cx, cy = w.center()
        key = (
            w.text,
            int(round(cx / tolerance_pt)),
            int(round(cy / tolerance_pt)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def page_text(page: Any) -> str:
    """Return the page's plain-text rendering (or empty string on error)."""
    try:
        return page.get_text("text") or ""
    except Exception:
        return ""


__all__ = [
    "PdfWord",
    "extract_page_words",
    "dedupe_words",
    "page_text",
]
