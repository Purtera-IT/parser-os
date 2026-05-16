"""Unit tests for app.takeoff.pdf_native — no real PDF required."""
from __future__ import annotations

from app.takeoff.pdf_native import PdfWord, dedupe_words


def _w(text: str, x0: float, y0: float, x1: float, y1: float, word_no: int = 0) -> PdfWord:
    return PdfWord(
        text=text,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        block_no=0,
        line_no=0,
        word_no=word_no,
    )


def test_dedupe_collapses_exact_duplicate_wn() -> None:
    # Two WN tokens with identical coordinates (Adobe outline+fill).
    words = [
        _w("WN", 100.0, 200.0, 116.0, 209.5, word_no=0),
        _w("WN", 100.0, 200.0, 116.0, 209.5, word_no=1),
    ]
    out = dedupe_words(words, tolerance_pt=0.5)
    assert len(out) == 1
    assert out[0].text == "WN"


def test_dedupe_keeps_distinct_wn_at_different_coords() -> None:
    words = [
        _w("WN", 100.0, 200.0, 116.0, 209.5),
        _w("WN", 300.0, 200.0, 316.0, 209.5),
        _w("WN", 100.0, 400.0, 116.0, 409.5),
    ]
    out = dedupe_words(words, tolerance_pt=0.5)
    assert len(out) == 3


def test_dedupe_tolerance_collapses_near_coords() -> None:
    # 0.1pt jitter -> rounds into the same 0.5pt bin -> collapsed.
    words = [
        _w("WN", 100.0, 200.0, 116.0, 209.5),
        _w("WN", 100.1, 200.1, 116.1, 209.6),
    ]
    out = dedupe_words(words, tolerance_pt=0.5)
    assert len(out) == 1


def test_dedupe_preserves_order_of_first_occurrence() -> None:
    words = [
        _w("WN", 100.0, 200.0, 116.0, 209.5, word_no=0),
        _w("CR", 200.0, 200.0, 220.0, 209.5, word_no=1),
        _w("WN", 100.0, 200.0, 116.0, 209.5, word_no=2),
    ]
    out = dedupe_words(words, tolerance_pt=0.5)
    assert [w.text for w in out] == ["WN", "CR"]


def test_dedupe_different_symbols_at_same_coord_stay_separate() -> None:
    words = [
        _w("WN", 100.0, 200.0, 116.0, 209.5),
        _w("CR", 100.0, 200.0, 116.0, 209.5),
    ]
    out = dedupe_words(words, tolerance_pt=0.5)
    assert len(out) == 2
