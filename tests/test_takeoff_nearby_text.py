"""Tests for :mod:`app.takeoff.nearby_text`."""
from __future__ import annotations

from app.takeoff.nearby_text import collect_nearby_text
from app.takeoff.pdf_native import PdfWord
from app.takeoff.schemas import BBox


def _word(text: str, x0: float, y0: float, w: float = 24.0, h: float = 12.0) -> PdfWord:
    return PdfWord(text=text, x0=x0, y0=y0, x1=x0 + w, y1=y0 + h, block_no=0, line_no=0, word_no=0)


def test_collect_returns_room_label_within_radius() -> None:
    bbox = BBox(x0=100, y0=100, x1=120, y1=112)
    words = [
        _word("EXISTING", 130, 100),
        _word("MDF", 165, 100),
        _word("ROOM", 195, 100),
        _word("STORAGE", 400, 400),  # far away — excluded
    ]
    result = collect_nearby_text(bbox=bbox, page_words=words, own_symbol="WN", radius_pt=80)
    assert any("MDF" in r for r in result), result
    assert "STORAGE" not in (" ".join(result))


def test_collect_filters_phrases_that_are_only_the_own_symbol() -> None:
    """Two adjacent 'WN' tokens must not become a 'WN WN' nearby_text entry."""
    bbox = BBox(x0=100, y0=100, x1=110, y1=110)
    words = [
        _word("WN", 90, 100, w=14),
        _word("WN", 115, 100, w=14),
        _word("CORRIDOR", 60, 100, w=80),
    ]
    result = collect_nearby_text(bbox=bbox, page_words=words, own_symbol="WN", radius_pt=80)
    assert all(("WN" not in r.split() or r != "WN WN") for r in result), result
    # The corridor label DOES survive.
    assert any("CORRIDOR" in r for r in result), result


def test_collect_excludes_header_noise() -> None:
    bbox = BBox(x0=100, y0=100, x1=110, y1=110)
    words = [
        _word("CABLE", 130, 100),
        _word("TYPE", 165, 100),
        _word("FROM", 200, 100),
    ]
    result = collect_nearby_text(bbox=bbox, page_words=words, own_symbol="WN", radius_pt=80)
    # All three are in _HEADER_NOISE — none should survive.
    assert result == [], result


def test_collect_keeps_idf_and_mdf_markers_even_if_short() -> None:
    bbox = BBox(x0=100, y0=100, x1=110, y1=110)
    words = [
        _word("IDF-2", 130, 100),
        _word("MDF", 165, 100),
    ]
    result = collect_nearby_text(bbox=bbox, page_words=words, own_symbol="WN", radius_pt=80)
    # Both short markers should be kept.
    joined = " ".join(result)
    assert "IDF-2" in joined or "MDF" in joined, result


def test_collect_returns_in_distance_order() -> None:
    bbox = BBox(x0=100, y0=100, x1=110, y1=110)
    words = [
        _word("FAR_LABEL", 200, 100, w=80),
        _word("CLOSER_LABEL", 130, 100, w=80),
    ]
    result = collect_nearby_text(bbox=bbox, page_words=words, own_symbol="WN", radius_pt=300)
    # 'CLOSER_LABEL' is closer to the bbox center.
    assert result.index("CLOSER_LABEL") < result.index("FAR_LABEL"), result
