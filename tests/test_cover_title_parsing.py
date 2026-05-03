"""Unit tests for RFP / cover-page title splitting in extract_overlay_text."""

from __future__ import annotations

from orbitbrief_page_os.segmentation.extract_overlay_text import (
    _parse_cover_page_title_subtitles,
    _polish_cover_year_line,
    _split_flat_cover_title_line,
)
from orbitbrief_page_os.segmentation.passes.cover_page_title_bands import (
    _cover_page_text_signal,
    _footer_line_index,
)


def test_split_flat_tsc_style_line() -> None:
    s = (
        "Tractor Supply Company (TSC) Wireless Access Point (AP) Refresh Program "
        "Request for Proposal (RFP) Confidential 2026–2027 Deployment"
    )
    parts = _split_flat_cover_title_line(s)
    assert parts[0] == "Tractor Supply Company (TSC)"
    assert "Wireless Access Point (AP) Refresh Program" in parts
    assert "Request for Proposal (RFP)" in parts
    assert any("Confidential" in p for p in parts)
    assert any("2026" in p and "2027" in p for p in parts)


def test_parse_multiline_preserves_lines() -> None:
    raw = "Line One\nLine Two\nLine Three"
    title, subs = _parse_cover_page_title_subtitles(raw)
    assert title == "Line One"
    assert subs == ["Line Two", "Line Three"]


def test_polish_year_range() -> None:
    assert _polish_cover_year_line("2026\u202f2027 Deployment") == "2026–2027 Deployment"
    assert _polish_cover_year_line("2026–2027 Deployment") == "2026–2027 Deployment"


def test_cover_page_text_signal_tsc_style() -> None:
    s = (
        "Tractor Supply Company (TSC) Wireless Access Point (AP) Refresh Program "
        "Request for Proposal (RFP) Confidential 2026–2027 Deployment"
    )
    assert _cover_page_text_signal(s)


def test_cover_page_text_signal_generic_cover() -> None:
    s = (
        "Acme Corporation\n"
        "Annual Strategy Review\n"
        "Prepared for the Board of Directors\n"
        "Q4 Planning Session"
    )
    assert _cover_page_text_signal(s)


def test_cover_page_text_signal_rejects_body_paragraph() -> None:
    s = (
        "this is a long paragraph of body copy without any short title-cased "
        "lines, it just keeps going and is full of sentence-ending punctuation."
    )
    assert not _cover_page_text_signal(s)


def test_footer_line_index_finds_confidential() -> None:
    lines = [
        {"text": "Title A", "max_size": 20.0, "bold_ratio": 1.0},
        {"text": "Subtitle", "max_size": 14.0, "bold_ratio": 1.0},
        {"text": "Confidential", "max_size": 11.0, "bold_ratio": 0.0},
        {"text": "2026–2027 Deployment", "max_size": 11.0, "bold_ratio": 0.0},
    ]
    assert _footer_line_index(lines) == 2
