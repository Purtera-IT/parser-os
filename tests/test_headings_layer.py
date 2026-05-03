"""Heading layer: template merge, heuristics, span hooks."""

from __future__ import annotations

from orbitbrief_page_os.segmentation.headings.builtin import TSC_FOLLOWON_HEADINGS
from orbitbrief_page_os.segmentation.headings.heuristics import (
    generic_heading_candidates,
    score_line_as_generic_heading,
)
from orbitbrief_page_os.segmentation.headings.template import (
    load_overlay_heading_template,
    resolve_effective_headings,
)


def test_resolve_effective_headings_appends_extra() -> None:
    overlay = {
        "heading_template": {
            "followon_headings_extra": ["Appendix A"],
            "major_band_titles_extra": ["Appendix A"],
        }
    }
    f, m = resolve_effective_headings(overlay)
    assert "Appendix A" in f
    assert "Appendix A" in m
    assert f[:3] == TSC_FOLLOWON_HEADINGS[:3]


def test_resolve_replace_followon() -> None:
    overlay = {"heading_template": {"followon_headings_replace": ["One", "Two"]}}
    f, _m = resolve_effective_headings(overlay)
    assert f == ("One", "Two")


def test_generic_heading_score_short_title_case() -> None:
    sc, reasons = score_line_as_generic_heading("Scope of Work")
    assert sc > 0
    assert reasons


def test_generic_candidates_non_empty_on_span_like_lines() -> None:
    lines = [
        {
            "text": "Site Survey",
            "bbox_pdf": [0, 0, 1, 1],
            "font_size_pt_max": 14.0,
            "bold_span_ratio": 1.0,
            "color_rgb_sample": [0, 0, 200],
        }
    ]
    c = generic_heading_candidates(lines)
    assert c and c[0]["text"] == "Site Survey"
