"""Structured Introduction / Project Scope extraction from flattened RFP text."""

from __future__ import annotations

from orbitbrief_page_os.segmentation.extract_overlay_text import (
    _structure_introduction_project_scope_sections,
)


def _sample_flat_page() -> str:
    pad = " " + "Additional context for length gate. " * 30
    return (
        "Introduction Tractor Supply Company (“TSC”) is seeking proposals. "
        "The partner(s) must demonstrate the ability to: • Execute a repeatable model "
        "• Maintain consistent quality This program will be phased in 2026. "
        "Partnership with TSC is critical."
        + pad
        + " Project Scope The selected partner(s) "
        "will execute a lifecycle model. This program requires ownership: "
        "• Warehousing and inventory • Onsite installation"
    )


def test_intro_scope_two_sections() -> None:
    secs = _structure_introduction_project_scope_sections(_sample_flat_page())
    assert secs is not None
    assert len(secs) == 2
    assert secs[0]["title"] == "Introduction"
    assert secs[1]["title"] == "Project Scope"
    assert len(secs[0]["bullet_items"]) == 2
    assert secs[0]["closing_paragraphs"]
    assert "This program will be phased" in secs[0]["closing_paragraphs"][0]
    assert len(secs[1]["bullet_items"]) == 2


def test_intro_scope_returns_none_without_markers() -> None:
    assert _structure_introduction_project_scope_sections("Hello world " * 50) is None
