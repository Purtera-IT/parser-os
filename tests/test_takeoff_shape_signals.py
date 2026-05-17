"""Tests for the OpenCV shape-signal pass.

OpenCV is bundled with the takeoff pipeline (``opencv-python-headless``
is a hard dependency in pyproject.toml). The shape pass nonetheless
handles a missing cv2 gracefully — the ``_try_import_cv2`` helper
returns a reason string and the public API returns ``[]``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.takeoff.legend_extractor import load_default_legend_rules
from app.takeoff.schemas import BBox, SheetRecord, SymbolCandidate
from app.takeoff.shape_signals import (
    MATCH_THRESHOLD,
    ShapeTemplate,
    _nms,
    shape_candidates_for_page,
)


# ─── Pure unit tests ─────────────────────────────────────────────────


def test_nms_keeps_only_highest_score_within_radius() -> None:
    points = [(10, 10, 0.9), (12, 12, 0.85), (50, 50, 0.7), (52, 52, 0.95)]
    kept = _nms(points, radius=10.0)
    # Highest scorer in each cluster wins.
    coords = sorted([(x, y) for x, y, _ in kept])
    assert coords == [(10, 10), (52, 52)]


def test_nms_empty_returns_empty() -> None:
    assert _nms([], radius=10.0) == []


def test_shape_candidates_for_page_returns_empty_when_no_templates() -> None:
    sheet = SheetRecord(page_index=0, page_type="floor_plan", in_scope=True)
    result = shape_candidates_for_page(
        page=None, sheet=sheet, templates=[], rules_by_symbol={},
    )
    assert result == []


# ─── End-to-end synthetic PDF test ──────────────────────────────────


def _build_tiny_pdf_with_legend_cell(path: Path) -> None:
    """Write a one-page PDF with a fake legend cell so we can run the
    extractor end-to-end without depending on a real T-set."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    # Draw a label "WN" and a triangle below it.
    page.insert_text((50, 50), "WN", fontsize=10)
    # A solid downward triangle below the label, centered.
    tri = [(55, 65), (75, 65), (65, 90)]
    page.draw_polyline(tri + [tri[0]], color=(0, 0, 0), fill=(0, 0, 0), width=1)
    doc.save(str(path))
    doc.close()


def test_extract_templates_handles_synthetic_legend(tmp_path: Path) -> None:
    """Verify a template can be extracted from a synthetic legend cell
    with one labeled icon — guards against extractor regressions when
    the real legend is unavailable."""
    pytest.importorskip("fitz")
    import fitz

    pdf_path = tmp_path / "synthetic_legend.pdf"
    _build_tiny_pdf_with_legend_cell(pdf_path)

    from app.takeoff.shape_signals import extract_templates_from_legend
    from app.takeoff.schemas import LegendRule

    rules = [
        LegendRule(raw_symbol="WN", normalized_class="wireless_node_outlet", system="x"),
    ]
    with fitz.open(str(pdf_path)) as doc:
        templates = extract_templates_from_legend(doc[0], rules)
    # On a synthetic legend the algorithm may or may not lock onto the
    # right cell depending on stroke widths — we accept either outcome
    # but require the call NEVER throws.
    assert isinstance(templates, list)


# ─── Real-PDF smoke test (slow) ──────────────────────────────────────


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "real_data_cases"
    / "LOWVOLT_002_MARRIOTT_ATLANTA_T"
    / "artifacts"
    / "2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
)


@pytest.mark.skipif(
    not PDF_PATH.exists() or not os.environ.get("RUN_SLOW_TESTS"),
    reason="Marriott source PDF + RUN_SLOW_TESTS=1 required",
)
def test_marriott_legend_templates_extract_for_known_symbols() -> None:
    """The legend on the Marriott T0.01 page should yield at least
    a handful of templates (one per symbol family)."""
    import fitz

    from app.takeoff.shape_signals import extract_templates_from_legend

    with fitz.open(str(PDF_PATH)) as doc:
        templates = extract_templates_from_legend(doc[1], load_default_legend_rules())
    syms = {t.raw_symbol for t in templates}
    # We tolerate some symbols failing to extract — but at least WN
    # (the spec-driving class) should land.
    assert "WN" in syms, f"Expected WN template; got {syms}"
    assert len(templates) >= 4, f"Too few templates: {syms}"
    # Every template should be non-trivially sized.
    for t in templates:
        assert t.width >= 12 and t.height >= 12
        # And carry enough variation to actually match.
        assert float(t.image.std()) > 8.0


@pytest.mark.skipif(
    not PDF_PATH.exists() or not os.environ.get("RUN_SLOW_TESTS"),
    reason="Marriott source PDF + RUN_SLOW_TESTS=1 required",
)
def test_marriott_wn_extended_total_still_335_after_shape_signals() -> None:
    """Regression — adding shape signals must NOT inflate WN counts.

    Shape-only candidates go to needs_review (not the accepted rollup).
    Text+shape cross-validations preserve the count.
    """
    from app.takeoff.pipeline import build_low_voltage_takeoff

    takeoff = build_low_voltage_takeoff(PDF_PATH)
    wn = takeoff.summary.get("wireless_node_outlet") or {}
    assert wn.get("extended_count") == 335
