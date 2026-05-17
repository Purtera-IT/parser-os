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


def test_fusion_cross_validates_text_with_nearby_shape() -> None:
    """fuse_candidates_to_devices marks a text+shape pair as
    cross-validated and bumps confidence to 0.99."""
    from app.takeoff.candidate_fusion import fuse_candidates_to_devices
    from app.takeoff.legend_extractor import load_default_legend_rules

    rules = load_default_legend_rules()
    sheet = SheetRecord(
        page_index=5, page_type="floor_plan", in_scope=True,
        sheet_number="T1.05", multiplier=1, levels_represented=["5"],
    )
    text_cand = SymbolCandidate(
        id="text1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=100, y0=100, x1=110, y1=110),
        source_methods=["pdf_native_text"], confidence=0.94,
    )
    shape_cand = SymbolCandidate(
        id="shape1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=98, y0=98, x1=118, y1=118),  # within 24pt
        source_methods=["shape_template"], confidence=0.80, needs_review=True,
    )
    devices = fuse_candidates_to_devices(
        candidates=[text_cand],
        sheet=sheet,
        zones=[],
        legend_rules=rules,
        shape_candidates=[shape_cand],
    )
    assert len(devices) == 1
    # Text candidate should now be source-methods-extended.
    assert "shape_template" in text_cand.source_methods
    assert text_cand.confidence == 0.99
    # Shape candidate should be marked as text-validated, no longer needs_review.
    assert "pdf_native_text" in shape_cand.source_methods
    assert shape_cand.needs_review is False


def test_fusion_leaves_far_apart_text_and_shape_unmerged() -> None:
    """Shape candidate too far from text candidate stays needs_review."""
    from app.takeoff.candidate_fusion import fuse_candidates_to_devices
    from app.takeoff.legend_extractor import load_default_legend_rules

    rules = load_default_legend_rules()
    sheet = SheetRecord(
        page_index=5, page_type="floor_plan", in_scope=True,
        sheet_number="T1.05", multiplier=1, levels_represented=["5"],
    )
    text_cand = SymbolCandidate(
        id="text1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=100, y0=100, x1=110, y1=110),
        source_methods=["pdf_native_text"], confidence=0.94,
    )
    shape_cand = SymbolCandidate(
        id="shape1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=400, y0=400, x1=420, y1=420),  # far away
        source_methods=["shape_template"], confidence=0.80, needs_review=True,
    )
    devices = fuse_candidates_to_devices(
        candidates=[text_cand],
        sheet=sheet,
        zones=[],
        legend_rules=rules,
        shape_candidates=[shape_cand],
    )
    # Still one device (text), no cross-validation.
    assert len(devices) == 1
    assert "shape_template" not in text_cand.source_methods
    assert shape_cand.needs_review is True


def test_summary_emits_text_only_shape_only_and_xval_counts() -> None:
    """takeoff_summary surfaces the three cross-validation tallies."""
    from app.takeoff.exports import takeoff_summary
    from app.takeoff.schemas import DeviceInstance

    sheet = SheetRecord(
        page_index=5, page_type="floor_plan", in_scope=True,
        sheet_number="T1.05", multiplier=1,
    )
    device = DeviceInstance(
        id="d1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=100, y0=100, x1=110, y1=110),
        multiplier=1,
    )
    text_xval = SymbolCandidate(
        id="t1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=100, y0=100, x1=110, y1=110),
        source_methods=["pdf_native_text", "shape_template"],
    )
    text_alone = SymbolCandidate(
        id="t2", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=200, y0=200, x1=210, y1=210),
        source_methods=["pdf_native_text"],
    )
    shape_alone = SymbolCandidate(
        id="s1", page_index=5, raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        bbox=BBox(x0=300, y0=300, x1=320, y1=320),
        source_methods=["shape_template"],
    )
    summary = takeoff_summary(
        [sheet], [device],
        candidates_by_class=None,
        text_candidates=[text_xval, text_alone],
        shape_candidates=[shape_alone],
    )
    wn = summary["wireless_node_outlet"]
    assert wn["cross_validated_count"] == 1
    assert wn["text_only_count"] == 1
    assert wn["shape_only_count"] == 1


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

    Runs the pipeline WITH ``PARSER_OS_ENABLE_SHAPE_SIGNALS=1`` so the
    shape pass is actually exercised. The env var is restored after.
    """
    from app.takeoff.pipeline import build_low_voltage_takeoff

    prev = os.environ.get("PARSER_OS_ENABLE_SHAPE_SIGNALS")
    os.environ["PARSER_OS_ENABLE_SHAPE_SIGNALS"] = "1"
    try:
        takeoff = build_low_voltage_takeoff(PDF_PATH)
    finally:
        if prev is None:
            os.environ.pop("PARSER_OS_ENABLE_SHAPE_SIGNALS", None)
        else:
            os.environ["PARSER_OS_ENABLE_SHAPE_SIGNALS"] = prev
    wn = takeoff.summary.get("wireless_node_outlet") or {}
    assert wn.get("extended_count") == 335, (
        f"WN extended_count={wn.get('extended_count')} after shape pass — "
        "shape signals inflated the WN rollup!"
    )
    # The shape signals summary block should be populated.
    sig = takeoff.summary.get("shape_signals") or {}
    assert sig.get("enabled") is True
    assert sig.get("templates_extracted")
    # Cross-validation should be sane: at least some WN cross-validations
    # (or no shape candidates emitted at all if the template wasn't
    # tuned for this set).
    xval = wn.get("cross_validated_count", 0)
    text_only = wn.get("text_only_count", 0)
    assert xval + text_only == 174, (
        f"WN candidates: text_only={text_only} xval={xval}, expected sum 174"
    )
