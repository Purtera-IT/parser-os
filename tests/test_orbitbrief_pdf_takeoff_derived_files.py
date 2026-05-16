"""Integration tests: ``parse_artifact`` emits takeoff alongside structured.

The fast tests synthesize a tiny one-page PDF in ``tmp_path`` so the
structured parser doesn't have to walk a 25-sheet drawing set on every
CI run. A separate slow test exercises the real Marriott PDF — gated
behind ``RUN_SLOW_TESTS=1`` so it stays out of the fast loop.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

MARRIOTT_PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "real_data_cases"
    / "LOWVOLT_002_MARRIOTT_ATLANTA_T"
    / "artifacts"
    / "2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
)

ENV_OVERLAY = "PARSER_OS_WRITE_TAKEOFF_QA"


def _make_tiny_lvolt_pdf(path: Path) -> None:
    """Write a one-page PDF that classifies as ``floor_plan`` and contains
    a single WN token inside the default plan viewport.

    Letter portrait (612 x 792 pt). Plan viewport defaults to
    ``[0, 0, w*0.84, h*0.94]`` so a token placed near ``(100, 200)`` is
    well inside.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Sheet title line — the classifier reads this exactly.
    page.insert_text(
        (40, 60),
        "SHEET NUMBER: T1.01 - LOWER LOBBY FLOOR PLAN",
        fontsize=10,
    )
    # A homerun zone note so the zone resolver assigns the device.
    page.insert_text(
        (40, 80),
        "HOMERUN ALL CABLES ON THIS LEVEL TO MDF ROOM, THIS LEVEL.",
        fontsize=8,
    )
    # Two WN tokens inside the plan viewport. (One of them with the same
    # coords as a duplicate to exercise the dedupe path indirectly.)
    page.insert_text((120, 200), "WN", fontsize=10)
    page.insert_text((220, 320), "WN", fontsize=10)
    doc.save(str(path))
    doc.close()


def _unset_overlay_env() -> str | None:
    """Remove the overlay env var. Returns its prior value (or None)."""
    return os.environ.pop(ENV_OVERLAY, None)


def _restore_env(name: str, prev: str | None) -> None:
    if prev is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = prev


# ──────────────────────────── Fast unit tests ────────────────────────────


def test_parse_artifact_emits_takeoff_without_overlay_by_default(tmp_path: Path) -> None:
    """parse_artifact emits structured.* and takeoff.* derived files, but
    does NOT spend time rendering QA overlays unless the env flag is set."""
    pytest.importorskip("fitz")
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "tiny_lvolt.pdf"
    _make_tiny_lvolt_pdf(pdf)

    prev = _unset_overlay_env()
    try:
        parser = OrbitBriefPdfParser()
        out = parser.parse_artifact(
            project_id="test_project",
            artifact_id="test_artifact",
            path=pdf,
        )
    finally:
        _restore_env(ENV_OVERLAY, prev)

    paths = {d.relative_path for d in out.derived_files}
    derived_stem = f"{pdf.stem}.derived"
    assert f"{derived_stem}/structured.json" in paths
    assert f"{derived_stem}/structured.md" in paths
    assert f"{derived_stem}/takeoff.json" in paths
    assert f"{derived_stem}/takeoff.md" in paths

    # QA overlay dir should not be populated when the env flag is off.
    qa_dir = pdf.parent / derived_stem / "qa_overlays"
    overlay_pngs = list(qa_dir.glob("*.png")) if qa_dir.exists() else []
    assert overlay_pngs == [], (
        f"QA overlays should not render without {ENV_OVERLAY}=1; found {overlay_pngs}"
    )

    # Takeoff failure path is silent on success.
    failures = [w for w in out.warnings if w.startswith("low_voltage_takeoff_failed")]
    assert failures == []


def test_parse_artifact_can_enable_overlay_with_env_flag(tmp_path: Path) -> None:
    """Setting PARSER_OS_WRITE_TAKEOFF_QA=1 makes parse_artifact render
    QA overlay PNGs for accepted-device pages."""
    pytest.importorskip("fitz")
    pytest.importorskip("PIL")
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "tiny_lvolt.pdf"
    _make_tiny_lvolt_pdf(pdf)

    prev = os.environ.get(ENV_OVERLAY)
    os.environ[ENV_OVERLAY] = "1"
    try:
        parser = OrbitBriefPdfParser()
        out = parser.parse_artifact(
            project_id="test_project",
            artifact_id="test_artifact",
            path=pdf,
        )
    finally:
        _restore_env(ENV_OVERLAY, prev)

    # Derived files still all present.
    paths = {d.relative_path for d in out.derived_files}
    derived_stem = f"{pdf.stem}.derived"
    assert f"{derived_stem}/takeoff.json" in paths
    assert f"{derived_stem}/takeoff.md" in paths

    # And at least one overlay PNG has been written for the accepted page.
    qa_dir = pdf.parent / derived_stem / "qa_overlays"
    assert qa_dir.exists(), (
        f"QA overlay dir should exist when {ENV_OVERLAY}=1, missing: {qa_dir}"
    )
    pngs = sorted(qa_dir.glob("*.png"))
    assert len(pngs) >= 1, (
        f"Expected >=1 overlay PNG when {ENV_OVERLAY}=1, got {pngs}"
    )

    failures = [w for w in out.warnings if w.startswith("low_voltage_takeoff_failed")]
    assert failures == []


# ──────────────────────────── Slow regression ────────────────────────────
#
# The full Marriott parse is real work (25 sheets through both the structured
# pipeline and the takeoff pipeline). It's gated behind RUN_SLOW_TESTS=1
# so it stays out of the normal fast loop. Run with:
#
#     RUN_SLOW_TESTS=1 pytest tests/test_orbitbrief_pdf_takeoff_derived_files.py
#


@pytest.mark.slow
@pytest.mark.skipif(
    not MARRIOTT_PDF_PATH.exists() or os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Marriott PDF missing or RUN_SLOW_TESTS!=1 (set RUN_SLOW_TESTS=1 to enable)",
)
def test_parse_artifact_emits_takeoff_alongside_structured_marriott() -> None:
    pytest.importorskip("fitz")
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    # Force overlays off even in the slow case — they're a separate
    # concern from the derived-file emission contract being verified here.
    prev = _unset_overlay_env()
    try:
        parser = OrbitBriefPdfParser()
        out = parser.parse_artifact(
            project_id="test_project",
            artifact_id="test_artifact",
            path=MARRIOTT_PDF_PATH,
        )
    finally:
        _restore_env(ENV_OVERLAY, prev)

    paths = {d.relative_path for d in out.derived_files}
    derived_stem = f"{MARRIOTT_PDF_PATH.stem}.derived"
    assert f"{derived_stem}/structured.json" in paths
    assert f"{derived_stem}/structured.md" in paths
    assert f"{derived_stem}/takeoff.json" in paths
    assert f"{derived_stem}/takeoff.md" in paths

    assert (MARRIOTT_PDF_PATH.parent / derived_stem / "takeoff.json").exists()
    assert (MARRIOTT_PDF_PATH.parent / derived_stem / "takeoff.md").exists()
    assert (MARRIOTT_PDF_PATH.parent / derived_stem / "structured.json").exists()
    assert (MARRIOTT_PDF_PATH.parent / derived_stem / "structured.md").exists()

    failures = [w for w in out.warnings if w.startswith("low_voltage_takeoff_failed")]
    assert failures == []
