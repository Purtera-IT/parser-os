"""Integration test: parse_artifact returns takeoff.json + takeoff.md in derived_files."""
from __future__ import annotations

from pathlib import Path

import pytest

PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "real_data_cases"
    / "LOWVOLT_002_MARRIOTT_ATLANTA_T"
    / "artifacts"
    / "2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
)


def test_parse_artifact_emits_takeoff_alongside_structured() -> None:
    if not PDF_PATH.exists():
        pytest.skip(f"Marriott source PDF not available: {PDF_PATH}")
    pytest.importorskip("fitz")
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    parser = OrbitBriefPdfParser()
    out = parser.parse_artifact(
        project_id="test_project",
        artifact_id="test_artifact",
        path=PDF_PATH,
    )

    paths = {d.relative_path for d in out.derived_files}
    derived_stem = f"{PDF_PATH.stem}.derived"
    assert f"{derived_stem}/structured.json" in paths
    assert f"{derived_stem}/structured.md" in paths
    assert f"{derived_stem}/takeoff.json" in paths
    assert f"{derived_stem}/takeoff.md" in paths

    # And the actual files were written.
    assert (PDF_PATH.parent / derived_stem / "takeoff.json").exists()
    assert (PDF_PATH.parent / derived_stem / "takeoff.md").exists()
    assert (PDF_PATH.parent / derived_stem / "structured.json").exists()
    assert (PDF_PATH.parent / derived_stem / "structured.md").exists()

    # The takeoff failure path is silent on success.
    failures = [w for w in out.warnings if w.startswith("low_voltage_takeoff_failed")]
    assert failures == []
