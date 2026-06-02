"""v57 parse-coverage gate: an artifact that a parser matches and runs
on WITHOUT error, yet yields 0 atoms, must surface a hard warning rather
than vanish silently. Silent data loss (a scanned/image-only PDF, an
empty sheet, an unextractable layout) is the most dangerous failure mode
because the reviewer never learns an input contributed nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.compiler import compile_project


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    # Keep the compile hermetic + fast — the gate is deterministic and
    # lives in parse_artifacts, well before any LLM stage.
    for var in (
        "SOWSMITH_MULTI_ENTITY_DISABLE",
        "SOWSMITH_TYPED_CLASSIFIER_DISABLE",
        "SOWSMITH_RETRIEVAL_DISABLE",
        "SOWSMITH_CONTRADICTION_DISABLE",
    ):
        monkeypatch.setenv(var, "1")


def test_zero_atom_artifact_emits_coverage_warning(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    # A real artifact that parses to atoms.
    (project_dir / "kickoff.txt").write_text(
        "[00:00:01] Jane: Install 5 cameras at Main Campus.\nDecisions:\n- Proceed.\n",
        encoding="utf-8",
    )
    # A header-only CSV: the xlsx parser MATCHES (by extension) and runs
    # cleanly, but there are no data rows so it yields 0 atoms.
    (project_dir / "asset_inventory.csv").write_text(
        "asset_id,serial,ip_address\n", encoding="utf-8"
    )

    result = compile_project(
        project_dir, project_id="cov", allow_unverified_receipts=True
    )

    coverage_warnings = [
        w
        for w in result.warnings
        if "asset_inventory.csv" in w and "yielded 0 atoms" in w
    ]
    assert coverage_warnings, (
        "expected a parse-coverage warning naming the 0-atom artifact; "
        f"got warnings: {result.warnings}"
    )


def test_healthy_artifact_does_not_trip_gate(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "kickoff.txt").write_text(
        "[00:00:01] Jane: Install 5 cameras at Main Campus.\nDecisions:\n- Proceed.\n",
        encoding="utf-8",
    )

    result = compile_project(
        project_dir, project_id="cov_ok", allow_unverified_receipts=True
    )

    assert not [w for w in result.warnings if "yielded 0 atoms" in w]
