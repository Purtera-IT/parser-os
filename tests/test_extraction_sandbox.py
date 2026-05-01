from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.core.compiler import compile_project
from app.experiments.freeze import freeze_experiment_output
from app.experiments.sandbox import run_extraction_sandbox


def test_sandbox_report_generated(demo_project: Path) -> None:
    run, report = run_extraction_sandbox(
        project_dir=demo_project,
        extractor_name="semantic_linker",
        extractor_version="exp_test",
    )
    assert run.experiment_id
    assert report["experiment_run"]["experiment_id"] == run.experiment_id
    assert "delta" in report


def test_baseline_compile_unchanged(demo_project: Path) -> None:
    baseline_before = compile_project(demo_project, project_id="sandbox_stable", allow_errors=True, allow_unverified_receipts=True)
    _run, _report = run_extraction_sandbox(
        project_dir=demo_project,
        extractor_name="semantic_linker",
        extractor_version="exp_test",
    )
    baseline_after = compile_project(demo_project, project_id="sandbox_stable", allow_errors=True, allow_unverified_receipts=True)
    assert baseline_before.manifest is not None and baseline_after.manifest is not None
    assert baseline_before.manifest.output_signature == baseline_after.manifest.output_signature


def test_hypothetical_packet_delta_reported(demo_project: Path) -> None:
    run, _report = run_extraction_sandbox(
        project_dir=demo_project,
        extractor_name="llm_candidate_extractor",
        extractor_version="exp_test",
    )
    delta = run.delta_vs_baseline
    assert delta.new_candidates >= 0
    assert delta.new_packets_if_accepted >= 0
    assert delta.changed_packets_if_accepted >= 0


def test_freeze_requires_approve(demo_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "experiment.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_extraction_experiment.py",
            "--project",
            str(demo_project),
            "--extractor",
            "semantic_linker",
            "--out",
            str(out),
        ],
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/freeze_extractor_output.py",
            "--experiment",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_frozen_change_creates_regression_fixture_metadata(demo_project: Path, tmp_path: Path) -> None:
    _run, report = run_extraction_sandbox(
        project_dir=demo_project,
        extractor_name="semantic_linker",
        extractor_version="exp_test",
    )
    exp_path = tmp_path / "experiment.json"
    exp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    freeze_dir = tmp_path / "freeze"
    result = freeze_experiment_output(experiment_path=exp_path, approve=True, out_dir=freeze_dir)
    assert result.status == "applied"
    assert (freeze_dir / "freeze_metadata.json").exists()
    assert (freeze_dir / "gold_regression.json").exists()


def test_normal_compile_remains_deterministic_after_freeze(demo_project: Path, tmp_path: Path) -> None:
    _run, report = run_extraction_sandbox(
        project_dir=demo_project,
        extractor_name="weak_supervision_rules",
        extractor_version="exp_test",
    )
    exp_path = tmp_path / "experiment.json"
    exp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _ = freeze_experiment_output(experiment_path=exp_path, approve=True, out_dir=tmp_path / "freeze")

    first = compile_project(demo_project, project_id="sandbox_deterministic", allow_errors=True, allow_unverified_receipts=True)
    second = compile_project(demo_project, project_id="sandbox_deterministic", allow_errors=True, allow_unverified_receipts=True)
    assert first.manifest is not None and second.manifest is not None
    assert first.manifest.output_signature == second.manifest.output_signature
