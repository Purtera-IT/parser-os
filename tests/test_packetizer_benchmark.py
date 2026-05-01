from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.eval.benchmark import BenchmarkReport, run_packetizer_benchmark, threshold_failures


def _write_gold(fixtures_dir: Path, *, scenario_id: str, project_dir: Path, family: str = "quantity_conflict", governing: str = "customer_current_authored") -> None:
    scenario = fixtures_dir / scenario_id
    scenario.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": scenario_id,
        "project_dir": str(project_dir),
        "expected_packets": [
            {
                "family": family,
                "anchor_key_contains": "site:west_wing",
                "must_contain_quantities": [91, 72],
                "expected_status": "needs_review",
                "forbidden_governing_authority": ["vendor_quote"],
            }
        ],
        "expected_governing": [
            {
                "family": "scope_exclusion",
                "anchor_key_contains": "site:west_wing",
                "governing_authority": governing,
            }
        ],
        "forbidden": [
            {"condition": "deleted_text_governs"},
            {"condition": "quoted_old_email_governs_current_conflict"},
        ],
    }
    (scenario / "gold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_benchmark_runs_on_demo_project() -> None:
    fixtures = Path("tests/fixtures/gold_scenarios")
    report = run_packetizer_benchmark(fixtures)
    assert report.scenario_count >= 1
    assert report.aggregate_metrics["compile_success_rate"] == 1.0


def test_benchmark_detects_missing_expected_packet(demo_project: Path, tmp_path: Path) -> None:
    fixtures = tmp_path / "gold"
    _write_gold(fixtures, scenario_id="missing_packet", project_dir=demo_project, family="non_existent_family")
    report = run_packetizer_benchmark(fixtures)
    scenario = report.scenario_results[0]
    assert scenario.metrics["packet_family_recall"] == 0.0


def test_benchmark_detects_bad_governing_authority(demo_project: Path, tmp_path: Path) -> None:
    fixtures = tmp_path / "gold"
    _write_gold(fixtures, scenario_id="bad_governing", project_dir=demo_project, governing="vendor_quote")
    report = run_packetizer_benchmark(fixtures)
    scenario = report.scenario_results[0]
    assert scenario.metrics["governing_accuracy"] == 0.0


def test_benchmark_report_serializes_json() -> None:
    report = BenchmarkReport(
        scenario_count=1,
        aggregate_metrics={"compile_success_rate": 1.0},
        scenario_results=[],
        failed_invariants=[],
        recommended_next_fixes=[],
    )
    payload = report.model_dump_json(indent=2)
    assert "compile_success_rate" in payload


def test_thresholds_fail_if_invalid_governance_exists() -> None:
    report = BenchmarkReport(
        scenario_count=1,
        aggregate_metrics={
            "packet_family_recall": 1.0,
            "governing_accuracy": 1.0,
            "contradiction_recall": 1.0,
            "invalid_governance_count": 1,
            "determinism_pass": True,
            "false_active_rate": 0.0,
            "compile_success_rate": 1.0,
        },
        scenario_results=[],
        failed_invariants=[],
        recommended_next_fixes=[],
    )
    failures = threshold_failures(report)
    assert any("invalid_governance_count" in failure for failure in failures)


def test_script_exits_non_zero_without_allow_fail(demo_project: Path, tmp_path: Path) -> None:
    fixtures = tmp_path / "gold"
    out = tmp_path / "report.json"
    _write_gold(fixtures, scenario_id="script_fail", project_dir=demo_project, family="non_existent_family")
    script = Path("scripts/run_packetizer_benchmark.py").resolve()
    proc_fail = subprocess.run(
        [sys.executable, str(script), "--fixtures", str(fixtures), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc_fail.returncode != 0

    proc_allow = subprocess.run(
        [sys.executable, str(script), "--fixtures", str(fixtures), "--out", str(out), "--allow-fail"],
        capture_output=True,
        text=True,
    )
    assert proc_allow.returncode == 0
