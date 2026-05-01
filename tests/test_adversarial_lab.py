from __future__ import annotations

import json
from pathlib import Path

from app.testing.scenarios import generate_scenario
from scripts.run_adversarial_lab import run_lab


def test_generate_five_scenarios_and_all_pass_invariants(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = run_lab(count=5, out=report_path, seed=2000)
    assert report["metrics"]["total_scenarios"] == 5
    assert report["metrics"]["compile_pass_count"] == 5
    assert report["metrics"]["hard_error_count"] == 0
    assert report["metrics"]["determinism_failures"] == 0
    assert all(scenario["status"] == "pass" for scenario in report["scenarios"])


def test_mutation_names_appear_in_metadata(tmp_path: Path) -> None:
    scenario_dir = generate_scenario(3001, output_root=tmp_path)
    metadata = json.loads((scenario_dir / "scenario_metadata.json").read_text(encoding="utf-8"))
    assert "mutations" in metadata
    mutation_names = metadata["mutations"]
    assert all(mutation_names.get(key) for key in ("spreadsheet", "email", "docx", "transcript", "quote"))


def test_report_contains_metrics_and_family_coverage(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = run_lab(count=5, out=report_path, seed=2100)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert "metrics" in loaded
    assert "expected_packet_recall_by_family" in loaded["metrics"]
    coverage = loaded["mutation_family_coverage"]
    assert all(family in coverage and coverage[family] for family in ("spreadsheet", "email", "docx", "transcript", "quote"))


def test_deterministic_seed_generates_same_fixture_metadata(tmp_path: Path) -> None:
    scenario_one = generate_scenario(4001, output_root=tmp_path / "a")
    scenario_two = generate_scenario(4001, output_root=tmp_path / "b")
    meta_one = json.loads((scenario_one / "scenario_metadata.json").read_text(encoding="utf-8"))
    meta_two = json.loads((scenario_two / "scenario_metadata.json").read_text(encoding="utf-8"))
    assert meta_one["seed"] == meta_two["seed"]
    assert meta_one["base_scenario"] == meta_two["base_scenario"]
    assert meta_one["mutations"] == meta_two["mutations"]
