from __future__ import annotations

import json
from pathlib import Path

from scripts.build_final_mvp_report import build_final_mvp_report, calculate_readiness


def _write_fake_inputs(base: Path, *, pytest_passed: bool = True, packet_family_recall: float = 0.97) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "compile_result.json").write_text(
        json.dumps(
            {
                "project_id": "demo_project",
                "compile_id": "cmp_demo",
                "compiler_version": "0.2.0",
                "atoms": [
                    {"id": "a1", "receipts": [{"replay_status": "verified"}]},
                    {"id": "a2", "receipts": [{"replay_status": "verified"}]},
                ],
                "packets": [
                    {
                        "id": "p1",
                        "family": "vendor_mismatch",
                        "anchor_key": "device:ip_camera",
                        "risk": {"review_priority": 1, "risk_score": 0.9, "severity": "high"},
                        "certificate": {"existence_reason": "exists"},
                    }
                ],
                "manifest": {
                    "input_signature": "in_sig",
                    "output_signature": "out_sig",
                    "domain_pack_id": "security_camera",
                    "domain_pack_version": "1.0.0",
                },
                "trace": {"total_duration_ms": 1200, "packet_family_counts": {"vendor_mismatch": 1}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (base / "trace.json").write_text(
        json.dumps({"total_duration_ms": 1200, "packet_family_counts": {"vendor_mismatch": 1}}, indent=2),
        encoding="utf-8",
    )
    (base / "coverage.json").write_text(
        json.dumps({"overall_coverage_rate": 0.8, "artifact_reports": []}, indent=2),
        encoding="utf-8",
    )
    (base / "packetizer_benchmark.json").write_text(
        json.dumps(
            {
                "aggregate_metrics": {
                    "packet_family_recall": packet_family_recall,
                    "governing_accuracy": 0.97,
                    "false_active_rate": 0.0,
                    "invalid_governance_count": 0,
                },
                "recommended_next_fixes": ["none"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (base / "parser_benchmark.json").write_text(
        json.dumps({"aggregate_metrics": {"source_ref_coverage": 1.0}}, indent=2),
        encoding="utf-8",
    )
    (base / "adversarial_report.json").write_text(
        json.dumps({"metrics": {"total_scenarios": 50, "compile_pass_count": 50}}, indent=2),
        encoding="utf-8",
    )
    (base / "domain_cert_security_camera.json").write_text(
        json.dumps({"pack_id": "security_camera", "pack_version": "1.0.0", "passed": True, "recommendations": []}, indent=2),
        encoding="utf-8",
    )
    (base / "experiment_semantic_linker.json").write_text(
        json.dumps({"experiment_run": {"delta_vs_baseline": {"new_packets_if_accepted": 1}}}, indent=2),
        encoding="utf-8",
    )
    (base / "experiment_llm_candidate.json").write_text(
        json.dumps({"experiment_run": {"delta_vs_baseline": {"new_packets_if_accepted": 0}}}, indent=2),
        encoding="utf-8",
    )
    (base / "active_learning_queue.json").write_text(
        json.dumps({"items": [{"item_id": "q1"}], "metadata": {}}, indent=2),
        encoding="utf-8",
    )
    (base / "perf_100_sites.json").write_text(
        json.dumps({"total_duration_ms": 25000.0, "sites": 100}, indent=2),
        encoding="utf-8",
    )
    (base / "pytest_summary.json").write_text(
        json.dumps({"passed": pytest_passed, "command": "python -m pytest -q"}, indent=2),
        encoding="utf-8",
    )
    (base / "source_replay_summary.json").write_text(
        json.dumps({"receipt_failed_count": 0}, indent=2),
        encoding="utf-8",
    )
    (base / "real_data_harness_summary.json").write_text(
        json.dumps({"passed": True, "skeleton_checked": True}, indent=2),
        encoding="utf-8",
    )


def test_final_report_builder_works_with_fake_inputs(tmp_path: Path) -> None:
    out = tmp_path / "final"
    _write_fake_inputs(out)
    json_path, md_path = build_final_mvp_report(out)
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["compile"]["compile_id"] == "cmp_demo"


def test_readiness_returns_yes_when_thresholds_pass() -> None:
    report = {
        "pytest": {"passed": True},
        "source_replay": {"receipt_failed_count": 0},
        "packetizer_benchmark": {
            "aggregate_metrics": {
                "invalid_governance_count": 0,
                "packet_family_recall": 0.99,
                "governing_accuracy": 0.97,
                "false_active_rate": 0.0,
            }
        },
        "adversarial_lab": {"compile_success_rate": 0.99},
        "parser_benchmark": {"aggregate_metrics": {"source_ref_coverage": 1.0}},
        "domain_certification": {"passed": True},
        "compile": {"total_duration_ms": 3000.0},
        "performance": {"total_duration_ms": 20000.0},
    }
    ready, thresholds = calculate_readiness(report)
    assert ready is True
    assert all(row["passed"] for row in thresholds)


def test_readiness_returns_no_when_critical_threshold_fails() -> None:
    report = {
        "pytest": {"passed": True},
        "source_replay": {"receipt_failed_count": 1},
        "packetizer_benchmark": {
            "aggregate_metrics": {
                "invalid_governance_count": 0,
                "packet_family_recall": 0.99,
                "governing_accuracy": 0.97,
                "false_active_rate": 0.0,
            }
        },
        "adversarial_lab": {"compile_success_rate": 0.99},
        "parser_benchmark": {"aggregate_metrics": {"source_ref_coverage": 1.0}},
        "domain_certification": {"passed": True},
        "compile": {"total_duration_ms": 3000.0},
        "performance": {"total_duration_ms": 20000.0},
    }
    ready, thresholds = calculate_readiness(report)
    assert ready is False
    assert any((not row["passed"]) and row["name"] == "receipt_failure_count" for row in thresholds)


def test_markdown_report_contains_key_sections(tmp_path: Path) -> None:
    out = tmp_path / "final"
    _write_fake_inputs(out, pytest_passed=False, packet_family_recall=0.5)
    _, md_path = build_final_mvp_report(out)
    body = md_path.read_text(encoding="utf-8")
    assert "Top 10 Risk Packets" in body
    assert "Readiness Thresholds" in body
    assert "Ready for OrbitBrief v0?" in body
