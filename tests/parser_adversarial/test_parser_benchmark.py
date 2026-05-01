from __future__ import annotations

import json
from pathlib import Path

from app.eval.parser_metrics import ParserBenchmarkReport, parser_threshold_failures
from scripts.run_parser_benchmark import run_parser_benchmark


def test_parser_benchmark_report_writes_aggregate_metrics(tmp_path: Path) -> None:
    out = tmp_path / "parser_benchmark.json"
    report = run_parser_benchmark(out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "aggregate_metrics" in payload
    assert isinstance(payload["aggregate_metrics"], dict)
    assert report.aggregate_metrics["source_ref_coverage"] >= 1.0


def test_parser_benchmark_thresholds_fail_when_metrics_drop() -> None:
    report = ParserBenchmarkReport(
        aggregate_metrics={
            "atom_recall_by_type": 0.5,
            "source_ref_coverage": 0.8,
            "entity_key_accuracy": 0.7,
            "quantity_accuracy": 0.5,
            "authority_class_accuracy": 0.8,
            "review_flag_accuracy": 0.5,
            "parse_crash_rate": 0.2,
            "unsupported_feature_warnings": 3.0,
        },
        parser_reports=[],
        threshold_failures=[],
    )
    failures = parser_threshold_failures(report)
    assert failures
