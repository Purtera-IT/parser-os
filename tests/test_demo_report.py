from __future__ import annotations

import json
from pathlib import Path

from scripts.build_demo_report import build_demo_report


def test_build_demo_report_works_with_fake_json(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "compile_result.json").write_text(
        json.dumps(
            {
                "compile_id": "cmp_demo",
                "manifest": {"input_signature": "in_sig", "output_signature": "out_sig"},
                "trace": {"parser_atom_counts": {"xlsx": 4}, "packet_family_counts": {"vendor_mismatch": 1}},
                "packets": [
                    {
                        "id": "pkt_1",
                        "family": "vendor_mismatch",
                        "anchor_key": "device:ip_camera",
                        "status": "needs_review",
                        "reason": "Quantity mismatch",
                        "risk": {"severity": "high", "risk_score": 0.9, "review_priority": 1},
                        "certificate": {"existence_reason": "Created because mismatch exists."},
                    }
                ],
                "warnings": ["WARNING: demo warning"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (demo_dir / "trace.json").write_text(
        json.dumps(
            {
                "total_duration_ms": 123.4,
                "parser_atom_counts": {"xlsx": 4},
                "packet_family_counts": {"vendor_mismatch": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (demo_dir / "packetizer_benchmark.json").write_text(
        json.dumps(
            {
                "aggregate_metrics": {"packet_family_recall": 1.0},
                "recommended_next_fixes": ["Keep improving authority scoring."],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (demo_dir / "adversarial_report.json").write_text(
        json.dumps({"metrics": {"total_scenarios": 10, "compile_pass_count": 9}}, indent=2),
        encoding="utf-8",
    )
    (demo_dir / "parser_benchmark.json").write_text(
        json.dumps({"aggregate_metrics": {"source_ref_coverage": 1.0}}, indent=2),
        encoding="utf-8",
    )

    packet_summary_path, demo_report_path = build_demo_report(demo_dir)
    assert packet_summary_path.exists()
    assert demo_report_path.exists()


def test_demo_report_contains_compile_id_packet_families_and_metrics(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "compile_result.json").write_text(
        json.dumps(
            {
                "compile_id": "cmp_demo_2",
                "manifest": {"input_signature": "in_sig_2", "output_signature": "out_sig_2"},
                "trace": {"parser_atom_counts": {"email": 2}, "packet_family_counts": {"scope_exclusion": 2}},
                "packets": [
                    {
                        "id": "pkt_2",
                        "family": "scope_exclusion",
                        "anchor_key": "site:west_wing",
                        "status": "needs_review",
                        "reason": "Exclusion conflict",
                        "risk": {"severity": "critical", "risk_score": 1.0, "review_priority": 1},
                        "certificate": {"existence_reason": "Created because exclusion conflict exists."},
                    }
                ],
                "warnings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (demo_dir / "trace.json").write_text(
        json.dumps(
            {
                "total_duration_ms": 222.2,
                "parser_atom_counts": {"email": 2},
                "packet_family_counts": {"scope_exclusion": 2},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (demo_dir / "packetizer_benchmark.json").write_text(
        json.dumps({"aggregate_metrics": {"governing_accuracy": 0.95}, "recommended_next_fixes": []}, indent=2),
        encoding="utf-8",
    )
    (demo_dir / "adversarial_report.json").write_text(
        json.dumps({"metrics": {"total_scenarios": 25, "compile_pass_count": 25}}, indent=2),
        encoding="utf-8",
    )
    (demo_dir / "parser_benchmark.json").write_text(
        json.dumps({"aggregate_metrics": {"authority_class_accuracy": 0.97}}, indent=2),
        encoding="utf-8",
    )

    _, demo_report_path = build_demo_report(demo_dir)
    report = demo_report_path.read_text(encoding="utf-8")
    assert "cmp_demo_2" in report
    assert "scope_exclusion" in report
    assert "governing_accuracy" in report
