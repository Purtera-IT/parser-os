from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _top_risk_packets(compile_payload: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    packets = [row for row in (compile_payload.get("packets") or []) if isinstance(row, dict)]
    ranked = sorted(
        packets,
        key=lambda packet: (
            _safe_int((packet.get("risk") or {}).get("review_priority"), 5),
            -_safe_float((packet.get("risk") or {}).get("risk_score"), 0.0),
            str(packet.get("id", "")),
        ),
    )
    return ranked[:limit]


def calculate_readiness(report: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    threshold_rows: list[dict[str, Any]] = []

    checks = [
        ("pytest_passed", bool(report.get("pytest", {}).get("passed")), True),
        ("receipt_failure_count", _safe_int(report.get("source_replay", {}).get("receipt_failed_count")) == 0, True),
        (
            "invalid_governance_count",
            _safe_int(report.get("packetizer_benchmark", {}).get("aggregate_metrics", {}).get("invalid_governance_count")) == 0,
            True,
        ),
        (
            "packet_family_recall",
            _safe_float(report.get("packetizer_benchmark", {}).get("aggregate_metrics", {}).get("packet_family_recall")) >= 0.95,
            True,
        ),
        (
            "governing_accuracy",
            _safe_float(report.get("packetizer_benchmark", {}).get("aggregate_metrics", {}).get("governing_accuracy")) >= 0.95,
            True,
        ),
        (
            "false_active_rate",
            _safe_float(report.get("packetizer_benchmark", {}).get("aggregate_metrics", {}).get("false_active_rate")) == 0.0,
            True,
        ),
        (
            "adversarial_compile_success_rate",
            _safe_float(report.get("adversarial_lab", {}).get("compile_success_rate")) >= 0.98,
            True,
        ),
        (
            "parser_source_ref_coverage",
            _safe_float(report.get("parser_benchmark", {}).get("aggregate_metrics", {}).get("source_ref_coverage")) >= 1.0,
            True,
        ),
        ("domain_certification_passed", bool(report.get("domain_certification", {}).get("passed")), True),
        ("demo_compile_under_5s", _safe_float(report.get("compile", {}).get("total_duration_ms")) <= 5000.0, True),
        ("perf_100_sites_under_30s", _safe_float(report.get("performance", {}).get("total_duration_ms")) <= 30000.0, True),
    ]
    for name, passed, critical in checks:
        threshold_rows.append({"name": name, "passed": bool(passed), "critical": critical})
    ready = all(row["passed"] for row in threshold_rows if row["critical"])
    return ready, threshold_rows


def build_final_mvp_report(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    compile_payload = _load_json(output_dir / "compile_result.json")
    trace_payload = _load_json(output_dir / "trace.json")
    coverage_payload = _load_json(output_dir / "coverage.json")
    parser_benchmark_payload = _load_json(output_dir / "parser_benchmark.json")
    packetizer_benchmark_payload = _load_json(output_dir / "packetizer_benchmark.json")
    adversarial_payload = _load_json(output_dir / "adversarial_report.json")
    domain_cert_payload = _load_json(output_dir / "domain_cert_security_camera.json")
    experiment_semantic_payload = _load_json(output_dir / "experiment_semantic_linker.json")
    experiment_llm_payload = _load_json(output_dir / "experiment_llm_candidate.json")
    queue_payload = _load_json(output_dir / "active_learning_queue.json")
    perf_payload = _load_json(output_dir / "perf_100_sites.json")
    pytest_payload = _load_json(output_dir / "pytest_summary.json")
    source_replay_payload = _load_json(output_dir / "source_replay_summary.json")
    real_data_payload = _load_json(output_dir / "real_data_harness_summary.json")

    manifest = compile_payload.get("manifest") if isinstance(compile_payload.get("manifest"), dict) else {}
    trace = trace_payload or (compile_payload.get("trace") if isinstance(compile_payload.get("trace"), dict) else {})

    packets = [row for row in (compile_payload.get("packets") or []) if isinstance(row, dict)]
    atoms = [row for row in (compile_payload.get("atoms") or []) if isinstance(row, dict)]
    receipt_total = 0
    receipt_verified = 0
    receipt_failed = 0
    for atom in atoms:
        for receipt in atom.get("receipts") or []:
            if not isinstance(receipt, dict):
                continue
            receipt_total += 1
            status = str(receipt.get("replay_status"))
            if status == "verified":
                receipt_verified += 1
            if status == "failed":
                receipt_failed += 1
    receipt_rate = round(receipt_verified / receipt_total, 6) if receipt_total else 0.0

    certificate_coverage = round(
        len([packet for packet in packets if isinstance(packet.get("certificate"), dict)]) / len(packets),
        6,
    ) if packets else 0.0

    adversarial_metrics = adversarial_payload.get("metrics") if isinstance(adversarial_payload.get("metrics"), dict) else {}
    adversarial_total = _safe_int(adversarial_metrics.get("total_scenarios"))
    adversarial_pass = _safe_int(adversarial_metrics.get("compile_pass_count"))
    adversarial_rate = round(adversarial_pass / adversarial_total, 6) if adversarial_total else 0.0

    report_payload: dict[str, Any] = {
        "build_timestamp": _now_iso(),
        "compiler_version": str(compile_payload.get("compiler_version") or "unknown"),
        "domain_pack_versions": {
            "compile": {
                "id": manifest.get("domain_pack_id"),
                "version": manifest.get("domain_pack_version"),
            },
            "security_camera": {
                "id": domain_cert_payload.get("pack_id"),
                "version": domain_cert_payload.get("pack_version"),
            },
        },
        "pytest": pytest_payload,
        "compile": {
            "compile_id": compile_payload.get("compile_id"),
            "input_signature": manifest.get("input_signature"),
            "output_signature": manifest.get("output_signature"),
            "total_duration_ms": trace.get("total_duration_ms"),
            "packet_family_counts": trace.get("packet_family_counts", {}),
        },
        "top_risk_packets": _top_risk_packets(compile_payload, limit=10),
        "certificate_coverage": {
            "packet_certificate_coverage": certificate_coverage,
            "packet_count": len(packets),
        },
        "source_replay": {
            "receipt_total": receipt_total,
            "receipt_verified_count": receipt_verified,
            "receipt_failed_count": receipt_failed,
            "receipt_verification_rate": receipt_rate,
            **source_replay_payload,
        },
        "coverage": coverage_payload,
        "parser_benchmark": parser_benchmark_payload,
        "packetizer_benchmark": packetizer_benchmark_payload,
        "adversarial_lab": {
            **adversarial_payload,
            "compile_success_rate": adversarial_rate,
        },
        "domain_certification": domain_cert_payload,
        "experimental_extractors": {
            "semantic_linker": experiment_semantic_payload.get("experiment_run", {}),
            "llm_candidate_extractor": experiment_llm_payload.get("experiment_run", {}),
        },
        "active_learning_queue": {
            "item_count": len(queue_payload.get("items", [])) if isinstance(queue_payload.get("items"), list) else 0,
            "metadata": queue_payload.get("metadata", {}),
        },
        "performance": perf_payload,
        "real_data_harness": real_data_payload,
        "remaining_known_limitations": [
            "LLM candidate extractor remains sandbox-only and does not affect production packets.",
            "Domain-pack promotion and freeze artifacts still require explicit human review workflows.",
            "Coverage diagnostics rely on available source artifacts for segment regeneration.",
        ],
    }

    ready, threshold_rows = calculate_readiness(report_payload)
    report_payload["readiness"] = {
        "ready_for_orbitbrief_v0": ready,
        "label": "YES" if ready else "NO",
        "thresholds": threshold_rows,
        "failed_thresholds": [row["name"] for row in threshold_rows if not row["passed"]],
    }
    if not ready:
        report_payload["recommended_next_fixes"] = list(
            sorted(
                set(
                    list(packetizer_benchmark_payload.get("recommended_next_fixes") or [])
                    + list(domain_cert_payload.get("recommendations") or [])
                    + [f"Fix threshold: {name}" for name in report_payload["readiness"]["failed_thresholds"]]
                )
            )
        )
    else:
        report_payload["recommended_next_fixes"] = []

    json_path = output_dir / "final_mvp_report.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Purtera Final MVP Report v2",
        "",
        "## Build Metadata",
        f"- Build timestamp: `{report_payload['build_timestamp']}`",
        f"- Compiler version: `{report_payload['compiler_version']}`",
        f"- Compile id: `{report_payload['compile']['compile_id']}`",
        "",
        "## Test Pass/Fail Summary",
        f"- Pytest passed: `{report_payload.get('pytest', {}).get('passed')}`",
        f"- Pytest command: `{report_payload.get('pytest', {}).get('command', 'n/a')}`",
        "",
        "## Compile Signatures",
        f"- Input signature: `{report_payload['compile']['input_signature']}`",
        f"- Output signature: `{report_payload['compile']['output_signature']}`",
        "",
        "## Packet Family Counts",
        f"- `{json.dumps(report_payload['compile'].get('packet_family_counts') or {}, sort_keys=True)}`",
        "",
        "## Top 10 Risk Packets",
    ]
    for packet in report_payload["top_risk_packets"]:
        risk = packet.get("risk") or {}
        md_lines.append(
            f"- `{packet.get('id')}` {packet.get('family')} `{packet.get('anchor_key')}` "
            f"(severity={risk.get('severity')}, risk_score={risk.get('risk_score')}, priority={risk.get('review_priority')})"
        )
    md_lines.extend(
        [
            "",
            "## Certificate Coverage",
            f"- Packet certificate coverage: `{report_payload['certificate_coverage']['packet_certificate_coverage']}`",
            "",
            "## Receipt Verification Rate",
            f"- Receipt verification rate: `{report_payload['source_replay']['receipt_verification_rate']}`",
            f"- Receipt failed count: `{report_payload['source_replay']['receipt_failed_count']}`",
            "",
            "## Parser Benchmark Metrics",
            f"- `{json.dumps((parser_benchmark_payload.get('aggregate_metrics') or {}), sort_keys=True)}`",
            "",
            "## Packetizer Benchmark Metrics",
            f"- `{json.dumps((packetizer_benchmark_payload.get('aggregate_metrics') or {}), sort_keys=True)}`",
            "",
            "## Adversarial Pass Rate",
            f"- Compile success rate: `{adversarial_rate}`",
            "",
            "## Domain Certification Status",
            f"- Passed: `{domain_cert_payload.get('passed')}`",
            f"- Pack: `{domain_cert_payload.get('pack_id')}` version `{domain_cert_payload.get('pack_version')}`",
            "",
            "## Experimental Extractor Deltas",
            f"- Semantic linker delta: `{json.dumps((report_payload['experimental_extractors']['semantic_linker'] or {}).get('delta_vs_baseline', {}), sort_keys=True)}`",
            f"- Fake LLM delta: `{json.dumps((report_payload['experimental_extractors']['llm_candidate_extractor'] or {}).get('delta_vs_baseline', {}), sort_keys=True)}`",
            "",
            "## Active Learning Queue Summary",
            f"- Queue item count: `{report_payload['active_learning_queue']['item_count']}`",
            "",
            "## Performance Metrics",
            f"- `{json.dumps(perf_payload, sort_keys=True)}`",
            "",
            "## Remaining Known Limitations",
        ]
    )
    for row in report_payload["remaining_known_limitations"]:
        md_lines.append(f"- {row}")
    md_lines.extend(
        [
            "",
            "## Readiness Thresholds",
            *[
                f"- {'PASS' if row['passed'] else 'FAIL'} `{row['name']}`"
                for row in report_payload["readiness"]["thresholds"]
            ],
            "",
            "## Ready for OrbitBrief v0?",
            f"- **{'YES' if ready else 'NO'}**",
        ]
    )
    if not ready:
        md_lines.extend(["", "### Failed Thresholds"])
        for row in report_payload["readiness"]["failed_thresholds"]:
            md_lines.append(f"- {row}")
        md_lines.extend(["", "### Recommended Next Fixes"])
        for row in report_payload["recommended_next_fixes"]:
            md_lines.append(f"- {row}")

    md_path = output_dir / "final_mvp_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final MVP gauntlet report from output artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Gauntlet output directory path")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = build_final_mvp_report(args.out_dir)
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path)}))


if __name__ == "__main__":
    main()
