from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _top_risk_packets(compile_payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    packets = compile_payload.get("packets", [])
    ranked = sorted(
        packets,
        key=lambda packet: (
            (packet.get("risk") or {}).get("review_priority", 5),
            -float((packet.get("risk") or {}).get("risk_score", 0.0)),
        ),
    )
    return ranked[:limit]


def _packet_summary_markdown(compile_payload: dict[str, Any]) -> str:
    lines: list[str] = ["# Packet Summary", ""]
    packets = compile_payload.get("packets", [])
    lines.append(f"- Packet count: {len(packets)}")
    lines.append("")
    lines.append("| Family | Anchor | Status | Severity | Risk Score | Reason |")
    lines.append("|---|---|---|---|---:|---|")
    for packet in _top_risk_packets(compile_payload, limit=10):
        risk = packet.get("risk") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(packet.get("family", "")),
                    str(packet.get("anchor_key", "")),
                    str(packet.get("status", "")),
                    str(risk.get("severity", "")),
                    str(risk.get("risk_score", "")),
                    str(packet.get("reason", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_demo_report(demo_dir: Path) -> tuple[Path, Path]:
    compile_payload = _load_json(demo_dir / "compile_result.json")
    trace_payload = _load_json(demo_dir / "trace.json")
    packetizer_payload = _load_json(demo_dir / "packetizer_benchmark.json")
    adversarial_payload = _load_json(demo_dir / "adversarial_report.json")
    parser_payload = _load_json(demo_dir / "parser_benchmark.json")

    packet_summary = _packet_summary_markdown(compile_payload)
    packet_summary_path = demo_dir / "packet_summary.md"
    packet_summary_path.write_text(packet_summary, encoding="utf-8")

    manifest = compile_payload.get("manifest") or {}
    trace = trace_payload or compile_payload.get("trace") or {}
    packet_family_counts = (trace.get("packet_family_counts") or {})
    parser_counts = (trace.get("parser_atom_counts") or {})
    warnings = compile_payload.get("warnings") or []
    packetizer_aggregate = packetizer_payload.get("aggregate_metrics") or {}
    parser_aggregate = parser_payload.get("aggregate_metrics") or {}
    adversarial_metrics = adversarial_payload.get("metrics") or {}
    compile_pass = int(adversarial_metrics.get("compile_pass_count", 0))
    total_scenarios = int(adversarial_metrics.get("total_scenarios", 0))
    adversarial_pass_rate = round((compile_pass / total_scenarios), 4) if total_scenarios else 0.0

    certificate_rows: list[str] = []
    for packet in _top_risk_packets(compile_payload, limit=5):
        cert = packet.get("certificate") or {}
        certificate_rows.append(
            f"- `{packet.get('id')}` {packet.get('family')} `{packet.get('anchor_key')}`: "
            f"{cert.get('existence_reason', packet.get('reason', ''))}"
        )

    top_risk_rows: list[str] = []
    for packet in _top_risk_packets(compile_payload, limit=5):
        risk = packet.get("risk") or {}
        top_risk_rows.append(
            f"- `{packet.get('id')}` {packet.get('family')} `{packet.get('anchor_key')}` "
            f"(severity={risk.get('severity')}, risk_score={risk.get('risk_score')}, priority={risk.get('review_priority')})"
        )

    recommended_fixes = packetizer_payload.get("recommended_next_fixes") or []
    if not recommended_fixes:
        recommended_fixes = ["No benchmark recommendations emitted."]

    report_lines = [
        "# Purtera MVP Demo Report",
        "",
        "## Compile Signatures",
        f"- compile_id: `{compile_payload.get('compile_id', '')}`",
        f"- input_signature: `{manifest.get('input_signature', '')}`",
        f"- output_signature: `{manifest.get('output_signature', '')}`",
        "",
        "## Trace And Counts",
        f"- total_duration_ms: {trace.get('total_duration_ms', '')}",
        f"- parser_atom_counts: `{json.dumps(parser_counts, sort_keys=True)}`",
        f"- packet_family_counts: `{json.dumps(packet_family_counts, sort_keys=True)}`",
        "",
        "## Top Risk Packets",
        *top_risk_rows,
        "",
        "## Packet Certificate Highlights",
        *(certificate_rows or ["- No packets found."]),
        "",
        "## Benchmark Metrics",
        f"- packetizer aggregate: `{json.dumps(packetizer_aggregate, sort_keys=True)}`",
        f"- parser aggregate: `{json.dumps(parser_aggregate, sort_keys=True)}`",
        "",
        "## Adversarial Lab",
        f"- scenarios: {total_scenarios}",
        f"- compile_pass_count: {compile_pass}",
        f"- pass_rate: {adversarial_pass_rate}",
        "",
        "## Warnings",
        *(f"- {row}" for row in warnings[:20]),
        "",
        "## Next Recommended Fixes",
        *(f"- {row}" for row in recommended_fixes),
        "",
    ]

    demo_report_path = demo_dir / "demo_report.md"
    demo_report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return packet_summary_path, demo_report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown summary reports for insane MVP demo outputs.")
    parser.add_argument("--demo-dir", type=Path, required=True, help="Demo output directory")
    args = parser.parse_args()
    args.demo_dir.mkdir(parents=True, exist_ok=True)
    packet_summary_path, demo_report_path = build_demo_report(args.demo_dir)
    print(
        json.dumps(
            {
                "packet_summary": str(packet_summary_path),
                "demo_report": str(demo_report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
