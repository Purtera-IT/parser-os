from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.core.compiler import compile_project
from app.core.risk import packet_pm_sort_key
from app.core.schemas import PacketFamily


def test_vendor_mismatch_ip_camera_cost_exposure(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    vendor_packets = [packet for packet in result.packets if packet.family == PacketFamily.vendor_mismatch]
    assert vendor_packets
    assert any(packet.risk and packet.risk.estimated_cost_exposure == 5700.0 for packet in vendor_packets)


def test_scope_exclusion_has_high_or_critical_severity(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    scope_packets = [packet for packet in result.packets if packet.family == PacketFamily.scope_exclusion]
    assert scope_packets
    assert all(packet.risk and packet.risk.severity in {"high", "critical"} for packet in scope_packets)


def test_site_access_packet_has_failed_dispatch_exposure(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    site_access = [packet for packet in result.packets if packet.family == PacketFamily.site_access]
    assert site_access
    assert all(packet.risk and packet.risk.estimated_cost_exposure == 400.0 for packet in site_access)


def test_risk_scores_bounded_and_queue_tiers_sorted(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert all(packet.risk is not None for packet in result.packets)
    assert all(0.0 <= packet.risk.risk_score <= 1.0 for packet in result.packets if packet.risk)
    triage_sorted = sorted(
        result.packets,
        key=lambda p: packet_pm_sort_key(p) if p.risk is not None else (99, 99, 0.0, p.anchor_key, p.id),
    )
    tiers = [packet.risk.queue_tier for packet in triage_sorted if packet.risk]
    assert tiers == sorted(tiers)


def test_inspect_script_prints_severity(demo_project: Path, tmp_path: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    output_path = tmp_path / "compiled.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "inspect_packets.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(output_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = proc.stdout.lower()
    assert "severity" in stdout
    assert any(level in stdout for level in ("critical", "high", "medium", "low"))


def test_vendor_mismatch_risk_reason_flag_present(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    vendor_packet = next(packet for packet in result.packets if packet.family == PacketFamily.vendor_mismatch)
    assert vendor_packet.risk is not None
    assert any("vendor_scope_quantity_mismatch" in reason for reason in vendor_packet.risk.risk_reasons)


def test_packet_has_certificate_and_risk(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert all(packet.certificate is not None and packet.risk is not None for packet in result.packets)
