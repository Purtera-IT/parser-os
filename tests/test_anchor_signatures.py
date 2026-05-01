from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.schemas import PacketFamily


def test_anchor_signature_hash_deterministic_across_compiles(demo_project: Path) -> None:
    first = compile_project(
        demo_project, project_id="demo_project", allow_unverified_receipts=True, use_cache=False
    )
    second = compile_project(
        demo_project, project_id="demo_project", allow_unverified_receipts=True, use_cache=False
    )
    first_hashes = sorted(packet.anchor_signature.hash for packet in first.packets if packet.anchor_signature is not None)
    second_hashes = sorted(packet.anchor_signature.hash for packet in second.packets if packet.anchor_signature is not None)
    assert first_hashes == second_hashes


def test_west_wing_exclusion_is_single_canonical_packet(demo_project: Path) -> None:
    result = compile_project(
        demo_project, project_id="demo_project", allow_unverified_receipts=True, use_cache=False
    )
    west = [
        packet
        for packet in result.packets
        if packet.family == PacketFamily.scope_exclusion
        and packet.anchor_signature is not None
        and "site:west_wing" in packet.anchor_signature.canonical_key
    ]
    assert len(west) == 1


def test_ip_camera_vendor_mismatch_single_packet(demo_project: Path) -> None:
    result = compile_project(
        demo_project, project_id="demo_project", allow_unverified_receipts=True, use_cache=False
    )
    mismatches = [
        packet
        for packet in result.packets
        if packet.family == PacketFamily.vendor_mismatch
        and packet.anchor_signature is not None
        and packet.anchor_signature.canonical_key == "device:ip_camera"
    ]
    assert len(mismatches) == 1


def test_ip_camera_quantity_conflict_packets(demo_project: Path) -> None:
    """Demo fixture should emit at least one quantity_conflict on ip_camera (roster vs vendor)."""
    result = compile_project(
        demo_project, project_id="demo_project", allow_unverified_receipts=True, use_cache=False
    )
    conflicts = [
        packet
        for packet in result.packets
        if packet.family == PacketFamily.quantity_conflict
        and packet.anchor_signature is not None
        and packet.anchor_signature.canonical_key == "device:ip_camera"
    ]
    assert conflicts
    assert any("72" in p.reason for p in conflicts)
