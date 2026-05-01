from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from app.core.compiler import compile_project
from app.core.diffing import diff_compile_results
from app.core.schemas import PacketFamily


def _compile(project_dir: Path):
    return compile_project(project_dir=project_dir, project_id="demo_project", allow_unverified_receipts=True)


def _set_vendor_quantity(project_dir: Path, qty: int) -> None:
    quote_path = project_dir / "vendor_quote.xlsx"
    workbook = load_workbook(quote_path)
    sheet = workbook.active
    sheet["C2"] = qty
    workbook.save(quote_path)
    workbook.close()


def test_changing_vendor_quote_resolves_vendor_mismatch(demo_project: Path) -> None:
    before = _compile(demo_project)
    before_vendor = [packet for packet in before.packets if packet.family == PacketFamily.vendor_mismatch]
    assert before_vendor

    _set_vendor_quantity(demo_project, 91)
    after = _compile(demo_project)
    diff = diff_compile_results(before, after)

    before_vendor_ids = {packet.id for packet in before_vendor}
    resolved = [packet_diff for packet_diff in diff.packet_diffs if packet_diff.packet_id in before_vendor_ids]
    assert resolved
    assert any(packet_diff.change_type == "removed" for packet_diff in resolved)


def test_changing_customer_exclusion_invalidates_scope_exclusion(demo_project: Path) -> None:
    before = _compile(demo_project)
    before_scope = next(
        packet
        for packet in before.packets
        if packet.family == PacketFamily.scope_exclusion and "site:west_wing" in packet.anchor_key
    )

    email_path = demo_project / "customer_email.txt"
    updated = email_path.read_text(encoding="utf-8").replace(
        "Please remove West Wing from scope.",
        "Please keep West Wing in scope.",
    )
    email_path.write_text(updated, encoding="utf-8")

    after = _compile(demo_project)
    diff = diff_compile_results(before, after)
    scope_diffs = [packet_diff for packet_diff in diff.packet_diffs if packet_diff.packet_id == before_scope.id]
    assert scope_diffs
    assert any(packet_diff.change_type == "invalidated" for packet_diff in scope_diffs)


def test_irrelevant_atom_change_does_not_invalidate_packets(demo_project: Path) -> None:
    before = _compile(demo_project)

    transcript_path = demo_project / "kickoff_transcript.txt"
    text = transcript_path.read_text(encoding="utf-8")
    transcript_path.write_text(
        text.replace(
            "[00:00:01] Purtera PM: Starting kickoff for the camera rollout.",
            "[00:00:01] Purtera PM: Starting kickoff for the rollout review.",
        ),
        encoding="utf-8",
    )

    after = _compile(demo_project)
    diff = diff_compile_results(before, after)
    assert not diff.invalidated_packet_ids


def test_removing_governing_atom_invalidates_packet(demo_project: Path) -> None:
    before = _compile(demo_project)
    before_access = next(
        packet
        for packet in before.packets
        if packet.family == PacketFamily.site_access and packet.anchor_key == "site:main_campus"
    )

    email_path = demo_project / "customer_email.txt"
    updated = email_path.read_text(encoding="utf-8").replace(
        "Main Campus requires escort access after 5pm.",
        "",
    )
    email_path.write_text(updated, encoding="utf-8")

    after = _compile(demo_project)
    diff = diff_compile_results(before, after)
    access_diffs = [packet_diff for packet_diff in diff.packet_diffs if packet_diff.packet_id == before_access.id]
    assert access_diffs
    assert any(packet_diff.change_type == "invalidated" for packet_diff in access_diffs)


def test_blast_radius_summary_includes_downstream_consumers(demo_project: Path) -> None:
    before = _compile(demo_project)
    _set_vendor_quantity(demo_project, 91)
    after = _compile(demo_project)
    diff = diff_compile_results(before, after)
    consumers = diff.blast_radius_summary.get("impacted_consumers", [])
    assert consumers
    assert "OrbitBrief.scope_truth" in consumers
