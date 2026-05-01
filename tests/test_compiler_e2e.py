from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app as cli_app
from app.core.compiler import compile_project
from app.core.schemas import AuthorityClass, PacketStatus
from app.main import app as fastapi_app


def test_compile_empty_directory_returns_valid_result(tmp_path: Path) -> None:
    project_dir = tmp_path / "empty_project"
    project_dir.mkdir()
    result = compile_project(project_dir=project_dir)
    assert result.project_id == "empty_project"
    assert isinstance(result.atoms, list)
    assert isinstance(result.entities, list)
    assert isinstance(result.edges, list)
    assert isinstance(result.packets, list)
    assert isinstance(result.warnings, list)
    assert result.candidate_summary is not None


def test_compiler_e2e_golden_regression(demo_project: Path, tmp_path: Path) -> None:
    result = compile_project(project_dir=demo_project, project_id="demo_project")
    expected_path = Path(__file__).parent / "fixtures" / "expected" / "demo_summary.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    # Atom + packet structural guarantees
    assert all(atom.source_refs for atom in result.atoms)
    assert all(
        packet.status in {PacketStatus.rejected, PacketStatus.invalidated} or packet.governing_atom_ids
        for packet in result.packets
    )

    # Expected family coverage from golden summary.
    families = {packet.family.value for packet in result.packets}
    for family in expected["packet_families"]:
        assert family in families

    # Deleted and quoted governance constraints.
    governed_ids = {atom_id for packet in result.packets for atom_id in packet.governing_atom_ids}
    atoms_by_id = {atom.id: atom for atom in result.atoms}
    governed_atoms = [atoms_by_id[atom_id] for atom_id in governed_ids if atom_id in atoms_by_id]
    assert all(atom.authority_class != AuthorityClass.deleted_text for atom in governed_atoms)
    assert all(atom.authority_class != AuthorityClass.quoted_old_email for atom in governed_atoms)

    # West Wing exclusion should be governed by current customer-authored exclusion.
    scope_exclusion_packets = [p for p in result.packets if p.family.value == "scope_exclusion"]
    assert scope_exclusion_packets
    west_wing_packet = next(
        (
            p
            for p in scope_exclusion_packets
            if "site:west_wing" in p.anchor_key
            or any(
                "site:west_wing" in atoms_by_id.get(aid).entity_keys
                for aid in (p.supporting_atom_ids + p.contradicting_atom_ids)
                if aid in atoms_by_id
            )
        ),
        None,
    )
    assert west_wing_packet is not None
    assert west_wing_packet.governing_atom_ids
    west_governing = atoms_by_id[west_wing_packet.governing_atom_ids[0]]
    assert west_governing.authority_class == AuthorityClass.customer_current_authored
    assert west_governing.atom_type.value == "exclusion"

    # Roster aggregate (91) vs vendor line (72) is emitted on vendor_mismatch; per-site vs vendor may be quantity_conflict.
    commercial_qty_packets = [
        p for p in result.packets if p.family.value in {"quantity_conflict", "vendor_mismatch"}
    ]
    assert any("91" in p.reason and "72" in p.reason for p in commercial_qty_packets)

    # Anchor sanity from golden summary (high-level, non-ID sensitive).
    for family, expected_anchors in expected["anchors"].items():
        family_packets = [p for p in result.packets if p.family.value == family]
        observed = {p.anchor_key for p in family_packets}
        assert any(anchor in observed for anchor in expected_anchors)

    # Output JSON write/load smoke test.
    out_path = tmp_path / "compiled_demo.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["project_id"] == "demo_project"
    assert isinstance(loaded["packets"], list)
    assert "candidate_summary" in loaded


def test_api_app_imports() -> None:
    assert fastapi_app is not None
    assert fastapi_app.title == "Purtera Evidence Compiler MVP"


def test_cli_compile_creates_json_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    out_path = tmp_path / "result.json"
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["compile", str(project_dir), "--out", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "project_id" in payload
    assert "atoms" in payload
    assert "entities" in payload
    assert "edges" in payload
    assert "packets" in payload
    assert "warnings" in payload
