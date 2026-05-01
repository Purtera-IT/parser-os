from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.schemas import PacketFamily
from app.core.validators import validate_compile_result


def test_every_demo_packet_has_certificate(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.packets
    assert all(packet.certificate is not None for packet in result.packets)


def test_quantity_conflict_certificate_minimal_set_and_reason(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    commercial = [
        p
        for p in result.packets
        if p.family in (PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch)
        and "91" in p.reason
        and "72" in p.reason
    ]
    assert commercial, "expected roster aggregate vs vendor quantity surfaced on a commercial packet"
    packet = commercial[0]
    assert packet.certificate is not None
    minimal_ids = packet.certificate.minimal_sufficient_atom_ids
    assert len(minimal_ids) >= 2
    atoms_by_id = {atom.id: atom for atom in result.atoms}
    minimal_atoms = [atoms_by_id[aid] for aid in minimal_ids if aid in atoms_by_id]
    assert any(atom.authority_class.value == "approved_site_roster" for atom in minimal_atoms)
    assert any(atom.authority_class.value == "vendor_quote" for atom in minimal_atoms)
    assert "91" in packet.certificate.existence_reason and "72" in packet.certificate.existence_reason


def test_scope_exclusion_rationale_mentions_customer_authored_when_governed(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    atoms_by_id = {atom.id: atom for atom in result.atoms}
    scope_packets = [packet for packet in result.packets if packet.family == PacketFamily.scope_exclusion]
    assert scope_packets
    governed_by_customer = next(
        (
            packet
            for packet in scope_packets
            if packet.governing_atom_ids
            and atoms_by_id[packet.governing_atom_ids[0]].authority_class.value == "customer_current_authored"
        ),
        None,
    )
    assert governed_by_customer is not None
    assert governed_by_customer.certificate is not None
    assert "customer_current_authored" in governed_by_customer.certificate.governing_rationale


def test_certificate_references_and_counterfactuals_are_consistent(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    atom_ids = {atom.id for atom in result.atoms}
    for packet in result.packets:
        assert packet.certificate is not None
        cert = packet.certificate
        for atom_id in cert.minimal_sufficient_atom_ids:
            assert atom_id in atom_ids
        cf_ids = {entry.get("atom_id") for entry in cert.counterfactuals}
        assert set(cert.minimal_sufficient_atom_ids).issubset(cf_ids)


def test_scope_packets_include_sowsmith_in_blast_radius_and_scores_are_bounded(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    for packet in result.packets:
        assert packet.certificate is not None
        cert = packet.certificate
        assert 0.0 <= cert.evidence_completeness_score <= 1.0
        assert 0.0 <= cert.ambiguity_score <= 1.0
        if packet.family in {PacketFamily.scope_exclusion, PacketFamily.scope_inclusion}:
            assert any(entry.startswith("SOWSmith.") for entry in cert.blast_radius)


def test_packet_validation_fails_if_certificate_missing(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.packets
    result.packets[0].certificate = None
    messages = validate_compile_result(result, source_files_available=True)
    assert any("missing certificate" in msg for msg in messages if msg.startswith("ERROR:"))


def test_packet_certificate_domain_pack_reflects_copper_selection(demo_project: Path) -> None:
    result = compile_project(
        demo_project,
        project_id="demo_project",
        allow_unverified_receipts=True,
        domain_pack="copper_cabling",
    )
    assert result.manifest and result.manifest.domain_pack_id == "copper_cabling"
    for packet in result.packets:
        assert packet.certificate is not None
        assert packet.certificate.domain_pack_id == "copper_cabling"
        assert packet.certificate.domain_pack_version == "0.4.0-generated"
