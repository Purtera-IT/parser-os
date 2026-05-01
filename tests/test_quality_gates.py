from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CompileResult,
    EvidenceReceipt,
    EvidenceAtom,
    EvidencePacket,
    PacketFamily,
    PacketStatus,
    ReviewStatus,
    SourceRef,
)
from app.core.validators import validate_compile_result


def _atom(
    atom_id: str,
    *,
    authority: AuthorityClass,
    atom_type: AtomType = AtomType.scope_item,
    confidence: float = 0.9,
    entity_keys: list[str] | None = None,
) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=f"{atom_type.value} text",
        normalized_text=f"{atom_type.value} text",
        value={"quantity": 91 if atom_type == AtomType.quantity else "value"},
        entity_keys=entity_keys or ["site:west_wing"],
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=authority,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_invalid_packet_with_deleted_text_governing_fails() -> None:
    deleted = _atom("a_deleted", authority=AuthorityClass.deleted_text)
    packet = EvidencePacket(
        id="p1",
        project_id="proj_1",
        family=PacketFamily.scope_inclusion,
        anchor_type="site",
        anchor_key="site:west_wing",
        governing_atom_ids=[deleted.id],
        supporting_atom_ids=[deleted.id],
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=0.5,
        status=PacketStatus.active,
        reason="bad governing",
        review_flags=[],
    )
    result = CompileResult(project_id="proj_1", atoms=[deleted], entities=[], edges=[], packets=[packet], warnings=[])
    messages = validate_compile_result(result)
    assert any("deleted_text governing atom" in msg for msg in messages)


def test_conflict_packet_with_no_contradicting_atoms_fails() -> None:
    atom = _atom("a1", authority=AuthorityClass.approved_site_roster, atom_type=AtomType.quantity)
    packet = EvidencePacket(
        id="p_conflict",
        project_id="proj_1",
        family=PacketFamily.quantity_conflict,
        anchor_type="device",
        anchor_key="device:ip_camera",
        governing_atom_ids=[atom.id],
        supporting_atom_ids=[atom.id],
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=0.9,
        status=PacketStatus.needs_review,
        reason="conflict missing details",
        review_flags=[],
    )
    result = CompileResult(project_id="proj_1", atoms=[atom], entities=[], edges=[], packets=[packet], warnings=[])
    messages = validate_compile_result(result)
    assert any("quantity_conflict packet" in msg and "no contradicting_atom_ids" in msg for msg in messages)


def test_low_confidence_atom_creates_warning() -> None:
    low = _atom("a_low", authority=AuthorityClass.machine_extractor, confidence=0.6)
    packet = EvidencePacket(
        id="p1",
        project_id="proj_1",
        family=PacketFamily.scope_inclusion,
        anchor_type="site",
        anchor_key="site:west_wing",
        governing_atom_ids=[low.id],
        supporting_atom_ids=[low.id],
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=0.6,
        status=PacketStatus.active,
        reason="low confidence",
        review_flags=[],
    )
    result = CompileResult(project_id="proj_1", atoms=[low], entities=[], edges=[], packets=[packet], warnings=[])
    messages = validate_compile_result(result)
    assert any("low confidence" in msg.lower() for msg in messages)


def test_failed_receipt_is_hard_error_when_sources_available() -> None:
    atom = _atom("a1", authority=AuthorityClass.approved_site_roster)
    atom.receipts = [
        EvidenceReceipt(
            atom_id=atom.id,
            artifact_id=atom.artifact_id,
            filename="fixture.txt",
            source_ref_id=atom.source_refs[0].id,
            replay_status="failed",
            extracted_snippet=None,
            locator={"line_start": 1, "line_end": 1},
            reason="line missing",
            verifier_version="test",
        )
    ]
    packet = EvidencePacket(
        id="p1",
        project_id="proj_1",
        family=PacketFamily.scope_inclusion,
        anchor_type="site",
        anchor_key="site:west_wing",
        governing_atom_ids=[atom.id],
        supporting_atom_ids=[atom.id],
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=0.8,
        status=PacketStatus.active,
        reason="scope",
        review_flags=[],
    )
    result = CompileResult(project_id="proj_1", atoms=[atom], entities=[], edges=[], packets=[packet], warnings=[])
    messages = validate_compile_result(result, source_files_available=True)
    assert any("failed receipt" in msg.lower() for msg in messages if msg.startswith("ERROR:"))


def test_demo_project_passes_hard_validation_but_has_needs_review_warnings(demo_project: Path) -> None:
    result = compile_project(project_dir=demo_project, project_id="demo_project")
    messages = validate_compile_result(result, source_files_available=True)
    hard_errors = [m for m in messages if m.startswith("ERROR:")]
    warn_messages = [m for m in messages if m.startswith("WARNING:")]
    assert not hard_errors
    assert any("needs_review" in msg or "unsupported receipt" in msg.lower() for msg in warn_messages)
