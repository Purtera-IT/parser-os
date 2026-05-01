from __future__ import annotations

from pathlib import Path

from app.core.schemas import AtomType, AuthorityClass, ReviewStatus
from app.parsers.transcript_parser import TranscriptParser
from scripts.make_demo_fixtures import create_demo_project


def test_transcript_parser_extracts_expected_atoms(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    project_dir = create_demo_project(root)
    transcript_path = project_dir / "kickoff_transcript.txt"

    atoms = TranscriptParser().parse_artifact(
        project_id="proj_1",
        artifact_id="art_transcript_1",
        path=transcript_path,
    )

    assert atoms
    assert all(atom.source_refs for atom in atoms)
    assert all("line_start" in atom.source_refs[0].locator for atom in atoms)
    assert all("line_end" in atom.source_refs[0].locator for atom in atoms)

    assert any(
        atom.atom_type == AtomType.constraint and "escort access after 5pm" in atom.normalized_text.lower()
        for atom in atoms
    )
    assert any(atom.atom_type == AtomType.exclusion and "west wing" in atom.normalized_text.lower() for atom in atoms)
    assert any(atom.atom_type == AtomType.action_item and "provide lift access" in atom.normalized_text.lower() for atom in atoms)
    assert any(atom.atom_type == AtomType.open_question and "mdf room requires badge access" in atom.normalized_text.lower() for atom in atoms)
    assert any(
        atom.atom_type in {AtomType.quantity, AtomType.scope_item}
        and "add 5 more ip cameras" in atom.normalized_text.lower()
        for atom in atoms
    )
    assert any(
        atom.atom_type in {AtomType.scope_item, AtomType.exclusion, AtomType.customer_instruction, AtomType.quantity}
        and atom.review_status == ReviewStatus.needs_review
        for atom in atoms
    )
    assert all(atom.authority_class == AuthorityClass.meeting_note for atom in atoms)
    assert any("customer_spoken_instruction" in atom.review_flags for atom in atoms if atom.atom_type == AtomType.customer_instruction)
    assert any("site:west_wing" in atom.entity_keys for atom in atoms)
    assert any("site:main_campus" in atom.entity_keys for atom in atoms)
    assert any("device:ip_camera" in atom.entity_keys for atom in atoms)
