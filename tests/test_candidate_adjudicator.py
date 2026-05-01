from __future__ import annotations

from pathlib import Path

from app.core.candidate_adjudicator import adjudicate_candidates
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CandidateAtom,
    ReviewStatus,
    SourceRef,
)


def _source_ref(artifact_id: str, filename: str, line_number: int = 1) -> SourceRef:
    return SourceRef(
        id=stable_id("src", artifact_id, line_number),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.txt,
        filename=filename,
        locator={"line_start": line_number, "line_end": line_number},
        extraction_method="deterministic_rule",
        parser_version="test_parser_v1",
    )


def _candidate(
    *,
    project_id: str,
    artifact_id: str,
    source_refs: list[SourceRef],
    raw_text: str,
    candidate_type: AtomType = AtomType.constraint,
    authority: AuthorityClass = AuthorityClass.customer_current_authored,
    confidence: float = 0.9,
    entity_keys: list[str] | None = None,
) -> CandidateAtom:
    return CandidateAtom(
        id=stable_id("cand", project_id, artifact_id, raw_text),
        project_id=project_id,
        artifact_id=artifact_id,
        candidate_type=candidate_type,
        raw_text=raw_text,
        proposed_normalized_text=raw_text.lower(),
        proposed_value={"text": raw_text},
        proposed_entity_keys=entity_keys or ["site:main_campus"],
        source_refs=source_refs,
        proposed_authority_class=authority,
        extractor_name="unit_test_extractor",
        extractor_version="unit_v1",
        extraction_method="deterministic_rule",
        confidence=confidence,
        evidence_span=raw_text,
        validation_status="pending",
        validation_reasons=[],
    )


def test_valid_candidate_becomes_evidence_atom(tmp_path: Path) -> None:
    artifact = tmp_path / "notes.txt"
    artifact.write_text("Escort required for access.", encoding="utf-8")
    artifact_id = stable_id("art", "proj", artifact.name)
    candidate = _candidate(
        project_id="proj",
        artifact_id=artifact_id,
        source_refs=[_source_ref(artifact_id, artifact.name)],
        raw_text="Escort required for access.",
    )
    result = adjudicate_candidates([candidate], {artifact_id: artifact})
    assert len(result.accepted_atoms) == 1
    assert not result.rejected_candidates
    assert result.accepted_atoms[0].atom_type == AtomType.constraint


def test_candidate_without_source_ref_is_rejected() -> None:
    candidate = _candidate(
        project_id="proj",
        artifact_id="art_missing",
        source_refs=[],
        raw_text="Need badge access.",
    )
    result = adjudicate_candidates([candidate], {})
    assert not result.accepted_atoms
    assert len(result.rejected_candidates) == 1
    assert "missing_source_ref" in result.rejected_candidates[0].validation_reasons


def test_low_confidence_candidate_rejected_or_needs_review(tmp_path: Path) -> None:
    artifact = tmp_path / "notes.txt"
    artifact.write_text("install 2 cameras\nNeed confirmation?", encoding="utf-8")
    artifact_id = stable_id("art", "proj", artifact.name)

    scope_candidate = _candidate(
        project_id="proj",
        artifact_id=artifact_id,
        source_refs=[_source_ref(artifact_id, artifact.name)],
        raw_text="install 2 cameras",
        candidate_type=AtomType.scope_item,
        authority=AuthorityClass.meeting_note,
        confidence=0.4,
    )
    question_candidate = _candidate(
        project_id="proj",
        artifact_id=artifact_id,
        source_refs=[_source_ref(artifact_id, artifact.name, line_number=2)],
        raw_text="Need confirmation?",
        candidate_type=AtomType.open_question,
        authority=AuthorityClass.meeting_note,
        confidence=0.4,
    )
    result = adjudicate_candidates([scope_candidate, question_candidate], {artifact_id: artifact})
    assert any("low_confidence_scope_impact" in row.validation_reasons for row in result.rejected_candidates)
    assert any(atom.review_status == ReviewStatus.needs_review for atom in result.accepted_atoms)


def test_meeting_note_scope_impacting_candidate_becomes_needs_review(tmp_path: Path) -> None:
    artifact = tmp_path / "meeting.txt"
    artifact.write_text("Please remove west wing from scope.", encoding="utf-8")
    artifact_id = stable_id("art", "proj", artifact.name)
    candidate = _candidate(
        project_id="proj",
        artifact_id=artifact_id,
        source_refs=[_source_ref(artifact_id, artifact.name)],
        raw_text="Please remove west wing from scope.",
        candidate_type=AtomType.exclusion,
        authority=AuthorityClass.meeting_note,
        confidence=0.95,
    )
    result = adjudicate_candidates([candidate], {artifact_id: artifact})
    assert len(result.accepted_atoms) == 1
    assert result.accepted_atoms[0].review_status == ReviewStatus.needs_review


def test_deleted_text_candidate_preserves_rejected_behavior(tmp_path: Path) -> None:
    artifact = tmp_path / "doc.txt"
    artifact.write_text("Old deleted sentence", encoding="utf-8")
    artifact_id = stable_id("art", "proj", artifact.name)
    candidate = _candidate(
        project_id="proj",
        artifact_id=artifact_id,
        source_refs=[_source_ref(artifact_id, artifact.name)],
        raw_text="Old deleted sentence",
        candidate_type=AtomType.scope_item,
        authority=AuthorityClass.deleted_text,
        confidence=0.9,
    )
    result = adjudicate_candidates([candidate], {artifact_id: artifact})
    assert len(result.accepted_atoms) == 1
    assert result.accepted_atoms[0].authority_class == AuthorityClass.deleted_text
    assert result.accepted_atoms[0].review_status == ReviewStatus.rejected
    assert "tracked_change_deleted_text" in result.accepted_atoms[0].review_flags
