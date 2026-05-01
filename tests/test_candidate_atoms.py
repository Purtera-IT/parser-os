from __future__ import annotations

from pathlib import Path

from app.core.candidates import candidate_from_evidence_atom, summarize_candidate_outcomes
from app.core.compiler import compile_project
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CandidateAtom,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom() -> EvidenceAtom:
    source = SourceRef(
        id="src_1",
        artifact_id="art_1",
        artifact_type=ArtifactType.txt,
        filename="demo.txt",
        locator={"line_start": 1, "line_end": 1},
        extraction_method="rule",
        parser_version="v1",
    )
    return EvidenceAtom(
        id="atm_1",
        project_id="proj",
        artifact_id="art_1",
        atom_type=AtomType.constraint,
        raw_text="Escort required",
        normalized_text="escort required",
        value={"text": "Escort required"},
        entity_keys=["site:main_campus"],
        source_refs=[source],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="v1",
    )


def test_candidate_from_evidence_atom_bridge() -> None:
    candidate = candidate_from_evidence_atom(_atom(), extractor_name="xlsx", extractor_version="v1")
    assert isinstance(candidate, CandidateAtom)
    assert candidate.candidate_type == AtomType.constraint
    assert candidate.validation_status == "pending"


def test_candidate_summary_counts() -> None:
    candidate = candidate_from_evidence_atom(_atom(), extractor_name="xlsx", extractor_version="v1")
    summary = summarize_candidate_outcomes(candidates=[candidate], accepted_atoms=[_atom()], rejected_candidates=[])
    assert summary.candidate_count == 1
    assert summary.accepted_count == 1
    assert summary.rejected_count == 0
    assert summary.by_extractor["xlsx"] == 1


def test_compiler_still_works_with_direct_atoms(tmp_path: Path) -> None:
    artifact = tmp_path / "customer_email.txt"
    artifact.write_text("From: customer@example.com\nSent: Monday\nSubject: Scope\nExclude west wing", encoding="utf-8")
    result = compile_project(tmp_path, allow_errors=True)
    assert isinstance(result.atoms, list)
    assert result.candidate_summary is not None
    assert result.candidate_summary.candidate_count >= 0
