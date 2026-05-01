from __future__ import annotations

from collections import Counter
from typing import Literal

from app.core.ids import stable_id
from app.core.schemas import CandidateAtom, CandidateSummary, EvidenceAtom, ReviewStatus


def candidate_from_evidence_atom(
    atom: EvidenceAtom,
    *,
    extractor_name: str,
    extractor_version: str,
    extraction_method: Literal[
        "deterministic_rule",
        "domain_pack_rule",
        "semantic_candidate",
        "llm_candidate",
        "human_label",
    ] = "deterministic_rule",
) -> CandidateAtom:
    return CandidateAtom(
        id=stable_id("cand", atom.project_id, atom.artifact_id, atom.atom_type.value, atom.raw_text, atom.id),
        project_id=atom.project_id,
        artifact_id=atom.artifact_id,
        candidate_type=atom.atom_type,
        raw_text=atom.raw_text,
        proposed_normalized_text=atom.normalized_text,
        proposed_value=dict(atom.value),
        proposed_entity_keys=list(atom.entity_keys),
        source_refs=list(atom.source_refs),
        proposed_authority_class=atom.authority_class,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        extraction_method=extraction_method,
        confidence=atom.confidence,
        evidence_span=atom.raw_text,
        validation_status="pending",
        validation_reasons=[],
    )


def summarize_candidate_outcomes(
    *,
    candidates: list[CandidateAtom],
    accepted_atoms: list[EvidenceAtom],
    rejected_candidates: list[CandidateAtom],
) -> CandidateSummary:
    by_extractor: Counter[str] = Counter(candidate.extractor_name for candidate in candidates)
    needs_review_count = sum(1 for atom in accepted_atoms if atom.review_status == ReviewStatus.needs_review)
    return CandidateSummary(
        candidate_count=len(candidates),
        accepted_count=len(accepted_atoms),
        rejected_count=len(rejected_candidates),
        needs_review_count=needs_review_count,
        by_extractor=dict(sorted(by_extractor.items())),
    )
