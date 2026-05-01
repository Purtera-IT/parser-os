from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    CandidateAtom,
    EvidenceAtom,
    ReviewStatus,
)
from app.core.source_replay import replay_source_ref

_SCOPE_IMPACTING_TYPES = {
    AtomType.scope_item,
    AtomType.exclusion,
    AtomType.customer_instruction,
}
_TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".eml", ".json", ".csv", ".vtt", ".srt"}
_ENTITY_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*:[a-z0-9_]+$")


class CandidateAdjudicationResult(BaseModel):
    accepted_atoms: list[EvidenceAtom] = Field(default_factory=list)
    rejected_candidates: list[CandidateAtom] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _candidate_atom_id(candidate: CandidateAtom) -> str:
    return stable_id(
        "atm",
        candidate.project_id,
        candidate.artifact_id,
        candidate.candidate_type.value,
        candidate.raw_text,
        candidate.proposed_normalized_text,
        candidate.proposed_value,
        sorted(candidate.proposed_entity_keys),
        [source_ref.id for source_ref in candidate.source_refs],
        candidate.extractor_name,
        candidate.extractor_version,
        candidate.extraction_method,
    )


def _make_evidence_atom(candidate: CandidateAtom, review_status: ReviewStatus) -> EvidenceAtom:
    review_flags: list[str] = list(candidate.validation_reasons)
    confidence = candidate.confidence
    if candidate.proposed_authority_class == AuthorityClass.deleted_text:
        review_status = ReviewStatus.rejected
        confidence = min(confidence, 0.2)
        review_flags.append("tracked_change_deleted_text")
    return EvidenceAtom(
        id=_candidate_atom_id(candidate),
        project_id=candidate.project_id,
        artifact_id=candidate.artifact_id,
        atom_type=candidate.candidate_type,
        raw_text=candidate.raw_text,
        normalized_text=candidate.proposed_normalized_text,
        value=dict(candidate.proposed_value),
        entity_keys=list(candidate.proposed_entity_keys),
        source_refs=list(candidate.source_refs),
        authority_class=candidate.proposed_authority_class,
        confidence=confidence,
        review_status=review_status,
        review_flags=sorted(set(flag for flag in review_flags if flag)),
        parser_version=candidate.extractor_version,
    )


def _is_valid_entity_key(entity_key: str) -> bool:
    return bool(_ENTITY_KEY_RE.match(entity_key))


def _span_matches_candidate(candidate: CandidateAtom, artifact_paths: dict[str, Path]) -> bool:
    if not candidate.evidence_span:
        return True
    span = normalize_text(candidate.evidence_span)
    if not span:
        return False
    checked = False
    for source_ref in candidate.source_refs:
        path = artifact_paths.get(source_ref.artifact_id)
        if path is None or not path.exists():
            continue
        if path.suffix.lower() in _TEXT_ARTIFACT_SUFFIXES:
            checked = True
            body = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
            if span in body:
                return True
    if not checked:
        return True
    return False


def adjudicate_candidates(
    candidates: list[CandidateAtom],
    artifact_paths: dict[str, Path],
) -> CandidateAdjudicationResult:
    accepted_atoms: list[EvidenceAtom] = []
    rejected_candidates: list[CandidateAtom] = []
    warnings: list[str] = []

    for candidate in candidates:
        reasons: list[str] = []

        if not candidate.source_refs:
            reasons.append("missing_source_ref")
        if not candidate.raw_text.strip():
            reasons.append("empty_raw_text")
        for entity_key in candidate.proposed_entity_keys:
            if not _is_valid_entity_key(entity_key):
                reasons.append(f"invalid_entity_key:{entity_key}")
                break
        if not _span_matches_candidate(candidate, artifact_paths):
            reasons.append("evidence_span_not_found")

        if reasons:
            rejected_candidates.append(
                candidate.model_copy(update={"validation_status": "rejected", "validation_reasons": sorted(set(reasons))})
            )
            continue

        receipt_failures = 0
        for source_ref in candidate.source_refs:
            provisional_atom = _make_evidence_atom(candidate, review_status=ReviewStatus.auto_accepted)
            receipt = replay_source_ref(provisional_atom, source_ref, artifact_paths)
            if receipt.replay_status == "failed":
                receipt_failures += 1
                reasons.append(f"source_replay_failed:{source_ref.id}")
            elif receipt.replay_status == "unsupported":
                warnings.append(f"WARNING: Candidate {candidate.id} has unsupported source replay for {source_ref.id}")

        if receipt_failures > 0:
            rejected_candidates.append(
                candidate.model_copy(update={"validation_status": "rejected", "validation_reasons": sorted(set(reasons))})
            )
            continue

        review_status = ReviewStatus.auto_accepted
        if candidate.confidence < 0.5:
            if candidate.candidate_type in _SCOPE_IMPACTING_TYPES:
                reasons.append("low_confidence_scope_impact")
                rejected_candidates.append(
                    candidate.model_copy(update={"validation_status": "rejected", "validation_reasons": sorted(set(reasons))})
                )
                continue
            reasons.append("low_confidence_needs_review")
            review_status = ReviewStatus.needs_review

        if (
            candidate.proposed_authority_class == AuthorityClass.meeting_note
            and candidate.candidate_type in _SCOPE_IMPACTING_TYPES
        ):
            reasons.append("meeting_note_scope_impact_needs_review")
            review_status = ReviewStatus.needs_review

        evidence_atom = _make_evidence_atom(candidate, review_status=review_status)
        accepted_atoms.append(evidence_atom)

    return CandidateAdjudicationResult(
        accepted_atoms=accepted_atoms,
        rejected_candidates=rejected_candidates,
        warnings=sorted(set(warnings)),
    )
