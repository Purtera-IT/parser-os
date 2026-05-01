from __future__ import annotations

from collections.abc import Iterable

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import AtomType, EdgeType, EvidenceAtom, SemanticLinkCandidate
from app.domain import get_active_domain_pack
from app.domain.schemas import DomainPack
from app.semantic.vectorizer import atom_representation, best_effort_similarity

ACCEPTED_THRESHOLD = 0.95
REVIEW_THRESHOLD = 0.82


def _entity_type(atom: EvidenceAtom) -> str | None:
    for key in atom.entity_keys:
        prefix, _, _ = key.partition(":")
        if prefix:
            return prefix
    if isinstance(atom.value, dict):
        raw = atom.value.get("entity_type")
        if raw:
            return normalize_text(str(raw))
    return None


def _device_alias_canonicals(text: str, pack: DomainPack) -> set[str]:
    lowered = normalize_text(text)
    matched: set[str] = set()
    for canonical, aliases in pack.device_aliases.items():
        for alias in aliases:
            token = normalize_text(alias)
            if token and token in lowered:
                matched.add(canonical)
                break
    return matched


def _pair_edge_type(left: EvidenceAtom, right: EvidenceAtom) -> tuple[EdgeType, str, str] | None:
    if left.atom_type == AtomType.entity and right.atom_type == AtomType.entity:
        from_id, to_id = sorted([left.id, right.id])
        return EdgeType.same_as, from_id, to_id
    if left.atom_type == AtomType.scope_item and right.atom_type == AtomType.scope_item:
        from_id, to_id = sorted([left.id, right.id])
        return EdgeType.supports, from_id, to_id
    if left.atom_type == AtomType.constraint and right.atom_type == AtomType.constraint:
        from_id, to_id = sorted([left.id, right.id])
        return EdgeType.supports, from_id, to_id
    if left.atom_type == AtomType.exclusion and right.atom_type == AtomType.scope_item:
        return EdgeType.excludes, left.id, right.id
    if left.atom_type == AtomType.scope_item and right.atom_type == AtomType.exclusion:
        return EdgeType.excludes, right.id, left.id
    return None


def _atom_by_id(atoms: Iterable[EvidenceAtom]) -> dict[str, EvidenceAtom]:
    return {atom.id: atom for atom in atoms}


def _adjust_similarity_for_aliases(score: float, left: EvidenceAtom, right: EvidenceAtom, pack: DomainPack) -> float:
    left_keys = {key for key in left.entity_keys if key.startswith("device:")}
    right_keys = {key for key in right.entity_keys if key.startswith("device:")}
    if left_keys and left_keys.intersection(right_keys):
        return max(score, 0.99)
    left_aliases = _device_alias_canonicals(left.raw_text, pack)
    right_aliases = _device_alias_canonicals(right.raw_text, pack)
    if left_aliases and left_aliases.intersection(right_aliases):
        return max(score, 0.96)
    return score


def propose_semantic_link_candidates(
    atoms: list[EvidenceAtom],
    *,
    domain_pack: DomainPack | None = None,
) -> list[SemanticLinkCandidate]:
    if not atoms:
        return []
    ordered = sorted(atoms, key=lambda atom: atom.id)
    texts = [atom_representation(atom, domain_pack=domain_pack) for atom in ordered]
    similarity_matrix, method = best_effort_similarity(texts)
    pack = domain_pack or get_active_domain_pack()
    candidates: list[SemanticLinkCandidate] = []

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            left = ordered[i]
            right = ordered[j]
            if left.project_id != right.project_id:
                continue
            pair = _pair_edge_type(left, right)
            if pair is None:
                continue
            edge_type, from_id, to_id = pair
            score = float(similarity_matrix[i][j])
            score = _adjust_similarity_for_aliases(score, left, right, pack)
            if score < REVIEW_THRESHOLD:
                continue

            status = "needs_review"
            entity_match = _entity_type(left) is not None and _entity_type(left) == _entity_type(right)
            if score >= ACCEPTED_THRESHOLD:
                if edge_type in {EdgeType.same_as, EdgeType.supports}:
                    status = "accepted" if entity_match else "needs_review"
                else:
                    status = "accepted"

            reason = (
                f"semantic_candidate_linker method={method} score={score:.3f} "
                f"for {left.atom_type.value}->{right.atom_type.value}"
            )
            candidate = SemanticLinkCandidate(
                id=stable_id("slk", left.project_id, from_id, to_id, edge_type.value, method, round(score, 6)),
                from_atom_id=from_id,
                to_atom_id=to_id,
                proposed_edge_type=edge_type,
                similarity_score=min(1.0, max(0.0, round(score, 6))),
                method=method,
                reason=reason,
                status=status,
            )
            if candidate.proposed_edge_type != EdgeType.contradicts:
                candidates.append(candidate)

    candidates.sort(key=lambda row: row.id)
    return candidates
