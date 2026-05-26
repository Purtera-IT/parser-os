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
    """Legacy single-call variant. Use ``_build_alias_lookup`` +
    ``_device_alias_canonicals_cached`` in hot loops — they pre-normalize
    the pack once and skip O(N*M) regex work per pair.
    """
    lowered = normalize_text(text)
    matched: set[str] = set()
    for canonical, aliases in pack.device_aliases.items():
        for alias in aliases:
            token = normalize_text(alias)
            if token and token in lowered:
                matched.add(canonical)
                break
    return matched


def _build_alias_lookup(pack: DomainPack) -> list[tuple[str, str]]:
    """Pre-normalize pack.device_aliases into a flat ``[(token, canonical)]``
    list so the per-pair hot path only does substring tests, never regex
    normalization. Sorted by token length descending so longer aliases
    (more specific) win the early-exit per canonical.
    """
    out: list[tuple[str, str]] = []
    for canonical, aliases in pack.device_aliases.items():
        canonical_token = normalize_text(canonical.replace("_", " "))
        if canonical_token:
            out.append((canonical_token, canonical))
        for alias in aliases or []:
            token = normalize_text(alias)
            if token:
                out.append((token, canonical))
    out.sort(key=lambda kv: -len(kv[0]))
    return out


def _device_alias_canonicals_cached(lowered: str, lookup: list[tuple[str, str]]) -> set[str]:
    """Hot-path companion to ``_build_alias_lookup``. ``lowered`` must
    already be the normalize_text result. Iterates the flat list once
    with substring tests."""
    matched: set[str] = set()
    for token, canonical in lookup:
        if canonical in matched:
            continue
        if token in lowered:
            matched.add(canonical)
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
    """Legacy per-pair entrypoint kept for tests / external callers.
    The fast path in ``propose_semantic_link_candidates`` uses the
    cached per-atom alias sets instead."""
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
    # Scale guard: at very large atom counts, the linker's N^2
    # similarity matrix dominates both memory and compute (50M entries
    # at N=10k). For S++ scale, skip the linker entirely above this
    # threshold — its findings on near-identical template rows in a
    # huge BOM are low-value and the cost is prohibitive.
    _LINKER_MAX_ATOMS = 2000
    if len(atoms) > _LINKER_MAX_ATOMS:
        return []
    ordered = sorted(atoms, key=lambda atom: atom.id)
    texts = [atom_representation(atom, domain_pack=domain_pack) for atom in ordered]
    similarity_matrix, method = best_effort_similarity(texts)
    pack = domain_pack or get_active_domain_pack()
    candidates: list[SemanticLinkCandidate] = []

    # Pre-compute per-atom data ONCE so the inner N^2 pair loop only does
    # set intersections + numeric comparisons.
    #
    # Before this optimization, ``_device_alias_canonicals`` ran inside
    # the inner loop and re-normalized every device alias for every
    # candidate pair — on OPTBOT (286 atoms, ~40k pairs) it accounted
    # for 12.8s of 32.7s total compile time. After precompute it drops
    # to roughly O(N) work.
    alias_lookup = _build_alias_lookup(pack)
    atom_device_keys: list[set[str]] = []
    atom_alias_sets: list[set[str]] = []
    atom_entity_types: list[str | None] = []
    for atom in ordered:
        atom_device_keys.append({k for k in atom.entity_keys if k.startswith("device:")})
        atom_alias_sets.append(
            _device_alias_canonicals_cached(normalize_text(atom.raw_text or ""), alias_lookup)
        )
        atom_entity_types.append(_entity_type(atom))

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
            # Inline alias adjustment using precomputed sets.
            if atom_device_keys[i] and atom_device_keys[i].intersection(atom_device_keys[j]):
                score = max(score, 0.99)
            elif (
                atom_alias_sets[i]
                and atom_alias_sets[i].intersection(atom_alias_sets[j])
            ):
                score = max(score, 0.96)
            if score < REVIEW_THRESHOLD:
                continue

            status = "needs_review"
            left_et = atom_entity_types[i]
            entity_match = left_et is not None and left_et == atom_entity_types[j]
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
