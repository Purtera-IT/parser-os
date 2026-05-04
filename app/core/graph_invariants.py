from __future__ import annotations

import re
from collections import defaultdict

from app.core.normalizers import normalize_text
from app.core.schemas import AtomType, EdgeType, EvidenceAtom, EvidenceEdge


def _active_pack_pattern_lists() -> tuple[list[str], list[str]]:
    """Return ``(exclusion_patterns, constraint_patterns)`` from the active pack.

    Mirrors what ``graph_builder.build_edges`` uses to decide whether a
    ``customer_instruction`` atom counts as an exclusion- or constraint-bearing
    endpoint.  Defensive: if no active pack is set (e.g. in a unit test that
    constructs edges directly), returns empty lists, which keeps the validator
    in its original strict mode (atom_type must literally be exclusion /
    constraint).
    """
    try:
        from app.domain import get_active_domain_pack
    except Exception:
        return [], []
    try:
        pack = get_active_domain_pack()
    except Exception:
        return [], []
    excl = [normalize_text(p) for p in (getattr(pack, "exclusion_patterns", None) or [])]
    cons: list[str] = []
    for patterns in (getattr(pack, "constraint_patterns", None) or {}).values():
        for pattern in patterns or []:
            cons.append(normalize_text(pattern))
    return excl, cons


def _atom_matches_any(atom: EvidenceAtom | None, patterns: list[str]) -> bool:
    if atom is None or not patterns:
        return False
    text = normalize_text(atom.raw_text)
    return any(p and p in text for p in patterns)


def _is_exclusion_endpoint(
    atom: EvidenceAtom | None, exclusion_patterns: list[str]
) -> bool:
    """Mirror ``graph_builder``'s exclusion membership rule.

    Either the atom is literally an ``exclusion`` atom, or it is a
    ``customer_instruction`` whose text matches one of the active pack's
    exclusion patterns (e.g. an ``A1.`` answer from a customer-current PDF
    that says "we will not be needing X").
    """
    if atom is None:
        return False
    if atom.atom_type == AtomType.exclusion:
        return True
    if atom.atom_type == AtomType.customer_instruction:
        return _atom_matches_any(atom, exclusion_patterns)
    return False


def _is_constraint_endpoint(
    atom: EvidenceAtom | None, constraint_patterns: list[str]
) -> bool:
    """Mirror ``graph_builder``'s constraint membership rule."""
    if atom is None:
        return False
    if atom.atom_type == AtomType.constraint:
        return True
    if atom.atom_type == AtomType.customer_instruction:
        return _atom_matches_any(atom, constraint_patterns)
    return False


def check_graph_invariants(atoms: list[EvidenceAtom], edges: list[EvidenceEdge]) -> list[str]:
    errors: list[str] = []
    atom_by_id = {atom.id: atom for atom in atoms}

    aggregate_conflict_by_device: dict[str, int] = defaultdict(int)
    exclusion_patterns, constraint_patterns = _active_pack_pattern_lists()

    for edge in edges:
        if edge.from_atom_id not in atom_by_id:
            errors.append(f"ERROR: Edge {edge.id} missing from_atom_id {edge.from_atom_id}")
        if edge.to_atom_id not in atom_by_id:
            errors.append(f"ERROR: Edge {edge.id} missing to_atom_id {edge.to_atom_id}")

        if edge.from_atom_id == edge.to_atom_id and edge.edge_type != EdgeType.same_as:
            errors.append(f"ERROR: Edge {edge.id} has disallowed self-loop for edge_type {edge.edge_type.value}")

        if edge.edge_type == EdgeType.contradicts and not edge.reason.strip():
            errors.append(f"ERROR: Edge {edge.id} contradicts edge must include reason")

        if edge.edge_type == EdgeType.excludes:
            from_atom = atom_by_id.get(edge.from_atom_id)
            to_atom = atom_by_id.get(edge.to_atom_id)
            if not (
                _is_exclusion_endpoint(from_atom, exclusion_patterns)
                or _is_exclusion_endpoint(to_atom, exclusion_patterns)
            ):
                errors.append(f"ERROR: Edge {edge.id} excludes edge must involve exclusion atom")

        if edge.edge_type == EdgeType.requires:
            from_atom = atom_by_id.get(edge.from_atom_id)
            to_atom = atom_by_id.get(edge.to_atom_id)
            if not (
                _is_constraint_endpoint(from_atom, constraint_patterns)
                or _is_constraint_endpoint(to_atom, constraint_patterns)
            ):
                errors.append(f"ERROR: Edge {edge.id} requires edge must involve constraint atom")

        if edge.edge_type == EdgeType.contradicts and "Aggregate scoped quantity" in edge.reason:
            match = re.search(r"for\s+([a-z0-9:_]+)$", edge.reason)
            device_key = match.group(1) if match else "unknown_device"
            aggregate_conflict_by_device[device_key] += 1

        if edge.edge_type == EdgeType.same_as:
            from_atom = atom_by_id.get(edge.from_atom_id)
            to_atom = atom_by_id.get(edge.to_atom_id)
            if from_atom and to_atom:
                from_types = {key.split(":", 1)[0] for key in from_atom.entity_keys if ":" in key}
                to_types = {key.split(":", 1)[0] for key in to_atom.entity_keys if ":" in key}
                if from_types and to_types and from_types.isdisjoint(to_types):
                    errors.append(f"ERROR: Edge {edge.id} same_as merges unrelated entity key types")

    for device_key, count in aggregate_conflict_by_device.items():
        if count > 1:
            errors.append(
                f"ERROR: Aggregate quantity contradiction duplicated {count} times for {device_key}"
            )

    return sorted(set(errors))
