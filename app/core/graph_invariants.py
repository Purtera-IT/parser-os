from __future__ import annotations

import re
from collections import defaultdict

from app.core.schemas import AtomType, EdgeType, EvidenceAtom, EvidenceEdge


def check_graph_invariants(atoms: list[EvidenceAtom], edges: list[EvidenceEdge]) -> list[str]:
    errors: list[str] = []
    atom_by_id = {atom.id: atom for atom in atoms}

    aggregate_conflict_by_device: dict[str, int] = defaultdict(int)

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
                (from_atom and from_atom.atom_type == AtomType.exclusion)
                or (to_atom and to_atom.atom_type == AtomType.exclusion)
            ):
                errors.append(f"ERROR: Edge {edge.id} excludes edge must involve exclusion atom")

        if edge.edge_type == EdgeType.requires:
            from_atom = atom_by_id.get(edge.from_atom_id)
            to_atom = atom_by_id.get(edge.to_atom_id)
            if not (
                (from_atom and from_atom.atom_type == AtomType.constraint)
                or (to_atom and to_atom.atom_type == AtomType.constraint)
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
