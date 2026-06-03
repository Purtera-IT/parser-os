"""Corroboration-graded truth (Gap F — Truth Gate).

A fact asserted by three independent documents is not the same as a fact
that appears once, in one place, with no echo anywhere else — yet the
rest of the pipeline treats them identically once they become atoms and
entities. The Truth Gate grades every entity (and, by extension, the
facts hanging off it) by **independent-source agreement**: how many
*distinct artifacts* assert it, whether any source contradicts it, and a
coarse tier the cockpit can colour-band.

This is deliberately deterministic and LLM-free — it is a structural
count over provenance, not a semantic judgement — so it runs in the
quality-gate tier and is fully unit-testable.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.core.schemas import EntityRecord, EvidenceAtom, EvidenceEdge

# Tier thresholds on the count of *distinct source artifacts*.
_WELL_CORROBORATED = 3
_CORROBORATED = 2

SINGLE_SOURCE = "single_source"
CORROBORATED = "corroborated"
WELL_CORROBORATED = "well_corroborated"


def distinct_source_artifacts(atom: EvidenceAtom) -> set[str]:
    """The set of distinct artifact ids that back a single atom.

    Falls back to the atom's own ``artifact_id`` when its source_refs
    carry no artifact id (older atoms / synthetic atoms)."""
    sources: set[str] = set()
    for sref in getattr(atom, "source_refs", None) or []:
        aid = getattr(sref, "artifact_id", None)
        if aid:
            sources.add(str(aid))
    if not sources:
        aid = getattr(atom, "artifact_id", None)
        if aid:
            sources.add(str(aid))
    return sources


def corroboration_tier(n_sources: int) -> str:
    if n_sources >= _WELL_CORROBORATED:
        return WELL_CORROBORATED
    if n_sources >= _CORROBORATED:
        return CORROBORATED
    return SINGLE_SOURCE


def _contested_atom_ids(edges: Iterable[EvidenceEdge] | None) -> set[str]:
    """Atom ids that participate in a contradiction/exclusion edge — the
    facts where independent sources actively disagree."""
    contested: set[str] = set()
    for edge in edges or []:
        etype = getattr(edge, "edge_type", None)
        etype_val = getattr(etype, "value", etype)
        meta = getattr(edge, "metadata", None) or {}
        fam = meta.get("edge_family") if isinstance(meta, dict) else None
        is_conflict = etype_val in ("contradicts", "excludes") or (
            isinstance(fam, str) and ("contradict" in fam or "conflict" in fam)
        )
        if is_conflict:
            if getattr(edge, "from_atom_id", None):
                contested.add(edge.from_atom_id)
            if getattr(edge, "to_atom_id", None):
                contested.add(edge.to_atom_id)
    return contested


def _entity_clusters_from_atoms(
    atoms: list[EvidenceAtom],
) -> list[tuple[str, str, list[str]]]:
    """Fallback when no resolved entities are supplied: cluster atoms by
    their entity_keys. Returns ``(entity_id, canonical_key, atom_ids)``."""
    by_key: dict[str, list[str]] = {}
    for atom in atoms:
        for key in getattr(atom, "entity_keys", None) or []:
            if not key or key.endswith(":unknown"):
                continue
            by_key.setdefault(key, []).append(atom.id)
    return [(f"key::{k}", k, ids) for k, ids in by_key.items()]


def build_truth_gate(
    *,
    atoms: list[EvidenceAtom],
    entities: list[EntityRecord] | None = None,
    edges: list[EvidenceEdge] | None = None,
) -> dict[str, Any]:
    """Grade every entity by independent-source corroboration.

    Returns a rollup mirroring the other cockpit surfaces
    (``site_readiness`` / ``stakeholder_load``): a per-entity list plus
    aggregate counts and the weakest (single-source, multi-fact) entities
    a reviewer should corroborate first.
    """
    atom_by_id = {a.id: a for a in atoms}
    contested = _contested_atom_ids(edges)

    if entities:
        clusters = [
            (e.id, e.canonical_key, list(e.source_atom_ids or []))
            for e in entities
        ]
    else:
        clusters = _entity_clusters_from_atoms(atoms)

    rows: list[dict[str, Any]] = []
    for entity_id, canonical_key, atom_ids in clusters:
        sources: set[str] = set()
        is_contested = False
        for aid in atom_ids:
            atom = atom_by_id.get(aid)
            if atom is None:
                continue
            sources |= distinct_source_artifacts(atom)
            if aid in contested:
                is_contested = True
        n = len(sources)
        rows.append({
            "entity_id": entity_id,
            "canonical_key": canonical_key,
            "corroboration": n,
            "tier": corroboration_tier(n),
            "contested": is_contested,
            "source_atom_count": len(atom_ids),
        })

    rows.sort(key=lambda r: (r["corroboration"], -r["source_atom_count"], r["canonical_key"]))

    single = sum(1 for r in rows if r["tier"] == SINGLE_SOURCE)
    corrob = sum(1 for r in rows if r["tier"] == CORROBORATED)
    well = sum(1 for r in rows if r["tier"] == WELL_CORROBORATED)
    contested_count = sum(1 for r in rows if r["contested"])
    total = len(rows)

    # The reviewer's worklist: load-bearing facts (assert many atoms) that
    # rest on a single source — most valuable to independently confirm.
    weakest = [
        r["canonical_key"]
        for r in sorted(
            (r for r in rows if r["tier"] == SINGLE_SOURCE),
            key=lambda r: -r["source_atom_count"],
        )[:5]
        if r["source_atom_count"] >= 1
    ]

    return {
        "entities": rows,
        "entity_count": total,
        "single_source_count": single,
        "corroborated_count": corrob,
        "well_corroborated_count": well,
        "contested_count": contested_count,
        "single_source_share": round(single / total, 3) if total else 0.0,
        "avg_corroboration": round(
            sum(r["corroboration"] for r in rows) / total, 3
        ) if total else 0.0,
        "weakest_entities": weakest,
    }


__all__ = [
    "build_truth_gate",
    "corroboration_tier",
    "distinct_source_artifacts",
    "SINGLE_SOURCE",
    "CORROBORATED",
    "WELL_CORROBORATED",
]
