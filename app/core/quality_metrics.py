"""Compile-quality metrics (PRODUCTION_GAPS.md P3.4).

Computes a ``CompileQuality`` record from a freshly-built ``CompileResult``
plus its parser-routing telemetry.  Every metric is a derived view —
nothing in here mutates atoms, entities, edges, or packets.

A separate module (rather than a method on the model) so we can call it
from both ``compile_project`` and tests / one-off audit scripts without
having to re-instantiate a full pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.core.schemas import (
    CompileManifest,
    CompileQuality,
    CompileResult,
    CompileTrace,
)


_LOW_PARSER_CONFIDENCE_THRESHOLD = 0.50


def _atom_has_real_entity_key(atom: Any) -> bool:
    keys = getattr(atom, "entity_keys", None) or []
    if not keys:
        return False
    for k in keys:
        if k and not k.endswith(":unknown"):
            return True
    return False


def _packet_has_real_anchor(packet: Any) -> bool:
    anchor_key = getattr(packet, "anchor_key", None) or ""
    anchor_type = getattr(packet, "anchor_type", None) or ""
    if not anchor_key:
        return False
    if anchor_key.endswith(":unknown"):
        return False
    flags = getattr(packet, "review_flags", None) or []
    if "unknown_anchor" in flags:
        return False
    if anchor_type == "" and ":" not in anchor_key:
        return False
    return True


def _stage_durations_ms(trace: CompileTrace | None) -> dict[str, float]:
    """Extract per-stage wall time from the compile trace.

    The CompileTrace stage model exposes the stage name on the
    ``stage_name`` attribute (not ``stage`` — that was a bug in the
    initial Week 4 implementation that produced empty durations).
    """
    if trace is None:
        return {}
    durations: dict[str, float] = {}
    for stage in trace.stages or []:
        # Try both attribute names so future schema renames don't
        # silently drop telemetry.
        name = getattr(stage, "stage_name", None) or getattr(stage, "stage", None)
        ms = getattr(stage, "duration_ms", None)
        if name is not None and ms is not None:
            durations[str(name)] = float(ms)
    return durations


def compute_quality(
    result: CompileResult,
    *,
    pack_routing_source: str = "unknown",
    pack_routing_confidence: float = 0.0,
) -> CompileQuality:
    """Build a :class:`CompileQuality` from a finished compile result."""
    atoms = list(result.atoms or [])
    entities = list(result.entities or [])
    edges = list(result.edges or [])
    packets = list(result.packets or [])
    manifest: CompileManifest | None = result.manifest

    atom_count = len(atoms)
    entity_count = len(entities)
    edge_count = len(edges)
    packet_count = len(packets)

    qty_conflict_count = 0
    cross_artifact_count = 0
    for edge in edges:
        meta = getattr(edge, "metadata", None) or {}
        if isinstance(meta, dict):
            family = meta.get("edge_family")
            if family == "part_number_quantity_conflict":
                qty_conflict_count += 1
            elif family == "quantity_contradiction":
                qty_conflict_count += 1
            if meta.get("cross_artifact"):
                cross_artifact_count += 1

    if atom_count > 0:
        with_real = sum(1 for a in atoms if _atom_has_real_entity_key(a))
        entity_resolution_rate = with_real / atom_count
    else:
        entity_resolution_rate = 0.0

    if packet_count > 0:
        real_anchored = sum(1 for p in packets if _packet_has_real_anchor(p))
        packet_specificity = real_anchored / packet_count
    else:
        packet_specificity = 0.0

    routing: list[dict[str, Any]] = []
    if manifest is not None:
        routing = list(getattr(manifest, "parser_routing", None) or [])
    if routing:
        total_conf = sum(float(r.get("confidence") or 0.0) for r in routing)
        parser_routing_confidence_avg = total_conf / len(routing)
    else:
        parser_routing_confidence_avg = 0.0

    # Per-parser atom yields.
    parser_atoms: dict[str, int] = Counter()
    for atom in atoms:
        version = getattr(atom, "parser_version", None)
        if not version:
            continue
        parser_atoms[str(version)] += 1
    parsers_with_zero_atoms: list[str] = []
    parsers_with_low_confidence: list[str] = []
    nonzero_parsers = 0
    total_parsers_seen = 0
    for entry in routing:
        chosen = str(entry.get("chosen_parser") or "")
        if not chosen or chosen == "none":
            continue
        total_parsers_seen += 1
        version = str(entry.get("parser_version") or "")
        atoms_for_parser = parser_atoms.get(version, 0)
        if atoms_for_parser == 0:
            parsers_with_zero_atoms.append(
                f"{entry.get('filename', '?')} ({chosen} v{version})"
            )
        else:
            nonzero_parsers += 1
        if float(entry.get("confidence") or 0.0) < _LOW_PARSER_CONFIDENCE_THRESHOLD:
            parsers_with_low_confidence.append(
                f"{entry.get('filename', '?')} ({chosen} conf={entry.get('confidence')})"
            )
    parser_atom_yield_rate = (
        nonzero_parsers / total_parsers_seen if total_parsers_seen else 0.0
    )

    artifact_count = len(routing) if routing else 0
    atoms_per_artifact = atom_count / artifact_count if artifact_count else 0.0

    pack_id = "unknown"
    if manifest is not None and manifest.domain_pack_id:
        pack_id = manifest.domain_pack_id

    return CompileQuality(
        atom_count=atom_count,
        packet_count=packet_count,
        edge_count=edge_count,
        entity_count=entity_count,
        quantity_conflict_edge_count=qty_conflict_count,
        cross_artifact_edge_count=cross_artifact_count,
        entity_resolution_rate=round(entity_resolution_rate, 4),
        packet_specificity=round(packet_specificity, 4),
        parser_routing_confidence_avg=round(parser_routing_confidence_avg, 4),
        parser_atom_yield_rate=round(parser_atom_yield_rate, 4),
        atoms_per_artifact=round(atoms_per_artifact, 4),
        pack_id=pack_id,
        pack_routing_source=pack_routing_source,
        pack_routing_confidence=round(float(pack_routing_confidence), 4),
        stage_durations_ms=_stage_durations_ms(result.trace),
        parsers_with_zero_atoms=sorted(set(parsers_with_zero_atoms)),
        parsers_with_low_confidence=sorted(set(parsers_with_low_confidence)),
    )


__all__ = ["compute_quality"]
