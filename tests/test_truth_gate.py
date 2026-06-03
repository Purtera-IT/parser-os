"""Corroboration-graded truth (Gap F — Truth Gate).

A fact backed by three independent documents must grade higher than the
same fact asserted once. These tests pin the per-entity corroboration
count, the tier banding, contested-fact detection, and the entity-less
fallback that clusters by entity_key.
"""

from __future__ import annotations

from app.core.truth_gate import (
    CORROBORATED,
    SINGLE_SOURCE,
    WELL_CORROBORATED,
    build_truth_gate,
    corroboration_tier,
    distinct_source_artifacts,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EntityRecord,
    EvidenceAtom,
    EvidenceEdge,
    ReviewStatus,
    SourceRef,
)


def _atom(rid, *, artifact_ids, entity_keys=None, text="some fact"):
    return EvidenceAtom(
        id=rid,
        project_id="p",
        artifact_id=artifact_ids[0],
        atom_type=AtomType.requirement,
        raw_text=text,
        normalized_text=text.lower(),
        value={},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id=f"src_{i}",
                artifact_id=aid,
                artifact_type=ArtifactType.txt,
                filename=f"{aid}.txt",
                locator={},
                extraction_method="test",
                parser_version="t",
            )
            for i, aid in enumerate(artifact_ids)
        ],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        confidence_raw=0.8,
        calibrated_confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


def _entity(eid, key, atom_ids):
    return EntityRecord(
        id=eid,
        project_id="p",
        entity_type="device",
        canonical_key=key,
        canonical_name=key.split(":", 1)[-1],
        aliases=[],
        source_atom_ids=atom_ids,
        confidence=1.0,
        review_status=ReviewStatus.auto_accepted,
    )


def test_tier_thresholds():
    assert corroboration_tier(0) == SINGLE_SOURCE
    assert corroboration_tier(1) == SINGLE_SOURCE
    assert corroboration_tier(2) == CORROBORATED
    assert corroboration_tier(3) == WELL_CORROBORATED
    assert corroboration_tier(9) == WELL_CORROBORATED


def test_distinct_source_artifacts_dedupes():
    atom = _atom("a1", artifact_ids=["doc1", "doc1", "doc2"])
    assert distinct_source_artifacts(atom) == {"doc1", "doc2"}


def test_entity_corroboration_counts_distinct_artifacts():
    atoms = [
        _atom("a1", artifact_ids=["doc1"], entity_keys=["device:display"]),
        _atom("a2", artifact_ids=["doc2"], entity_keys=["device:display"]),
        _atom("a3", artifact_ids=["doc3"], entity_keys=["device:display"]),
    ]
    entities = [_entity("e1", "device:display", ["a1", "a2", "a3"])]
    gate = build_truth_gate(atoms=atoms, entities=entities, edges=[])
    assert gate["entity_count"] == 1
    row = gate["entities"][0]
    assert row["corroboration"] == 3
    assert row["tier"] == WELL_CORROBORATED
    assert gate["well_corroborated_count"] == 1
    assert gate["single_source_count"] == 0


def test_single_source_entity_is_flagged_weak():
    atoms = [
        _atom("a1", artifact_ids=["doc1"], entity_keys=["device:rare"]),
        _atom("a2", artifact_ids=["doc1"], entity_keys=["device:rare"]),
    ]
    entities = [_entity("e1", "device:rare", ["a1", "a2"])]
    gate = build_truth_gate(atoms=atoms, entities=entities, edges=[])
    row = gate["entities"][0]
    assert row["corroboration"] == 1  # same artifact twice = still one source
    assert row["tier"] == SINGLE_SOURCE
    assert gate["single_source_count"] == 1
    assert "device:rare" in gate["weakest_entities"]
    assert gate["single_source_share"] == 1.0


def test_contradiction_edge_marks_entity_contested():
    atoms = [
        _atom("a1", artifact_ids=["doc1"], entity_keys=["device:display"]),
        _atom("a2", artifact_ids=["doc2"], entity_keys=["device:display"]),
    ]
    entities = [_entity("e1", "device:display", ["a1", "a2"])]
    edge = EvidenceEdge(
        id="edg1",
        project_id="p",
        from_atom_id="a1",
        to_atom_id="a2",
        edge_type=EdgeType.contradicts,
        reason="qty mismatch",
        confidence=0.9,
        metadata={},
    )
    gate = build_truth_gate(atoms=atoms, entities=entities, edges=[edge])
    assert gate["entities"][0]["contested"] is True
    assert gate["contested_count"] == 1


def test_fallback_clusters_by_entity_key_without_entities():
    atoms = [
        _atom("a1", artifact_ids=["doc1"], entity_keys=["device:display"]),
        _atom("a2", artifact_ids=["doc2"], entity_keys=["device:display"]),
        _atom("a3", artifact_ids=["doc1"], entity_keys=["device:unknown"]),
    ]
    gate = build_truth_gate(atoms=atoms, entities=None, edges=[])
    keys = {r["canonical_key"] for r in gate["entities"]}
    assert "device:display" in keys
    assert "device:unknown" not in keys  # :unknown clusters are skipped
    row = next(r for r in gate["entities"] if r["canonical_key"] == "device:display")
    assert row["corroboration"] == 2
    assert row["tier"] == CORROBORATED


def test_empty_inputs_safe():
    gate = build_truth_gate(atoms=[], entities=[], edges=[])
    assert gate["entity_count"] == 0
    assert gate["avg_corroboration"] == 0.0
    assert gate["single_source_share"] == 0.0
    assert gate["weakest_entities"] == []
