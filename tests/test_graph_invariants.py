from __future__ import annotations

from app.core.graph_invariants import check_graph_invariants
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EvidenceAtom,
    EvidenceEdge,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_id: str, atom_type: AtomType) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=f"{atom_type.value} text",
        normalized_text=f"{atom_type.value} text",
        value={},
        entity_keys=["site:west_wing"],
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=AuthorityClass.approved_site_roster,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def _edge(edge_id: str, edge_type: EdgeType, from_id: str, to_id: str, reason: str) -> EvidenceEdge:
    return EvidenceEdge(
        id=edge_id,
        project_id="proj_1",
        from_atom_id=from_id,
        to_atom_id=to_id,
        edge_type=edge_type,
        reason=reason,
        confidence=0.9,
    )


def test_graph_invariant_missing_atom_refs_detected() -> None:
    a1 = _atom("a1", AtomType.scope_item)
    edges = [_edge("e1", EdgeType.supports, "a1", "missing", "support")]
    errors = check_graph_invariants([a1], edges)
    assert any("missing to_atom_id" in error for error in errors)


def test_excludes_without_exclusion_atom_fails() -> None:
    a1 = _atom("a1", AtomType.scope_item)
    a2 = _atom("a2", AtomType.scope_item)
    edges = [_edge("e1", EdgeType.excludes, "a1", "a2", "bad excludes")]
    errors = check_graph_invariants([a1, a2], edges)
    assert any("must involve exclusion atom" in error for error in errors)


def test_requires_without_constraint_atom_fails() -> None:
    a1 = _atom("a1", AtomType.scope_item)
    a2 = _atom("a2", AtomType.quantity)
    edges = [_edge("e1", EdgeType.requires, "a1", "a2", "bad requires")]
    errors = check_graph_invariants([a1, a2], edges)
    assert any("must involve constraint atom" in error for error in errors)
