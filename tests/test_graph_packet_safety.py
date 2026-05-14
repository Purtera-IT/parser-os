"""Regression tests for PR9 graph + packet safety helpers.

These guard the new gates that prevent:
  * quantity_conflict packets backed by < 2 quantity atoms,
  * quantity_conflict packets without a contradicts edge,
  * quantity_conflict packets with single-source provenance,
  * scope_exclusion packets without explicit exclusion evidence,
  * Base vs Add-Alternate quantity contradictions in graph_builder.
"""
from __future__ import annotations

from app.core.graph_builder import (
    _quantity_atoms_are_comparable,
    _scope_dimension,
    _scope_dimensions_compatible,
)
from app.core.packetizer import (
    _valid_quantity_conflict_group,
    _valid_scope_exclusion_group,
)
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


def _qty(
    *,
    aid: str = "atm1",
    artifact: str = "art1",
    authority: AuthorityClass = AuthorityClass.contractual_scope,
    text: str = "186 drops",
    qty: float = 186,
    entity_keys: list[str] | None = None,
) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid,
        project_id="P",
        artifact_id=artifact,
        atom_type=AtomType.quantity,
        raw_text=text,
        normalized_text=text.lower(),
        value={"quantity": qty},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id=f"src_{aid}",
                artifact_id=artifact,
                artifact_type=ArtifactType.txt,
                filename="x.txt",
                locator={"line": 1},
                extraction_method="t",
                parser_version="t",
            )
        ],
        receipts=[],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="t",
    )


def _exclusion_atom(text: str) -> EvidenceAtom:
    a = _qty(aid="atmEx", text=text)
    a.atom_type = AtomType.exclusion
    return a


def _contradicts_edge(a_id: str, b_id: str) -> EvidenceEdge:
    return EvidenceEdge(
        id="e1",
        project_id="P",
        from_atom_id=a_id,
        to_atom_id=b_id,
        edge_type=EdgeType.contradicts,
        reason="qty mismatch",
        confidence=0.9,
        metadata={},
    )


def test_quantity_conflict_requires_two_quantity_atoms():
    a = _qty(aid="a")
    not_qty = _qty(aid="b")
    not_qty.atom_type = AtomType.scope_item
    edge = _contradicts_edge("a", "b")
    assert _valid_quantity_conflict_group([a, not_qty], [edge]) is False


def test_quantity_conflict_requires_contradicts_edge():
    a = _qty(aid="a", artifact="x")
    b = _qty(aid="b", artifact="y")
    assert _valid_quantity_conflict_group([a, b], []) is False


def test_quantity_conflict_requires_multi_source_provenance():
    """Same artifact AND same authority_class → no conflict packet."""
    a = _qty(aid="a", artifact="X", authority=AuthorityClass.vendor_quote)
    b = _qty(aid="b", artifact="X", authority=AuthorityClass.vendor_quote)
    edge = _contradicts_edge("a", "b")
    assert _valid_quantity_conflict_group([a, b], [edge]) is False


def test_quantity_conflict_accepts_two_artifacts():
    a = _qty(aid="a", artifact="X")
    b = _qty(aid="b", artifact="Y")
    edge = _contradicts_edge("a", "b")
    assert _valid_quantity_conflict_group([a, b], [edge]) is True


def test_quantity_conflict_accepts_two_authorities():
    a = _qty(aid="a", artifact="X", authority=AuthorityClass.vendor_quote)
    b = _qty(
        aid="b",
        artifact="X",
        authority=AuthorityClass.approved_site_roster,
    )
    edge = _contradicts_edge("a", "b")
    assert _valid_quantity_conflict_group([a, b], [edge]) is True


def test_scope_exclusion_requires_explicit_evidence():
    only_quantities = [_qty(aid="a"), _qty(aid="b")]
    assert _valid_scope_exclusion_group(only_quantities) is False


def test_scope_exclusion_accepts_exclusion_atom():
    g = [_exclusion_atom("Fire alarm work is excluded")]
    assert _valid_scope_exclusion_group(g) is True


def test_scope_exclusion_rejects_quantity_with_explicit_text():
    """PR3 post-v3 — only ``exclusion`` atoms can govern a
    scope_exclusion packet. A ``quantity`` atom that happens to
    contain "not in scope" text in its raw_text is NOT enough —
    we want a real exclusion atom to anchor the packet."""
    a = _qty(text="186 drops not in scope")
    assert _valid_scope_exclusion_group([a]) is False


def test_scope_exclusion_accepts_inclusion_status_field():
    a = _qty(text="Camera install")
    a.value["inclusion_status"] = "excluded"
    assert _valid_scope_exclusion_group([a]) is True


def test_scope_dimension_detects_alternate_and_base():
    base = _qty(text="Base bid: 186 drops")
    alt = _qty(text="Add Alternate: 12 additional drops")
    assert _scope_dimension(base) == "base"
    assert _scope_dimension(alt) == "alternate"


def test_scope_dimensions_compatible_blocks_base_vs_alt():
    base = _qty(text="Base bid: 186 drops")
    alt = _qty(text="Add Alternate: 12 additional drops")
    assert _scope_dimensions_compatible(base, alt) is False


def test_scope_dimensions_compatible_allows_unspecified():
    a = _qty(text="186 drops")
    b = _qty(text="Add Alternate: 12 drops")
    assert _scope_dimensions_compatible(a, b) is True


def test_quantity_atoms_are_comparable_blocks_base_vs_alt():
    a = _qty(aid="a", text="Base: 186 drops", entity_keys=["site:s1"])
    b = _qty(aid="b", text="Add Alternate: 12 drops", entity_keys=["site:s1"])
    assert _quantity_atoms_are_comparable(a, b) is False


def test_quantity_atoms_are_comparable_blocks_disjoint_entities():
    a = _qty(aid="a", entity_keys=["site:school_a"])
    b = _qty(aid="b", entity_keys=["site:school_b"])
    assert _quantity_atoms_are_comparable(a, b) is False
