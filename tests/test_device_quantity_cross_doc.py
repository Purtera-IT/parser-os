"""Unit tests for the cross-artifact device-only quantity conflict
detector (``EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC``).

The detector pairs atoms across DIFFERENT artifacts that share a
``device:*`` key and have different ``quantity:N`` keys when no
``part_number:*`` anchors a part-level conflict.

Strict false-positive guards:
  - cross-artifact required
  - >1 device key per atom: skip (ambiguous binding)
  - >1 quantity key per atom: skip (ambiguous binding)
  - shared part_number: skip (part-number path handles it)
  - disjoint sites: skip (independent deployments)
  - quantity < 5: skip (template / "1 each" noise)
"""
from __future__ import annotations

from app.core.graph_builder import (
    EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC,
    build_edges,
)
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(
    *,
    id: str,
    artifact_id: str,
    text: str,
    atom_type: AtomType = AtomType.scope_item,
    authority: AuthorityClass = AuthorityClass.contractual_scope,
    entity_keys: list[str] | None = None,
) -> EvidenceAtom:
    """Build a minimal EvidenceAtom for graph_builder tests."""
    return EvidenceAtom(
        id=id,
        project_id="test_project",
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value={"kind": "paragraph"},
        entity_keys=sorted(entity_keys or []),
        source_refs=[
            SourceRef(
                id=stable_id("src", id),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.txt,
                filename=f"{artifact_id}.txt",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        receipts=[],
        authority_class=authority,
        confidence=0.9,
        confidence_raw=0.9,
        calibrated_confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test_v1",
    )


def _edge_families(edges, family: str) -> list:
    return [e for e in edges if (e.metadata or {}).get("edge_family") == family]


# ── Positive case: real BOM vs SOW conflict ──────────────────────


def test_conflict_fires_for_bom_vs_sow_same_site():
    a = _atom(
        id=stable_id("atm", "a"),
        artifact_id="art_bom",
        text="PurTera shall ship 50 wireless access points to ATL-HQ-01.",
        authority=AuthorityClass.vendor_quote,
        entity_keys=["device:access_point", "quantity:50", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b"),
        artifact_id="art_sow",
        text="PurTera will install 60 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:60", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    conflicts = _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)
    assert len(conflicts) == 1
    assert "device:access_point" in conflicts[0].reason
    assert "50" in conflicts[0].reason and "60" in conflicts[0].reason


# ── Negative cases (all guards must hold) ───────────────────────


def test_no_conflict_when_same_artifact():
    a = _atom(
        id=stable_id("atm", "a1"),
        artifact_id="art_same",
        text="50 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b1"),
        artifact_id="art_same",
        text="60 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:60", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_no_conflict_when_disjoint_sites():
    a = _atom(
        id=stable_id("atm", "a2"),
        artifact_id="art_bom2",
        text="50 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b2"),
        artifact_id="art_sow2",
        text="60 access points at ATL-WEST-02.",
        entity_keys=["device:access_point", "quantity:60", "site:atl_west_02"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_no_conflict_when_atom_has_multiple_devices():
    # Same atom names BOTH access_point AND switch — quantity binds
    # ambiguously, so the detector skips.
    a = _atom(
        id=stable_id("atm", "a3"),
        artifact_id="art_bom3",
        text="50 access points and 5 switches at ATL-HQ-01.",
        entity_keys=["device:access_point", "device:switch", "quantity:50", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b3"),
        artifact_id="art_sow3",
        text="60 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:60", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_no_conflict_when_atom_has_multiple_quantities():
    a = _atom(
        id=stable_id("atm", "a4"),
        artifact_id="art_bom4",
        text="50 access points and 60 cables at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "quantity:60", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b4"),
        artifact_id="art_sow4",
        text="70 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:70", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_no_conflict_when_shared_part_number():
    # When both atoms cite the SAME part number, the part-number
    # conflict path handles it — device-only conflict should not fire.
    a = _atom(
        id=stable_id("atm", "a5"),
        artifact_id="art_bom5",
        text="50 WAP-9180AX at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "part_number:wap_9180ax", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b5"),
        artifact_id="art_sow5",
        text="60 WAP-9180AX at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:60", "part_number:wap_9180ax", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_no_conflict_when_quantity_below_threshold():
    # qty=1 or 2 is too generic — likely template "1 each" rows.
    a = _atom(
        id=stable_id("atm", "a6"),
        artifact_id="art_bom6",
        text="1 access point sample at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:1", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b6"),
        artifact_id="art_sow6",
        text="2 access points spare at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:2", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_no_conflict_when_same_quantity():
    # If both quantities agree, no contradiction.
    a = _atom(
        id=stable_id("atm", "a7"),
        artifact_id="art_bom7",
        text="50 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b7"),
        artifact_id="art_sow7",
        text="50 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "site:atl_hq_01"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)


def test_conflict_works_when_only_one_atom_has_site():
    # If one atom is site-scoped and the other isn't, the site-scope
    # guard should NOT block the conflict (unknown site != disjoint).
    a = _atom(
        id=stable_id("atm", "a8"),
        artifact_id="art_bom8",
        text="50 access points at ATL-HQ-01.",
        entity_keys=["device:access_point", "quantity:50", "site:atl_hq_01"],
    )
    b = _atom(
        id=stable_id("atm", "b8"),
        artifact_id="art_sow8",
        text="60 access points total.",
        entity_keys=["device:access_point", "quantity:60"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    conflicts = _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)
    assert len(conflicts) == 1


def test_conflict_skips_when_device_is_unknown():
    a = _atom(
        id=stable_id("atm", "a9"),
        artifact_id="art_bom9",
        text="50 items.",
        entity_keys=["device:unknown", "quantity:50"],
    )
    b = _atom(
        id=stable_id("atm", "b9"),
        artifact_id="art_sow9",
        text="60 items.",
        entity_keys=["device:unknown", "quantity:60"],
    )
    edges = build_edges(project_id="test_project", atoms=[a, b], entities=[])
    assert not _edge_families(edges, EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC)
