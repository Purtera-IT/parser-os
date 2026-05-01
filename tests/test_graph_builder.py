from __future__ import annotations

from app.core.graph_builder import build_edges
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(
    atom_id: str,
    *,
    atom_type: AtomType,
    authority: AuthorityClass,
    entity_keys: list[str],
    quantity: float | None = None,
    text: str = "text",
    value_extra: dict | None = None,
) -> EvidenceAtom:
    value: dict[str, object] = {"text": text}
    if quantity is not None:
        value["quantity"] = quantity
    if value_extra:
        value.update(value_extra)
    locator = {}
    if authority == AuthorityClass.quoted_old_email:
        locator["quoted"] = True
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value,
        entity_keys=entity_keys,
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator=locator,
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_site_specific_quantities_do_not_contradict() -> None:
    a1 = _atom(
        "q1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:main_campus", "device:ip_camera"],
        quantity=50,
        text="Main Campus IP Camera quantity 50",
    )
    a2 = _atom(
        "q2",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:west_wing", "device:ip_camera"],
        quantity=41,
        text="West Wing IP Camera quantity 41",
    )
    edges = build_edges("proj_1", [a1, a2], [])
    direct = [
        e
        for e in edges
        if e.edge_type == EdgeType.contradicts and {e.from_atom_id, e.to_atom_id} == {"q1", "q2"}
    ]
    assert not direct


def test_aggregate_scoped_quantity_contradicts_vendor_quantity() -> None:
    s1 = _atom(
        "s1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:main_campus", "device:ip_camera"],
        quantity=50,
        text="Main Campus quantity 50",
    )
    s2 = _atom(
        "s2",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:west_wing", "device:ip_camera"],
        quantity=41,
        text="West Wing quantity 41",
    )
    v1 = _atom(
        "v1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["device:ip_camera", "part:cam_ip_001"],
        quantity=72,
        text="Vendor quantity 72",
    )
    edges = build_edges("proj_1", [s1, s2, v1], [])
    contradictions = [e for e in edges if e.edge_type == EdgeType.contradicts]
    assert any(
        e.reason == "Aggregate scoped quantity 91 does not match vendor quantity 72 for device:ip_camera"
        for e in contradictions
    )


def test_different_plate_roster_line_items_same_material_no_contradiction() -> None:
    """Different plates are different scope rows; same site + same identity is not a conflict."""
    a1 = _atom(
        "p1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:spring_lake", "plate:avl_1"],
        quantity=1,
        text="RJ45 plate AVL-1",
        value_extra={"normalized_item": "rj45"},
    )
    a2 = _atom(
        "p2",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:spring_lake", "plate:avl_2"],
        quantity=2,
        text="RJ45 plate AVL-2",
        value_extra={"normalized_item": "rj45"},
    )
    edges = build_edges("proj_plates", [a1, a2], [])
    assert not any(e.edge_type == EdgeType.contradicts for e in edges)


def test_same_plate_roster_vs_customer_quantity_contradicts() -> None:
    """Same plate + same material + different qty across authorities is a real conflict."""
    roster = _atom(
        "r_pl",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:spring_lake", "plate:avl_1"],
        quantity=1,
        text="RJ45 AVL-1",
        value_extra={"normalized_item": "rj45"},
    )
    revised = _atom(
        "c_pl",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:spring_lake", "plate:avl_1"],
        quantity=2,
        text="RJ45 AVL-1 revised",
        value_extra={"normalized_item": "rj45"},
    )
    edges = build_edges("proj_same_plate", [roster, revised], [])
    assert any(e.edge_type == EdgeType.contradicts for e in edges)


def test_material_identity_roster_vendor_quantity_contradictions() -> None:
    roster_rj = _atom(
        "r_rj",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["connector:rj45"],
        quantity=72,
        text="RJ45 total",
        value_extra={"normalized_item": "rj45", "aggregate": True},
    )
    vendor_rj = _atom(
        "v_rj",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["part:line1"],
        quantity=68,
        text="vendor rj45",
        value_extra={"normalized_item": "rj45"},
    )
    roster_utp = _atom(
        "r_utp",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["material:cat6_utp"],
        quantity=66,
        text="utp total",
        value_extra={"normalized_item": "cat6_utp", "aggregate": True},
    )
    vendor_utp = _atom(
        "v_utp",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["part:utp"],
        quantity=60,
        text="vendor utp",
        value_extra={"normalized_item": "cat6_utp"},
    )
    roster_stp = _atom(
        "r_stp",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["cable:cat6_stp"],
        quantity=6,
        text="stp total",
        value_extra={"normalized_item": "cat6_stp", "aggregate": True},
    )
    vendor_stp = _atom(
        "v_stp",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["part:stp"],
        quantity=8,
        text="vendor stp",
        value_extra={"normalized_item": "cat6_stp"},
    )
    edges = build_edges(
        "proj_copper",
        [roster_rj, vendor_rj, roster_utp, vendor_utp, roster_stp, vendor_stp],
        [],
    )
    mat_edges = [
        e
        for e in edges
        if e.edge_type == EdgeType.contradicts
        and (e.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    assert len(mat_edges) == 3
    by_id = {(e.from_atom_id, e.to_atom_id): e for e in mat_edges}
    assert ("r_rj", "v_rj") in by_id
    assert ("r_utp", "v_utp") in by_id
    assert ("r_stp", "v_stp") in by_id
    assert all(e.from_atom_id.startswith("r_") for e in mat_edges)
    assert all(e.to_atom_id.startswith("v_") for e in mat_edges)
    for e in mat_edges:
        md = e.metadata
        assert md.get("identity") in {"rj45", "cat6_utp", "cat6_stp"}
        assert "roster_quantity" in md and "vendor_quantity" in md and "delta" in md
        assert md.get("preferred_packet_family") in ("quantity_conflict", "vendor_mismatch")


def test_material_identity_no_edge_when_totals_match() -> None:
    roster = _atom(
        "r1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["connector:rj45"],
        quantity=10,
        value_extra={"normalized_item": "rj45", "aggregate": True},
    )
    vendor = _atom(
        "v1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["part:p"],
        quantity=10,
        value_extra={"normalized_item": "rj45"},
    )
    edges = build_edges("p", [roster, vendor], [])
    mat = [
        e
        for e in edges
        if (e.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    assert not mat


def test_material_identity_does_not_cross_link_items() -> None:
    roster_rj = _atom(
        "r_rj",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=[],
        quantity=72,
        value_extra={"normalized_item": "rj45", "aggregate": True},
    )
    vendor_utp = _atom(
        "v_utp",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=[],
        quantity=60,
        value_extra={"normalized_item": "cat6_utp"},
    )
    edges = build_edges("p", [roster_rj, vendor_utp], [])
    mat = [
        e
        for e in edges
        if (e.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    assert not mat


def test_vendor_optional_line_excluded_from_primary_material_total() -> None:
    """Optional vendor rows must not inflate primary vendor total used for roster comparison."""
    roster = _atom(
        "r_rj",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:sl"],
        quantity=100,
        text="rj45 roster",
        value_extra={"normalized_item": "rj45", "aggregate": True},
    )
    base = _atom(
        "v_base",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:sl"],
        quantity=10,
        text="included rj45",
        value_extra={"normalized_item": "rj45", "inclusion_status": "included"},
    )
    optional_line = _atom(
        "v_opt",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:sl"],
        quantity=999,
        text="optional rj45",
        value_extra={"normalized_item": "rj45", "inclusion_status": "optional"},
    )
    edges = build_edges("proj_opt", [roster, base, optional_line], [])
    mat = [
        e
        for e in edges
        if e.edge_type == EdgeType.contradicts
        and (e.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    assert len(mat) == 1
    e = mat[0]
    assert e.metadata.get("vendor_quantity") == 10.0
    assert e.metadata.get("roster_quantity") == 100.0
    assert "v_base" in e.metadata.get("vendor_atom_ids", [])
    assert "v_opt" in e.metadata.get("vendor_excluded_atom_ids", [])


def test_exclusion_creates_excludes_edge_and_constraint_requires() -> None:
    exclusion = _atom(
        "ex1",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Exclude west wing cameras",
    )
    scope = _atom(
        "sc1",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Install west wing cameras",
    )
    constraint = _atom(
        "ct1",
        atom_type=AtomType.constraint,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:west_wing"],
        text="Escort required at west wing",
    )
    qty = _atom(
        "q1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:west_wing", "device:ip_camera"],
        quantity=41,
        text="West wing quantity 41",
    )

    edges = build_edges("proj_1", [exclusion, scope, constraint, qty], [])
    assert any(e.edge_type == EdgeType.excludes and e.from_atom_id == "ex1" and e.to_atom_id == "sc1" for e in edges)
    assert any(e.edge_type == EdgeType.requires and e.from_atom_id == "ct1" for e in edges)
    assert all(e.reason for e in edges)
    assert all(e.confidence >= 0.0 for e in edges)


def test_transcript_exclusion_creates_excludes_edge_against_scope() -> None:
    transcript_exclusion = _atom(
        "tx_ex1",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="West Wing removed from scope.",
    )
    roster_scope = _atom(
        "rs_scope",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Install west wing cameras",
    )
    edges = build_edges("proj_1", [transcript_exclusion, roster_scope], [])
    assert any(e.edge_type == EdgeType.excludes and e.from_atom_id == "tx_ex1" for e in edges)


def test_transcript_constraint_creates_requires_edge() -> None:
    transcript_constraint = _atom(
        "tx_c1",
        atom_type=AtomType.constraint,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus"],
        text="Main Campus requires escort access after 5pm.",
    )
    scope_item = _atom(
        "sc_main",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:main_campus", "device:ip_camera"],
        text="Install cameras main campus",
    )
    edges = build_edges("proj_1", [transcript_constraint, scope_item], [])
    assert any(e.edge_type == EdgeType.requires and e.from_atom_id == "tx_c1" for e in edges)


def test_transcript_quantity_can_conflict_with_existing_quantity() -> None:
    transcript_qty = _atom(
        "tx_q",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus", "device:ip_camera"],
        quantity=5,
        text="Add 5 more IP cameras at Main Campus",
    )
    roster_qty = _atom(
        "rs_q",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:main_campus", "device:ip_camera"],
        quantity=50,
        text="Main campus quantity 50",
    )
    edges = build_edges("proj_1", [transcript_qty, roster_qty], [])
    assert any(e.edge_type == EdgeType.contradicts for e in edges)


def test_transcript_open_question_does_not_create_false_scope_inclusion_edge() -> None:
    open_q = _atom(
        "tx_oq",
        atom_type=AtomType.open_question,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus"],
        text="Confirm whether MDF room requires badge access?",
    )
    edges = build_edges("proj_1", [open_q], [])
    assert not any(e.edge_type in {EdgeType.excludes, EdgeType.requires, EdgeType.supports} for e in edges)


def test_semantic_edges_include_metadata_and_no_contradictions() -> None:
    e1 = _atom(
        "e1",
        atom_type=AtomType.entity,
        authority=AuthorityClass.machine_extractor,
        entity_keys=["device:ip_camera"],
        text="IP Camera",
    )
    e2 = _atom(
        "e2",
        atom_type=AtomType.entity,
        authority=AuthorityClass.machine_extractor,
        entity_keys=["device:ip_camera"],
        text="security camera",
    )
    edges = build_edges("proj_1", [e1, e2], [])
    semantic_edges = [edge for edge in edges if "semantic_candidate_linker" in edge.reason.lower()]
    assert semantic_edges
    assert all("method=" in edge.reason.lower() for edge in semantic_edges)
    assert all(edge.edge_type != EdgeType.contradicts for edge in semantic_edges)
