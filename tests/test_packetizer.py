from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.core.packetizer import build_packets
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    PacketFamily,
    PacketStatus,
    ReviewStatus,
    EvidenceAtom,
    EvidenceEdge,
    SourceRef,
)


def _atom(
    atom_id: str,
    *,
    atom_type: AtomType,
    authority: AuthorityClass,
    entity_keys: list[str],
    text: str,
    confidence: float = 0.9,
    quantity: float | None = None,
    value_extra: dict | None = None,
) -> EvidenceAtom:
    value = {"text": text}
    if quantity is not None:
        value["quantity"] = quantity
    if value_extra:
        value.update(value_extra)
    locator = {"quoted": authority == AuthorityClass.quoted_old_email}
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
                id=f"src_{atom_id}",
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator=locator,
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=authority,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def _edge(
    edge_id: str,
    edge_type: EdgeType,
    from_id: str,
    to_id: str,
    reason: str,
    *,
    metadata: dict | None = None,
) -> EvidenceEdge:
    return EvidenceEdge(
        id=edge_id,
        project_id="proj_1",
        from_atom_id=from_id,
        to_atom_id=to_id,
        edge_type=edge_type,
        reason=reason,
        confidence=0.9,
        metadata=dict(metadata or {}),
    )


def test_packetizer_v0_conflicts_and_governing_rules() -> None:
    scope_west = _atom(
        "scope_west",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Install cameras at west wing",
    )
    exclusion_customer = _atom(
        "excl_customer",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Please remove west wing from scope",
    )
    exclusion_quoted = _atom(
        "excl_quoted",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.quoted_old_email,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Include west wing",
    )
    qty_approved = _atom(
        "qty_approved",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["device:ip_camera", "site:main_campus"],
        text="Scoped qty 91",
        quantity=91,
    )
    qty_vendor = _atom(
        "qty_vendor",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["device:ip_camera", "part:cam_ip_001"],
        text="Vendor qty 72",
        quantity=72,
    )
    access = _atom(
        "access_1",
        atom_type=AtomType.constraint,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:main_campus"],
        text="Escort access required after 5pm",
    )
    deleted = _atom(
        "deleted_1",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.deleted_text,
        entity_keys=["site:west_wing"],
        text="Install AV displays",
    )

    edges = [
        _edge(
            "e_contra_qty",
            EdgeType.contradicts,
            "qty_approved",
            "qty_vendor",
            "Aggregate scoped quantity 91 does not match vendor quantity 72 for device:ip_camera",
        ),
        _edge("e_excludes", EdgeType.excludes, "excl_customer", "scope_west", "Exclusion applies"),
    ]

    packets = build_packets(
        project_id="proj_1",
        atoms=[scope_west, exclusion_customer, exclusion_quoted, qty_approved, qty_vendor, access, deleted],
        entities=[],
        edges=edges,
    )
    families = {p.family for p in packets}
    assert PacketFamily.quantity_conflict in families
    assert PacketFamily.vendor_mismatch in families
    assert PacketFamily.scope_exclusion in families
    assert PacketFamily.site_access in families

    scope_exclusion_packet = next(p for p in packets if p.family == PacketFamily.scope_exclusion)
    assert scope_exclusion_packet.governing_atom_ids == ["excl_customer"]
    assert "exclusion_present" in scope_exclusion_packet.review_flags
    assert "excl_quoted" in (scope_exclusion_packet.supporting_atom_ids + scope_exclusion_packet.contradicting_atom_ids)

    assert "deleted_1" not in scope_exclusion_packet.governing_atom_ids
    assert all(len(p.supporting_atom_ids) + len(p.contradicting_atom_ids) > 0 for p in packets)
    assert all(
        not (p.status in {PacketStatus.active, PacketStatus.needs_review} and not p.governing_atom_ids)
        for p in packets
    )
    assert all(p.certificate is not None for p in packets)
    assert all(p.risk is not None for p in packets)
    assert all(p.anchor_signature is not None for p in packets)
    qty_packet = next(p for p in packets if p.family == PacketFamily.quantity_conflict)
    assert qty_packet.certificate is not None
    assert "91" in qty_packet.certificate.existence_reason and "72" in qty_packet.certificate.existence_reason
    assert qty_packet.certificate.authority_path
    assert "dimensions" in qty_packet.certificate.authority_path[0]
    assert 0.0 <= qty_packet.risk.risk_score <= 1.0


def test_transcript_packets_and_governance_rules() -> None:
    email_exclusion = _atom(
        "email_excl",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="Please remove west wing from scope.",
    )
    transcript_exclusion = _atom(
        "tx_excl",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:west_wing", "device:ip_camera"],
        text="West wing removed from scope for now.",
    )
    transcript_access = _atom(
        "tx_access",
        atom_type=AtomType.constraint,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus"],
        text="Main Campus requires escort access after 5pm.",
    )
    transcript_open_q = _atom(
        "tx_q",
        atom_type=AtomType.open_question,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus"],
        text="Confirm whether MDF room requires badge access?",
    )
    transcript_action = _atom(
        "tx_action",
        atom_type=AtomType.action_item,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus"],
        text="Customer to provide lift access.",
    )
    transcript_decision = _atom(
        "tx_decision",
        atom_type=AtomType.decision,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:west_wing"],
        text="Decision: West Wing removed from scope.",
    )
    transcript_qty = _atom(
        "tx_qty",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:main_campus", "device:ip_camera"],
        text="We may add 5 more IP cameras at Main Campus.",
        quantity=5,
    )
    roster_qty = _atom(
        "roster_qty",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:main_campus", "device:ip_camera"],
        text="Main Campus quantity 50",
        quantity=50,
    )
    edges = [
        _edge("e_ex", EdgeType.excludes, "tx_excl", "email_excl", "Transcript exclusion supports removal"),
        _edge("e_con", EdgeType.contradicts, "tx_qty", "roster_qty", "Quantity mismatch 5 vs 50"),
    ]

    packets = build_packets(
        project_id="proj_1",
        atoms=[
            email_exclusion,
            transcript_exclusion,
            transcript_access,
            transcript_open_q,
            transcript_action,
            transcript_decision,
            transcript_qty,
            roster_qty,
        ],
        entities=[],
        edges=edges,
    )

    families = {p.family for p in packets}
    assert PacketFamily.scope_exclusion in families
    assert PacketFamily.site_access in families
    assert PacketFamily.missing_info in families
    assert PacketFamily.action_item in families
    assert PacketFamily.meeting_decision in families or PacketFamily.quantity_conflict in families

    scope_packet = next(p for p in packets if p.family == PacketFamily.scope_exclusion)
    assert scope_packet.governing_atom_ids
    assert scope_packet.governing_atom_ids[0] == "email_excl"
    assert "tx_excl" in (scope_packet.supporting_atom_ids + scope_packet.contradicting_atom_ids)

    missing_packet = next(p for p in packets if p.family == PacketFamily.missing_info)
    assert missing_packet.status == PacketStatus.needs_review

    decision_packets = [p for p in packets if p.family == PacketFamily.meeting_decision]
    assert all(p.status == PacketStatus.needs_review for p in decision_packets)

    assert all(
        not (
            p.family in {PacketFamily.scope_exclusion, PacketFamily.scope_inclusion, PacketFamily.meeting_decision}
            and any(aid.startswith("tx_") for aid in p.governing_atom_ids)
            and p.status == PacketStatus.active
        )
        for p in packets
    )
    assert all(p.certificate is not None for p in packets)
    assert all(p.risk is not None for p in packets)
    assert all(p.anchor_signature is not None for p in packets)
    scope_packet_cert = scope_packet.certificate
    assert scope_packet_cert is not None
    assert "customer_current_authored" in scope_packet_cert.governing_rationale
    assert scope_packet_cert.authority_path
    assert "dimensions" in scope_packet_cert.authority_path[0]


def test_material_aggregate_edge_one_packet_roster_governs_vendor_contradicts() -> None:
    roster = _atom(
        "r_mat",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:sl"],
        text="rj45 aggregate 72",
        quantity=72,
        value_extra={"normalized_item": "rj45", "aggregate": True},
    )
    vendor = _atom(
        "v_mat",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:sl"],
        text="rj45 vendor 68",
        quantity=68,
        value_extra={"normalized_item": "rj45", "inclusion_status": "included"},
    )
    md = {
        "identity": "rj45",
        "roster_quantity": 72.0,
        "vendor_quantity": 68.0,
        "delta": 4.0,
        "roster_atom_id": "r_mat",
        "vendor_atom_ids": ["v_mat"],
        "vendor_excluded_atom_ids": [],
        "roster_authority_class": AuthorityClass.approved_site_roster.value,
        "vendor_authority_class": AuthorityClass.vendor_quote.value,
        "comparison_basis": "aggregate_roster_vs_summed_vendor_quote",
        "included_vendor_line_filter": "primary",
        "preferred_packet_family": "quantity_conflict",
    }
    edge = _edge(
        "e_mat",
        EdgeType.contradicts,
        "r_mat",
        "v_mat",
        "RJ45: approved_site_roster aggregate 72 vs vendor_quote primary-line total 68; vendor quote short by 4.",
        metadata=md,
    )
    packets = build_packets("proj_1", [roster, vendor], [], [edge], attach_metadata=True)
    mat_pkts = [p for p in packets if p.anchor_key.startswith("material:rj45")]
    assert len(mat_pkts) == 1
    p = mat_pkts[0]
    assert p.family == PacketFamily.quantity_conflict
    assert p.status == PacketStatus.needs_review
    assert p.governing_atom_ids == ["r_mat"]
    assert p.contradicting_atom_ids == ["v_mat"]
    assert p.certificate is not None
    cert = p.certificate
    assert "72" in cert.existence_reason and "68" in cert.existence_reason
    assert cert.contradiction_summary and "delta=4" in cert.contradiction_summary
    assert "approved_site_roster" in cert.governing_rationale.lower()
    assert "RunbookGen.site_steps" in cert.blast_radius
    assert "AtlasDispatch.site_readiness" in cert.blast_radius
    assert "OrbitBrief.scope_truth" in cert.blast_radius
    assert "SOWSmith.scope_clause" in cert.blast_radius
    assert set(cert.minimal_sufficient_atom_ids) == {"r_mat", "v_mat"}
    assert "roster_vendor_aggregate_mismatch" in p.review_flags


def test_vendor_only_power_line_no_active_scope_inclusion_pollution_packet() -> None:
    power = _atom(
        "v_power",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit"],
        text="20 amp power locations",
        quantity=4,
        value_extra={
            "item_kind": "power",
            "is_scope_pollution_candidate": True,
            "scope_relevance": "scope_pollution_candidate",
            "inclusion_status": "included",
        },
    )
    packets = build_packets("proj_pollute", [power], [], [])
    assert not any(p.family == PacketFamily.scope_inclusion for p in packets)
    pol = [p for p in packets if "vendor_scope_pollution_candidate" in p.review_flags]
    assert pol and all(p.family == PacketFamily.scope_exclusion for p in pol)
    assert any("scope:power" in (p.anchor_key or "") for p in pol)


def test_vendor_only_20_amp_regex_no_pollution_flags_still_not_scope_inclusion() -> None:
    """Branch-circuit / receptacle wording without parser pollution flags must not become scope_inclusion."""
    line = _atom(
        "v_20a",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit"],
        text="Install 20 amp duplex receptacles at each AV location",
        quantity=6,
        value_extra={"inclusion_status": "included"},
    )
    packets = build_packets("proj_20a_only", [line], [], [])
    assert not any(p.family == PacketFamily.scope_inclusion for p in packets)
    assert any(
        p.family == PacketFamily.scope_exclusion and "vendor_scope_pollution_candidate" in p.review_flags
        for p in packets
    )


def test_customer_excludes_power_vendor_20_amp_emits_scope_exclusion_vendor_contradicts() -> None:
    gov = _atom(
        "cust_power_out",
        atom_type=AtomType.customer_instruction,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:hall_a"],
        text="Electrical contractor power and 20 amp outlets are not in scope; by others.",
    )
    vendor_power = _atom(
        "v_20a_hall",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:hall_a"],
        text="20 amp power locations",
        quantity=4,
        value_extra={"item_kind": "power", "inclusion_status": "included"},
    )
    packets = build_packets("proj_power_pair", [gov, vendor_power], [], [])
    pwr = next(
        p
        for p in packets
        if p.family == PacketFamily.scope_exclusion and "power_vendor_scope_mismatch" in p.review_flags
    )
    assert pwr.status == PacketStatus.needs_review
    assert pwr.governing_atom_ids[0] == "cust_power_out"
    assert pwr.contradicting_atom_ids == ["v_20a_hall"]
    assert "scope:power" in (pwr.anchor_key or "")


def test_poe_switch_vendor_does_not_trigger_power_vendor_scope_mismatch() -> None:
    gov = _atom(
        "cust_power_out",
        atom_type=AtomType.exclusion,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:hall_a"],
        text="120V branch circuit power is excluded and not included.",
    )
    poe = _atom(
        "v_poe",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:hall_a"],
        text="48-port PoE switch with 802.3at injectors",
        quantity=2,
        value_extra={"normalized_item": "poe_switch", "inclusion_status": "included"},
    )
    packets = build_packets("proj_poe", [gov, poe], [], [])
    assert not any("power_vendor_scope_mismatch" in (p.review_flags or []) for p in packets)


def test_roster_cat6_vendor_20_amp_power_not_grouped_into_scope_inclusion() -> None:
    roster = _atom(
        "r_cat6",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:sl"],
        text="Provide Cat6 UTP cabling per approved roster",
        value_extra={"inclusion_status": "included"},
    )
    contractual = _atom(
        "sow_power",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.contractual_scope,
        entity_keys=["site:sl"],
        text="Utility electrical power at outlets is by others; not in contractor scope.",
    )
    vendor_power = _atom(
        "v_20a_sl",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:sl"],
        text="20 amp receptacle circuits",
        quantity=3,
        value_extra={"inclusion_status": "included"},
    )
    packets = build_packets("proj_mix", [roster, contractual, vendor_power], [], [])
    inc = [p for p in packets if p.family == PacketFamily.scope_inclusion]
    assert inc
    linked_all: set[str] = set()
    for p in inc:
        linked_all |= set(p.governing_atom_ids + p.supporting_atom_ids + p.contradicting_atom_ids)
    assert "v_20a_sl" not in linked_all
    assert any("power_vendor_scope_mismatch" in (p.review_flags or []) for p in packets)


def test_vendor_excluded_certification_no_scope_inclusion_missing_info() -> None:
    roster_req = _atom(
        "r_cert",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:audit", "testing:fluke"],
        text="Fluke certification required",
        quantity=1,
        value_extra={"item_kind": "certification"},
    )
    vendor_ex = _atom(
        "v_cert",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit", "testing:cert_export"],
        text="Cable certification report exports",
        quantity=0,
        value_extra={
            "item_kind": "certification",
            "inclusion_status": "excluded",
            "included": False,
        },
    )
    packets = build_packets("proj_cert", [roster_req, vendor_ex], [], [])
    cert_packets = [
        p
        for p in packets
        if p.family == PacketFamily.missing_info
        and "certification_testing_export_missing_info" in (p.review_flags or [])
    ]
    assert cert_packets
    p = cert_packets[0]
    assert "requirement:certification" in (p.anchor_key or "")
    assert "vendor_excluded_line" in (p.review_flags or [])
    assert p.governing_atom_ids == ["r_cert"]
    assert "v_cert" in p.contradicting_atom_ids
    for p2 in packets:
        if p2.family == PacketFamily.scope_inclusion:
            assert "v_cert" not in set(p2.governing_atom_ids + p2.supporting_atom_ids)


def test_scope_requires_certification_quote_includes_no_cert_export_missing_info() -> None:
    roster_req = _atom(
        "r_cert",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:audit"],
        text="Fluke certification and test reports required",
        quantity=1,
        value_extra={"item_kind": "certification"},
    )
    vendor_ok = _atom(
        "v_cert",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit"],
        text="Cable certification report exports",
        quantity=1,
        value_extra={
            "item_kind": "certification",
            "inclusion_status": "included",
            "included": True,
        },
    )
    packets = build_packets("proj_cert_ok", [roster_req, vendor_ok], [], [])
    assert not any(
        "certification_testing_export_missing_info" in (p.review_flags or []) for p in packets
    )


def test_meeting_only_certification_confirm_emits_action_item_not_cert_missing_info() -> None:
    meeting_ai = _atom(
        "m_cert_fmt",
        atom_type=AtomType.action_item,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:audit"],
        text="Vendor to confirm certification report format.",
        value_extra={"owner": "vendor"},
    )
    packets = build_packets("proj_m_only", [meeting_ai], [], [])
    assert any(p.family == PacketFamily.action_item for p in packets)
    assert not any(
        "certification_testing_export_missing_info" in (p.review_flags or []) for p in packets
    )


def test_certification_missing_info_prefers_roster_over_meeting_action_item() -> None:
    roster_req = _atom(
        "r_cert",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:audit"],
        text="Fluke pass/fail reports required per spec",
        quantity=1,
        value_extra={"item_kind": "certification"},
    )
    meeting_ai = _atom(
        "m_cert_fmt",
        atom_type=AtomType.action_item,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:audit"],
        text="Vendor to confirm certification report format.",
        value_extra={"owner": "vendor"},
    )
    vendor_ex = _atom(
        "v_cert",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit"],
        text="Cable certification report exports",
        quantity=0,
        value_extra={
            "item_kind": "certification",
            "inclusion_status": "excluded",
            "included": False,
        },
    )
    packets = build_packets("proj_cert_gov", [roster_req, meeting_ai, vendor_ex], [], [])
    p = next(
        x
        for x in packets
        if x.family == PacketFamily.missing_info
        and "certification_testing_export_missing_info" in (x.review_flags or [])
    )
    assert p.governing_atom_ids == ["r_cert"]
    assert "v_cert" in p.contradicting_atom_ids
    assert "m_cert_fmt" in p.supporting_atom_ids


def test_site_access_catwalk_plus_lift_same_site() -> None:
    catwalk_loc = _atom(
        "sc_catwalk",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:aud"],
        text="Cable tray route uses auditorium catwalk per kickoff notes.",
    )
    lift_note = _atom(
        "k_lift",
        atom_type=AtomType.customer_instruction,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:aud"],
        text="Contractor must coordinate boom lift for ceiling cable bundles above the catwalk.",
    )
    packets = build_packets("sa_cat_lift", [catwalk_loc, lift_note], [], [])
    sa = [p for p in packets if p.family == PacketFamily.site_access]
    assert sa
    p0 = sa[0]
    blob = (p0.reason or "").lower()
    assert "catwalk" in blob or "lift" in blob
    assert "site_access_catwalk_context" in (p0.review_flags or [])
    assert "site_access_lift_equipment" in (p0.review_flags or [])


def test_site_access_after_hours_only() -> None:
    cust = _atom(
        "ah_only",
        atom_type=AtomType.customer_instruction,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:aud"],
        text="Field work is limited to weekends and after-hours; no weekday outages in occupied spaces.",
    )
    packets = build_packets("sa_after", [cust], [], [])
    assert any(p.family == PacketFamily.site_access for p in packets)
    p0 = next(p for p in packets if p.family == PacketFamily.site_access)
    assert "after-hours" in (p0.reason or "").lower() or "after" in (p0.reason or "").lower()


def test_site_access_customer_provided_lift_also_emits_action_item() -> None:
    ai = _atom(
        "lift_ai",
        atom_type=AtomType.action_item,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:aud"],
        text="Customer to provide scissor lift for ceiling access runs.",
        value_extra={"owner": "customer"},
    )
    packets = build_packets("sa_lift_ai", [ai], [], [])
    assert any(p.family == PacketFamily.site_access for p in packets)
    assert any(p.family == PacketFamily.action_item for p in packets)


def test_badge_mdf_unknown_open_question_emits_access_gate_missing_info() -> None:
    oq = _atom(
        "mdf_q",
        atom_type=AtomType.open_question,
        authority=AuthorityClass.meeting_note,
        entity_keys=["site:aud"],
        text="Is MDF badge access still TBD for the low-voltage contractor?",
    )
    packets = build_packets("sa_gate", [oq], [], [])
    assert not any(p.family == PacketFamily.site_access for p in packets)
    gate = next(
        p
        for p in packets
        if p.family == PacketFamily.missing_info and "missing_info_access_gate" in (p.review_flags or [])
    )
    assert gate.anchor_key == "missing_info:access:site_gate"
    assert gate.status == PacketStatus.needs_review


def test_catwalk_location_only_no_site_access_packet() -> None:
    loc = _atom(
        "cat_only",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:aud"],
        text="Horizontal cable pathway follows the auditorium catwalk per drawing C-4.",
    )
    packets = build_packets("sa_cat_only", [loc], [], [])
    assert not any(p.family == PacketFamily.site_access for p in packets)


def test_vendor_excluded_raceway_no_scope_inclusion() -> None:
    raceway = _atom(
        "v_race",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit"],
        text="Raceway/conduit allowance",
        quantity=1,
        value_extra={
            "item_kind": "raceway",
            "inclusion_status": "excluded",
            "included": False,
        },
    )
    packets = build_packets("proj_race", [raceway], [], [])
    assert not any(p.family == PacketFamily.scope_inclusion for p in packets)
    p = next(x for x in packets if x.family == PacketFamily.missing_info and "vendor_excluded_line" in x.review_flags)
    assert "raceway_conduit" in (p.anchor_key or "")
    assert "raceway_conduit_pathway_missing_info" in p.review_flags


def test_customer_raceway_allowance_question_emits_raceway_conduit_missing_info() -> None:
    cust = _atom(
        "cust_rw",
        atom_type=AtomType.customer_instruction,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:hall"],
        text="Do we price new raceway per affected plate or as a unit allowance? TBD.",
    )
    packets = build_packets("proj_rw_q", [cust], [], [])
    p = next(x for x in packets if "raceway_conduit_pathway_missing_info" in (x.review_flags or []))
    assert p.family == PacketFamily.missing_info
    assert p.anchor_key == "missing_info:raceway_conduit"
    assert p.governing_atom_ids == ["cust_rw"]


def test_existing_conduit_unknown_contractual_emits_raceway_conduit_missing_info() -> None:
    note = _atom(
        "n_con",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.contractual_scope,
        entity_keys=["site:hall"],
        text="Existing conduit condition at hallways is unknown; verify before bidding.",
    )
    packets = build_packets("proj_con_u", [note], [], [])
    p = next(x for x in packets if "raceway_conduit_pathway_missing_info" in (x.review_flags or []))
    assert "missing_info:raceway_conduit" in (p.anchor_key or "")


def test_customer_raceway_question_vendor_excluded_supports_same_missing_info() -> None:
    cust = _atom(
        "cust_rw",
        atom_type=AtomType.customer_instruction,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:hall"],
        text="Confirm raceway allowance basis — per plate or lump sum?",
    )
    vendor = _atom(
        "v_race",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:hall"],
        text="Raceway/conduit allowance",
        quantity=0,
        value_extra={
            "item_kind": "raceway",
            "inclusion_status": "excluded",
            "included": False,
        },
    )
    packets = build_packets("proj_rw_pair", [cust, vendor], [], [])
    p = next(x for x in packets if "raceway_conduit_pathway_missing_info" in (x.review_flags or []))
    assert p.governing_atom_ids == ["cust_rw"]
    assert "v_race" in p.supporting_atom_ids
    assert not any(
        p2.family == PacketFamily.scope_inclusion and "v_race" in (p2.governing_atom_ids + p2.supporting_atom_ids)
        for p2 in packets
    )


def test_known_included_raceway_with_clear_qty_no_raceway_pathway_missing_info() -> None:
    cust = _atom(
        "cust_rw",
        atom_type=AtomType.customer_instruction,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:hall"],
        text="Provide 120 LF surface raceway; included per drawing A-12; quantity 120.",
        value_extra={"inclusion_status": "included"},
    )
    packets = build_packets("proj_rw_clear", [cust], [], [])
    assert not any("raceway_conduit_pathway_missing_info" in (p.review_flags or []) for p in packets)


def test_vendor_patch_panel_only_no_active_scope_inclusion() -> None:
    patch = _atom(
        "v_patch",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["site:audit"],
        text="48-port patch panel",
        quantity=2,
        value_extra={"item_kind": "patch_panel", "inclusion_status": "included"},
    )
    packets = build_packets("proj_patch", [patch], [], [])
    assert not any(p.family == PacketFamily.scope_inclusion for p in packets)


def test_approved_roster_still_emits_active_scope_inclusion() -> None:
    roster = _atom(
        "r_inc",
        atom_type=AtomType.scope_item,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:audit", "room:mdF"],
        text="Provide Cat6 drops per approved roster",
        value_extra={"inclusion_status": "included"},
    )
    packets = build_packets("proj_roster_inc", [roster], [], [])
    inc = [p for p in packets if p.family == PacketFamily.scope_inclusion]
    assert inc
    assert all(p.status == PacketStatus.active for p in inc)
    assert all(p.governing_atom_ids == ["r_inc"] for p in inc)


COPPER_ROOT = Path(
    os.environ.get(
        "COPPER_VALIDATION_ROOT",
        r"c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs"
        r"\purtera_copper_low_voltage_validation_packs\real_data_cases",
    )
)
COPPER_ART = COPPER_ROOT / "COPPER_001_SPRING_LAKE_AUDITORIUM" / "artifacts"


@pytest.mark.skipif(not (COPPER_ART / "extracted").is_dir(), reason="COPPER_001 artifacts not present")
def test_copper_001_scope_inclusion_not_vendor_governed() -> None:
    result = compile_project(
        project_dir=COPPER_ART,
        project_id="COPPER_001_SPRING_LAKE_AUDITORIUM",
        allow_errors=True,
        allow_unverified_receipts=True,
    )
    atom_by_id = {a.id: a for a in result.atoms}
    allowed_governors = {
        AuthorityClass.approved_site_roster,
        AuthorityClass.customer_current_authored,
        AuthorityClass.contractual_scope,
    }
    for p in result.packets:
        if p.family != PacketFamily.scope_inclusion:
            continue
        assert p.governing_atom_ids
        for gid in p.governing_atom_ids:
            gov = atom_by_id[gid]
            assert gov.authority_class != AuthorityClass.vendor_quote
            assert gov.authority_class in allowed_governors or (
                gov.authority_class == AuthorityClass.meeting_note and gov.review_status == ReviewStatus.approved
            )
