from __future__ import annotations

from app.core.risk import (
    compute_pm_queue_tier,
    packet_pm_sort_key,
    pm_material_mismatch_order,
    score_packet_risk,
)
from app.core.schemas import (
    AnchorSignature,
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    EvidencePacket,
    PacketFamily,
    PacketStatus,
    ReviewStatus,
    SourceRef,
)
from app.core.packet_certificates import build_packet_certificate


def _minimal_atom(
    atom_id: str,
    *,
    atom_type: AtomType,
    text: str,
    authority: AuthorityClass = AuthorityClass.meeting_note,
) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="p1",
        artifact_id="a1",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value={"text": text},
        entity_keys=["site:test"],
        source_refs=[
            SourceRef(
                id=f"src_{atom_id}",
                artifact_id="a1",
                artifact_type=ArtifactType.txt,
                filename="t.txt",
                locator={},
                extraction_method="t",
                parser_version="t",
            )
        ],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="t",
    )


def _packet_with_cert(
    *,
    pid: str,
    family: PacketFamily,
    anchor_key: str,
    flags: list[str],
    atoms: list[EvidenceAtom],
) -> EvidencePacket:
    atom_by_id = {a.id: a for a in atoms}
    edge_by_id: dict = {}
    p = EvidencePacket(
        id=pid,
        project_id="p1",
        family=family,
        anchor_type="site",
        anchor_key=anchor_key,
        governing_atom_ids=[atoms[0].id] if atoms else [],
        supporting_atom_ids=[a.id for a in atoms],
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=0.9,
        status=PacketStatus.needs_review,
        reason="test",
        review_flags=flags,
        anchor_signature=AnchorSignature(
            anchor_type="test",
            canonical_key=anchor_key,
            entity_keys=[anchor_key],
            normalized_topic="t",
            scope_dimension=None,
            hash="h",
        ),
    )
    p.certificate = build_packet_certificate(p, atom_by_id, edge_by_id=edge_by_id)
    p.risk = score_packet_risk(p, atoms, [])
    return p


def test_vendor_mismatch_queue_tier_before_action_item() -> None:
    vm_atom_a = _minimal_atom("qa", atom_type=AtomType.quantity, text="rj45 66")
    vm_atom_b = _minimal_atom("qb", atom_type=AtomType.quantity, text="rj45 60")
    vm = _packet_with_cert(
        pid="pkt_vm",
        family=PacketFamily.vendor_mismatch,
        anchor_key="material:cat6_utp",
        flags=["vendor_scope_quantity_mismatch", "contradiction_present"],
        atoms=[vm_atom_a, vm_atom_b],
    )
    ai_atom = _minimal_atom("ai1", atom_type=AtomType.action_item, text="Please confirm meeting minutes formatting.")
    ai = _packet_with_cert(
        pid="pkt_ai",
        family=PacketFamily.action_item,
        anchor_key="action_item:owner:notes",
        flags=[],
        atoms=[ai_atom],
    )
    assert vm.risk is not None and ai.risk is not None
    assert vm.risk.queue_tier < ai.risk.queue_tier
    assert packet_pm_sort_key(vm) < packet_pm_sort_key(ai)


def test_power_scope_exclusion_tier_before_generic_action_item() -> None:
    ex_atoms = [_minimal_atom("e1", atom_type=AtomType.exclusion, text="20A power excluded")]
    se = _packet_with_cert(
        pid="pkt_se",
        family=PacketFamily.scope_exclusion,
        anchor_key="site:hall|scope:power",
        flags=["power_vendor_scope_mismatch", "vendor_quote_not_scope_governor"],
        atoms=ex_atoms,
    )
    ai_atom = _minimal_atom("ai2", atom_type=AtomType.action_item, text="Update distribution list for weekly email.")
    ai = _packet_with_cert(
        pid="pkt_ai2",
        family=PacketFamily.action_item,
        anchor_key="action_item:owner:comms",
        flags=[],
        atoms=[ai_atom],
    )
    assert se.risk and ai.risk
    assert se.risk.queue_tier < ai.risk.queue_tier


def test_raceway_missing_info_tier_before_generic_open_question() -> None:
    mi_r = _packet_with_cert(
        pid="pkt_rw",
        family=PacketFamily.missing_info,
        anchor_key="missing_info:raceway_conduit",
        flags=["raceway_conduit_pathway_missing_info"],
        atoms=[_minimal_atom("rw1", atom_type=AtomType.customer_instruction, text="Raceway allowance TBD")],
    )
    mi_o = _packet_with_cert(
        pid="pkt_oq",
        family=PacketFamily.missing_info,
        anchor_key="missing_info:random_topic_slug",
        flags=[],
        atoms=[_minimal_atom("oq1", atom_type=AtomType.open_question, text="What color cable labels?")],
    )
    assert mi_r.risk and mi_o.risk
    assert mi_r.risk.queue_tier < mi_o.risk.queue_tier


def test_pm_material_order_rj45_before_cat6_utp() -> None:
    assert pm_material_mismatch_order("material:rj45") < pm_material_mismatch_order("material:cat6_utp")


def test_device_unknown_high_queue_tier() -> None:
    t = compute_pm_queue_tier(
        family="missing_info",
        anchor_key="device:unknown",
        review_flags=[],
        status="needs_review",
    )
    assert t >= 90
