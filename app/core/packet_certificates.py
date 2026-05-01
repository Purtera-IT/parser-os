from __future__ import annotations

from typing import Any

from app.core.authority import authority_rank, score_authority
from app.domain import get_active_domain_pack
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, EvidenceEdge, EvidencePacket, PacketCertificate, PacketFamily, PacketStatus

CERTIFICATE_VERSION = "packet_certificate_v1"

_BLAST_RADIUS_BY_FAMILY: dict[PacketFamily, list[str]] = {
    PacketFamily.quantity_conflict: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
        "RunbookGen.site_steps",
        "AtlasDispatch.site_readiness",
    ],
    PacketFamily.scope_exclusion: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
        "SOWSmith.exclusion_clause",
        "RunbookGen.site_steps",
    ],
    PacketFamily.scope_inclusion: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
        "RunbookGen.site_steps",
        "AtlasDispatch.site_readiness",
    ],
    PacketFamily.site_access: [
        "RunbookGen.site_steps",
        "AtlasDispatch.site_readiness",
        "VisionQC.photo_requirements",
    ],
    PacketFamily.vendor_mismatch: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
        "AtlasDispatch.site_readiness",
        "RunbookGen.site_steps",
    ],
    PacketFamily.missing_info: [
        "OrbitBrief.scope_truth",
        "RunbookGen.site_steps",
        "AtlasDispatch.site_readiness",
    ],
    PacketFamily.meeting_decision: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
        "RunbookGen.site_steps",
    ],
    PacketFamily.action_item: [
        "RunbookGen.site_steps",
        "AtlasDispatch.site_readiness",
        "VisionQC.photo_requirements",
    ],
    PacketFamily.customer_override: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
        "SOWSmith.exclusion_clause",
    ],
    PacketFamily.quantity_claim: [
        "OrbitBrief.scope_truth",
        "SOWSmith.scope_clause",
    ],
}


def _atom_ids(packet: EvidencePacket) -> list[str]:
    return sorted(set(packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids))


def _material_edge_metadata(packet: EvidencePacket, edge_by_id: dict[str, EvidenceEdge]) -> dict[str, Any] | None:
    for eid in packet.related_edge_ids or []:
        edge = edge_by_id.get(eid)
        if edge is None:
            continue
        md = edge.metadata or {}
        if md.get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote":
            return md
    return None


def _exists_reason(packet: EvidencePacket, _atom_by_id: dict[str, EvidenceAtom], md: dict[str, Any] | None) -> str:
    if md:
        ident = md.get("identity", "material")
        rq = md.get("roster_quantity")
        vq = md.get("vendor_quantity")
        d = md.get("delta")
        return (
            f"Created because roster aggregate and vendor primary-line totals diverge for {ident}: "
            f"roster_qty={rq}, vendor_primary_sum={vq}, delta={d}. {packet.reason.rstrip('.')}."
        )
    if packet.family == PacketFamily.quantity_conflict:
        return f"Created because {packet.reason.rstrip('.')}."
    if packet.family == PacketFamily.scope_exclusion:
        return (
            f"Created because exclusion evidence conflicts with scope inclusion evidence for {packet.anchor_key}."
        )
    if packet.family == PacketFamily.site_access:
        return f"Created because access constraints were identified for {packet.anchor_key}."
    if packet.family == PacketFamily.vendor_mismatch:
        base = f"Created because vendor_quote quantity does not align with scoped quantity for {packet.anchor_key}."
        detail = packet.reason.strip() if packet.reason else ""
        if detail:
            return f"{base} Detail: {detail.rstrip('.')}."
        return base
    if packet.family == PacketFamily.missing_info:
        return f"Created because open question evidence remains unresolved for {packet.anchor_key}."
    if packet.family == PacketFamily.meeting_decision:
        return f"Created because meeting decision/commitment evidence was detected for {packet.anchor_key}."
    if packet.family == PacketFamily.action_item:
        return f"Created because actionable owner/task evidence was extracted for {packet.anchor_key}."
    if "semantic_candidate_linker" in packet.review_flags:
        return (
            f"{packet.reason.rstrip('.')} (Includes semantic_candidate_linker neighborhood support; "
            "requires deterministic validation)."
        )
    return packet.reason


def _governing_rationale(packet: EvidencePacket, atom_by_id: dict[str, EvidenceAtom], md: dict[str, Any] | None) -> str:
    if not packet.governing_atom_ids:
        return "No governing atom selected; packet remains non-governing."
    governing = atom_by_id.get(packet.governing_atom_ids[0])
    if governing is None:
        return "Governing atom id is missing from atom map."
    if md or (
        packet.anchor_key.startswith("material:")
        and packet.family in (PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch)
    ):
        return (
            "approved_site_roster (approved addendum / site roster aggregate) governs scoped material quantity; "
            "vendor_quote reveals mismatch or coverage gaps and must not govern in-scope quantity."
        )
    if packet.family == PacketFamily.scope_exclusion and governing.authority_class == AuthorityClass.customer_current_authored:
        return "customer_current_authored outranks approved_site_roster for exclusion decisions."
    if packet.family == PacketFamily.scope_exclusion:
        if packet.status == PacketStatus.needs_review:
            return (
                f"{governing.authority_class.value} exclusion evidence currently governs, "
                "but exclusion remains needs_review due to contradiction or lower-confidence context."
            )
        return f"{governing.authority_class.value} exclusion evidence governs this packet."
    if packet.family == PacketFamily.quantity_conflict:
        return "approved_site_roster governs scope quantity; vendor_quote can support or contradict procurement coverage."
    return (
        f"{governing.authority_class.value} governs due to higher authority rank "
        f"({authority_rank(governing.authority_class)}) and deterministic tie-breakers."
    )


def _minimal_sufficient_ids(
    packet: EvidencePacket, atom_by_id: dict[str, EvidenceAtom], md: dict[str, Any] | None = None
) -> list[str]:
    if md and md.get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote":
        minimal: list[str] = []
        rid = md.get("roster_atom_id")
        if isinstance(rid, str) and rid in atom_by_id:
            minimal.append(rid)
        for vid in sorted(md.get("vendor_atom_ids") or []):
            if isinstance(vid, str) and vid in atom_by_id:
                minimal.append(vid)
        allowed_ids = set(packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids)
        return sorted([aid for aid in dict.fromkeys(minimal) if aid in allowed_ids])

    minimal = []
    governing_ids = [aid for aid in packet.governing_atom_ids if aid in atom_by_id]
    contradicting_ids = [aid for aid in packet.contradicting_atom_ids if aid in atom_by_id]
    support_ids = [aid for aid in packet.supporting_atom_ids if aid in atom_by_id]

    if packet.family == PacketFamily.quantity_conflict:
        gov_qty = next((aid for aid in governing_ids if atom_by_id[aid].atom_type == AtomType.quantity), None)
        if gov_qty:
            minimal.append(gov_qty)
        contradict_qty = next(
            (aid for aid in contradicting_ids if aid != gov_qty and atom_by_id[aid].atom_type == AtomType.quantity),
            None,
        )
        if contradict_qty:
            minimal.append(contradict_qty)
    elif packet.family == PacketFamily.scope_exclusion:
        gov_exclusion = next((aid for aid in governing_ids if atom_by_id[aid].atom_type == AtomType.exclusion), None)
        if gov_exclusion:
            minimal.append(gov_exclusion)
        contradicted_inclusion = next(
            (
                aid
                for aid in contradicting_ids
                if atom_by_id[aid].atom_type in {AtomType.scope_item, AtomType.quantity}
            ),
            None,
        )
        if contradicted_inclusion:
            minimal.append(contradicted_inclusion)
    elif packet.family == PacketFamily.site_access:
        for aid in governing_ids + support_ids:
            a = atom_by_id.get(aid)
            if a and a.atom_type in {
                AtomType.constraint,
                AtomType.customer_instruction,
                AtomType.action_item,
                AtomType.open_question,
                AtomType.scope_item,
            }:
                minimal.append(aid)
                break
    elif packet.family == PacketFamily.vendor_mismatch:
        scope_qty = sorted(
            {
                aid
                for aid in support_ids + governing_ids
                if atom_by_id[aid].atom_type == AtomType.quantity
                and atom_by_id[aid].authority_class != AuthorityClass.vendor_quote
            }
        )
        minimal.extend(scope_qty[:1])
        vendor_qty = next(
            (
                aid
                for aid in support_ids + contradicting_ids + governing_ids
                if atom_by_id[aid].atom_type == AtomType.quantity
                and atom_by_id[aid].authority_class == AuthorityClass.vendor_quote
            ),
            None,
        )
        if vendor_qty:
            minimal.append(vendor_qty)
    elif packet.family == PacketFamily.missing_info:
        question_atom = next(
            (aid for aid in support_ids + governing_ids if atom_by_id[aid].atom_type == AtomType.open_question),
            None,
        )
        if question_atom:
            minimal.append(question_atom)
        elif packet.review_flags and "raceway_conduit_pathway_missing_info" in packet.review_flags:
            for aid in governing_ids + support_ids:
                a = atom_by_id.get(aid)
                if a and a.atom_type in {
                    AtomType.customer_instruction,
                    AtomType.action_item,
                    AtomType.scope_item,
                    AtomType.exclusion,
                }:
                    minimal.append(aid)
                    break
        elif packet.review_flags and "certification_testing_export_missing_info" in packet.review_flags:
            for aid in governing_ids + support_ids:
                a = atom_by_id.get(aid)
                if a and a.authority_class in {
                    AuthorityClass.approved_site_roster,
                    AuthorityClass.customer_current_authored,
                    AuthorityClass.contractual_scope,
                    AuthorityClass.meeting_note,
                } and a.atom_type in {
                    AtomType.customer_instruction,
                    AtomType.action_item,
                    AtomType.scope_item,
                    AtomType.exclusion,
                    AtomType.quantity,
                }:
                    minimal.append(aid)
                    break
        elif packet.review_flags and "missing_info_access_gate" in packet.review_flags:
            q_atom = next(
                (aid for aid in governing_ids + support_ids if atom_by_id[aid].atom_type == AtomType.open_question),
                None,
            )
            if q_atom:
                minimal.append(q_atom)
    elif packet.family == PacketFamily.meeting_decision:
        decision_atom = next(
            (
                aid
                for aid in support_ids + governing_ids
                if atom_by_id[aid].atom_type in {AtomType.decision, AtomType.meeting_commitment}
            ),
            None,
        )
        if decision_atom:
            minimal.append(decision_atom)

    if not minimal:
        minimal = sorted(set(governing_ids[:1] + contradicting_ids[:1] + support_ids[:1]))
    allowed_ids = set(packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids)
    return sorted([aid for aid in dict.fromkeys(minimal) if aid in allowed_ids])


def _counterfactuals(packet: EvidencePacket, minimal_atom_ids: list[str], atom_by_id: dict[str, EvidenceAtom]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for atom_id in minimal_atom_ids:
        atom = atom_by_id.get(atom_id)
        if atom is None:
            continue
        if_removed = "confidence would drop"
        explanation = "Removing this atom reduces certificate sufficiency."
        if packet.family in {PacketFamily.quantity_conflict, PacketFamily.scope_exclusion, PacketFamily.vendor_mismatch}:
            if_removed = "packet would not exist"
            explanation = "Removing this atom removes a required side of the conflict/override."
        elif atom_id in packet.governing_atom_ids:
            if_removed = "governing decision would change"
            explanation = "Removing governing evidence changes authority selection."
        elif packet.status == PacketStatus.active:
            if_removed = "status would become active/needs_review"
            explanation = "Without this supporting atom, confidence and status safety degrade."
        outputs.append(
            {
                "atom_id": atom_id,
                "if_removed": if_removed,
                "explanation": explanation,
            }
        )
    return sorted(outputs, key=lambda x: x["atom_id"])


def _authority_path(packet: EvidencePacket, atom_by_id: dict[str, EvidenceAtom]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, Any]] = []
    for atom_id in _atom_ids(packet):
        atom = atom_by_id.get(atom_id)
        if atom is None:
            continue
        score = score_authority(
            atom,
            [atom_by_id[aid] for aid in _atom_ids(packet) if aid in atom_by_id],
            context={"packet_family": packet.family.value},
        )
        rows.append(
            {
                "atom_id": atom.id,
                "authority_class": atom.authority_class.value,
                "authority_rank": authority_rank(atom.authority_class),
                "confidence": float(atom.confidence),
                "final_score": score.final_score,
                "dimensions": score.dimensions,
                "score_explanation": score.explanation,
                "role": (
                    "governing"
                    if atom.id in packet.governing_atom_ids
                    else "contradicting"
                    if atom.id in packet.contradicting_atom_ids
                    else "supporting"
                ),
            }
        )
    return sorted(rows, key=lambda x: (str(x["role"]), str(x["atom_id"])))


def _evidence_completeness(packet: EvidencePacket, atom_by_id: dict[str, EvidenceAtom]) -> float:
    involved = [atom_by_id[aid] for aid in _atom_ids(packet) if aid in atom_by_id]
    all_receipts_verified = all(
        atom.receipts and all(receipt.replay_status == "verified" for receipt in atom.receipts)
        for atom in involved
    ) if involved else False
    score = 1.0 if (packet.governing_atom_ids and packet.supporting_atom_ids and all_receipts_verified) else 0.8
    contradiction_unresolved = bool(packet.contradicting_atom_ids) and packet.status.value == "needs_review"
    if contradiction_unresolved:
        score -= 0.2
    has_unsupported = any(
        receipt.replay_status == "unsupported"
        for atom in involved
        for receipt in atom.receipts
    )
    if has_unsupported:
        score -= 0.15
    if not packet.anchor_key or packet.anchor_key == "unknown":
        score -= 0.2
    return max(0.0, min(1.0, round(score, 4)))


def _ambiguity_score(packet: EvidencePacket, atom_by_id: dict[str, EvidenceAtom], completeness: float) -> float:
    involved = [atom_by_id[aid] for aid in _atom_ids(packet) if aid in atom_by_id]
    authority_tie = False
    if involved:
        ranked = sorted((authority_rank(atom.authority_class), atom.confidence) for atom in involved)
        if len(ranked) > 1:
            top_rank = ranked[-1][0]
            top_rank_count = sum(1 for rank, _ in ranked if rank == top_rank)
            authority_tie = top_rank_count > 1
    low_confidence_present = any(atom.confidence < 0.75 for atom in involved)
    score = 1.0 - completeness
    if authority_tie:
        score += 0.2
    if low_confidence_present:
        score += 0.2
    return max(0.0, min(1.0, round(score, 4)))


def build_packet_certificate(
    packet: EvidencePacket,
    atom_by_id: dict[str, EvidenceAtom],
    edge_by_id: dict[str, EvidenceEdge] | None = None,
) -> PacketCertificate:
    pack = get_active_domain_pack()
    edges_map = edge_by_id or {}
    md = _material_edge_metadata(packet, edges_map)
    minimal_ids = _minimal_sufficient_ids(packet, atom_by_id, md)
    if md:
        contradiction_summary = (
            f"identity={md.get('identity')} roster_qty={md.get('roster_quantity')} "
            f"vendor_primary_sum={md.get('vendor_quantity')} delta={md.get('delta')}"
        )
    elif packet.contradicting_atom_ids:
        contradiction_summary = f"{len(packet.contradicting_atom_ids)} contradicting atom(s) linked."
    else:
        contradiction_summary = None
    existence_reason = _exists_reason(packet, atom_by_id, md)
    completeness = _evidence_completeness(packet, atom_by_id)
    ambiguity = _ambiguity_score(packet, atom_by_id, completeness)
    return PacketCertificate(
        packet_id=packet.id,
        certificate_version=CERTIFICATE_VERSION,
        domain_pack_id=pack.pack_id,
        domain_pack_version=pack.version,
        existence_reason=existence_reason,
        governing_rationale=_governing_rationale(packet, atom_by_id, md),
        minimal_sufficient_atom_ids=minimal_ids,
        contradiction_summary=contradiction_summary,
        authority_path=_authority_path(packet, atom_by_id),
        counterfactuals=_counterfactuals(packet, minimal_ids, atom_by_id),
        blast_radius=_BLAST_RADIUS_BY_FAMILY.get(packet.family, ["OrbitBrief.scope_truth"]),
        evidence_completeness_score=completeness,
        ambiguity_score=ambiguity,
    )
