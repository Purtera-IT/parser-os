from __future__ import annotations

import re

from app.domain import get_active_domain_pack
from app.core.normalizers import normalize_text
from app.core.schemas import AtomType, EvidenceAtom, EvidenceEdge, EvidencePacket, PacketFamily, PacketRisk, PacketStatus

_BASE_RISK: dict[PacketFamily, float] = {
    PacketFamily.quantity_conflict: 0.82,
    PacketFamily.vendor_mismatch: 0.83,
    PacketFamily.scope_exclusion: 0.85,
    PacketFamily.site_access: 0.68,
    PacketFamily.missing_info: 0.64,
    PacketFamily.meeting_decision: 0.55,
    PacketFamily.action_item: 0.36,
    PacketFamily.scope_inclusion: 0.20,
    PacketFamily.customer_override: 0.70,
    PacketFamily.quantity_claim: 0.35,
}

_OPS_IMPACT: dict[PacketFamily, list[str]] = {
    PacketFamily.quantity_conflict: ["commercial_quote", "procurement_alignment"],
    PacketFamily.vendor_mismatch: ["commercial_quote", "procurement_alignment", "schedule_risk"],
    PacketFamily.scope_exclusion: ["scope_baseline", "change_order_risk"],
    PacketFamily.site_access: ["dispatch_readiness", "onsite_execution"],
    PacketFamily.missing_info: ["decision_latency", "schedule_risk"],
    PacketFamily.meeting_decision: ["scope_alignment"],
    PacketFamily.action_item: ["owner_followup"],
    PacketFamily.scope_inclusion: ["baseline_tracking"],
    PacketFamily.customer_override: ["scope_baseline", "commercial_alignment"],
    PacketFamily.quantity_claim: ["baseline_tracking"],
}


def pm_material_mismatch_order(anchor_key: str | None) -> int:
    """Tie-break among COPPER-style material anchors (lower = earlier in queue)."""
    a = (anchor_key or "").lower()
    if "rj45" in a:
        return 0
    if "cat6_utp" in a or ("cat6" in a and "utp" in a):
        return 1
    if "cat6_stp" in a or ("cat6" in a and ("stp" in a or "shield" in a)):
        return 2
    return 50


def compute_pm_queue_tier(
    *,
    family: str,
    anchor_key: str | None,
    review_flags: list[str] | None,
    status: str | None,
) -> int:
    """PM-facing ordering: lower tier = review sooner. Aligns with COPPER_001 expectations."""
    flags = set(review_flags or [])
    anchor = (anchor_key or "").lower()

    if "device:unknown" in anchor or anchor in ("site:unknown", "entity:unknown"):
        return 92
    if family == "scope_inclusion" and status == "active":
        return 85

    if family == "quantity_conflict":
        return 0
    if family == "vendor_mismatch" and "vendor_scope_quantity_mismatch" in flags:
        return 1

    if family == "scope_exclusion" and (
        "power_vendor_scope_mismatch" in flags
        or "scope_pollution_vendor_vs_written_exclusion" in flags
        or "vendor_scope_pollution_candidate" in flags
    ):
        return 2

    if family == "missing_info":
        if "raceway_conduit_pathway_missing_info" in flags:
            return 3
        if "certification_testing_export_missing_info" in flags:
            return 4
        if "missing_info_access_gate" in flags or "site_access_gate_unknown" in flags:
            return 5
        return 6

    if family == "site_access":
        return 7

    if family == "scope_exclusion":
        return 8

    if family == "meeting_decision":
        return 9
    if family == "customer_override":
        return 10

    if family == "action_item":
        return 11

    return 40


def packet_pm_sort_key(packet: EvidencePacket) -> tuple[int, int, float, str, str]:
    """Sort key for packet lists: ascending tuple = higher PM priority first."""
    risk = packet.risk
    tier = risk.queue_tier if risk is not None else 50
    mat = pm_material_mismatch_order(packet.anchor_key)
    score = -(risk.risk_score if risk is not None else 0.0)
    return (tier, mat, score, packet.anchor_key, packet.id)


def _unit_exposure_from_atoms(atoms: list[EvidenceAtom]) -> float:
    parts: list[str] = []
    for atom in atoms:
        parts.extend(
            [
                atom.raw_text,
                str(atom.value.get("item", "")),
                str(atom.value.get("description", "")),
                " ".join(atom.entity_keys),
            ]
        )
    text_blob = normalize_text(" ".join(parts))
    pack = get_active_domain_pack()
    pack_defaults = pack.risk_defaults
    if "ip camera" in text_blob or "camera" in text_blob:
        return float(pack_defaults.get("ip_camera_unit_exposure", 300.0))
    if "access point" in text_blob or " ap" in text_blob or "ap:" in text_blob:
        return float(pack_defaults.get("access_point_unit_exposure", 250.0))
    if "switch" in text_blob:
        return float(pack_defaults.get("switch_unit_exposure", 500.0))
    if "ip camera" in text_blob or "camera" in text_blob:
        return 300.0
    if "access point" in text_blob or " ap" in text_blob or "ap:" in text_blob:
        return 250.0
    if "switch" in text_blob:
        return 500.0
    return 200.0


def _estimate_cost(packet: EvidencePacket, atoms: list[EvidenceAtom]) -> float | None:
    pack_defaults = get_active_domain_pack().risk_defaults
    if packet.family in {PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch}:
        reason_numbers = [float(token) for token in re.findall(r"\d+(?:\.\d+)?", packet.reason)]
        if len(reason_numbers) >= 2:
            diff = abs(reason_numbers[0] - reason_numbers[1])
            return round(diff * _unit_exposure_from_atoms(atoms), 2)
        quantities = [
            float(atom.value.get("quantity"))
            for atom in atoms
            if atom.atom_type == AtomType.quantity and isinstance(atom.value.get("quantity"), (int, float))
        ]
        if len(quantities) >= 2:
            diff = abs(quantities[0] - quantities[1])
            return round(diff * _unit_exposure_from_atoms(atoms), 2)
        return None
    if packet.family == PacketFamily.site_access:
        return float(pack_defaults.get("failed_dispatch_exposure", 400.0))
    if packet.family == PacketFamily.scope_exclusion:
        quantities = [
            float(atom.value.get("quantity"))
            for atom in atoms
            if atom.atom_type == AtomType.quantity and isinstance(atom.value.get("quantity"), (int, float))
        ]
        if quantities:
            return round(quantities[0] * _unit_exposure_from_atoms(atoms), 2)
        return float(pack_defaults.get("unpriced_scope_default", 5000.0))
    return None


def _severity(score: float) -> str:
    if score >= 0.90:
        return "critical"
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _action_item_atom_low_priority(atoms: list[EvidenceAtom], packet: EvidencePacket) -> bool:
    if packet.family != PacketFamily.action_item or len(atoms) != 1:
        return False
    atom = atoms[0]
    if atom.atom_type != AtomType.action_item:
        return False
    text = normalize_text(atom.raw_text)
    risky = any(
        token in text for token in ("scope", "add", "remove", "price", "cost", "commercial", "change")
    )
    access_lift = any(
        token in text
        for token in (
            "lift",
            "catwalk",
            "access",
            "badge",
            "mdf",
            "idf",
            "escort",
            "after-hours",
            "after hours",
            "ceiling",
            "boom",
            "scissor",
        )
    )
    return not risky and not access_lift


def _priority(severity: str, packet: EvidencePacket, atoms: list[EvidenceAtom]) -> int:
    if packet.review_flags and "roster_vendor_aggregate_mismatch" in packet.review_flags:
        if severity == "low":
            return 3
        if severity == "medium":
            return 2
    if packet.family == PacketFamily.scope_inclusion and packet.status == PacketStatus.active:
        return 5
    if packet.family == PacketFamily.action_item and _action_item_atom_low_priority(atoms, packet):
        return 4
    if severity == "critical":
        return 1
    if severity == "high":
        return 2
    if severity == "medium":
        return 3
    return 4


def score_packet_risk(packet: EvidencePacket, atoms: list[EvidenceAtom], edges: list[EvidenceEdge]) -> PacketRisk:
    del edges
    score = _BASE_RISK.get(packet.family, 0.50)
    reasons: list[str] = [f"base:{packet.family.value}={score:.2f}"]
    flags = set(packet.review_flags or [])

    if packet.status == PacketStatus.needs_review:
        score += 0.10
        reasons.append("status:needs_review")
    if "contradiction_present" in flags:
        score += 0.10
        reasons.append("flag:contradiction_present")
    if "customer_current_override" in flags:
        score += 0.10
        reasons.append("flag:customer_current_override")
    if "exclusion_present" in flags:
        score += 0.10
        reasons.append("flag:exclusion_present")
    if "vendor_scope_quantity_mismatch" in flags:
        score += 0.15
        reasons.append("flag:vendor_scope_quantity_mismatch")
    if "roster_vendor_aggregate_mismatch" in flags:
        score += 0.08
        reasons.append("flag:roster_vendor_aggregate_mismatch")
    if "low_confidence_atom" in flags:
        score += 0.05
        reasons.append("flag:low_confidence_atom")

    if packet.family in {PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch}:
        score += 0.04
        reasons.append("commercial_procurement_impact")

    if packet.family == PacketFamily.scope_exclusion and flags & {
        "power_vendor_scope_mismatch",
        "scope_pollution_vendor_vs_written_exclusion",
        "vendor_scope_pollution_candidate",
    }:
        score += 0.10
        reasons.append("vendor_written_scope_power_contradiction")

    if packet.family == PacketFamily.missing_info:
        if flags & {
            "raceway_conduit_pathway_missing_info",
            "certification_testing_export_missing_info",
            "missing_info_access_gate",
            "site_access_gate_unknown",
        }:
            score += 0.08
            reasons.append("missing_info_blocks_quote_schedule_or_testing")

    if packet.family == PacketFamily.site_access:
        score += 0.03
        reasons.append("site_access_mobilization_impact")

    if packet.family == PacketFamily.action_item and _action_item_atom_low_priority(atoms, packet):
        score = max(0.0, score - 0.14)
        reasons.append("generic_action_item_demotion")

    if packet.certificate is not None:
        if packet.certificate.ambiguity_score > 0.5:
            score += 0.10
            reasons.append("certificate:ambiguity_gt_0.5")
        no_contradiction = len(packet.contradicting_atom_ids) == 0
        if packet.certificate.evidence_completeness_score > 0.9 and no_contradiction:
            score -= 0.10
            reasons.append("certificate:high_completeness_discount")

    score = max(0.0, min(1.0, round(score, 4)))
    severity = _severity(score)
    queue_tier = compute_pm_queue_tier(
        family=packet.family.value,
        anchor_key=packet.anchor_key,
        review_flags=list(packet.review_flags or []),
        status=packet.status.value,
    )
    return PacketRisk(
        risk_score=score,
        severity=severity,  # type: ignore[arg-type]
        risk_reasons=sorted(reasons),
        estimated_cost_exposure=_estimate_cost(packet, atoms),
        operational_impact=_OPS_IMPACT.get(packet.family, ["general_review"]),
        review_priority=_priority(severity, packet, atoms),
        queue_tier=queue_tier,
    )
