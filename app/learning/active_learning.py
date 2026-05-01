from __future__ import annotations

from typing import Any

from app.core.ids import stable_id
from app.core.risk import compute_pm_queue_tier, pm_material_mismatch_order
from app.review.schemas import ReviewQueueItem


def _queue_created_at(compile_payload: dict[str, Any]) -> str:
    manifest = compile_payload.get("manifest")
    if isinstance(manifest, dict):
        for key in ("completed_at", "started_at"):
            value = manifest.get(key)
            if isinstance(value, str) and value:
                return value
    return "1970-01-01T00:00:00+00:00"


def _severity_boost(severity: str | None) -> float:
    mapping = {"critical": 0.55, "high": 0.4, "medium": 0.2, "low": 0.05}
    return mapping.get(str(severity or "").lower(), 0.0)


def _failure_boost(records: list[dict[str, Any]]) -> tuple[float, list[str]]:
    boost = 0.0
    reasons: list[str] = []
    for record in records:
        category = str(record.get("category", "unknown"))
        severity = str(record.get("severity", "low")).lower()
        if severity == "critical":
            boost += 0.35
        elif severity == "high":
            boost += 0.2
        else:
            boost += 0.08
        reasons.append(f"failure_category:{category}")
    return boost, sorted(set(reasons))


def _novelty_score_for_packet(packet: dict[str, Any], manifest: dict[str, Any] | None) -> tuple[float, list[str]]:
    reasons: list[str] = []
    anchor_key = str(packet.get("anchor_key", "")).lower()
    score = 0.0
    # Do not boost unknown anchors — they are often false positives (device:unknown should not jump the queue).
    if anchor_key.startswith("device:") and any(token in anchor_key for token in ("custom", "new", "other")):
        score += 0.25
        reasons.append("unseen_device_alias")
    domain_pack_id = (manifest or {}).get("domain_pack_id")
    if domain_pack_id and domain_pack_id != "default_pack":
        score += 0.2
        reasons.append("new_domain_pack")
    return min(1.0, score), reasons


def _question_for_packet(family: str, anchor_key: str | None) -> str:
    key = anchor_key or "this anchor"
    mapping = {
        "quantity_conflict": f"Which quantity should govern for {key}?",
        "scope_exclusion": f"Should {key} be excluded from active scope?",
        "missing_info": "Is this open question required before dispatch/SOW?",
    }
    return mapping.get(family, f"Should this {family} packet be accepted?")


def _question_for_candidate(candidate: dict[str, Any]) -> str:
    if str(candidate.get("extraction_method")) == "llm_candidate":
        return "Is this proposed extracted claim correct?"
    return "Should this candidate evidence be promoted to accepted evidence?"


def _packet_queue_item(
    packet: dict[str, Any],
    *,
    manifest: dict[str, Any] | None,
    failure_records: list[dict[str, Any]],
    created_at: str,
) -> ReviewQueueItem:
    risk = packet.get("risk") or {}
    certificate = packet.get("certificate") or {}
    severity = str(risk.get("severity", "low")).lower()
    risk_score = float(risk.get("risk_score") or 0.0)
    ambiguity_score = float(certificate.get("ambiguity_score") or 0.0)
    completeness = float(certificate.get("evidence_completeness_score") or 0.0)
    novelty_score, novelty_reasons = _novelty_score_for_packet(packet, manifest)
    failure_boost, failure_reasons = _failure_boost(failure_records)

    score = 0.0
    reasons: list[str] = []
    if severity in {"critical", "high"}:
        reasons.append(f"severity:{severity}")
    score += _severity_boost(severity)
    score += min(0.4, risk_score * 0.35)
    if ambiguity_score > 0.0:
        score += min(0.35, ambiguity_score * 0.3)
        reasons.append("high_ambiguity")
    if completeness < 1.0:
        score += min(0.3, (1.0 - completeness) * 0.3)
        reasons.append("low_evidence_completeness")
    if packet.get("status") == "needs_review":
        score += 0.12
        reasons.append("status_needs_review")
    if str(packet.get("family")) == "missing_info":
        score += 0.18
        reasons.append("missing_info_requires_human_resolution")
    if "calibration_abstain" in (packet.get("review_flags") or []):
        score += 0.2
        reasons.append("calibration_abstain")
    if "semantic_candidate_linker" in (packet.get("review_flags") or []):
        score += 0.05
        reasons.append("semantic_link_support")
    if int(len(packet.get("contradicting_atom_ids") or [])) > 0:
        score += 0.08
        reasons.append("conflicting_evidence")
    score += novelty_score * 0.3
    reasons.extend(novelty_reasons)
    score += failure_boost
    reasons.extend(failure_reasons)
    score = round(min(1.0, score), 6)

    family = str(packet.get("family", "packet"))
    anchor_key = packet.get("anchor_key")
    anchor_s = str(anchor_key or "").lower()
    queue_tier = compute_pm_queue_tier(
        family=family,
        anchor_key=str(anchor_key) if anchor_key else "",
        review_flags=list(packet.get("review_flags") or []),
        status=str(packet.get("status") or ""),
    )
    anchor_sort_key = pm_material_mismatch_order(str(anchor_key) if anchor_key else None)

    if "device:unknown" in anchor_s or anchor_s in ("site:unknown", "entity:unknown"):
        score -= 0.35
        reasons.append("penalize_unknown_anchor")

    if family in {"quantity_conflict", "vendor_mismatch"}:
        score += 0.22
        reasons.append("commercial_procurement_queue_boost")
    if family == "scope_exclusion" and set(packet.get("review_flags") or []) & {
        "power_vendor_scope_mismatch",
        "scope_pollution_vendor_vs_written_exclusion",
        "vendor_scope_pollution_candidate",
    }:
        score += 0.18
        reasons.append("power_scope_pollution_queue_boost")
    if family == "missing_info" and set(packet.get("review_flags") or []) & {
        "raceway_conduit_pathway_missing_info",
        "certification_testing_export_missing_info",
        "missing_info_access_gate",
        "site_access_gate_unknown",
    }:
        score += 0.12
        reasons.append("blocking_missing_info_queue_boost")
    if family == "site_access":
        score += 0.08
        reasons.append("site_access_queue_boost")
    if family == "action_item":
        score -= 0.12
        reasons.append("action_item_default_demotion")

    score = round(min(1.0, max(0.0, score)), 6)

    suggested_question = _question_for_packet(family, anchor_key)
    return ReviewQueueItem(
        item_id=stable_id("rq", "packet", packet.get("id"), score, family),
        item_type="packet",
        target_id=str(packet.get("id")),
        priority_score=score,
        priority_reasons=sorted(set(reasons)),
        suggested_question=suggested_question,
        family_or_type=family,
        anchor_key=anchor_key,
        risk_score=risk_score if risk else None,
        ambiguity_score=ambiguity_score if certificate else None,
        novelty_score=novelty_score,
        created_at=created_at,
        queue_tier=queue_tier,
        anchor_sort_key=anchor_sort_key,
    )


def _candidate_queue_item(candidate: dict[str, Any], *, created_at: str) -> ReviewQueueItem:
    extraction_method = str(candidate.get("extraction_method", "deterministic_rule"))
    status = str(candidate.get("validation_status", "pending"))
    confidence = float(candidate.get("confidence") or 0.0)
    score = 0.1
    reasons: list[str] = []
    if extraction_method in {"llm_candidate", "semantic_candidate"}:
        score += 0.45
        reasons.append(f"extractor:{extraction_method}")
    if status == "needs_review":
        score += 0.25
        reasons.append("candidate_needs_review")
    elif status == "rejected":
        score += 0.18
        reasons.append("candidate_rejected_feedback")
    if confidence < 0.6:
        score += 0.2
        reasons.append("low_confidence_candidate")
    novelty_score = min(1.0, max(0.0, 1.0 - confidence))
    score += novelty_score * 0.1
    score = round(min(1.0, score), 6)
    return ReviewQueueItem(
        item_id=stable_id("rq", "candidate", candidate.get("id"), score, extraction_method),
        item_type="candidate",
        target_id=str(candidate.get("id")),
        priority_score=score,
        priority_reasons=sorted(set(reasons)),
        suggested_question=_question_for_candidate(candidate),
        family_or_type=str(candidate.get("candidate_type", "candidate")),
        anchor_key=(candidate.get("proposed_entity_keys") or [None])[0],
        risk_score=None,
        ambiguity_score=round(1.0 - confidence, 6),
        novelty_score=novelty_score,
        created_at=created_at,
        queue_tier=70,
        anchor_sort_key=50,
    )


def _semantic_queue_item(link: dict[str, Any], *, created_at: str) -> ReviewQueueItem:
    proposed_edge_type = str(link.get("proposed_edge_type", "edge"))
    item_type = "entity_alias" if proposed_edge_type == "same_as" else "edge"
    score = round(min(1.0, 0.25 + float(link.get("similarity_score") or 0.0) * 0.5), 6)
    question = (
        "Are these entity names the same real-world object?"
        if item_type == "entity_alias"
        else "Should this proposed relationship edge be accepted?"
    )
    return ReviewQueueItem(
        item_id=stable_id("rq", item_type, link.get("id"), score),
        item_type=item_type,
        target_id=str(link.get("id")),
        priority_score=score,
        priority_reasons=[f"semantic_link:{link.get('method', 'unknown')}"],
        suggested_question=question,
        family_or_type=proposed_edge_type,
        anchor_key=None,
        risk_score=None,
        ambiguity_score=round(1.0 - float(link.get("similarity_score") or 0.0), 6),
        novelty_score=None,
        created_at=created_at,
        queue_tier=75,
        anchor_sort_key=50,
    )


def _failure_index_by_target(records: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    packet_map: dict[str, list[dict[str, Any]]] = {}
    candidate_map: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        packet_id = record.get("packet_id")
        if packet_id:
            packet_map.setdefault(str(packet_id), []).append(record)
        atom_id = record.get("atom_id")
        if atom_id:
            candidate_map.setdefault(str(atom_id), []).append(record)
    return packet_map, candidate_map


def build_active_learning_queue(
    compile_payload: dict[str, Any],
    *,
    max_items: int | None = None,
) -> list[ReviewQueueItem]:
    created_at = _queue_created_at(compile_payload)
    packets = [row for row in (compile_payload.get("packets") or []) if isinstance(row, dict)]
    candidates = [row for row in (compile_payload.get("rejected_candidates") or []) if isinstance(row, dict)]
    candidates.extend([row for row in (compile_payload.get("candidates") or []) if isinstance(row, dict)])
    semantic_links = [row for row in (compile_payload.get("semantic_link_candidates") or []) if isinstance(row, dict)]
    semantic_links = [row for row in semantic_links if row.get("status") == "needs_review"]
    failures = [row for row in (compile_payload.get("failure_records") or []) if isinstance(row, dict)]
    manifest = compile_payload.get("manifest") if isinstance(compile_payload.get("manifest"), dict) else None

    failure_packets, _failure_candidates = _failure_index_by_target(failures)
    queue: list[ReviewQueueItem] = []

    for packet in packets:
        queue.append(
            _packet_queue_item(
                packet,
                manifest=manifest,
                failure_records=failure_packets.get(str(packet.get("id")), []),
                created_at=created_at,
            )
        )

    for candidate in candidates:
        method = str(candidate.get("extraction_method", ""))
        status = str(candidate.get("validation_status", "pending"))
        if status in {"needs_review", "rejected"} or method in {"llm_candidate", "semantic_candidate"}:
            queue.append(_candidate_queue_item(candidate, created_at=created_at))

    for link in semantic_links:
        queue.append(_semantic_queue_item(link, created_at=created_at))

    queue.sort(
        key=lambda item: (
            item.queue_tier,
            item.anchor_sort_key,
            -item.priority_score,
            item.item_type,
            item.target_id,
            item.item_id,
        )
    )
    if max_items is not None:
        queue = queue[: max(0, max_items)]
    return queue
