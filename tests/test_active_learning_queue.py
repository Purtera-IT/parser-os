from __future__ import annotations

from app.learning.active_learning import build_active_learning_queue


def _compile_payload() -> dict:
    return {
        "compile_id": "cmp_demo",
        "manifest": {"completed_at": "2026-04-28T12:00:00+00:00", "domain_pack_id": "security_ops_v1"},
        "packets": [
            {
                "id": "pkt_vendor_high",
                "family": "vendor_mismatch",
                "anchor_key": "device:new_camera_model",
                "status": "needs_review",
                "review_flags": ["calibration_abstain"],
                "contradicting_atom_ids": ["atm_2"],
                "risk": {"severity": "critical", "risk_score": 0.95},
                "certificate": {"ambiguity_score": 0.7, "evidence_completeness_score": 0.35},
            },
            {
                "id": "pkt_scope_normal",
                "family": "scope_inclusion",
                "anchor_key": "site:main_campus",
                "status": "active",
                "review_flags": [],
                "contradicting_atom_ids": [],
                "risk": {"severity": "low", "risk_score": 0.2},
                "certificate": {"ambiguity_score": 0.1, "evidence_completeness_score": 0.95},
            },
            {
                "id": "pkt_missing",
                "family": "missing_info",
                "anchor_key": "scope:dispatch_window",
                "status": "needs_review",
                "review_flags": [],
                "contradicting_atom_ids": [],
                "risk": {"severity": "medium", "risk_score": 0.3},
                "certificate": {"ambiguity_score": 0.3, "evidence_completeness_score": 0.6},
            },
        ],
        "rejected_candidates": [
            {
                "id": "cand_llm_1",
                "candidate_type": "scope_item",
                "proposed_entity_keys": ["site:main_campus"],
                "extraction_method": "llm_candidate",
                "validation_status": "needs_review",
                "confidence": 0.41,
            }
        ],
        "failure_records": [
            {
                "failure_id": "f_1",
                "category": "PACKET_BAD_SEVERITY",
                "severity": "high",
                "packet_id": "pkt_vendor_high",
            }
        ],
    }


def test_high_risk_vendor_mismatch_ranks_above_scope_inclusion() -> None:
    queue = build_active_learning_queue(_compile_payload())
    packet_rows = [row for row in queue if row.item_type == "packet"]
    ids = [row.target_id for row in packet_rows]
    assert ids.index("pkt_vendor_high") < ids.index("pkt_scope_normal")


def test_llm_candidate_needing_review_enters_queue() -> None:
    queue = build_active_learning_queue(_compile_payload())
    llm_rows = [row for row in queue if row.item_type == "candidate" and row.target_id == "cand_llm_1"]
    assert llm_rows
    assert llm_rows[0].suggested_question == "Is this proposed extracted claim correct?"


def test_missing_info_enters_queue() -> None:
    queue = build_active_learning_queue(_compile_payload())
    missing_rows = [row for row in queue if row.item_type == "packet" and row.family_or_type == "missing_info"]
    assert missing_rows
    assert "dispatch/SOW" in missing_rows[0].suggested_question


def test_review_queue_deterministic() -> None:
    first = build_active_learning_queue(_compile_payload())
    second = build_active_learning_queue(_compile_payload())
    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]


def test_suggested_question_populated() -> None:
    queue = build_active_learning_queue(_compile_payload())
    assert all(row.suggested_question.strip() for row in queue)


def test_max_items_parameter_works() -> None:
    queue = build_active_learning_queue(_compile_payload(), max_items=2)
    assert len(queue) == 2
