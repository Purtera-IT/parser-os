from __future__ import annotations

from app.learning.active_learning import build_active_learning_queue


def _pm_ordering_payload() -> dict:
    cert = {"ambiguity_score": 0.2, "evidence_completeness_score": 0.75}
    return {
        "compile_id": "cmp_pm",
        "manifest": {"completed_at": "2026-04-29T12:00:00+00:00", "domain_pack_id": "default_pack"},
        "packets": [
            {
                "id": "p_action_generic",
                "family": "action_item",
                "anchor_key": "action_item:owner:notes",
                "status": "needs_review",
                "review_flags": [],
                "contradicting_atom_ids": [],
                "risk": {"severity": "medium", "risk_score": 0.62},
                "certificate": cert,
            },
            {
                "id": "p_vendor_utp",
                "family": "vendor_mismatch",
                "anchor_key": "material:cat6_utp",
                "status": "needs_review",
                "review_flags": ["vendor_scope_quantity_mismatch", "contradiction_present"],
                "contradicting_atom_ids": ["a", "b"],
                "risk": {"severity": "high", "risk_score": 0.88},
                "certificate": cert,
            },
            {
                "id": "p_qty_rj45",
                "family": "quantity_conflict",
                "anchor_key": "material:rj45",
                "status": "needs_review",
                "review_flags": ["contradiction_present"],
                "contradicting_atom_ids": ["x", "y"],
                "risk": {"severity": "critical", "risk_score": 0.95},
                "certificate": cert,
            },
            {
                "id": "p_power_ex",
                "family": "scope_exclusion",
                "anchor_key": "site:aud|scope:power",
                "status": "needs_review",
                "review_flags": ["power_vendor_scope_mismatch", "vendor_quote_not_scope_governor"],
                "contradicting_atom_ids": ["v1"],
                "risk": {"severity": "high", "risk_score": 0.9},
                "certificate": cert,
            },
            {
                "id": "p_missing_raceway",
                "family": "missing_info",
                "anchor_key": "missing_info:raceway_conduit",
                "status": "needs_review",
                "review_flags": ["raceway_conduit_pathway_missing_info"],
                "contradicting_atom_ids": [],
                "risk": {"severity": "medium", "risk_score": 0.7},
                "certificate": cert,
            },
            {
                "id": "p_missing_cert",
                "family": "missing_info",
                "anchor_key": "missing_info:requirement:certification",
                "status": "needs_review",
                "review_flags": ["certification_testing_export_missing_info"],
                "contradicting_atom_ids": [],
                "risk": {"severity": "medium", "risk_score": 0.72},
                "certificate": cert,
            },
            {
                "id": "p_site_access",
                "family": "site_access",
                "anchor_key": "site:aud",
                "status": "needs_review",
                "review_flags": ["site_access_physical_constraints"],
                "contradicting_atom_ids": [],
                "risk": {"severity": "medium", "risk_score": 0.65},
                "certificate": cert,
            },
            {
                "id": "p_unknown_fp",
                "family": "missing_info",
                "anchor_key": "device:unknown",
                "status": "needs_review",
                "review_flags": [],
                "contradicting_atom_ids": [],
                "risk": {"severity": "high", "risk_score": 0.91},
                "certificate": cert,
            },
            {
                "id": "p_open_generic",
                "family": "missing_info",
                "anchor_key": "missing_info:generic_question_slug",
                "status": "needs_review",
                "review_flags": [],
                "contradicting_atom_ids": [],
                "risk": {"severity": "medium", "risk_score": 0.68},
                "certificate": cert,
            },
        ],
        "rejected_candidates": [],
        "failure_records": [],
    }


def test_review_queue_material_mismatch_before_generic_action_item() -> None:
    q = build_active_learning_queue(_pm_ordering_payload())
    packet_rows = [r for r in q if r.item_type == "packet"]
    ids = [r.target_id for r in packet_rows]
    assert ids.index("p_qty_rj45") < ids.index("p_action_generic")
    assert ids.index("p_vendor_utp") < ids.index("p_action_generic")


def test_power_exclusion_before_generic_action_item() -> None:
    q = build_active_learning_queue(_pm_ordering_payload())
    ids = [r.target_id for r in q if r.item_type == "packet"]
    assert ids.index("p_power_ex") < ids.index("p_action_generic")


def test_raceway_and_cert_missing_info_before_generic_open_question() -> None:
    q = build_active_learning_queue(_pm_ordering_payload())
    ids = [r.target_id for r in q if r.item_type == "packet"]
    assert ids.index("p_missing_raceway") < ids.index("p_open_generic")
    assert ids.index("p_missing_cert") < ids.index("p_open_generic")


def test_device_unknown_not_in_top_queue_slots() -> None:
    q = build_active_learning_queue(_pm_ordering_payload(), max_items=6)
    top = [r.target_id for r in q[:6] if r.item_type == "packet"]
    assert "p_unknown_fp" not in top


def test_top_ten_contains_core_commercial_issues() -> None:
    q = build_active_learning_queue(_pm_ordering_payload(), max_items=10)
    top = [r.target_id for r in q[:10] if r.item_type == "packet"]
    for required in ("p_qty_rj45", "p_vendor_utp", "p_power_ex", "p_missing_raceway", "p_missing_cert"):
        assert required in top
