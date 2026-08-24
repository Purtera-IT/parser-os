"""The review queue can rank by teaching value, and stays unchanged without it."""
from __future__ import annotations

from app.learning.active_learning import build_active_learning_queue


def _payload() -> dict:
    return {
        "manifest": {"completed_at": "2026-08-04T00:00:00+00:00"},
        "candidates": [
            {
                "id": "c_saturated",
                "extraction_method": "llm_candidate",
                "validation_status": "needs_review",
                "confidence": 0.72,
                "candidate_type": "scope_item",
                "text": "Contractor to patch penetrations.",
            },
            {
                "id": "c_unseen",
                "extraction_method": "llm_candidate",
                "validation_status": "needs_review",
                "confidence": 0.72,
                "candidate_type": "submission_req",
                "text": "Submit closeout package within 10 days.",
            },
        ],
    }


def test_without_class_counts_behaviour_is_unchanged():
    """Existing callers must keep their exact ordering — this is additive."""
    base = build_active_learning_queue(_payload())
    same = build_active_learning_queue(_payload(), class_counts=None)
    assert [i.item_id for i in base] == [i.item_id for i in same]
    assert all(not any(r.startswith("gold:") for r in i.priority_reasons) for i in base)


def test_with_class_counts_the_unseen_class_is_asked_first():
    """Both are equally uncertain; only the training log distinguishes them, and
    the empty class is the one that actually moves coverage."""
    counts = {("atom_type", "scope_item"): 500}
    q = build_active_learning_queue(_payload(), class_counts=counts)
    ranked = [i for i in q if i.target_id in {"c_saturated", "c_unseen"}]
    assert ranked, "candidates should be queued"
    assert ranked[0].target_id == "c_unseen"


def test_rescoring_explains_itself():
    counts = {("atom_type", "scope_item"): 500}
    q = build_active_learning_queue(_payload(), class_counts=counts)
    unseen = next(i for i in q if i.target_id == "c_unseen")
    assert any(r.startswith("gold:") for r in unseen.priority_reasons)
    assert "gold:unseen_class" in unseen.priority_reasons


def test_a_rescoring_failure_never_drops_an_item_from_review():
    # Non-numeric counts blow up inside the scorer; the item must survive.
    q = build_active_learning_queue(_payload(), class_counts={("atom_type", "scope_item"): "nope"})
    assert {i.target_id for i in q} >= {"c_saturated", "c_unseen"}
