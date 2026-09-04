"""The `gap` head finally has a consumer.

A PM rejecting a question stored a correction the pipeline never read, so the
same ask returned on the next deal. `question_screen` is what reads it. These
pin the contract: it fires on a stored rejection, it stays silent when it is
unsure, and it never hides a question because something was broken.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.core.pm_feedback import HEAD_REGISTRY
from app.core.question_screen import (
    GAP_RELATION,
    drop_learned_bad_questions,
    screen_question,
    screen_questions,
    suppressed_ids,
)


class FakeStore:
    """Resolves like the real store: a hit only for the relation it knows,
    only for a verdict inside the caller's candidate set."""

    def __init__(self, hits=None, raises=False):
        self.hits = hits or {}
        self.raises = raises
        self.calls = []

    def resolve(self, *, relation, text, candidates, context, scope, instruction, relations, facts=None):
        self.calls.append(
            {"relation": relation, "text": text, "scope": scope, "candidates": candidates, "facts": facts}
        )
        if self.raises:
            raise RuntimeError("embedder unreachable")
        if relation != GAP_RELATION:
            return None
        verdict = self.hits.get(text)
        if verdict is None or verdict not in candidates:
            return None
        return SimpleNamespace(verdict=verdict, correction_id="pm_gap_abc123", confidence=0.91)


def test_the_relation_matches_the_head_registry() -> None:
    assert HEAD_REGISTRY["gap"].relation == GAP_RELATION


def test_a_stored_rejection_suppresses_the_question() -> None:
    q = "Who provides the customer bridge / remote support dial-in for each install?"
    store = FakeStore({q: "invalid"})
    out = screen_question(q, deal_id="deal-1", store=store)
    assert out["verdict"] == "invalid"
    assert out["correction_id"] == "pm_gap_abc123"
    assert out["confidence"] == 0.91
    assert store.calls[0]["relation"] == GAP_RELATION
    assert store.calls[0]["scope"].deal_id == "deal-1"


def test_a_question_nobody_judged_is_shown() -> None:
    store = FakeStore({"something else": "invalid"})
    assert screen_question("Who signs site acceptance?", store=store)["verdict"] is None


def test_an_explicit_keep_is_reported_as_valid() -> None:
    q = "Confirm the approved cutover window."
    assert screen_question(q, store=FakeStore({q: "valid"}))["verdict"] == "valid"


def test_the_deals_facts_reach_the_store_so_a_conditional_lesson_can_judge() -> None:
    q = "Who pays for the project management effort?"
    store = FakeStore({q: "invalid"})
    screen_question(q, deal_id="d1", store=store, facts={"owner": "Chase Whitfield"})
    assert store.calls[0]["facts"] == {"owner": "Chase Whitfield"}


def test_no_store_means_show_everything() -> None:
    assert screen_question("Anything at all?", store=None)["verdict"] is None


def test_an_unreachable_embedder_never_hides_a_question() -> None:
    out = screen_question("Who signs acceptance?", store=FakeStore(raises=True))
    assert out["verdict"] is None


def test_empty_text_is_not_screened() -> None:
    store = FakeStore({})
    assert screen_question("   ", store=store)["verdict"] is None
    assert store.calls == []


def test_screen_many_keeps_order_and_carries_ids() -> None:
    qs = [
        {"id": "q1", "rule_id": "site.A.acceptance", "text": "Who signs acceptance at A?"},
        {"id": "q2", "rule_id": "mode.staff_aug.bridge", "text": "Who provides the bridge?"},
        {"id": "q3", "rule_id": "pmcover.work_hours", "text": "What are the approved work hours?"},
    ]
    store = FakeStore({"Who provides the bridge?": "invalid"})
    results = screen_questions(qs, deal_id="deal-9", store=store)
    assert [r["id"] for r in results] == ["q1", "q2", "q3"]
    assert suppressed_ids(results) == ["q2"]
    assert results[1]["rule_id"] == "mode.staff_aug.bridge"


def test_generated_gaps_are_filtered_at_the_source() -> None:
    gaps = [
        {"summary": "Who is the day-of onsite contact, and how do we reach them?"},
        {"summary": "Confirm technician parking — any fees customer-reimbursed?"},
    ]
    store = FakeStore({"Confirm technician parking — any fees customer-reimbursed?": "invalid"})
    asked, dropped = drop_learned_bad_questions(gaps, deal_id="deal-2", store=store)
    assert len(asked) == 1 and len(dropped) == 1
    assert "onsite contact" in asked[0]["summary"]


def test_generated_gaps_pass_through_untouched_without_a_store() -> None:
    gaps = [{"summary": "Who signs acceptance?"}, {"summary": "What are the work hours?"}]
    asked, dropped = drop_learned_bad_questions(gaps, store=None)
    assert asked == gaps and dropped == []
