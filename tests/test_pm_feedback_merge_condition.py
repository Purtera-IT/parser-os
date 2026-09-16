"""A merged correction carries a condition only when every contributor agreed on it.

Live 2026-09-16: one POST carried ``when: {owner: deal_kit}`` into the global
``task`` correction that fifty other kit-taught lines had merged into without a
condition. decide() passes no facts, so condition_holds() answered False for
every caller and the type-learning loop went silent (000020 Binghamton fell from
11 of 11 kit lines to 4).
"""
from __future__ import annotations

import numpy as np

from app.core.feedback_store import FeedbackStore, condition_holds
from app.core.pm_feedback import apply_pm_correction


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:", embed_fn=lambda texts: np.full((len(texts), 4), 0.5, np.float32),
                         reachable_fn=lambda: True)


_BASE = {"head": "type", "oldValue": "", "newValue": "task", "scope": "global", "context": "",
         "rationale": "the finished Deal Kit priced this line", "candidates": [], "pm": "deal-kit:test"}
_WHEN = {"when": {"field": "owner", "equals": "deal_kit"}}


def _find(st: FeedbackStore, cid: str):
    return next(c for c in st.all_corrections(active_only=True) if c.id == cid)


def test_a_merged_correction_drops_a_condition_its_contributors_disagree_on():
    st = _store()
    first = apply_pm_correction(st, {**_BASE, "dealId": "deal-a", "text": "Reset the gateway for the new subnet", "relations": {}})
    second = apply_pm_correction(st, {**_BASE, "dealId": "deal-b", "text": "Install the access switch", "relations": dict(_WHEN)})
    assert first == second, "same head, scope and verdict merge into one correction"
    c = _find(st, second)
    assert len(c.exemplars) == 2
    assert "when" not in c.relations
    assert condition_holds(c.relations, None) is True, "the unconditional contributor's lines still fire"


def test_a_condition_every_contributor_agrees_on_is_kept():
    st = _store()
    apply_pm_correction(st, {**_BASE, "dealId": "deal-a", "text": "Reset the gateway for the new subnet", "relations": dict(_WHEN)})
    cid = apply_pm_correction(st, {**_BASE, "dealId": "deal-b", "text": "Install the access switch", "relations": dict(_WHEN)})
    c = _find(st, cid)
    assert c.relations.get("when") == _WHEN["when"]
    assert condition_holds(c.relations, None) is False


def test_a_lone_conditional_lesson_keeps_its_condition():
    st = _store()
    cid = apply_pm_correction(st, {**_BASE, "dealId": "deal-a", "text": "Reset the gateway for the new subnet", "relations": dict(_WHEN)})
    assert _find(st, cid).relations.get("when") == _WHEN["when"]
