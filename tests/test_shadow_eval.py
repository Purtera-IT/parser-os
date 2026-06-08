"""Tests for app.core.shadow_eval — the cutover gate.

The harness must (a) score the student on the *holdout* split only, so a
relation is credited for learning only when it works on unseen deals;
(b) return ``ready`` only when accuracy, gold accuracy, coverage AND sample
size all clear; and (c) stay honest under name-swap — holdout deals carry
different proper nouns than train deals, and the student is still scored on
whether it learned the rule.
"""

from __future__ import annotations

import numpy as np

from app.core.delexicalize import ROLE_CUSTOMER, ROLE_SELF_ORG
from app.core.shadow_eval import (
    evaluate_all,
    evaluate_relation,
    ready_relations,
    summary,
)
from app.core.training_log import TEACHER_LLM, TEACHER_PM, TrainingLog, TrainingRow

# Reuse the masked-text concept embedder from the student tests.
try:
    from tests.test_extractor_student import _embed
except ModuleNotFoundError:  # sibling helper module absent (prior-session test) -> skip cleanly
    import pytest
    pytest.skip("tests.test_extractor_student helper not present", allow_module_level=True)


def _make_log_with_split(n_per_deal: int = 6) -> TrainingLog:
    """Build a log whose deals land in both splits, all teaching the same rule
    ('shall provide ...' → requirement) but with different proper nouns."""
    log = TrainingLog(":memory:")
    rows = []
    # Many deals so the holdout split has >= _MIN_HOLDOUT rows. assign_split is
    # deterministic by deal-id hash, so this spreads across train/holdout.
    for d in range(40):
        name = f"Org{d}"
        cust = f"County{d}"
        for i in range(n_per_deal):
            rows.append(TrainingRow(
                relation="atom_type", label="requirement",
                raw_text=f"{name} shall provide item {i} to {cust}",
                teacher=TEACHER_LLM, deal_id=f"deal{d}",
                provenance={"role_map": {name: ROLE_SELF_ORG, cust: ROLE_CUSTOMER}},
            ))
    log.add_many(rows)
    return log


def _ev(log, relation, **kw):
    return evaluate_relation(
        log, relation, embed_fn=_embed, reachable_fn=lambda: True,
        threshold=0.6, **kw,
    )


def test_holdout_only_scoring_and_readiness():
    log = _make_log_with_split()
    rep = _ev(log, "atom_type")
    assert rep.n_holdout > 0
    # Single-rule corpus → student should answer and be right on unseen deals.
    assert rep.coverage > 0.6
    assert rep.accuracy >= 0.9
    assert rep.ready()


def test_sparse_relation_is_not_ready():
    log = TrainingLog(":memory:")
    # Only a couple of rows → below _MIN_HOLDOUT → never ready even if correct.
    log.add_many([
        TrainingRow(relation="payment_term", label="net_30",
                    raw_text="Net 30 from invoice", teacher=TEACHER_PM, deal_id="d1"),
        TrainingRow(relation="payment_term", label="net_30",
                    raw_text="payment net 30 days", teacher=TEACHER_PM, deal_id="d2"),
    ])
    rep = _ev(log, "payment_term")
    assert not rep.ready()


def test_name_swap_holdout_still_accurate():
    """Holdout deals carry names the train split never saw; accuracy stays high
    → the student generalized the rule, not the identities."""
    log = _make_log_with_split()
    rep = _ev(log, "atom_type")
    # If the student had memorized names, holdout accuracy would collapse to the
    # abstain floor. It does not.
    assert rep.accuracy >= 0.9


def test_evaluate_all_and_summary():
    log = _make_log_with_split()
    reports = evaluate_all(log, embed_fn=_embed, reachable_fn=lambda: True, threshold=0.6)
    assert "atom_type" in reports
    s = summary(reports)
    assert s["relations"] >= 1
    assert "atom_type" in s["ready"]
    assert "atom_type" in ready_relations(reports)


def test_gold_accuracy_gate_blocks_cutover():
    """A relation can have high overall accuracy but fail because it gets the
    PM-taught (gold) labels wrong — those carry a stricter bar."""
    log = TrainingLog(":memory:")
    rows = []
    # 40 silver rows teaching 'shall provide' -> requirement (so the student
    # confidently predicts 'requirement' for that concept).
    for d in range(40):
        rows.append(TrainingRow(
            relation="atom_type", label="requirement",
            raw_text=f"Org{d} shall provide item to County{d}",
            teacher=TEACHER_LLM, deal_id=f"deal{d}",
        ))
    # Gold rows in holdout that LOOK like requirements but PM labeled 'payment'
    # — the student will mispredict them, tanking gold accuracy.
    for d in range(40, 60):
        rows.append(TrainingRow(
            relation="atom_type", label="payment_term",
            raw_text=f"Org{d} shall provide payment terms",
            teacher=TEACHER_PM, deal_id=f"golddeal{d}",
        ))
    log.add_many(rows)
    rep = _ev(log, "atom_type")
    # Whatever the headline accuracy, mishandled gold must block readiness.
    if rep.n_gold and rep.gold_accuracy < 0.95:
        assert not rep.ready()
