"""The envelope's ledger says how many there were, not just the first 300.

MEASURED on live dev 2026-09-30, across the 18 compiles that finished that day:
13 shipped exactly 300 `suppressed` rows and 5 shipped exactly 600
`rule_decisions`. Those are the caps, to the row. So the caps bind on most real
deals, and nothing on the envelope said so -- a deal that suppressed exactly
300 atoms and one that suppressed 4,591 both shipped 300, and no field
distinguished them.

That matters beyond the UI. The suppression ledger is the instrument every
content-loss audit reads; an audit run against a silently truncated ledger
under-reports loss while looking thorough.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app.core import orbitbrief_envelope as env
from app.core import semantic_rules as sr


class _Atom:
    def __init__(self, text: str) -> None:
        self.raw_text = text
        self.review_flags = ["suppressed:semantic_dedup"]


@pytest.fixture
def carrying_the_ledger(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")


def test_the_total_is_the_real_count_not_the_capped_one(carrying_the_ledger):
    dropped = [_Atom(f"line {i}") for i in range(env._SUPPRESSED_MAX + 1_200)]
    result = SimpleNamespace(suppressed_atoms=dropped, project_id="d1")
    shown = env._suppressed_for_review(result, [])
    total = env._suppressed_total(result)
    assert len(shown) == env._SUPPRESSED_MAX, "the cap should still bind"
    assert total == env._SUPPRESSED_MAX + 1_200, "the total must be the real one"
    # The pair is the whole point: 300 alone is not an answer to "how much was
    # thrown away", and 300 of 300 means something different from 300 of 1500.
    assert total > len(shown)


def test_a_ledger_that_fits_reports_itself_exactly(carrying_the_ledger):
    dropped = [_Atom(f"line {i}") for i in range(7)]
    result = SimpleNamespace(suppressed_atoms=dropped, project_id="d1")
    assert env._suppressed_total(result) == 7
    assert len(env._suppressed_for_review(result, [])) == 7


def test_the_total_is_zero_when_the_ledger_is_not_carried(monkeypatch):
    # Carrying dropped atoms is opt-in and off by default; the total must not
    # imply a ledger that was never built.
    monkeypatch.delenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", raising=False)
    result = SimpleNamespace(suppressed_atoms=[_Atom("x")], project_id="d1")
    assert env._suppressed_total(result) == 0


def test_counting_decisions_does_not_go_through_all_decisions(monkeypatch):
    """`all_decisions(limit=0)` returns ONE row, not none and not all.

    The guard inside it is `len(out) >= limit`, so zero trips on the first
    append. Counting through it would have reported 1 decision for every
    compile -- a plausible-looking number that is always wrong.
    """
    sr._DECISIONS.clear()
    sr._DECISIONS["quantity"] = [{"rule": "r1"}, {"rule": "r2"}]
    sr._DECISIONS["cost"] = [{"rule": "r1"}]
    assert sr.decision_count() == 3
    assert len(sr.all_decisions(limit=0)) == 1, "the trap this avoids"
    assert env._rule_decisions_total() == 3
    sr._DECISIONS.clear()


def test_no_decisions_is_zero():
    sr._DECISIONS.clear()
    assert sr.decision_count() == 0
    assert env._rule_decisions_total() == 0
