# -*- coding: utf-8 -*-
"""A rule judgment does not need an atom to be judgeable.

Rule decisions reached the labelling workspace only by being hung on an atom
whose `raw_text` matched the judged text EXACTLY. That join is honest in
principle -- a fuzzy one would attribute a decision to text it was never made
about -- and empty in practice, because a rule judges a CELL or a LINE while an
atom carries assembled text. The rules judge things like

    'Quantity'   'Cost'   'INTRODUCTION'   '150ft CAT6 (non plenum) Materials'

Live dev compile cmp_863df5d394c69f33: **1220 decisions recorded, 0 attached to
atoms.** So the Rules stage read 0/0 and a labeller would conclude there was
nothing to do, while 132 of those 1220 sat within 0.08 of a hand-tuned
threshold -- the only decisions where moving the number changes the output.

What a person needs in order to say "this rule was right" is the text it judged
and how close the call was. The decision carries both. So the envelope ships
them directly, beside `suppressed`, ambiguous first.

The key is sha256(rule | text) and deliberately carries no deal: a threshold is
a property of the RULE, so one verdict should count on every deal that line
appears in.
"""
from __future__ import annotations

import pytest

from app.core import semantic_rules as SR
from app.core.orbitbrief_envelope import _rule_decisions_for_review


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setenv("SOWSMITH_RULE_DECISIONS", "1")
    SR.reset_decisions()
    yield
    SR.reset_decisions()


def _record(rule, text, best_pos, threshold, fired):
    SR._record_decision(rule, text, best_pos, 0.1, threshold, fired)


def test_a_decision_reaches_the_envelope_without_an_atom() -> None:
    _record("hours_effort_metric_label", "Quantity", 0.61, 0.60, True)
    rows = _rule_decisions_for_review()
    assert len(rows) == 1
    row = rows[0]
    assert row["rule"] == "hours_effort_metric_label"
    assert row["text"] == "Quantity"
    assert row["fired"] is True
    assert row["unsure"] is True and row["margin"] == 0.01


def test_the_ambiguous_calls_come_first() -> None:
    """A decision 0.001 from its threshold is worth a person's attention; one
    at 0.4 is not. Ordering is the whole value of the stage."""
    _record("a", "clear call", 0.95, 0.60, True)
    _record("b", "coin flip", 0.601, 0.60, True)
    _record("c", "near miss", 0.58, 0.60, False)
    rows = _rule_decisions_for_review()
    assert [r["text"] for r in rows][:2] == ["coin flip", "near miss"]
    assert rows[-1]["text"] == "clear call"


def test_the_key_is_the_rule_and_the_text_not_the_deal() -> None:
    """A threshold is a property of the rule, so one verdict counts everywhere
    that line appears."""
    _record("list_lead_in", "The following applies:", 0.62, 0.60, True)
    first = _rule_decisions_for_review()[0]["target_key"]
    SR.reset_decisions()
    _record("list_lead_in", "The following applies:", 0.62, 0.60, True)
    assert _rule_decisions_for_review()[0]["target_key"] == first

    SR.reset_decisions()
    _record("section_title", "The following applies:", 0.62, 0.60, True)
    assert _rule_decisions_for_review()[0]["target_key"] != first


def test_a_confident_call_is_not_marked_unsure() -> None:
    _record("qa_answer_block", "Answer: yes", 0.99, 0.60, True)
    assert _rule_decisions_for_review()[0]["unsure"] is False


def test_the_gate_holds(monkeypatch) -> None:
    _record("r", "text", 0.6, 0.6, True)
    monkeypatch.delenv("SOWSMITH_RULE_DECISIONS", raising=False)
    assert _rule_decisions_for_review() == []


def test_nothing_recorded_is_an_empty_list() -> None:
    assert _rule_decisions_for_review() == []
