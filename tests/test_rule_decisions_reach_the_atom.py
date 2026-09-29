# -*- coding: utf-8 -*-
"""What a rule decided must reach the person who can judge it.

`SemanticRule`s decide how atoms FORM -- section title, money column header,
table lead-in -- so they run before an atom exists, and a wrong call detaches
everything under a heading from it.

Their thresholds are hand-tuned, and a harvest over six live deals put 435 of
847 distinct decisions (51%) within 0.08 of their boundary. On 01491cca the
line "SOW - Premise Wiring, Bldg. 704 ii" misses `meeting_section_header` at
0.553 against a threshold of 0.550.

`_log_decision` already wrote all of this to `SOWSMITH_RULE_LOG` -- a file on
whichever box ran the compile. Nothing carried it to the envelope, so the
labelling UI had nothing to show and the thresholds could not be judged by the
person best placed to judge them. These attach the decisions to the atom whose
text they were made about.
"""
from __future__ import annotations

import pytest

from app.core import semantic_rules
from app.core.compiler import _attach_rule_decisions


class _Atom:
    """Only what `_attach_rule_decisions` touches."""

    def __init__(self, raw_text: str, value: object = None) -> None:
        self.raw_text = raw_text
        self.value = {} if value is None else value


@pytest.fixture(autouse=True)
def _clean():
    semantic_rules.reset_decisions()
    yield
    semantic_rules.reset_decisions()


def _record(text: str, rule: str, fired: bool, best_pos: float, threshold: float) -> None:
    semantic_rules._record_decision(rule, text, best_pos, 0.4, threshold, fired)


def test_off_by_default_so_a_normal_compile_pays_nothing(monkeypatch) -> None:
    monkeypatch.delenv("SOWSMITH_RULE_DECISIONS", raising=False)
    assert semantic_rules.decisions_enabled() is False
    _record("a line", "section_title", True, 0.9, 0.5)
    assert semantic_rules.decisions_for("a line") == []

    atom = _Atom("a line")
    assert _attach_rule_decisions([atom]) == 0
    assert "rule_decisions" not in atom.value


def test_the_decision_lands_on_the_atom_it_was_made_about(monkeypatch) -> None:
    monkeypatch.setenv("SOWSMITH_RULE_DECISIONS", "1")
    line = "SOW - Premise Wiring, Bldg. 704 ii"
    _record(line, "meeting_section_header", False, 0.553, 0.550)

    judged, untouched = _Atom(line), _Atom("something else entirely")
    assert _attach_rule_decisions([judged, untouched]) == 1

    (d,) = judged.value["rule_decisions"]
    assert d["rule"] == "meeting_section_header"
    assert d["fired"] is False
    # The margin is the whole reason to look: three thousandths decided
    # whether this line was a heading.
    assert d["margin"] == pytest.approx(0.003, abs=1e-6)
    assert "rule_decisions" not in untouched.value


def test_matching_is_exact_not_fuzzy(monkeypatch) -> None:
    """A rule judges a LINE. Attributing its verdict to text it was never
    shown would put words in its mouth."""
    monkeypatch.setenv("SOWSMITH_RULE_DECISIONS", "1")
    _record("Scope of Work", "section_title", True, 0.8, 0.5)

    assert _attach_rule_decisions([_Atom("Scope of Work — Building 704")]) == 0
    assert _attach_rule_decisions([_Atom("  Scope of Work  ")]) == 1, (
        "surrounding whitespace is not a different line"
    )


def test_one_entry_per_rule_however_often_it_is_asked(monkeypatch) -> None:
    """A parse asks the same rule about the same line repeatedly; the UI
    should show that rule once, not once per call."""
    monkeypatch.setenv("SOWSMITH_RULE_DECISIONS", "1")
    for _ in range(5):
        _record("a heading", "section_title", True, 0.8, 0.5)
    _record("a heading", "list_lead_in", False, 0.3, 0.5)

    atom = _Atom("a heading")
    _attach_rule_decisions([atom])
    rules = [d["rule"] for d in atom.value["rule_decisions"]]
    assert sorted(rules) == ["list_lead_in", "section_title"]


def test_a_reset_forgets_the_previous_compile(monkeypatch) -> None:
    monkeypatch.setenv("SOWSMITH_RULE_DECISIONS", "1")
    _record("a line", "section_title", True, 0.9, 0.5)
    semantic_rules.reset_decisions()
    assert semantic_rules.decisions_for("a line") == []


def test_it_can_never_fail_a_compile(monkeypatch) -> None:
    """A labelling aid must not be able to take a compile down."""
    monkeypatch.setenv("SOWSMITH_RULE_DECISIONS", "1")
    _record("a line", "section_title", True, 0.9, 0.5)

    class Hostile:
        @property
        def raw_text(self):
            raise RuntimeError("this atom is unhappy")

    assert _attach_rule_decisions([Hostile()]) == 0

    # An atom whose value is not a dict is skipped, not crashed on.
    frozen = _Atom("a line", value="not a dict")
    assert _attach_rule_decisions([frozen]) == 0
