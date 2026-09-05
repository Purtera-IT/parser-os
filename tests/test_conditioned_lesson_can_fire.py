"""A lesson written "when Chase owns the deal" has to know who owns the deal.

`condition_holds` is deliberately closed by default — no facts, no fire — so a
caller that omits them does not weaken the gate, it silences it. Every
conditioned lesson is stored, matched, and then discarded for want of a
circumstance nobody supplied.

Found 2026-09-05 by dry-running a real note against live 010300:

    "stop asking who stages the hardware: when Chase owns the deal we stage it
     ourselves from the depot"

routed to gap=invalid with `condition={"field": "owner", "equals": "chase"}` —
correct in every respect, and unable to fire, because the compile-time screener
was called without facts while the terminology call one line below already
passed them.
"""

from __future__ import annotations

import inspect

from app.core.feedback_store import condition_holds


def test_the_compile_screener_supplies_the_deal_s_people() -> None:
    from app.core import orbitbrief_core

    src = inspect.getsource(orbitbrief_core)
    start = src.index("gap_questions, _dropped_gaps = drop_learned_bad_questions(")
    call = src[start : src.index("\n        )", start)]
    assert "facts=_pm_facts(atoms)" in call, (
        "a conditioned lesson cannot fire without the circumstance it names"
    )


def test_a_condition_needs_facts_to_hold() -> None:
    cond = {"when": {"field": "owner", "equals": "chase_whitfield"}}
    # The shape the screener was passing: nothing.
    assert condition_holds(cond, None) is False
    assert condition_holds(cond, {}) is False


def test_it_holds_when_the_named_person_is_on_the_deal() -> None:
    cond = {"when": {"field": "owner", "equals": "chase_whitfield"}}
    assert condition_holds(cond, {"owner": ["Chase Whitfield", "Carl Painter"]}) is True


def test_it_does_not_hold_for_somebody_else_s_deal() -> None:
    cond = {"when": {"field": "owner", "equals": "chase_whitfield"}}
    assert condition_holds(cond, {"owner": ["Carl Painter"]}) is False


def test_an_unconditioned_lesson_still_fires_with_no_facts() -> None:
    """Most lessons carry no condition; they must not be gated by this."""
    assert condition_holds({}, None) is True
    assert condition_holds({"when": {}}, None) is True


def test_pm_facts_reads_the_people_off_the_atoms() -> None:
    from app.core.orbitbrief_core import _pm_facts

    class A:
        def __init__(self, value):
            self.value = value

    facts = _pm_facts([A({"owner": "Chase Whitfield"}), A({"name": "Carl Painter"}), A(None)])
    assert "Chase Whitfield" in facts["owner"]
    assert "Carl Painter" in facts["owner"]


def test_no_atoms_is_not_a_crash() -> None:
    from app.core.orbitbrief_core import _pm_facts

    assert _pm_facts(None) == {"owner": []}
    assert _pm_facts([]) == {"owner": []}
