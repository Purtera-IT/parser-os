"""Taught quote-line tiers win over the word-list heuristics."""
import pytest

from app.core import decide as decide_mod
from app.core.decide import Decision
from app.core.task_tier_classifier import infer_task_tier


class _Store:
    def __init__(self, taught):
        self.taught = taught

    def resolve(self, *, relation, text, candidates, **_):
        v = self.taught.get(text) if relation == "task_tier" else None
        return Decision(verdict=v, confidence=0.9, source="store") if v in candidates else None


@pytest.fixture
def store():
    prev = decide_mod.get_store()
    yield lambda taught: decide_mod.set_store(_Store(taught))
    decide_mod.set_store(prev)


def test_taught_parent_beats_a_leading_confirm(store):
    line = "Confirm QS1 connectivity between workstations and host"
    assert infer_task_tier(text=line) == ("child", False)  # cold start: the word list
    store({line: "parent"})
    assert infer_task_tier(text=line) == ("parent", True)


def test_taught_child_beats_the_long_label_rule(store):
    line = "Floor plans should highlight the office section within the warehouse and AP placements"
    store({line: "child"})
    assert infer_task_tier(text=line) == ("child", False)


def test_explicit_tier_still_wins(store):
    line = "Confirm QS1 connectivity between workstations and host"
    store({line: "parent"})
    assert infer_task_tier(text=line, structured={"task_tier": "child"}) == ("child", False)


def test_no_store_keeps_the_heuristics():
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    try:
        assert infer_task_tier(text="Install 39 new CAT6 drops") == ("parent", True)
    finally:
        decide_mod.set_store(prev)
