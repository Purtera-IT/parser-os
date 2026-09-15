"""A lesson taught for one deal fires on that deal's atoms -- and only there.

Live 000061 (compile 3689321 → 401c85a, 2026-09-15): nine lessons taught
deal-scoped from the first quote ("this recap line is scope, not a task")
never applied, while the two global ones did. Taught typing and learned
hours asked the store with no scope at all, so it searched its global tier
only.
"""
from types import SimpleNamespace

from app.core import decide as decide_mod
from app.core import task_hours, typed_atom_classifier as tac
from app.core.decide import Decision
from app.core.schemas import AtomType

DEAL = "221a2bae-7b2c-420b-9d98-9ad2a69685e3"


class _ScopedStore:
    """Answers only when asked with the deal the lesson was taught for."""

    def __init__(self, lessons):
        self.lessons = lessons  # (text, verdict, deal_id)
        self.scopes = []

    def resolve(self, *, relation, text, candidates, scope, **_):
        self.scopes.append(getattr(scope, "deal_id", None))
        for t, v, deal in self.lessons:
            if t == text and v in candidates and deal == getattr(scope, "deal_id", None):
                return Decision(verdict=v, confidence=0.98, source="store", correction_id="corr_deal")
        return None

    def few_shot(self, **_):
        return []

    def all_corrections(self, active_only=True):
        return [SimpleNamespace(relation="task_hours", verdict="hours=4;role=r1")]


def _atom(text, deal=DEAL, atom_type=AtomType.scope_item):
    return SimpleNamespace(id="a", atom_type=atom_type, raw_text=text, value={}, entity_keys=[],
                           source_refs=[], project_id=deal)


def _with_store(monkeypatch, store):
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    prev = decide_mod.get_store(); decide_mod.set_store(store)
    return prev


def test_a_deal_scoped_type_lesson_fires_for_its_deal_only(monkeypatch):
    line = "AP relocation to lower height between 15 to 20 feet targets better user signal coverage"
    store = _ScopedStore([(line, "scope_item", DEAL)])
    prev = _with_store(monkeypatch, store)
    try:
        ours, theirs = _atom(line), _atom(line, deal="other-deal")
        decided = tac._apply_taught_types([ours, theirs])
    finally:
        decide_mod.set_store(prev)
    assert decided == {id(ours): "scope_item"}
    assert DEAL in store.scopes and "other-deal" in store.scopes


def test_a_deal_scoped_hours_lesson_fires_for_its_deal_only(monkeypatch):
    line = "Conduct site survey during regular business hours"
    store = _ScopedStore([(line, "hours=4;role=r1", DEAL)])
    prev = _with_store(monkeypatch, store)
    try:
        ours, theirs = _atom(line, atom_type=AtomType.task), _atom(line, deal="other-deal", atom_type=AtomType.task)
        n = task_hours.estimate_task_hours([ours, theirs], store=store)
    finally:
        decide_mod.set_store(prev)
    assert n == 1
    assert ours.value["estimated_hours"] == 4 and "estimated_hours" not in theirs.value
