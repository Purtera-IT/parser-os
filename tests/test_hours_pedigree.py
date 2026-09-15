"""A learned rate says how many taught lines stand behind it."""
from types import SimpleNamespace

from app.core import decide as decide_mod, task_hours
from app.core.decide import Decision
from app.core.schemas import AtomType


class _Store:
    def __init__(self):
        self.row = SimpleNamespace(id="c1", relation="task_hours", verdict="hours=5.33;per=camera;role=r1",
                                   exemplars=["3 Verkada cameras intsall.", "install 2 cameras above the doors", "camera install x4"],
                                   created_by="deal-kit:010043")
    def all_corrections(self, active_only=True): return [self.row]
    def get(self, cid): return self.row if cid == "c1" else None
    def resolve(self, *, relation, text, candidates, **_):
        return Decision(verdict=self.row.verdict, confidence=0.93, source="store", correction_id="c1")
    def few_shot(self, **_): return []


def test_the_rate_carries_its_pedigree(monkeypatch):
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    store = _Store(); prev = decide_mod.get_store(); decide_mod.set_store(store)
    try:
        t = SimpleNamespace(atom_type=AtomType.task, raw_text="3 Verkada cameras intsall.", value={}, project_id="p")
        assert task_hours.estimate_task_hours([t], store=store) == 1
    finally:
        decide_mod.set_store(prev)
    assert t.value["estimated_hours"] == 15.99 and t.value["hours_evidence"] == 3
    assert t.value["hours_taught_by"] == "deal-kit:010043"
