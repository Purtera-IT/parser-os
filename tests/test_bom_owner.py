"""Who supplies a hardware line: a store hit or a confident model verdict stamps
it; an undecided line carries nothing and nothing is dropped.

Live 010095 (2026-09-15): four SHI-supplied lines from a note reached the Deal
Kit prefill as BOM rows the kit never ordered.
"""
from types import SimpleNamespace

from app.core import bom_owner, decide as decide_mod, semantic_role
from app.core.decide import Decision
from app.core.schemas import AtomType


def _line(text, atom_type=AtomType.bom_line, value=None):
    return SimpleNamespace(atom_type=atom_type, raw_text=text, value=dict(value or {}), source_filename="note.txt", project_id="p")


class _Store:
    def __init__(self, taught): self.taught = taught; self.scopes = []
    def resolve(self, *, relation, text, candidates, scope, **_):
        self.scopes.append(scope.deal_id)
        v = self.taught.get(text)
        return Decision(verdict=v, confidence=0.96, source="store", correction_id="corr_kit") if v else None
    def few_shot(self, **_): return []


def test_a_taught_line_is_stamped_and_nothing_is_dropped(monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: (None, 0.0))
    store = _Store({"11JN0089US - Lenovo ThinkCentre M75q Gen 2": "customer_furnished"})
    prev = decide_mod.get_store(); decide_mod.set_store(store)
    try:
        theirs = _line("11JN0089US - Lenovo ThinkCentre M75q Gen 2")
        unknown = _line("CAT6 Plenum cable - 1000 ft box")
        n, verdicts = bom_owner.stamp_bom_owners([theirs, unknown, _line("x", atom_type=AtomType.task)], project_id="deal-1")
    finally:
        decide_mod.set_store(prev)
    assert n == 1
    assert theirs.value["supplied_by"] == "customer_furnished" and theirs.value["supplied_by_source"] == "store"
    assert "supplied_by" not in unknown.value
    assert [v["verdict"] for v in verdicts] == ["customer_furnished", None]
    assert store.scopes == ["deal-1", "deal-1"]


def test_the_model_needs_confidence(monkeypatch):
    decide_mod.set_store(None)
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: ("customer_furnished", 0.6))
    line = _line("ED41000P2-01 Lantronix Device Server EDS4100")
    assert bom_owner.stamp_bom_owners([line]) == (0, [{"text": line.raw_text, "verdict": None, "model_verdict": "customer_furnished", "confidence": 0.6, "source": "llm"}])
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: ("customer_furnished", 0.9))
    n, _ = bom_owner.stamp_bom_owners([line])
    assert n == 1 and line.value["supplied_by"] == "customer_furnished"


def test_the_switch_and_the_context(monkeypatch):
    monkeypatch.setenv("SOWSMITH_BOM_OWNER", "0")
    assert bom_owner.enabled() is False
    monkeypatch.delenv("SOWSMITH_BOM_OWNER")
    assert bom_owner.enabled() is True
    ctx = bom_owner._context(_line("x", value={"title": "Here is the model and complete list of equipment", "author_affiliation": "internal"}))
    assert "title: Here is the model" in ctx and "written by: internal" in ctx and "document: note.txt" in ctx
