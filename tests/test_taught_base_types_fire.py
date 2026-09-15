"""A verdict a Deal Kit taught fires whatever type it names, not only the
types the model may promote to.

Live 000061 (2026-09-15): the first quote priced the survey alone, so the
recap line "Expected four-hour survey will confirm AP count" was taught
scope_item. The store held it; the classifier resolved it against the
model's 44 promotion targets plus _keep, where scope_item does not appear, so
the store abstained and the model typed the line task on every compile.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.core import decide as decide_mod
from app.core import typed_atom_classifier as tac
from app.core.decide import Decision
from app.core.schemas import AtomType


class _Store:
    """Answers with what was taught for a text; records the candidates it was offered."""

    def __init__(self, taught):
        self.taught = taught
        self.offered: list[list[str]] = []

    def resolve(self, *, relation, text, candidates, **_):
        self.offered.append(list(candidates))
        v = self.taught.get(text)
        return Decision(verdict=v, confidence=0.97, source="store", correction_id="corr_kit") if v else None

    def few_shot(self, **_):
        return []


def _atom(aid, text, atom_type=AtomType.scope_item):
    return SimpleNamespace(id=aid, atom_type=atom_type, raw_text=text, value={"kind": "hubspot_note_body"},
                           entity_keys=[], source_refs=[SimpleNamespace(locator={"line_start": 3})], project_id="p")


def _past_every_layer(monkeypatch):
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    monkeypatch.delenv("SOWSMITH_TYPED_CLASSIFIER_DISABLE", raising=False)
    monkeypatch.setattr(tac, "_atom_type_deflect_enabled", lambda: False)
    monkeypatch.setattr(tac, "_typed_student_enabled", lambda: False)
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    try:
        from app.core import rubric_gate
        monkeypatch.setattr(rubric_gate, "keep_deflect_flags", lambda texts: [False] * len(texts))
    except Exception:
        pass
    batches = []
    monkeypatch.setattr(tac, "_classify_batch", lambda batch: batches.append(list(batch)) or {
        a.id: {"atom_type": "task", "value": {}} for a in batch})
    return batches


def test_a_line_taught_as_scope_stays_scope_and_never_reaches_the_model(monkeypatch):
    survey_note = _atom("n1", "Expected four-hour survey will confirm AP count")
    untaught = _atom("n2", "Relocate 10 APs from 35 ft to 15-20 ft in the office area")
    store = _Store({"Expected four-hour survey will confirm AP count": "scope_item"})
    prev = decide_mod.get_store(); decide_mod.set_store(store)
    try:
        batches = _past_every_layer(monkeypatch)
        tac.classify_atoms([survey_note, untaught])
    finally:
        decide_mod.set_store(prev)
    assert survey_note.atom_type == AtomType.scope_item
    assert untaught.atom_type == AtomType.task
    assert [a.id for b in batches for a in b] == ["n2"]
    assert all("scope_item" in c and "raw_utterance" in c and "_keep" in c for c in store.offered)


def test_a_line_taught_as_speech_becomes_speech(monkeypatch):
    said = _atom("u1", "We might have to install like, you know, or something.")
    store = _Store({"We might have to install like, you know, or something.": "raw_utterance"})
    prev = decide_mod.get_store(); decide_mod.set_store(store)
    try:
        batches = _past_every_layer(monkeypatch)
        tac.classify_atoms([said])
    finally:
        decide_mod.set_store(prev)
    assert said.atom_type == AtomType.raw_utterance
    assert batches == [] or all(a.id != "u1" for b in batches for a in b)


def test_the_taught_candidates_cover_every_atom_type():
    cands = tac._taught_type_candidates()
    assert set(t.value for t in AtomType) <= set(cands)
    assert cands[-1] == "_keep"
