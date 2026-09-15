"""Taught corrections type atoms before any head or LLM does."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core import typed_atom_classifier as tac
from app.core.decide import Decision
from app.core.schemas import AtomType


class _Store:
    def __init__(self, taught):
        self.taught = taught

    def resolve(self, *, relation, text, candidates, **_):
        v = self.taught.get(text) if relation == "atom_type" else None
        return Decision(verdict=v, confidence=0.9, source="store") if v in candidates else None


@pytest.fixture
def store():
    prev = decide_mod.get_store()
    yield lambda taught: decide_mod.set_store(_Store(taught))
    decide_mod.set_store(prev)


def _atom(text):
    return SimpleNamespace(atom_type=AtomType.scope_item, raw_text=text, value={}, review_flags=[], entity_keys=[])


def test_taught_types_apply_with_the_model_switched_off(store, monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    store({"Test systems with the client and troubleshoot as needed": "dependency",
           "Reset Ubiquiti gateway and switch for the new subnet": "task"})
    atoms = [_atom("Test systems with the client and troubleshoot as needed"),
             _atom("Reset Ubiquiti gateway and switch for the new subnet"),
             _atom("Something nobody taught")]
    assert tac.classify_atoms(atoms) == 2
    assert atoms[0].atom_type == AtomType.dependency
    assert atoms[1].atom_type == AtomType.task
    assert atoms[2].atom_type == AtomType.scope_item


def test_taught_keep_leaves_the_type(store, monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    monkeypatch.setenv("SOWSMITH_ATOM_TYPE_DEFLECT", "1")
    store({"Project: Consolidate the 1518 location tech": "_keep"})
    atoms = [_atom("Project: Consolidate the 1518 location tech")]
    assert tac.classify_atoms(atoms) == 0
    assert atoms[0].atom_type == AtomType.scope_item


def test_when_everything_was_taught_no_model_layer_runs(store, monkeypatch):
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    store({"Reset Ubiquiti gateway and switch for the new subnet": "task"})
    called = []
    monkeypatch.setattr(tac, "_atom_type_deflect_enabled", lambda: called.append(1) or False)
    atoms = [_atom("Reset Ubiquiti gateway and switch for the new subnet")]
    assert tac.classify_atoms(atoms) == 1
    assert atoms[0].atom_type == AtomType.task
    assert called == []


def test_no_store_changes_nothing(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    try:
        atoms = [_atom("Reset Ubiquiti gateway and switch for the new subnet")]
        assert tac.classify_atoms(atoms) == 0
        assert atoms[0].atom_type == AtomType.scope_item
    finally:
        decide_mod.set_store(prev)
