"""Taught corrections outrank trained admission heads when re-typing atoms."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core import span_admission
from app.core.decide import Decision
from app.core.schemas import AtomType


class _Store:
    """Answers only for texts it was taught; abstains otherwise."""

    def __init__(self, taught):
        self.taught = taught

    def resolve(self, *, relation, text, candidates, **_):
        verdict = self.taught.get(text) if relation == "atom_type" else None
        if verdict in candidates:
            return Decision(verdict=verdict, confidence=0.95, source="store")
        return None


def _atom(text, atom_type=AtomType.scope_item):
    return SimpleNamespace(atom_type=atom_type, raw_text=text, value={}, review_flags=[])


@pytest.fixture
def heads(monkeypatch):
    seen = []

    def fake_heads(atoms, heads, weak):
        seen.extend(a.raw_text for a in atoms
                    if getattr(a.atom_type, "value", a.atom_type) in weak)
        return 0

    monkeypatch.setattr(span_admission, "_load_admission_heads", lambda: {"requirements": object()})
    monkeypatch.setattr(span_admission, "_readmit_via_heads", fake_heads)
    return seen


@pytest.fixture
def store():
    prev = decide_mod.get_store()
    yield lambda taught: decide_mod.set_store(_Store(taught))
    decide_mod.set_store(prev)


def test_taught_task_fires_even_when_heads_are_registered(heads, store):
    line = "Reset Ubiquiti gateway and switch for the new subnet"
    store({line: "task"})
    atoms = [_atom(line), _atom("Medicine Shoppe 1517 Meraki switch")]
    assert span_admission.readmit_atom_types(atoms) == 1
    assert atoms[0].atom_type == AtomType.task
    # The store's answer is final for that atom; the heads see only the rest.
    assert heads == ["Medicine Shoppe 1517 Meraki switch"]


def test_store_abstains_heads_still_decide(heads, store):
    store({})
    atoms = [_atom("Confirm QS1 connectivity between workstations and host")]
    span_admission.readmit_atom_types(atoms)
    assert atoms[0].atom_type == AtomType.scope_item
    assert heads == ["Confirm QS1 connectivity between workstations and host"]


def test_no_store_wired_is_a_no_op_for_the_store_pass(heads):
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    try:
        atoms = [_atom("Update QS1 Host PC static IP for the 1517 subnet")]
        span_admission.readmit_atom_types(atoms)
        assert atoms[0].atom_type == AtomType.scope_item
    finally:
        decide_mod.set_store(prev)


def test_task_is_a_recoverable_type():
    assert "task" in span_admission.RECOVERABLE_ATOM_TYPES
