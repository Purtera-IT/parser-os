"""A row contact_property_block already reads correctly must not also reach
the LLM, which read it wrong on deal 010215: name="OSC phone" (a label),
email="843-423-2571 x3645" (a phone number).

Asserts directly on what _classify_batch is called with -- the actual LLM
batching call -- rather than trying to intercept network I/O, which the
function short-circuits around entirely when no model is reachable and would
make a naive "the LLM was never called" test pass for the wrong reason.
"""

from __future__ import annotations

from app.core.schemas import AtomType


class _Atom:
    def __init__(self, atom_type, value):
        self.atom_type = atom_type
        self.value = value
        self.entity_keys = []


def _osc_row_atom():
    return _Atom(AtomType.scope_item, {
        "kind": "table_row",
        "cells": {
            "Requestor Information": "OSC phone",
            "Requestor Information__1": "843-423-2571 x3645",
            "Requestor Information__2": "Backup Contact",
            "Requestor Information__3": "Bernard Donnelly",
        },
    })


def _unrelated_row_atom():
    return _Atom(AtomType.scope_item, {
        "kind": "table_row",
        "cells": {"col": "Freight surcharge applies to all shipments over 500 lbs"},
    })


def _patched(monkeypatch, tac):
    """Force past every deflect layer except the one under test, and past the
    reachability gate, so we can see exactly what reaches _classify_batch."""
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    monkeypatch.delenv("SOWSMITH_TYPED_CLASSIFIER_DISABLE", raising=False)
    monkeypatch.setattr(tac, "_atom_type_deflect_enabled", lambda: False)
    monkeypatch.setattr(tac, "_typed_student_enabled", lambda: False)
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    try:
        from app.core import rubric_gate
        monkeypatch.setattr(rubric_gate, "keep_deflect_flags",
                             lambda texts: [False] * len(texts))
    except Exception:
        pass
    seen_batches = []

    def _fake_classify_batch(batch):
        seen_batches.append(list(batch))
        return {}

    monkeypatch.setattr(tac, "_classify_batch", _fake_classify_batch)
    return seen_batches


def test_a_readable_contact_row_never_reaches_the_batch_call(monkeypatch):
    import app.core.typed_atom_classifier as tac

    seen = _patched(monkeypatch, tac)
    tac.classify_atoms([_osc_row_atom()])
    total = sum(len(b) for b in seen)
    assert total == 0, f"the OSC row reached the LLM batch: {seen}"


def test_an_unrelated_row_still_reaches_the_batch_call(monkeypatch):
    """The deflect is one-sided: it may only remove a row it recognises,
    never block one it does not. Proves the harness above actually exercises
    the batching path (a regression here would mean the previous test passed
    for the wrong reason)."""
    import app.core.typed_atom_classifier as tac

    seen = _patched(monkeypatch, tac)
    tac.classify_atoms([_unrelated_row_atom()])
    total = sum(len(b) for b in seen)
    assert total == 1, f"expected the unrelated row to reach the batch call: {seen}"
