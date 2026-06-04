"""Grounded-Extractor #70: the trained STUDENT fronts the atom-type LLM call.

Same one-sided contract as the store deflection (test_atom_type_deflect.py): the
student may only DROP an atom it confidently classifies ``_keep`` from the LLM
batch — never fabricate a promotion (those need the LLM's value synthesis). The
gate is ``SOWSMITH_TYPED_STUDENT``; off / empty-log / embedder-down → the student
abstains everywhere and the stage is byte-identical to the LLM-only path.

Hermetic: a deterministic content-addressed embedder (cosine 1.0 only on the
exact taught masked text), an in-memory training log, and a tripwire
``_call_ollama`` to prove whether the LLM round-trip happened. No network.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.typed_atom_classifier as tac
from app.core.extractor_student import ExtractionStudent
from app.core.training_log import TEACHER_LLM, TEACHER_PM, TrainingLog, TrainingRow

_D = 128
_REL = "atom_type"
_KEEP_TEXT = "miscellaneous note that maps to no taxonomy entry"


def _fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower().strip()))
        out[i, h % _D] = 1.0
        out[i, (h // _D) % _D] += 0.5
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


class _Type:
    def __init__(self, value):
        self.value = value


class _Atom:
    def __init__(self, atom_id, raw_text, type_str="scope_item"):
        self.id = atom_id
        self.raw_text = raw_text
        self.atom_type = _Type(type_str)
        self.value = {}
        self.source_refs = []  # empty section path → decide text == raw_text


def _student_with(*rows: TrainingRow) -> ExtractionStudent:
    log = TrainingLog(":memory:")
    log.add_many(list(rows))
    return ExtractionStudent(
        log, embed_fn=_fake_embed, reachable_fn=lambda: True, threshold=0.6,
    )


def _keep_rows(n: int = 3) -> list[TrainingRow]:
    # Several deals all teaching this exact shape → _keep (so the vote is firm).
    return [
        TrainingRow(relation=_REL, label="_keep", raw_text=_KEEP_TEXT,
                    teacher=TEACHER_LLM, deal_id=f"d{i}")
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("SOWSMITH_TYPED_STUDENT", "1")
    monkeypatch.delenv("SOWSMITH_ATOM_TYPE_DEFLECT", raising=False)
    monkeypatch.delenv("SOWSMITH_TYPED_CLASSIFIER_DISABLE", raising=False)
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    yield


def test_confident_keep_deflects_no_llm(monkeypatch):
    """An atom the student learned as _keep is pulled from the batch — the LLM
    is never called, and nothing is promoted."""
    monkeypatch.setattr(tac, "_get_typed_student", lambda: _student_with(*_keep_rows()))

    def _tripwire(prompt, *, max_tokens=4096):
        raise AssertionError("LLM must not be called on a confident student _keep")

    monkeypatch.setattr(tac, "_call_ollama", _tripwire)
    atoms = [_Atom("a1", _KEEP_TEXT)]
    assert tac.classify_atoms(atoms) == 0
    assert atoms[0].atom_type.value == "scope_item"  # unchanged


def test_untaught_atom_reaches_llm(monkeypatch):
    """An atom unlike anything taught → student abstains → LLM is consulted and
    promotes it. The student never fabricates a promotion."""
    monkeypatch.setattr(tac, "_get_typed_student", lambda: _student_with(*_keep_rows()))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return ('{"results": [{"atom_id": "a2", "atom_type": "milestone_phase", '
                '"value": {"name": "Phase 1 Kickoff"}}]}')

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    atoms = [_Atom("a2", "Phase 1 Kickoff runs Jan through March, owner PMO")]
    promoted = tac.classify_atoms(atoms)
    assert calls["n"] == 1
    assert promoted == 1
    assert atoms[0].atom_type.value == "milestone_phase"


def test_student_promotion_label_is_not_fabricated(monkeypatch):
    """Even when the student would confidently predict a PROMOTION label, the
    deflection is one-sided: only _keep removes an atom. A promotion-shaped
    prediction still reaches the LLM (which alone synthesizes the value)."""
    promo_text = "Contractor shall furnish and install 20 cameras"
    rows = [
        TrainingRow(relation=_REL, label="milestone_phase", raw_text=promo_text,
                    teacher=TEACHER_PM, deal_id=f"d{i}")
        for i in range(3)
    ]
    monkeypatch.setattr(tac, "_get_typed_student", lambda: _student_with(*rows))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return ('{"results": [{"atom_id": "a3", "atom_type": "milestone_phase", '
                '"value": {"name": "Install"}}]}')

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    atoms = [_Atom("a3", promo_text)]
    tac.classify_atoms(atoms)
    assert calls["n"] == 1  # promotion went to the LLM, not auto-applied


def test_flag_off_is_byte_identical(monkeypatch):
    """Flag off → student never consulted; the taught _keep atom hits the LLM."""
    monkeypatch.setenv("SOWSMITH_TYPED_STUDENT", "0")
    monkeypatch.setattr(tac, "_get_typed_student", lambda: _student_with(*_keep_rows()))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return '{"results": [{"atom_id": "a1", "atom_type": "_keep", "value": {}}]}'

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    atoms = [_Atom("a1", _KEEP_TEXT)]
    assert tac.classify_atoms(atoms) == 0
    assert calls["n"] == 1


def test_no_log_no_deflection(monkeypatch):
    """No training log → no student → LLM runs as usual (byte-identical)."""
    monkeypatch.setattr(tac, "_get_typed_student", lambda: None)
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return '{"results": [{"atom_id": "a1", "atom_type": "_keep", "value": {}}]}'

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    atoms = [_Atom("a1", _KEEP_TEXT)]
    assert tac.classify_atoms(atoms) == 0
    assert calls["n"] == 1
