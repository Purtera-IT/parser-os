"""Warm-store deflection for the typed-atom classifier (upgrade #3).

The atom-TYPE promotion is a batched LLM call. Its keep/promote half is a
role/shape judgment the feedback store learns, so we front it with a STORE-ONLY
(``llm=False``) pre-filter: an atom the store CONFIDENTLY classifies ``_keep``
(no taxonomy entry fits → keep current type) is dropped from the LLM batch — no
round-trip, no value extraction. One-sided: the store can only ever remove a
no-op keep, never fabricate a promotion (those still go to the LLM, which alone
synthesizes the value payload).

Hermetic: deterministic content-addressed embedder (cosine 1.0 only on an exact
taught exemplar) + a tripwire ``_call_ollama`` to prove the LLM is/ isn't called.
``_ollama_reachable`` is forced True so the only thing gating the LLM is the
store. No network.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.typed_atom_classifier as tac
from app.core.decide import set_store
from app.core.feedback_store import Correction, FeedbackStore

_D = 128
_REL = "atom_type"


def _fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower().strip()))
        out[i, h % _D] = 1.0
        out[i, (h // _D) % _D] += 0.5
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


def _store_with(*corrections: Correction) -> FeedbackStore:
    s = FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)
    for c in corrections:
        s.add(c)
    return s


class _Type:
    def __init__(self, value):
        self.value = value


class _Atom:
    """Minimal atom matching the accessors classify_atoms uses."""

    def __init__(self, atom_id, raw_text, type_str="scope_item"):
        self.id = atom_id
        self.raw_text = raw_text
        self.atom_type = _Type(type_str)
        self.value = {}
        self.source_refs = []  # → empty section path → decide text == raw_text


_KEEP_TEXT = "miscellaneous note that maps to no taxonomy entry"
_KEEP = Correction(
    id="at_keep", relation=_REL, verdict="_keep", created_by="pm",
    exemplars=[_KEEP_TEXT],
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("SOWSMITH_ATOM_TYPE_DEFLECT", "1")
    monkeypatch.delenv("SOWSMITH_TYPED_CLASSIFIER_DISABLE", raising=False)
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    monkeypatch.delenv("SOWSMITH_TEACHER_CACHE", raising=False)
    # The LLM is reachable as far as the classifier knows — only the store
    # decides whether a round-trip happens.
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    yield
    set_store(None)


def test_confident_keep_deflects_no_llm(monkeypatch):
    """An atom the store learned as _keep is pulled from the batch — the LLM is
    never called, and nothing is promoted."""
    set_store(_store_with(_KEEP))

    def _tripwire(prompt, *, max_tokens=4096):
        raise AssertionError("LLM must not be called on a confident store _keep")

    monkeypatch.setattr(tac, "_call_ollama", _tripwire)
    atoms = [_Atom("a1", _KEEP_TEXT)]
    assert tac.classify_atoms(atoms) == 0
    assert atoms[0].atom_type.value == "scope_item"  # unchanged


def test_untaught_atom_reaches_llm_and_promotes(monkeypatch):
    """An untaught atom abstains in the store → reaches the LLM → gets promoted.
    The store never fabricates a promotion on its own."""
    set_store(_store_with(_KEEP))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return (
            '{"results": [{"atom_id": "a2", "atom_type": "milestone_phase", '
            '"value": {"name": "Phase 1 Kickoff"}}]}'
        )

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    atoms = [_Atom("a2", "Phase 1 Kickoff runs Jan through March, owner PMO")]
    promoted = tac.classify_atoms(atoms)
    assert calls["n"] == 1
    assert promoted == 1
    assert atoms[0].atom_type.value == "milestone_phase"


def test_flag_off_is_byte_identical(monkeypatch):
    """Flag off → store never consulted; every promotable atom hits the LLM."""
    monkeypatch.setenv("SOWSMITH_ATOM_TYPE_DEFLECT", "0")
    set_store(_store_with(_KEEP))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return '{"results": [{"atom_id": "a1", "atom_type": "_keep", "value": {}}]}'

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    # Even the taught _keep atom reaches the LLM when deflection is disabled.
    atoms = [_Atom("a1", _KEEP_TEXT)]
    assert tac.classify_atoms(atoms) == 0
    assert calls["n"] == 1


def test_self_teach_warms_keep(monkeypatch):
    """With teacher-cache ON, an LLM _keep verdict is folded back so the same
    atom shape deflects on the next pass with no LLM call."""
    monkeypatch.setenv("SOWSMITH_TEACHER_CACHE", "1")
    set_store(_store_with())  # empty → abstains on first sight
    calls = {"n": 0}
    novel = "freeform commentary with no taxonomy home whatsoever"

    def _fake_ollama(prompt, *, max_tokens=4096):
        calls["n"] += 1
        return '{"results": [{"atom_id": "a3", "atom_type": "_keep", "value": {}}]}'

    monkeypatch.setattr(tac, "_call_ollama", _fake_ollama)
    assert tac.classify_atoms([_Atom("a3", novel)]) == 0
    assert calls["n"] == 1

    def _tripwire(prompt, *, max_tokens=4096):
        raise AssertionError("learned _keep should deflect without an LLM call")

    monkeypatch.setattr(tac, "_call_ollama", _tripwire)
    assert tac.classify_atoms([_Atom("a3", novel)]) == 0
