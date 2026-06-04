"""Warm-store keep/drop deflection for the enrich_entities canonicalize loop.

The per-candidate canonicalize LLM call is the dominant cost of a compile. Its
keep/drop half is a role/shape judgment the feedback store learns, so we front
the LLM with a STORE-ONLY (``llm=False``) check: a CONFIDENT learned ``drop``
skips the LLM entirely, while ``keep`` or any abstain still pays for it (the
canonical form is generative — the store can't synthesize it). One-sided by
design: the store can only ever REMOVE an LLM call on a high-confidence reject.

These tests are hermetic: a deterministic content-addressed embedder gives each
distinct text a near-orthogonal vector (cosine 1.0 only on an exact taught
exemplar), and ``_call_ollama`` is monkeypatched to a tripwire so we can prove
it is NOT called on a deflection and IS called on an abstain. No network.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.multi_entity_llm as me
from app.core.decide import set_store
from app.core.feedback_store import Correction, FeedbackStore

_D = 128
_REL = "entity_keep:site"


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


# A single taught reject exemplar: a roster phrase that is NOT a real site.
_GHOST = "OPTBOT Facilities escort contact"
_DROP = Correction(
    id="ek_drop", relation=_REL, verdict="drop", created_by="pm",
    exemplars=[_GHOST],
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # Deflection on; teacher-cache off (its own test exercises learning).
    monkeypatch.setenv("SOWSMITH_ENRICH_STORE_DEFLECT", "1")
    monkeypatch.delenv("SOWSMITH_TEACHER_CACHE", raising=False)
    yield
    set_store(None)


def test_flag_off_always_calls_llm(monkeypatch):
    """With the flag off, behavior is byte-identical to before — the store is
    never consulted and the LLM decides every candidate."""
    monkeypatch.setenv("SOWSMITH_ENRICH_STORE_DEFLECT", "0")
    set_store(_store_with(_DROP))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=256):
        calls["n"] += 1
        return '{"keep": false}'

    monkeypatch.setattr(me, "_call_ollama", _fake_ollama)
    # Even the taught ghost reaches the LLM when deflection is disabled.
    assert me._canonicalize_candidate(_GHOST, "site") is None
    assert calls["n"] == 1


def test_confident_drop_deflects_no_llm(monkeypatch):
    """A taught reject shape is dropped by the store — the LLM is NEVER called."""
    set_store(_store_with(_DROP))

    def _tripwire(prompt, *, max_tokens=256):
        raise AssertionError("LLM must not be called on a confident store drop")

    monkeypatch.setattr(me, "_call_ollama", _tripwire)
    assert me._canonicalize_candidate(_GHOST, "site") is None


def test_untaught_shape_abstains_calls_llm(monkeypatch):
    """An untaught candidate gets store-abstain → falls through to the LLM,
    which keeps it. The store never fabricates a keep."""
    set_store(_store_with(_DROP))
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=256):
        calls["n"] += 1
        return '{"keep": true, "canonical": "ATL-HQ-01 OPTBOT Atlanta HQ", "city": "Atlanta", "state": "GA"}'

    monkeypatch.setattr(me, "_call_ollama", _fake_ollama)
    out = me._canonicalize_candidate("ATL-HQ-01 OPTBOT Atlanta HQ", "site")
    assert calls["n"] == 1
    assert isinstance(out, dict) and out.get("keep") is True


def test_no_store_calls_llm(monkeypatch):
    """No store wired → no deflection possible → LLM decides (safe fallback)."""
    set_store(None)
    calls = {"n": 0}

    def _fake_ollama(prompt, *, max_tokens=256):
        calls["n"] += 1
        return '{"keep": false}'

    monkeypatch.setattr(me, "_call_ollama", _fake_ollama)
    assert me._canonicalize_candidate(_GHOST, "site") is None
    assert calls["n"] == 1


def test_self_teach_warms_store(monkeypatch):
    """With teacher-cache ON, the LLM's keep/drop is folded back so the SAME
    shape deflects on the next pass without an LLM call."""
    monkeypatch.setenv("SOWSMITH_TEACHER_CACHE", "1")
    store = _store_with()  # empty: store abstains on first sight
    set_store(store)
    calls = {"n": 0}
    novel = "weekends only, after-hours by request"

    def _fake_ollama(prompt, *, max_tokens=256):
        calls["n"] += 1
        return '{"keep": false}'

    monkeypatch.setattr(me, "_call_ollama", _fake_ollama)

    # First pass: store abstains → LLM fires → drop is learned.
    assert me._canonicalize_candidate(novel, "site") is None
    assert calls["n"] == 1

    # Second pass: the learned drop should deflect — LLM must not fire again.
    def _tripwire(prompt, *, max_tokens=256):
        raise AssertionError("learned drop should deflect without an LLM call")

    monkeypatch.setattr(me, "_call_ollama", _tripwire)
    assert me._canonicalize_candidate(novel, "site") is None
