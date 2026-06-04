"""LLM gating + teacher-cache: the LLM fires only on hard cases, and its own
confident answers train the store so its call-rate decays.

Uses the real FeedbackStore (in-memory, fake embedder) and a stubbed LLM so we
can count calls deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.decide as decide
import app.core.semantic_role as semantic_role
from app.core.decide import (
    DecisionScope,
    decide as decide_fn,
    get_decide_stats,
    reset_decide_stats,
    set_store,
)
from app.core.feedback_store import FeedbackStore

_D = 32


def _fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        tl = t.lower()
        out[i, 0] = 3.0
        out[i, 1] = 1.0 if "vendor" in tl or "billing" in tl else (
            -1.0 if "site" in tl or "project" in tl else 0.0)
        out[i, 2 + (abs(hash(t)) % (_D - 2))] = 0.05
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


@pytest.fixture(autouse=True)
def _wire(monkeypatch):
    monkeypatch.setenv("SOWSMITH_TEACHER_CACHE", "1")
    reset_decide_stats()
    store = FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)
    set_store(store)
    yield store
    set_store(None)
    reset_decide_stats()


def _make_llm(counter):
    """A stubbed classifier that always confidently answers by keyword, and
    increments a call counter so we can watch the LLM call-rate."""
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        counter["n"] += 1
        tl = text.lower()
        if "vendor" in tl or "billing" in tl:
            return ("vendor_or_billing_address", 0.95)
        return ("job_site", 0.95)
    return _clf


def test_llm_trains_store_and_call_rate_decays(monkeypatch, _wire):
    counter = {"n": 0}
    monkeypatch.setattr(semantic_role, "classify_role", _make_llm(counter))
    cands = ["job_site", "vendor_or_billing_address"]

    # Feed a stream of vendor/site decisions. Early on the store is empty so the
    # LLM is consulted; each confident answer is cached as a teacher correction.
    vendor_texts = [f"vendor billing letterhead office {i}" for i in range(8)]
    site_texts = [f"customer project install site {i}" for i in range(8)]
    stream = [t for pair in zip(vendor_texts, site_texts) for t in pair]

    for t in stream:
        decide_fn("physical_site", t, cands, instruction="role?", scope=DecisionScope())

    early_calls = counter["n"]
    assert early_calls > 0  # LLM did real work on the cold start

    # Now the head should have engaged (>=3 per class learned). A fresh batch of
    # SAME-REGION texts should be decided by the store with far fewer LLM calls.
    counter["n"] = 0
    reset_decide_stats()
    fresh = [f"vendor billing remittance address {i}" for i in range(6)] + \
            [f"field project install site premises {i}" for i in range(6)]
    for t in fresh:
        decide_fn("physical_site", t, cands, instruction="role?", scope=DecisionScope())

    stats = get_decide_stats()
    # The crux: the LLM is no longer the workhorse — most decisions are now
    # answered by the learned store, and the LLM call-rate has dropped.
    assert stats["store_hits"] > 0
    assert stats["llm_call_rate"] < 1.0
    assert counter["n"] < len(fresh)


def test_teacher_write_is_idempotent(_wire, monkeypatch):
    counter = {"n": 0}
    monkeypatch.setattr(semantic_role, "classify_role", _make_llm(counter))
    store = _wire
    # Two identical hard decisions → one teacher row (content-addressed id).
    for _ in range(2):
        store.learn_from_teacher(
            relation="physical_site", text="vendor billing letterhead",
            verdict="vendor_or_billing_address", confidence=0.95,
            scope=DecisionScope(),
        )
    teacher = [c for c in store.all_corrections() if c.created_by == "teacher"]
    assert len(teacher) == 1


def test_low_confidence_llm_answer_not_cached(_wire, monkeypatch):
    counter = {"n": 0}

    def _weak(text, candidates, *, instruction, context="", timeout=None, model=None):
        counter["n"] += 1
        return ("job_site", 0.5)  # below SOWSMITH_TEACHER_MIN_CONF default 0.85

    monkeypatch.setattr(semantic_role, "classify_role", _weak)
    decide_fn("physical_site", "ambiguous address blob", ["job_site", "vendor_or_billing_address"],
              instruction="role?", scope=DecisionScope())
    teacher = [c for c in _wire.all_corrections() if c.created_by == "teacher"]
    assert teacher == []  # never learn from an unsure teacher
