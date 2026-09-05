"""A verdict the head has no class for is contamination, not a lesson.

Found live 2026-09-05. The gap head declares
``HeadSpec("gap_valid", candidates=("valid", "invalid"))``, and the dev store
held 7 corrections for it — 5 of them reading ``not_relevant``, written by a
browser client that had invented its own vocabulary.

That is not inert. ``FeedbackStore._relation_head`` fits over EVERY stored
verdict for a relation on purpose ("the head learns the full boundary"), while
consumers filter to the declared set at decision time. So those 5 rows:

  * pulled the fitted boundary toward a class nothing can act on, and
  * left the head with ZERO exemplars of ``invalid`` — every dismissal a PM had
    ever made was missing from the vocabulary that decides.

The endpoint checked that the HEAD existed and never that the VERDICT did.
Refusing it here, once, is what stops the next client repeating it.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_feedback as rf
from app.core.decide import set_store
from app.core.feedback_store import FeedbackStore
from app.core.pm_feedback import HEAD_REGISTRY

_D = 32


def _embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, ch in enumerate(t.lower()[:_D]):
            out[i, j] = (ord(ch) % 17) / 17.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


def _client() -> TestClient:
    api = FastAPI()
    api.include_router(rf.router)
    return TestClient(api)


@pytest.fixture(autouse=True)
def _store():
    set_store(FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True))
    yield
    set_store(None)


def _post(verdict: str, head: str = "gap"):
    return _client().post(
        "/projects/p1/feedback/correction",
        json={
            "head": head,
            "deal_id": "d1",
            "target_id": "pmcover.hardware_furnish",
            "text": "What hardware is customer-furnished vs PurTera-furnished?",
            "old_value": "valid",
            "new_value": verdict,
            "scope": "deal",
        },
    )


def test_the_verdict_the_browser_was_sending_is_refused() -> None:
    r = _post("not_relevant")
    assert r.status_code == 422
    detail = r.json()["detail"]
    # The refusal has to say what IS allowed, or the caller can only guess.
    assert "valid" in detail and "invalid" in detail
    assert "not_relevant" in detail


def test_the_head_s_own_verdicts_are_accepted() -> None:
    for verdict in HEAD_REGISTRY["gap"].candidates:
        r = _post(verdict)
        assert r.status_code == 200, r.text
        assert r.json()["relation"] == "gap_valid"


def test_a_head_that_declares_no_candidates_is_left_alone() -> None:
    """Extract heads (`norm`, `terminology`) have an open vocabulary — the
    verdict IS the value. Constraining them would break them."""
    spec = HEAD_REGISTRY["norm"]
    assert not spec.candidates
    r = _post("$4,200.00", head="norm")
    assert r.status_code == 200, r.text


def test_every_classify_head_with_candidates_is_covered() -> None:
    """Guards the rule rather than one head: any head that declares a closed
    set gets the same protection, including ones added later."""
    closed = [k for k, s in HEAD_REGISTRY.items() if s.candidates]
    assert "gap" in closed and "admission" in closed
    for key in closed:
        r = _post("definitely-not-a-real-verdict", head=key)
        assert r.status_code == 422, f"{key} accepted a verdict it has no class for"
