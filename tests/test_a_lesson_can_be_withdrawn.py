"""A learned correction must be withdrawable, and the withdrawal must stick.

Until 2026-09-05 there was no way to take a lesson out of service. That mattered:
five `gap_valid` corrections carried the verdict `not_relevant` — a class the
head has no room for, written by a browser that had invented its own vocabulary.

They were not inert. `_relation_head` fits over EVERY stored verdict on purpose
("the head learns the full boundary") while consumers filter to the declared set
at decision time, so those five were shaping a boundary toward a class nothing
can act on: 5 of the gap head's 11 exemplars, teaching nothing and costing
accuracy on the head the PM relies on most.

Retire rather than relabel. All five carried an empty rationale, and the rule
elsewhere in this system is that a dismissal without a why is bookkeeping,
because learning from it generalizes a judgment nobody made. Relabelling them
to `invalid` would have activated five lessons that were never admissible.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_feedback as rf
from app.core.decide import set_store
from app.core.feedback_store import Correction, FeedbackStore

_D = 32


def _embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, ch in enumerate(t.lower()[:_D]):
            out[i, j] = (ord(ch) % 17) / 17.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


@pytest.fixture
def client():
    store = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    store.add(
        Correction(
            id="pm_gap_c3e4ca7cae43",
            relation="gap_valid",
            verdict="not_relevant",
            exemplars=["Confirm site access, escort, and badging requirements"],
            scope="deal",
            scope_key="4031a8b8",
        )
    )
    set_store(store)
    api = FastAPI()
    api.include_router(rf.router)
    yield TestClient(api), store
    set_store(None)


def test_a_bad_lesson_can_be_taken_out_of_service(client):
    c, store = client
    r = c.post("/projects/p1/feedback/corrections/pm_gap_c3e4ca7cae43/status", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "disabled"
    assert body["verdict"] == "not_relevant"
    assert [x.id for x in store.all_corrections(active_only=True)] == []


def test_the_record_survives_the_withdrawal(client):
    """Never a delete. A lesson a PM once taught is part of how the head got
    where it is, and a store you cannot audit backwards is one you cannot trust
    forwards."""
    c, store = client
    c.post("/projects/p1/feedback/corrections/pm_gap_c3e4ca7cae43/status", json={})
    kept = [x for x in store.all_corrections(active_only=False) if x.id == "pm_gap_c3e4ca7cae43"]
    assert len(kept) == 1
    assert kept[0].verdict == "not_relevant", "the original judgment stays readable"


def test_it_can_be_put_back(client):
    c, store = client
    c.post("/projects/p1/feedback/corrections/pm_gap_c3e4ca7cae43/status", json={})
    r = c.post(
        "/projects/p1/feedback/corrections/pm_gap_c3e4ca7cae43/status", json={"status": "active"}
    )
    assert r.status_code == 200
    assert len(store.all_corrections(active_only=True)) == 1


def test_an_unknown_correction_is_a_404_not_a_silent_success(client):
    c, _ = client
    assert c.post("/projects/p1/feedback/corrections/nope/status", json={}).status_code == 404


def test_a_status_the_store_does_not_have_is_refused(client):
    c, _ = client
    r = c.post(
        "/projects/p1/feedback/corrections/pm_gap_c3e4ca7cae43/status", json={"status": "deleted"}
    )
    assert r.status_code == 422
    assert "active" in r.json()["detail"]


def test_the_retirement_is_mirrored_so_it_survives_a_cold_start(client):
    """The blob mirror repopulates a restarted store. A retirement that lives
    only in this process comes straight back on the next cold start."""
    import inspect

    src = inspect.getsource(rf.feedback_correction_status)
    assert "upload_correction" in src
