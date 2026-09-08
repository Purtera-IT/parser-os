"""Ask a head what it would say, without teaching it anything.

The tier head fills the six rubric dimensions OrbitBrief cannot derive — but
only once a PM has decided some. Until then it must abstain, and an abstention
has to be returned as one rather than as a guess.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_feedback as rf
from app.core.decide import set_store
from app.core.feedback_store import FeedbackStore


def _embed(texts):
    """Deterministic stand-in: a stable pseudo-random unit vector per string.

    Not a character histogram — these exemplars share a long prefix ("Contractual
    risk · …"), so a histogram makes every one of them nearly parallel, the
    margin collapses and the head abstains on everything. That is a property of
    the double, not of the head, and it would have read as a broken endpoint.

    This double separates distinct strings and maps identical ones together, so
    it exercises the plumbing. It deliberately models NO generalisation between
    similar-but-different shapes — the real embedder's job, tested where the
    real embedder is.
    """
    import hashlib

    import numpy as np

    out = []
    for t in texts:
        digest = hashlib.blake2b(str(t).encode("utf-8"), digest_size=32).digest()
        vec = [(b - 127.5) / 127.5 for b in digest]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        out.append([v / norm for v in vec])
    # An ARRAY, not a list. `FeedbackStore.resolve` does `q.shape[0]`, and the
    # whole method is wrapped in a bare `except` — so a list-returning double
    # raises AttributeError, is swallowed, and every prediction comes back None.
    # That read exactly like a broken endpoint for half an hour.
    return np.array(out, dtype=float)


@pytest.fixture()
def client():
    set_store(FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True))
    api = FastAPI()
    api.include_router(rf.router)
    yield TestClient(api)
    set_store(None)


def _predict(client, **body):
    return client.post("/projects/p1/feedback/predict", json=body)


def test_an_untaught_head_abstains_rather_than_guessing(client) -> None:
    r = _predict(client, head="tier", texts=["Contractual risk · 1 site · single metro"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["relation"] == "project_tier"
    assert body["candidates"] == ["1", "2", "3"]
    assert body["predictions"][0]["verdict"] is None


def test_an_unknown_head_is_refused(client) -> None:
    r = _predict(client, head="not_a_head", texts=["x"])
    assert r.status_code == 422


def test_a_blank_exemplar_is_an_abstention_not_an_error(client) -> None:
    r = _predict(client, head="tier", texts=["   "])
    assert r.status_code == 200
    assert r.json()["predictions"][0]["verdict"] is None


def test_it_answers_one_prediction_per_exemplar(client) -> None:
    texts = ["Contractual risk · a", "Concurrency · b", "Scope novelty · c"]
    r = _predict(client, head="tier", texts=texts)
    assert [p["text"] for p in r.json()["predictions"]] == texts


def _teach(client, text, verdict):
    return client.post(
        "/projects/p1/feedback/correction",
        json={
            "head": "tier", "deal_id": "d1", "target_id": "project_tier:contractual_risk",
            "text": text, "old_value": "", "new_value": verdict,
            "scope": "global", "candidates": ["1", "2", "3"],
        },
    )


def test_one_lesson_answers_an_identical_deal(client) -> None:
    """Two paths, two bars.

    `_relation_head` fits a learned boundary only once it holds two DISTINCT
    verdicts. Below that, `resolve` still falls through to cosine, which fires
    on a correction whose exemplar the query matches — so one lesson answers a
    deal of exactly that shape, and it takes two before anything is
    generalised to a shape nobody has judged.
    """
    big = "Contractual risk · 16+ sites · multi-region or national · over 8 weeks"
    assert _teach(client, big, "3").status_code == 200
    r = _predict(client, head="tier", texts=[big])
    assert r.status_code == 200, r.text
    assert r.json()["predictions"][0]["verdict"] == "3"


def test_one_lesson_does_not_answer_a_different_deal(client) -> None:
    """One PM decision is an anecdote. It must not become an opinion about
    every deal — which is why the panel keeps these six blank until the head
    has actually seen a boundary."""
    _teach(client, "Contractual risk · 16+ sites · multi-region", "3")
    r = _predict(client, head="tier", texts=["Contractual risk · 1 site · single metro"])
    assert r.status_code == 200
    assert r.json()["predictions"][0]["verdict"] is None


def test_two_unlike_lessons_let_it_answer(client) -> None:
    """The whole point: the six dimensions OrbitBrief cannot derive get filled
    from what PMs decided on deals that looked like this one."""
    big = "Contractual risk · 16+ sites · multi-region or national · over 8 weeks"
    small = "Contractual risk · 1 site · single metro · under 1 week"
    assert _teach(client, big, "3").status_code == 200
    assert _teach(client, small, "1").status_code == 200

    r = _predict(client, head="tier", texts=[big, small])
    assert r.status_code == 200, r.text
    verdicts = [p["verdict"] for p in r.json()["predictions"]]
    assert verdicts == ["3", "1"], verdicts
