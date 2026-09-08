"""A reason can carry a circumstance, and a circumstance is the nuance.

"Customer paper, because this customer always insists on their own MSA" is a
blanket rule. "…because Chase negotiated this one" is not — it holds when
Chase owns the deal and nowhere else. `condition_holds` already gates a
correction on exactly that.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_feedback as rf
from app.core.decide import set_store
from app.core.feedback_store import FeedbackStore, condition_holds


def _embed(texts):
    out = []
    for t in texts:
        digest = hashlib.blake2b(str(t).encode("utf-8"), digest_size=32).digest()
        vec = [(b - 127.5) / 127.5 for b in digest]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        out.append([v / norm for v in vec])
    return np.array(out, dtype=float)


@pytest.fixture()
def client():
    set_store(FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True))
    api = FastAPI()
    api.include_router(rf.router)
    yield TestClient(api)
    set_store(None)


def _correct(client, rationale: str):
    return client.post(
        "/projects/p1/feedback/correction",
        json={
            "head": "tier", "deal_id": "d1", "target_id": "project_tier:contractual_risk",
            "text": "contractual_risk · 16+ · Multi-region", "old_value": "1",
            "new_value": "3", "scope": "global", "candidates": ["1", "2", "3"],
            "rationale": rationale,
        },
    )


def _stored_condition(client):
    from app.core.decide import get_store

    corrections = get_store().all_corrections(active_only=True)
    assert corrections, "the correction was not stored"
    return (corrections[-1].relations or {}).get("when") or {}


def test_a_reason_naming_a_person_becomes_a_condition(client) -> None:
    assert _correct(client, "because Chase negotiated this one").status_code == 200
    condition = _stored_condition(client)
    assert condition.get("field") == "owner"
    assert "chase" in str(condition.get("equals", ""))


def test_a_conditional_lesson_stays_silent_on_another_deal(client) -> None:
    """Firing one person's way of working on everybody's deals is the failure
    this prevents."""
    _correct(client, "because Chase negotiated this one")
    condition = {"when": _stored_condition(client)}
    assert condition_holds(condition, {"owner": "chase_whitfield"}) is True
    assert condition_holds(condition, {"owner": "dana_lee"}) is False
    # No facts supplied at all -> silence, not a guess.
    assert condition_holds(condition, None) is False


def test_a_general_reason_stays_a_blanket_rule(client) -> None:
    assert _correct(client, "because this customer always insists on their own MSA").status_code == 200
    assert _stored_condition(client) == {}


def test_no_reason_is_still_a_valid_correction(client) -> None:
    assert _correct(client, "").status_code == 200
    assert _stored_condition(client) == {}


def test_an_explicit_condition_from_the_caller_wins(client) -> None:
    """A caller that already knows the circumstance must not have it
    second-guessed by a sentence."""
    r = client.post(
        "/projects/p1/feedback/correction",
        json={
            "head": "tier", "deal_id": "d1", "target_id": "t",
            "text": "x", "old_value": "1", "new_value": "3", "scope": "global",
            "candidates": ["1", "2", "3"],
            "rationale": "because Chase negotiated this one",
            "relations": {"when": {"field": "owner", "equals": "dana_lee"}},
        },
    )
    assert r.status_code == 200
    assert _stored_condition(client).get("equals") == "dana_lee"
