"""PM feedback loop endpoints (upgrade #4).

End-to-end over the HTTP surface, but hermetic: a deterministic marker-clustering
embedder backs an in-memory feedback store (wired via decide.set_store), the LLM
synthesizer is monkeypatched to a fixed proposal, and the compile-result loader
is stubbed. No network, no DB file, no Ollama. We mount ONLY the feedback router
so app startup / init_db never runs.

What we prove:
* POST /feedback/rule commits a clean rule and refuses a collateral-damaging one.
* GET  /feedback/corrections surfaces the committed rule (provenance).
* POST /feedback/complaint never commits ungated (no controls), and commits when
  control probes are supplied and the gate passes.
* Every endpoint 409s when no store is active (never guesses).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_feedback as rf
import app.core.plain_rule_compiler as prc
from app.core.decide import set_store
from app.core.feedback_store import FeedbackStore
from app.core.schemas import CompileResult

_D = 64
_MARKERS = ["purtera", "pricebook"]


def _embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        tl = t.lower()
        placed = False
        for j, m in enumerate(_MARKERS):
            if m in tl:
                out[i, j] = 1.0
                placed = True
        if not placed:
            h = abs(hash(tl))
            out[i, len(_MARKERS) + (h % (_D - len(_MARKERS)))] = 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


def _client() -> TestClient:
    api = FastAPI()
    api.include_router(rf.router)
    return TestClient(api)


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)


def _clean_synth(sentence: str) -> dict:
    return {
        "relation": "physical_site",
        "verdict": "drop",
        "candidates": ["keep", "drop"],
        "exemplar": "PurTera headquarters",
        "paraphrases": ["our company PurTera", "PurTera Inc office"],
        "controls": ["Atlanta data center MDF", "Santa Fe warehouse"],
        "scope": "global",
        "scope_key": "",
    }


@pytest.fixture(autouse=True)
def _isolate():
    yield
    set_store(None)


def test_rule_route_commits_clean_rule(monkeypatch):
    monkeypatch.setattr(prc, "_default_synthesizer", _clean_synth)
    set_store(_store())
    r = _client().post(
        "/projects/p1/feedback/rule",
        json={"sentence": "PurTera is our company, never a site.", "created_by": "pm"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed"] is True
    assert body["correction_id"]
    assert body["relation"] == "physical_site" and body["verdict"] == "drop"
    assert body["failed_invariants"] == []


def test_rule_route_refuses_collateral(monkeypatch):
    def _greedy(sentence: str) -> dict:
        p = _clean_synth(sentence)
        p["controls"] = ["PurTera adjacent REAL site", "Santa Fe warehouse"]
        return p

    monkeypatch.setattr(prc, "_default_synthesizer", _greedy)
    set_store(_store())
    r = _client().post(
        "/projects/p1/feedback/rule", json={"sentence": "PurTera is ours."}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed"] is False
    assert "D" in body["failed_invariants"]


def test_corrections_listing_surfaces_committed_rule(monkeypatch):
    monkeypatch.setattr(prc, "_default_synthesizer", _clean_synth)
    set_store(_store())
    client = _client()
    client.post(
        "/projects/p1/feedback/rule", json={"sentence": "PurTera is our company."}
    )
    r = client.get("/projects/p1/feedback/corrections")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] >= 1
    rels = {c["relation"] for c in body["items"]}
    assert "physical_site" in rels


def test_complaint_ungated_does_not_commit(monkeypatch):
    monkeypatch.setattr(
        rf, "_load_compile_result", lambda pid: CompileResult(project_id=pid, atoms=[], entities=[], edges=[], packets=[])
    )
    set_store(_store())
    r = _client().post(
        "/projects/p1/feedback/complaint",
        json={
            "relation": "physical_site",
            "desired_verdict": "drop",
            "candidates": ["keep", "drop"],
            "text": "PurTera headquarters",
            # no controls → cannot prove no-collateral → must NOT commit
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed"] is False
    assert "UNGATED" in body["report"]
    # Store learned nothing.
    assert _client().get("/projects/p1/feedback/corrections").json()["total"] == 0


def test_complaint_gated_commits_with_controls(monkeypatch):
    monkeypatch.setattr(
        rf, "_load_compile_result", lambda pid: CompileResult(project_id=pid, atoms=[], entities=[], edges=[], packets=[])
    )
    set_store(_store())
    r = _client().post(
        "/projects/p1/feedback/complaint",
        json={
            "relation": "physical_site",
            "desired_verdict": "drop",
            "candidates": ["keep", "drop"],
            "text": "PurTera headquarters",
            "paraphrases": ["our company PurTera"],
            "controls": [
                {"text": "Atlanta data center MDF", "candidates": ["keep", "drop"]},
                {"text": "Santa Fe warehouse", "candidates": ["keep", "drop"]},
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed"] is True, body
    assert body["failed_invariants"] == []


def test_correction_chip_commits(monkeypatch):
    monkeypatch.setattr(
        rf,
        "_load_compile_result",
        lambda pid: CompileResult(project_id=pid, atoms=[], entities=[], edges=[], packets=[]),
    )
    set_store(_store())
    r = _client().post(
        "/projects/p1/feedback/correction",
        json={
            "head": "type",
            "text": "Install forty-eight wireless access points",
            "old_value": "scope_item",
            "new_value": "work_scope_item",
            "candidates": ["scope_item", "work_scope_item"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed"] is True
    assert body["relation"] == "atom_type"
    assert body["verdict"] == "work_scope_item"
    assert _client().get("/projects/p1/feedback/corrections").json()["total"] == 1


def test_endpoints_409_without_store(monkeypatch):
    # Ensure no store and no env-driven wiring.
    set_store(None)
    monkeypatch.delenv("SOWSMITH_FEEDBACK_STORE_DB", raising=False)
    client = _client()
    r1 = client.post("/projects/p1/feedback/rule", json={"sentence": "x"})
    r2 = client.get("/projects/p1/feedback/corrections")
    r3 = client.post(
        "/projects/p1/feedback/correction",
        json={"head": "type", "text": "x", "new_value": "work_scope_item"},
    )
    assert r1.status_code == 409
    assert r2.status_code == 409
    assert r3.status_code == 409
