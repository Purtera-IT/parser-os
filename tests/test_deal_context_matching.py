"""Lessons from the same customer or partner answer first.

The Deal Kit sends every lesson with its deal's context and, before a compile,
the context of the deal being compiled. Among lessons that already match the
work, the one taught for the same customer (then partner, then a similar
industry / deal type / vendor / kind of work) wins. The context never lets a
lesson fire that did not match the work on its own.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_feedback as rf
from app.core import decide as decide_mod
from app.core.decide import DecisionScope, decide
from app.core.feedback_store import FeedbackStore, context_affinity
from app.core.pm_feedback import apply_pm_correction

_AXES = ["access point", "ceiling", "cat6", "rack"]


def _embed(texts):
    out = np.zeros((len(texts), len(_AXES) + 1), dtype=np.float32)
    for i, t in enumerate(texts):
        low = (t or "").lower()
        hit = False
        for j, k in enumerate(_AXES):
            if k in low:
                out[i, j] = 1.0
                hit = True
        if not hit:
            out[i, len(_AXES)] = 1.0
        out[i] /= max(float(np.linalg.norm(out[i])), 1e-9)
    return out


def _store() -> FeedbackStore:
    s = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    s._enable_head = False
    return s


CDW = {"customer": "Five Below", "customerId": "a1", "channel": "partner", "partner": "CDW", "industry": "Retail", "dealType": "fixed", "vendors": ["meraki"], "work": ["wireless"]}
DIRECT = {"customer": "Chase", "customerId": "a3", "channel": "direct", "industry": "Banking", "dealType": "tm", "vendors": ["cisco"], "work": ["network"]}


def _teach(store, deal_id, text, verdict, ctx):
    return apply_pm_correction(store, {
        "head": "hours", "dealId": deal_id, "compileId": "", "targetId": f"task:{verdict}",
        "text": text, "oldValue": "", "newValue": verdict, "scope": "global", "context": "",
        "rationale": "", "relations": {"outcome": "new", "deal_context": ctx}, "pm": "pm@x", "candidates": [],
    })


def test_affinity_levels():
    assert context_affinity(CDW, {**CDW, "customer": "Ulta", "customerId": "a2"}) == ("partner", 0.05)
    assert context_affinity(CDW, {"customerId": "a1"}) == ("customer", 0.06)
    assert context_affinity(CDW, {"industry": "Retail", "work": ["wireless", "cabling"]}) == ("similar", 0.02)
    assert context_affinity(CDW, DIRECT) == ("", 0.0)
    assert context_affinity(None, CDW) == ("", 0.0)


def test_the_same_partners_lesson_answers_first_and_context_never_clears_a_threshold():
    store = _store()
    decide_mod.set_store(store)
    try:
        # Two lessons about AP installs, equally close to the new line: one from a
        # direct Chase deal, one from a CDW deal.
        _teach(store, "chase-1", "Mount access points in the ceiling", "hours=1;per=AP;qty=10;role=L2", DIRECT)
        _teach(store, "cdw-1", "Mount access points in the ceiling", "hours=1.5;per=AP;qty=12;role=L2", CDW)
        line = "Mount 12 access points in the ceiling"

        # A new CDW deal (another customer) gets CDW's answer.
        store.set_deal_context("cdw-new", {**CDW, "customer": "Ulta", "customerId": "a2"})
        d = decide("task_hours", line, ["hours=1;per=AP;qty=10;role=L2", "hours=1.5;per=AP;qty=12;role=L2"],
                   instruction="hours", llm=False, scope=DecisionScope(deal_id="cdw-new"))
        assert d.verdict == "hours=1.5;per=AP;qty=12;role=L2"
        assert "same partner (CDW)" in d.rationale

        # A new Chase deal gets Chase's.
        store.set_deal_context("chase-new", DIRECT)
        d = decide("task_hours", line, ["hours=1;per=AP;qty=10;role=L2", "hours=1.5;per=AP;qty=12;role=L2"],
                   instruction="hours", llm=False, scope=DecisionScope(deal_id="chase-new"))
        assert d.verdict == "hours=1;per=AP;qty=10;role=L2"
        assert "same customer (Chase)" in d.rationale

        # Different work: the CDW lesson does not fire just because the partner matches.
        d = decide("task_hours", "Rack and stack the core switch", ["hours=1.5;per=AP;qty=12;role=L2"],
                   instruction="hours", llm=False, scope=DecisionScope(deal_id="cdw-new"))
        assert d.verdict is None
    finally:
        decide_mod.set_store(None)


def test_the_brief_sends_the_deal_context_and_a_lesson_records_its_own():
    store = _store()
    rf_store = store
    api = FastAPI()
    api.include_router(rf.router)
    decide_mod.set_store(rf_store)
    try:
        c = TestClient(api)
        r = c.post("/projects/deal-9/feedback/deal-context", json={"context": {**CDW, "secret": "x"}})
        assert r.status_code == 200 and r.json()["stored"] is True
        assert store.get_deal_context("deal-9")["partner"] == "CDW"
        assert "secret" not in store.get_deal_context("deal-9")
        r = c.post("/projects/deal-10/feedback/correction", json={
            "head": "hours", "deal_id": "deal-10", "text": "Mount access points", "new_value": "hours=1;per=AP;qty=4",
            "scope": "global", "relations": {"outcome": "new", "deal_context": DIRECT}, "pm": "pm@x",
        })
        assert r.status_code == 200
        assert store.get_deal_context("deal-10")["customer"] == "Chase"
        corr = store.get(r.json()["correction_id"])
        assert corr.relations["deal_context"]["customer"] == "Chase"
    finally:
        decide_mod.set_store(None)
