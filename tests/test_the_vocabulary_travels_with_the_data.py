"""A relation's closed set has to live on the data, not in a static table.

The audit on 2026-09-05: the live store held **15 relations** against **10
declared heads**, and only two of those declared a candidate set. The other
eight — `sgate_*`, `physical_site`, `physical_site_city_shape`,
`*_noise_admission`, `task_site_anchor`, `site_candidate_role` — are minted at
runtime by `plain_rule_compiler` and can never be in the table.

So a verdict was checkable for `gap` and `admission` and nothing else. That is
how `not_relevant` reached the gap head and sat there for weeks, shaping a
boundary toward a class the consumers filter out.

A static registry cannot fix this, because the relations that need it do not
exist until a rule is synthesised. The vocabulary travels with the correction
instead.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_feedback as rf
from app.core.decide import set_store
from app.core.feedback_store import Correction, FeedbackStore

_D = 16


def _embed(texts: list[str]) -> np.ndarray:
    out = np.ones((len(texts), _D), dtype=np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _store(path: str = ":memory:") -> FeedbackStore:
    return FeedbackStore(path, embed_fn=_embed, reachable_fn=lambda: True)


def test_a_correction_remembers_its_own_vocabulary() -> None:
    st = _store()
    st.add(
        Correction(
            id="c1", relation="sgate_vendor", verdict="real_site",
            candidates=["real_site", "reject"], exemplars=["a vendor line"],
        )
    )
    (got,) = st.all_corrections(active_only=False)
    assert got.candidates == ["real_site", "reject"]


def test_an_open_vocabulary_is_still_allowed() -> None:
    """Extract heads have no closed set — the verdict IS the value. Empty means
    open, and constraining them would break them."""
    st = _store()
    st.add(Correction(id="c2", relation="value_norm", verdict="$4,200.00", exemplars=["x"]))
    assert st.all_corrections(active_only=False)[0].candidates == []


def test_a_july_database_still_opens_and_still_takes_writes() -> None:
    """`CREATE TABLE IF NOT EXISTS` does not add a column to an existing table,
    and these DBs outlive deploys. Without the migration every insert against an
    older file fails on the new column."""
    path = os.path.join(tempfile.mkdtemp(), "old.db")
    con = sqlite3.connect(path)
    con.executescript(
        """CREATE TABLE corrections (
             id TEXT PRIMARY KEY, relation TEXT NOT NULL, verdict TEXT NOT NULL,
             scope TEXT NOT NULL DEFAULT 'global', scope_key TEXT NOT NULL DEFAULT '',
             exemplars TEXT NOT NULL DEFAULT '[]', threshold REAL NOT NULL DEFAULT 0.82,
             relations TEXT NOT NULL DEFAULT '{}', instruction TEXT NOT NULL DEFAULT '',
             complaint_id TEXT, created_by TEXT NOT NULL DEFAULT '',
             created_at REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0,
             status TEXT NOT NULL DEFAULT 'active', supersedes TEXT,
             confidence_floor REAL NOT NULL DEFAULT 0, hit_count INTEGER NOT NULL DEFAULT 0,
             last_fired REAL, wrongful_override_count INTEGER NOT NULL DEFAULT 0);"""
    )
    con.execute(
        "INSERT INTO corrections (id,relation,verdict,exemplars) "
        "VALUES ('old1','gap_valid','invalid','[\"a July lesson\"]')"
    )
    con.commit()
    con.close()

    st = _store(path)
    kept = st.all_corrections(active_only=False)
    assert [c.id for c in kept] == ["old1"], "an existing lesson must survive the migration"
    assert kept[0].candidates == [], "unknown, not wrong — nothing to enforce against"
    st.add(Correction(id="new1", relation="gap_valid", verdict="valid", exemplars=["b"]))
    assert {c.id for c in st.all_corrections(active_only=False)} == {"old1", "new1"}


@pytest.fixture
def client():
    set_store(_store())
    api = FastAPI()
    api.include_router(rf.router)
    yield TestClient(api)
    set_store(None)


def _post(client, **over):
    body = {
        "head": "gap", "deal_id": "d1", "target_id": "t1",
        "text": "Who signs site acceptance?", "new_value": "invalid", "scope": "deal",
    }
    body.update(over)
    return client.post("/projects/p1/feedback/correction", json=body)


def test_the_callers_own_closed_set_is_enforced(client) -> None:
    """A rule synthesised at runtime knows a vocabulary the registry cannot."""
    r = _post(client, head="norm", new_value="maybe", candidates=["yes", "no"])
    assert r.status_code == 422
    assert "yes" in r.json()["detail"]


def test_it_is_recorded_so_the_next_check_needs_no_table(client) -> None:
    from app.core.decide import get_store

    assert _post(client, candidates=["valid", "invalid"]).status_code == 200
    (c,) = get_store().all_corrections(active_only=False)
    assert c.candidates == ["valid", "invalid"]


def test_the_head_declaration_still_applies_when_the_caller_sends_none(client) -> None:
    assert _post(client, new_value="not_relevant").status_code == 422
    assert _post(client, new_value="invalid").status_code == 200
