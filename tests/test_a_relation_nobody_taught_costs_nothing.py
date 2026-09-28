"""A decision about a relation nobody ever taught must be free.

The dev store holds 443 corrections and not one of them is `sheet_role`. That
head is asked once per sheet of every spreadsheet on a deal, and every call
did this before answering None:

    * probe the embedder's reachability,
    * `SELECT * FROM corrections` and deserialise all 443 rows,
    * filter them,
    * and then read and deserialise all 443 rows AGAIN to build the "0 of 443"
      log line.

886 row deserialisations per decision, for an answer guaranteed by the
contents of the store. It surfaced as `parse_artifacts` being slow -- the
stage where the sheet parsers run -- which is where a 45-document deal spent
the budget it then died for want of.
"""
from __future__ import annotations

import time

import pytest

from app.core.feedback_store import (
    SCOPE_GLOBAL,
    Correction,
    DecisionScope,
    FeedbackStore,
)


@pytest.fixture()
def store(tmp_path):
    s = FeedbackStore(str(tmp_path / "fb.db"))
    s._reachable_fn = lambda: True          # no network in a unit test
    return s


def _taught(store, relation="atom_type", verdict="task"):
    store.add(Correction(
        id=f"c_{relation}_{verdict}", relation=relation, verdict=verdict,
        scope=SCOPE_GLOBAL, exemplars=["install the cabling"],
        candidates=[verdict, "scope_item"],
    ))


def _resolve(store, relation, candidates=("a", "b")):
    return store.resolve(
        relation=relation, text="some sheet name", candidates=list(candidates),
        context="", scope=DecisionScope(deal_id="d"), instruction="",
        relations=None,
    )


def _counting(store):
    """Count the reachability probes and table reads a resolve performs."""
    calls = {"reach": 0, "read": 0}
    store._reachable_fn = lambda: (calls.__setitem__("reach", calls["reach"] + 1) or True)
    real = store.all_corrections

    def counted(**kw):
        calls["read"] += 1
        return real(**kw)

    store.all_corrections = counted
    return calls


def test_an_untaught_relation_costs_nothing_once_warm(store):
    """No probe, no table read. The relation index warms once per write, not
    once per decision."""
    _taught(store)
    store._relations_present()               # warm, as the first decision does
    calls = _counting(store)
    for _ in range(50):
        assert _resolve(store, "sheet_role") is None
    assert calls["reach"] == 0, "probed the embedder for a relation nobody taught"
    assert calls["read"] == 0, "read the corrections table to answer a guaranteed None"


def test_a_taught_relation_still_gets_the_full_path(store):
    """The short-circuit must not swallow a relation that HAS lessons: it has
    to get past the index and reach the embedder probe."""
    _taught(store)
    store._relations_present()
    calls = _counting(store)
    _resolve(store, "atom_type", candidates=("task", "scope_item"))
    assert calls["reach"] == 1, "a taught relation was short-circuited"


def test_teaching_a_relation_makes_it_visible_immediately(store):
    """The cache must not outlive the write that changes the answer."""
    assert "bom_owner" not in store._relations_present()
    _taught(store, relation="bom_owner", verdict="we_supply")
    assert "bom_owner" in store._relations_present()


def test_disabling_the_last_lesson_retires_the_relation(store):
    """`set_status` is the other write that changes what a resolve can see."""
    _taught(store, relation="travel_scope", verdict="yes")
    assert "travel_scope" in store._relations_present()
    store.set_status("c_travel_scope_yes", "disabled")
    assert "travel_scope" not in store._relations_present()


def test_a_hit_does_not_clear_the_cache(store):
    """`_record_hit` bumps a counter and cannot change a verdict. Invalidating
    on it would clear the cache on every successful match -- the common case."""
    _taught(store)
    store.all_corrections(active_only=True)
    store._record_hit("c_atom_type_task")
    assert store._corrections_cache, "a hit cleared the corrections cache"


def test_the_table_is_read_once_per_decision(store):
    """The log line recomputed it, doubling the cost of every miss."""
    _taught(store)
    store._relations_present()
    calls = _counting(store)
    # taught relation, non-matching verdicts -> filtered to zero: the path
    # that used to read the table twice
    _resolve(store, "atom_type", candidates=("nothing_like_this",))
    assert calls["read"] == 1, f"read the corrections table {calls['read']} times"


def test_it_is_fast_enough_to_call_per_sheet(store):
    """A thousand untaught decisions is what a spreadsheet-heavy deal does."""
    for i in range(200):
        _taught(store, relation=f"rel_{i}", verdict="v")
    store._relations_present()
    start = time.monotonic()
    for _ in range(1000):
        _resolve(store, "sheet_role")
    assert time.monotonic() - start < 0.5
