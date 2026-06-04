"""Tests for app.core.shadow_history — the metrics-over-time ledger.

The store must (a) append one row per relation per snapshot and never overwrite
(the table IS the time series); (b) read the curve back chronologically per
relation; (c) report the *latest* snapshot per relation as the dashboard;
(d) surface the timestamp a relation FIRST cleared every readiness bar — not an
earlier not-ready one; (e) capture total dataset size per relation from the log;
and (f) stay a no-op when unconfigured (default-off, byte-identical prod).
"""

from __future__ import annotations

from app.core.shadow_eval import RelationReport
from app.core.shadow_history import (
    ShadowHistory,
    get_shadow_history,
    record,
    set_shadow_history,
)
from app.core.training_log import TEACHER_LLM, TrainingLog, TrainingRow


def _ready_report(relation: str = "atom_type") -> RelationReport:
    """A report that clears every bar: holdout>=20, cov>=.6, acc>=.9, gold>=.95."""
    return RelationReport(
        relation=relation, n_holdout=30, n_answered=25, n_correct=24,
        n_gold=10, n_gold_correct=10,
    )


def _not_ready_report(relation: str = "atom_type") -> RelationReport:
    """Too few holdout rows → never ready even though accuracy is perfect."""
    return RelationReport(
        relation=relation, n_holdout=5, n_answered=5, n_correct=5,
        n_gold=2, n_gold_correct=2,
    )


def test_snapshot_writes_one_row_per_relation():
    h = ShadowHistory(":memory:")
    reports = {"atom_type": _ready_report("atom_type"),
               "payment_term": _not_ready_report("payment_term")}
    sid = h.snapshot(reports, created_at=100.0)
    assert sid.startswith("snp_")
    assert set(h.relations()) == {"atom_type", "payment_term"}
    assert h.snapshot_count() == 1
    at = h.trend("atom_type")
    assert len(at) == 1
    assert at[0].ready is True
    assert at[0].snapshot_id == sid
    pt = h.trend("payment_term")
    assert pt[0].ready is False


def test_append_only_accumulates():
    h = ShadowHistory(":memory:")
    h.snapshot({"atom_type": _not_ready_report()}, created_at=100.0)
    h.snapshot({"atom_type": _not_ready_report()}, created_at=200.0)
    h.snapshot({"atom_type": _ready_report()}, created_at=300.0)
    trend = h.trend("atom_type")
    assert len(trend) == 3
    # Chronological order preserved.
    assert [r.created_at for r in trend] == [100.0, 200.0, 300.0]
    assert h.snapshot_count() == 3


def test_latest_returns_most_recent_per_relation():
    h = ShadowHistory(":memory:")
    h.snapshot({"atom_type": _not_ready_report()}, created_at=100.0)
    h.snapshot({"atom_type": _ready_report()}, created_at=200.0)
    latest = h.latest()
    assert latest["atom_type"].created_at == 200.0
    assert latest["atom_type"].ready is True


def test_first_ready_picks_earliest_ready_not_before():
    h = ShadowHistory(":memory:")
    # Not ready at t=100, t=200; becomes ready at t=300 and stays.
    h.snapshot({"atom_type": _not_ready_report()}, created_at=100.0)
    h.snapshot({"atom_type": _not_ready_report()}, created_at=200.0)
    h.snapshot({"atom_type": _ready_report()}, created_at=300.0)
    h.snapshot({"atom_type": _ready_report()}, created_at=400.0)
    assert h.first_ready("atom_type") == 300.0


def test_first_ready_none_when_never_ready():
    h = ShadowHistory(":memory:")
    h.snapshot({"atom_type": _not_ready_report()}, created_at=100.0)
    assert h.first_ready("atom_type") is None


def test_n_rows_captured_from_log():
    log = TrainingLog(":memory:")
    log.add_many([
        TrainingRow(relation="atom_type", label="requirement",
                    raw_text=f"Org{i} shall provide item", teacher=TEACHER_LLM,
                    deal_id=f"deal{i}")
        for i in range(7)
    ])
    h = ShadowHistory(":memory:")
    h.snapshot({"atom_type": _ready_report()}, log=log, label="post-compile",
               created_at=100.0)
    row = h.trend("atom_type")[0]
    assert row.n_rows == 7
    assert row.label == "post-compile"
    # holdout count comes from the report, independent of the log total.
    assert row.n_holdout == 30


def test_record_noop_when_unconfigured():
    set_shadow_history(None)
    try:
        # No SOWSMITH_SHADOW_HISTORY_DB env and no injected store → no-op.
        assert get_shadow_history() is None
        assert record({"atom_type": _ready_report()}) is None
    finally:
        set_shadow_history(None)


def test_record_uses_injected_store():
    h = ShadowHistory(":memory:")
    set_shadow_history(h)
    try:
        sid = record({"atom_type": _ready_report()}, label="injected")
        assert sid is not None
        assert h.snapshot_count() == 1
        assert h.latest()["atom_type"].label == "injected"
    finally:
        set_shadow_history(None)


def test_snapshot_row_as_dict_rounds():
    rep = RelationReport(relation="atom_type", n_holdout=30, n_answered=25,
                         n_correct=24, n_gold=10, n_gold_correct=10)
    h = ShadowHistory(":memory:")
    h.snapshot({"atom_type": rep}, created_at=100.0)
    d = h.trend("atom_type")[0].as_dict()
    assert d["relation"] == "atom_type"
    assert d["ready"] is True
    assert 0.0 <= d["accuracy"] <= 1.0
    assert 0.0 <= d["coverage"] <= 1.0
