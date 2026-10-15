"""app/learning/c3_report.py: the C3 base's daily report from the compile
manifests. Made-up manifests; counts and times only."""
from __future__ import annotations

from app.learning import c3_report as R


def _m(state="read", read_ms=2000.0, llm_ms=9000.0, taken=5, agree=30, disagree=2, pairs=None):
    return {"deflect_counts": {
        "typed_atom": {"c3": taken, "llm_ms": llm_ms},
        "c3": {"mode": "shadow", "types": ["task"], "state": state, "read_ms": read_ms, "lines": 100,
               "reason": "MemoryError: 900 MB free" if state == "no_memory" else "",
               "shadow": {"by_answer": {"task": {"agree": agree, "disagree": disagree}},
                          "disagree_pairs": pairs or {"task->scope_item": disagree}}}}}


def test_pool_counts_health_speed_take_and_agreement():
    rep = R.pool([_m(), _m(read_ms=12000.0), _m(state="no_memory", read_ms=None, taken=0, agree=0,
                                                disagree=0, pairs={}), {"deflect_counts": {}}])
    assert rep["compiles"] == 3 and rep["health"] == {"read": 2, "no_memory": 1}
    assert rep["failure_reasons"] == {"MemoryError": 1}
    assert rep["speed"]["compiles_slower_than_llm"] == 1
    assert rep["take"]["atoms_typed_by_base"] == 10
    assert rep["agreement"]["task"]["n"] == 64 and rep["weak"] == []


def test_an_answer_under_the_bar_is_where_to_label():
    rep = R.pool([_m(agree=10, disagree=15)])
    assert rep["weak"] == ["task"]
    md = R.markdown(rep, "2026-10-15")
    assert "Where to label" in md and "**task**" in md


def test_no_storage_means_no_report_and_no_error(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    assert R.run() is None
