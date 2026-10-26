"""ml.c3.weak_report: where the base is weak and what to label next.
Made-up reads, routes and log lines; keys and counts only."""
from __future__ import annotations

import json

from ml.c3.weak_report import from_reads, from_routes, from_shadow, markdown


def _read():
    def line(t, p, about_p=0.9):
        return {"text": "never reported", "reads": {"col:label_type": {"answer": t, "p": p},
                                                   "col:about": {"answer": "x", "p": about_p}}}
    return {"deal_id": "d1", "lines": {"k1": line("task", 0.95), "k2": line("requirement", 0.4, 0.3),
                                       "k3": line("task", 0.5)}}


def test_reads_rank_unsure_heads_and_queue_least_sure_lines_by_key():
    r = from_reads([_read()], 0.6, 10)
    assert r["deals"]["d1"]["unsure_type"] == 2
    assert [q["key"] for q in r["queue"]] == ["k2", "k3"]
    assert {h["head"] for h in r["heads"]} == {"col:label_type", "col:about"}
    assert "never reported" not in json.dumps(r) and "never reported" not in markdown({"reads": r})


def test_routes_say_why_a_type_is_not_taken():
    route = {"checkpoint": "c", "min_n": 20, "bar": 0.85, "keep_bar": 0.97, "per_type": {
        "task": {"acts": True, "replace": False, "base_n": 5, "base_precision": 1.0,
                 "classifier_precision": 0.9, "gold_n": 9},
        "requirement": {"acts": True, "replace": False, "base_n": 40, "base_precision": 0.7,
                        "classifier_precision": 0.6, "gold_n": 50},
        "pricing_line": {"acts": False, "replace": False, "base_n": 40, "base_precision": 0.9,
                         "classifier_precision": 0.6, "gold_n": 50},
        "exclusion": {"acts": True, "replace": True, "base_n": 40, "base_precision": 0.95,
                      "classifier_precision": 0.6, "gold_n": 50}}}
    got = {r["type"]: r["why"] for r in from_routes([route])}
    assert set(got) == {"task", "requirement"}
    assert "label more" in got["task"] and "confusions" in got["requirement"]


def test_shadow_logs_pool_agreement_pairs_and_time():
    log = "\n".join([
        "noise line",
        json.dumps({"event": "c3_type_shadow", "by_answer": {"task": {"agree": 3, "disagree": 1}},
                    "disagree_pairs": {"task->scope_item": 1}}),
        json.dumps({"event": "typed_atom_deflect", "llm_ms": 9000.0, "layer_ms": {"c3": 1500.0}}),
    ])
    s = from_shadow([log, log])
    assert s["compiles"] == 2 and s["agreement"]["task"]["rate"] == 0.75
    assert s["top_disagreements"] == {"task->scope_item": 2} and s["c3_ms"] == 3000.0
