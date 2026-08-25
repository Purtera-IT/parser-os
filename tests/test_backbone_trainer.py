"""The GO #5 trainer's pure half: data honoring, loss namespacing, paired gate.

The GPU half (SupCon fit) is proven by --smoke; these tests pin the decisions
that must survive a rewrite: the split column governs, PM rows never train,
task namespacing separates classes, and the paired gate scores abstention as
a miss on both sides.
"""

import sqlite3
from pathlib import Path

from runpod_detector.train_multitask_backbone import (
    load_rows, namespaced, paired_vs_baseline,
)


def _table(tmp_path: Path) -> Path:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE multitask_rows (task TEXT, text TEXT, label TEXT, "
        "deal_id TEXT, split TEXT, teacher TEXT, source_db TEXT, repr_version INT)")
    rows = [
        ("atom_type", "no work on roof", "exclusion", "d1", "train", "pipeline", "x", 0),
        ("atom_type", "install 4 APs", "scope_item", "d2", "holdout", "pipeline", "x", 2),
        # PM row carrying split='train' -- must STILL be eval-only
        ("atom_type", "customer must provide lift", "constraint", "d1", "train", "pm", "x", 2),
        # bad label -> dropped entirely
        ("atom_type", "junk", "_keep", "d1", "train", "pipeline", "x", 0),
        # empty text -> dropped
        ("atom_type", "   ", "exclusion", "d1", "train", "pipeline", "x", 0),
        ("edge", "a vs b", "contradicts", "d3", "train", "llm", "x", 0),
    ]
    conn.executemany("INSERT INTO multitask_rows VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()
    return db


def test_split_column_governs_and_pm_is_eval_only(tmp_path):
    data = load_rows(_table(tmp_path))
    at = data["atom_type"]
    assert [r[1] for r in at["train"]] == ["exclusion"]      # bad/empty/pm gone
    assert [r[1] for r in at["holdout"]] == ["scope_item"]
    # the PM row sat in split='train'; it must land in the pm slot regardless
    assert [r[1] for r in at["pm"]] == ["constraint"]


def test_task_filter(tmp_path):
    data = load_rows(_table(tmp_path), tasks=("edge",))
    assert set(data) == {"edge"}


def test_namespacing_keeps_same_label_apart_across_tasks():
    assert namespaced("atom_type", "exclusion") != namespaced("edge", "exclusion")


def test_paired_gate_scores_abstention_as_miss():
    base = {"_answered": [True, False, True, False],
            "_correct": [True, False, False, False]}
    cand = {"_answered": [True, True, False, False],
            "_correct": [True, True, False, False]}
    r = paired_vs_baseline(base, cand)
    # row 2: candidate answers correctly where base abstained -> win
    # row 3: base answered WRONG (scores as miss), candidate abstained -> no pair
    assert (r["wins"], r["losses"]) == (1, 0)
    assert r["p"] is not None


def test_paired_gate_refuses_without_discordant_pairs():
    same = {"_answered": [True], "_correct": [True]}
    assert paired_vs_baseline(same, same)["p"] is None
