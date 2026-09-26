"""Rows must survive the last step, which is where two of them did not.

`assemble` deduplicates, and its key decided what counts as the same
assertion. It was (relation, text), which is wrong twice: a second label from
the same labeler is usually a second truth, not a contradiction, and an edge's
target is not in its text at all.
"""
from __future__ import annotations

import json
import sqlite3

from app.learning.multitask_table import assemble


def _db(tmp_path, rows):
    p = tmp_path / "t.db"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE training_rows (relation TEXT, label TEXT, "
                 "raw_text TEXT, deal_id TEXT, split TEXT, teacher TEXT, "
                 "provenance TEXT, weight REAL)")
    conn.executemany("INSERT INTO training_rows VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return p


def test_one_teacher_giving_two_answers_keeps_both(tmp_path):
    """An atom is decided by its own words AND by who said it. On 010288 the
    (relation, text) key read the second as a conflict and dropped 83 of 148."""
    text = "we provide the parts but not the relay to the lock"
    db = _db(tmp_path, [
        ("decided_from", "own_words", text, "d1", "train", "human", "{}", 1.0),
        ("decided_from", "who_said_it", text, "d1", "train", "human", "{}", 1.0),
    ])
    rows = assemble([db], tasks=("decided_from",)).rows
    assert {r.label for r in rows} == {"own_words", "who_said_it"}


def test_six_edges_out_of_one_sentence_are_six_rows(tmp_path):
    """An edge row's text is the sentence and its label is the relation, so the
    target -- which lives in the provenance -- is the rest of the assertion."""
    text = "Here are the details for the small job I was discussing earlier."
    db = _db(tmp_path, [
        ("edge_relation", "governs", text, "d1", "train", "human",
         json.dumps({"to_atom_id": f"atm_{i}"}), 1.0)
        for i in range(6)
    ])
    rows = assemble([db], tasks=("edge_relation",)).rows
    assert len(rows) == 6


def test_the_same_assertion_twice_is_still_one_row(tmp_path):
    db = _db(tmp_path, [
        ("atom_type", "bom_line", "Mag Lock Cable", "d1", "train", "human",
         json.dumps({"to_atom_id": "atm_1"}), 1.0),
    ] * 3)
    assert len(assemble([db], tasks=("atom_type",)).rows) == 1


def test_a_weaker_teacher_does_not_stand_beside_a_human(tmp_path):
    """Dropped whole, not merged. Silence from the weaker teacher is not
    evidence, and a union nobody asserted is what a multi-label head would
    otherwise learn."""
    text = "Mag Lock Cable"
    db = _db(tmp_path, [
        ("atom_type", "bom_line", text, "d1", "train", "human", "{}", 1.0),
        ("atom_type", "deal_metadata", text, "d1", "train", "deepseek", "{}", 1.0),
    ])
    rows = assemble([db], tasks=("atom_type",)).rows
    assert [r.label for r in rows] == ["bom_line"]
