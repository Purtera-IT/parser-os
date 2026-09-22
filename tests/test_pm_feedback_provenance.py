"""A PM/labeler type correction's gold row must say what its exemplar IS.

The atom labeler sends the decide-text v2 string as the exemplar plus
relations naming its version, compile, atom, label key and hint chips.
Without these in provenance a trainer sees a v2 row among v0 rows and
cannot tell them apart (the multitask table reads decide_text_version).
"""
from __future__ import annotations

import numpy as np

from app.core import training_log
from app.core.feedback_store import FeedbackStore
from app.core.pm_feedback import apply_pm_correction


class _Capture:
    def __init__(self):
        self.rows = []

    def add_many(self, rows):
        self.rows.extend(rows)
        return len(rows)


def _store():
    return FeedbackStore(":memory:", embed_fn=lambda texts: np.full((len(texts), 4), 0.5, np.float32),
                         reachable_fn=lambda: True)


def test_labeler_relations_ride_into_the_gold_rows_provenance(monkeypatch):
    cap = _Capture()
    training_log.set_training_log(cap)
    try:
        apply_pm_correction(_store(), {
            "head": "type", "dealId": "d1", "compileId": "c9", "targetId": "atm_1",
            "text": "Mount 110 TVs [table: blk_1] [section: SOW]",
            "oldValue": "scope_item", "newValue": "task", "scope": "deal",
            "pm": "labeler@purtera-it.com",
            "relations": {"decide_text_version": 2, "label_key": "lbl_x", "hints": ["section"],
                          "source": "atom_labeler"},
        })
    finally:
        training_log.set_training_log(None)
    assert cap.rows, "a gold row is logged"
    prov = cap.rows[0].provenance
    assert prov["decide_text_version"] == 2
    assert prov["compile_id"] == "c9" and prov["atom_id"] == "atm_1"
    assert prov["label_key"] == "lbl_x" and prov["source"] == "atom_labeler"
    assert prov["hints"] == ["section"]


def test_junk_version_is_dropped_not_fatal():
    cap = _Capture()
    training_log.set_training_log(cap)
    try:
        apply_pm_correction(_store(), {
            "head": "type", "dealId": "d1", "targetId": "a", "text": "x y z",
            "oldValue": "", "newValue": "task", "scope": "deal", "pm": "a@b.com",
            "relations": {"decide_text_version": "two"},
        })
    finally:
        training_log.set_training_log(None)
    assert cap.rows and "decide_text_version" not in cap.rows[0].provenance
