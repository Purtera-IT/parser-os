"""Nothing a labeler writes may be lost between Postgres, blob and corpus.

Four times in one day a field was produced correctly and dropped one step
later, and every time the symptom was silence:

  * the mirror stubbed `doc.judgments = []` -- 94 verdicts written and lost;
  * `buildBlobDoc` maps explicit fields, so four new columns on atom_labels
    reached Postgres and stopped;
  * the judgment `reason` went the same way;
  * `DEFAULT_TASKS` never listed the judgement heads, so 892 of 1,241 rows were
    skipped one step from the model.

Counting records would have caught none of them. The records were all present;
a field was missing, or a relation was unlisted. So these tests compare FIELDS
and RELATIONS, which is the grain the losses actually happen at.
"""
from __future__ import annotations

from app.learning.corpus_integrity import check_admitted, check_fields, check_rows


def test_a_filled_column_absent_from_the_blob_is_named():
    """The `consumer` case: written on every label, mapped by nobody."""
    pg = [{"label_key": "lbl_1", "label_type": "bom_line", "consumer": "deal_kit"}]
    blob = [{"label_key": "lbl_1", "label_type": "bom_line"}]
    out = check_fields(pg, blob, what="labels")
    assert len(out) == 1
    assert "labels.consumer" in out[0]


def test_an_empty_column_is_not_a_loss():
    """A column nobody filled is not a drop -- only silence about real data is."""
    pg = [{"label_key": "lbl_1", "rejected": None, "decided_by": ""}]
    blob = [{"label_key": "lbl_1"}]
    assert check_fields(pg, blob, what="labels") == []


def test_bookkeeping_columns_are_not_supervision():
    pg = [{"label_key": "lbl_1", "id": 7, "labeler": "a@b.c", "labeled_at": "now"}]
    blob = [{"label_key": "lbl_1"}]
    assert check_fields(pg, blob, what="labels") == []


def test_records_that_produce_no_row_are_named():
    """The judgments case: 94 verdicts in the blob, zero rows out."""
    out = check_rows({"judgments": 94}, {"atom_type": 65},
                     expect={"judgments": "gap_valid"})
    assert len(out) == 1 and "94 judgments" in out[0]


def test_records_that_do_produce_rows_are_quiet():
    out = check_rows({"judgments": 94}, {"gap_valid": 81, "gap_valid_reason": 81},
                     expect={"judgments": "gap_valid"})
    assert out == []


def test_a_relation_no_head_accepts_is_named():
    """The DEFAULT_TASKS case. `gap_valid` was emitted, mirrored and ingested,
    then matched nothing -- and the only symptom was a skip counter nobody
    read."""
    out = check_admitted({"atom_type": 65, "gap_valid": 81}, ("atom_type",))
    assert len(out) == 1 and "gap_valid" in out[0]


def test_spans_and_rationales_are_held_back_on_purpose():
    """A span is an extraction problem and a rationale a generative one.
    Neither belongs in a classifier, and neither is a loss."""
    out = check_admitted(
        {"atom_type": 65, "evidence_span:own_words": 60,
         "reads_value:expansion": 1, "rationale:atom": 65},
        ("atom_type",))
    assert out == []
