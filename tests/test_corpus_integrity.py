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

from app.learning.corpus_integrity import (
    check_admitted, check_assembled, check_fields, check_rows, check_spans,
)


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


def test_a_span_absent_from_its_own_prompt_is_named():
    """The fifth loss, and the largest: all 159 of 010288's pointers were
    emitted as spans, but 56 name the envelope or the document type and 21 name
    another surface. None of that is text on the page the head is holding, so a
    third of the span supervision was an instruction to invent."""
    out = check_spans([
        {"relation": "evidence_span:who_said_it",
         "label": "alec@cdw.com (reseller, theirs) -> aj@ours",
         "raw_text": "Mag Lock Cable [section: Provided by Club/installer]"},
    ])
    assert len(out) == 1 and "evidence_span:who_said_it" in out[0]


def test_a_span_its_prompt_contains_is_quiet():
    out = check_spans([
        {"relation": "evidence_span:own_words",
         "label": "but not the relay to the lock",
         "raw_text": "we provide the parts that connect the PC to the relay, "
                     "but not the relay to the lock"},
    ])
    assert out == []


def test_the_check_only_judges_spans():
    """A rationale is meant to be words that are not on the page."""
    out = check_spans([
        {"relation": "rationale:atom", "label": "this is an argument nobody wrote",
         "raw_text": "Mag Lock Cable"},
    ])
    assert out == []


def test_rows_lost_inside_assemble_are_named():
    """`decided_from`: 148 emitted, 65 reached the table, and the only trace
    was a counter reading "duplicate"."""
    out = check_assembled({"decided_from": 148, "atom_type": 65},
                          {"decided_from": 65, "atom_type": 65},
                          ("decided_from", "atom_type"))
    assert len(out) == 1 and "83 lost inside assemble" in out[0]


def test_an_intact_assemble_is_quiet():
    assert check_assembled({"edge_relation": 80}, {"edge_relation": 80},
                           ("edge_relation",)) == []
