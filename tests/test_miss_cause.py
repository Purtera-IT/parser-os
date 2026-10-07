"""`miss_cause`: why the parser lost a line a person added by hand (Missed tab).

A closed class on the base (universal) layer, read by the structure head next
to admission. It trains only on hand-added rows (origin "labeler"), and it
stays as history once a later parse produces the line: the blob's
``missed_history`` rows train the cause and nothing else. Invented rows only.
"""
from __future__ import annotations

import json

import pytest

from app.core.atom_type_registry import load_registry
from app.core.label_heads import head, head_of_read, head_of_task
from app.learning.human_labels import CLOSED_READS, MISS_CAUSE, IngestReport, rows_for_deal

CAUSES = ["glued_paragraph", "table_cell_glued", "table_row_dropped", "heading_as_section",
          "checkbox_cell", "folded_copy", "pasted_quote", "split_mid_sentence",
          "spreadsheet_cell", "other"]


def _missed(**kw):
    base = {
        "label_key": "lbl_m1", "origin": "labeler", "text": "Crew arrives at the dock by 7am.",
        "label_type": "work_scope_item", "doc_type": "email", "filename": "thread.eml",
        "labeler": "a@example.com", "section": ["Re: install"],
        "neighbors_above": ["Hello all,"], "neighbors_below": ["Thanks,"],
        "reads_set": {MISS_CAUSE: "glued_paragraph"},
    }
    base.update(kw)
    return base


def _cause_rows(rows):
    return [r for r in rows if r["relation"] == f"reads:{MISS_CAUSE}"]


def test_registry_declares_the_closed_class_on_the_base_layer():
    read = next(r for r in load_registry()["reads"] if r["key"] == MISS_CAUSE)
    assert [v.strip() for v in read["values"].split("|")] == CAUSES
    assert read["layer"] == "universal"
    assert sorted(read["value_desc"]) == sorted(CAUSES)
    assert all(len(d) > 20 for d in read["value_desc"].values())
    assert CLOSED_READS[MISS_CAUSE] == set(CAUSES)


def test_the_structure_head_owns_it_next_to_admission():
    assert head_of_read(MISS_CAUSE) == "structure.formation"
    assert head_of_task(f"reads:{MISS_CAUSE}") == "structure.formation"
    h = head("structure.formation")
    assert h["layer"] == "universal" and "admission" in h["tasks"]


def test_a_missed_row_trains_its_cause_as_a_class():
    rows = rows_for_deal({"deal_id": "d-miss", "labels": [_missed()]})
    [row] = _cause_rows(rows)
    assert row["label"] == "glued_paragraph" and row["label_kind"] == "judgment"
    # Where the line sat decides the cause, so the neighbours are served.
    assert "Hello all," in row["raw_text"] and "Thanks," in row["raw_text"]
    # The row still trains as a fact: admission keep for a line the parser missed.
    adm = [r for r in rows if r["relation"] == "admission"]
    assert [r["label"] for r in adm] == ["keep"]


@pytest.mark.parametrize("cause", CAUSES)
def test_every_listed_cause_trains(cause):
    rows = rows_for_deal({"deal_id": "d-miss", "labels": [_missed(reads_set={MISS_CAUSE: cause})]})
    assert [r["label"] for r in _cause_rows(rows)] == [cause]


def test_a_parser_row_never_teaches_a_cause():
    report = IngestReport()
    lb = _missed(origin="parser", reads_shown=[MISS_CAUSE],
                 rejected_reads={MISS_CAUSE: "not missed"})
    rows = rows_for_deal({"deal_id": "d-miss", "labels": [lb]}, report=report)
    assert _cause_rows(rows) == []
    assert any(r["relation"] == "atom_type" for r in rows)


def test_a_value_outside_the_list_is_skipped():
    rows = rows_for_deal({"deal_id": "d-miss", "labels": [_missed(reads_set={MISS_CAUSE: "gremlins"})]})
    assert _cause_rows(rows) == []


def test_history_row_trains_the_cause_and_nothing_else():
    hist = _missed(label_key="lbl_m2", note="DUPLICATE of parser atom lbl_p1: now emitted.",
                   duplicate_of="lbl_p1", reads_set={MISS_CAUSE: "table_cell_glued"})
    rows = rows_for_deal({"deal_id": "d-miss", "labels": [], "missed_history": [hist]})
    assert [(r["relation"], r["label"]) for r in rows] == [(f"reads:{MISS_CAUSE}", "table_cell_glued")]
    prov = json.loads(rows[0]["provenance"])
    assert prov["kind"] == "missed_history" and prov["duplicate_of"] == "lbl_p1"


def test_history_rows_respect_exclusion_drafts_and_the_list():
    rows = rows_for_deal({"deal_id": "d-miss", "labels": [_missed()], "missed_history": [
        _missed(label_key="h1", note="EXCLUDE_FROM_TRAINING: old manual sheet.",
                reads_set={MISS_CAUSE: "spreadsheet_cell"}),
        _missed(label_key="h2", labeler="draft (assistant)", reads_set={MISS_CAUSE: "other"}),
        _missed(label_key="h3", reads_set={MISS_CAUSE: "gremlins"}),
        _missed(label_key="h4", origin="parser", reads_set={MISS_CAUSE: "other"}),
        _missed(label_key="h5", reads_set={}),
    ]})
    assert [r["label"] for r in _cause_rows(rows)] == ["glued_paragraph"]


def test_c3_trains_the_cause_on_missed_lines_only():
    pytest.importorskip("torch")
    from ml.c3.data import IGNORE, DealExample, featurize
    from ml.c3.schema import load_schema

    schema = load_schema()
    opp = schema.by_key()[f"read:{MISS_CAUSE}"]
    assert opp.universal
    atoms = [{"label_key": "p1", "atom_id": "at-p1", "text": "Install four panels in the lobby.",
              "entered_at": "2026-05-01T10:00:00Z", "doc_kind": "email"}]
    blob = {"labels": [
        {"label_key": "p1", "label_type": "work_scope_item", "reads_set": {MISS_CAUSE: "other"},
         "reads_shown": [MISS_CAUSE]},
        _missed(label_key="m1", reads_set={MISS_CAUSE: "heading_as_section"}),
    ]}
    batch = featurize(DealExample.from_training_blob(blob, atoms, deal_id="d-miss"), schema)
    i_parser = next(i for i, t in enumerate(batch.texts) if "four panels" in t)
    i_missed = next(i for i, t in enumerate(batch.texts) if "dock by 7am" in t)
    col = batch.targets[f"read:{MISS_CAUSE}"]
    assert col[i_parser] == IGNORE
    assert col[i_missed] == opp.index("heading_as_section")
