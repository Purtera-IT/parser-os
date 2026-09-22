"""Human labels -> training rows: context kept, eval deals locked to holdout."""
from __future__ import annotations

import json
import sqlite3

from app.learning.human_labels import decide_text, rows_for_deal, write_db
from app.learning.multitask_table import assemble


def _label(**kw):
    base = {
        "label_key": "lbl_x", "text": "Mount 110 TVs in the lobby", "label_type": "work_scope_item",
        "section": ["Scope of Work", "Install"], "lead_in": ["Provider will:"],
        "neighbors_above": ["Provider will:"], "hints": ["lead_in", "section"],
        "doc_type": "SOW", "filename": "SOW.pdf", "page": 2, "labeler": "a@purtera-it.com",
    }
    base.update(kw)
    return base


def test_decide_text_matches_the_v2_representation():
    assert decide_text(_label()) == (
        "Mount 110 TVs in the lobby [section: Scope of Work > Install] [intro: Provider will:]"
    )


def test_one_label_makes_type_coarse_and_facet_rows_with_context():
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label()]})
    by = {r["relation"]: r for r in rows}
    assert by["atom_type"]["label"] == "work_scope_item"
    assert by["atom_type_coarse"]["label"] == "scope"
    assert by["facet"]["label"] == "WORK"
    prov = json.loads(by["atom_type"]["provenance"])
    assert prov["hints"] == ["lead_in", "section"] and prov["decide_text_version"] == 2
    assert all(r["teacher"] == "human" for r in rows)


def test_eval_deal_is_holdout_even_if_it_hashes_to_train():
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label(purpose="eval"), _label(label_key="b")]})
    assert {r["split"] for r in rows} == {"holdout"}


def test_proposed_type_keeps_its_coarse_but_gets_no_facet():
    rows = rows_for_deal({"deal_id": "d1", "labels": [
        _label(label_type="warranty_term", coarse="governance", is_new_type=True)]})
    assert {r["relation"] for r in rows} == {"atom_type", "atom_type_coarse"}


def test_written_db_feeds_the_multitask_table_and_human_wins_dedup(tmp_path):
    human = tmp_path / "_training_human.db"
    write_db([{"deal_id": "d1", "labels": [_label()],
               "deal_answers": [{"labeler": "a", "primary_service": "audio_visual",
                                 "declared_site_count": "3"}]}], human)
    llm = tmp_path / "_training_llm.db"
    c = sqlite3.connect(llm)
    c.execute("CREATE TABLE training_rows (relation TEXT, label TEXT, raw_text TEXT, deal_id TEXT, "
              "split TEXT, teacher TEXT, provenance TEXT)")
    c.execute("INSERT INTO training_rows VALUES ('atom_type','task',?,'d1','train','pm','{}')",
              (decide_text(_label()),))
    c.commit(); c.close()
    table = assemble([llm, human])
    types = [r for r in table.rows if r.task == "atom_type"]
    assert len(types) == 1 and types[0].label == "work_scope_item" and types[0].teacher == "human"
    assert types[0].repr_version == 2
    gold = sqlite3.connect(human).execute("SELECT primary_service, declared_site_count FROM deal_gold").fetchall()
    assert gold == [("audio_visual", 3)]


def test_stored_decide_text_wins_and_bare_rows_are_version_0():
    rows = rows_for_deal({"deal_id": "d1", "labels": [
        _label(decide_text="Mount 110 TVs [table: blk_1] [section: SOW]"),
        {"label_key": "b", "text": "Customer provides lift", "label_type": "task"},
    ]})
    by = [r for r in rows if r["relation"] == "atom_type"]
    assert by[0]["raw_text"] == "Mount 110 TVs [table: blk_1] [section: SOW]"
    assert json.loads(by[0]["provenance"])["decide_text_version"] == 2
    assert by[1]["raw_text"] == "Customer provides lift"
    assert json.loads(by[1]["provenance"])["decide_text_version"] == 0


def test_offline_gold_export_folds_in_as_bare_text():
    from app.learning.human_labels import docs_from_gold_export

    docs = docs_from_gold_export({"atoms": [
        {"deal": "d1", "doc": "SOW.pdf", "atom": "Mount 110 TVs", "label": "task", "parser_guess": "scope_item"},
        {"deal": "d2", "doc": "Q.xlsx", "atom": "65in TV | 110", "label": "bom_line"},
        {"deal": "d2", "atom": "", "label": "task"},
    ]}, labeler="pilot")
    assert {d["deal_id"] for d in docs} == {"d1", "d2"}
    rows = rows_for_deal(docs[0])
    prov = json.loads(rows[0]["provenance"])
    assert prov["source"] == "offline_gold_labeler" and prov["decide_text_version"] == 0


def test_judgments_become_rows_for_their_own_heads():
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        {"head": "conflict", "target_key": "e1", "text": "Cat 6 patch cords || in Building B704",
         "parser_value": "contradicts", "verdict": "unrelated"},
        {"head": "site", "target_key": "p1", "text": "columbus afb ms || columbus ms 39710", "verdict": "same_site"},
        {"head": "gap", "target_key": "g1", "text": "Who is the on-site contact?", "verdict": "valid"},
        {"head": "gap", "target_key": "g2", "text": "Junk question here", "verdict": "maybe"},
        {"head": "tier", "target_key": "t", "text": "whatever text", "verdict": "1"},
    ]})
    by = {(r["relation"], r["label"]) for r in rows}
    assert ("edge_relation", "unrelated") in by
    assert ("same_physical_site", "same_site") in by
    assert ("gap_valid", "valid") in by
    assert all(r["label"] != "maybe" for r in rows), "a verdict outside the head's classes is dropped"
    assert all(r["teacher"] == "human" for r in rows)
