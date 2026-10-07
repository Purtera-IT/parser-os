"""Human labels -> training rows: context kept, eval deals locked to holdout."""
from __future__ import annotations

import json
import sqlite3

from app.learning.human_labels import IngestReport, decide_text, rows_for_deal, write_db
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


def test_a_model_drafted_judgment_note_trains_at_draft_weight():
    """why_author: machine_draft scales the judgment's rationale, as it does an
    atom label's WHY in ml/c3; the verdict keeps its full weight. A row with no
    why_author (every row from before the column) is a person's."""
    from ml.c3.data import DRAFT_WHY_WEIGHT

    note = "The two windows cannot both hold for the same crew on site."

    def rows(**kw):
        out = rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
            {"head": "conflict", "target_key": "e1", "text": "Saturday only || weekdays after 6pm",
             "verdict": "contradicts", "note": note, "labeler": "a@purtera-it.com", **kw}]})
        return ({r["relation"]: r["weight"] for r in out if r["label_kind"] == "rationale"},
                {r["label"]: r["weight"] for r in out if r["relation"] == "edge_relation"})

    why, verdict = rows(why_author="machine_draft")
    assert why == {"rationale:conflict": DRAFT_WHY_WEIGHT}
    assert verdict == {"contradicts": 1.0}
    for author in ({}, {"why_author": None}, {"why_author": "person"}, {"why_author": "accepted_draft"}):
        why, verdict = rows(**author)
        assert why == {"rationale:conflict": 1.0}, author
        assert verdict == {"contradicts": 1.0}


def test_the_dropped_stage_teaches_the_admission_head():
    """Judging what the compile threw away is the `admission` question.

    The labelling workspace has offered a Dropped stage ("What the compile
    threw away") since it shipped, and `suppression` was in no registry. So
    every verdict made there was stored in Postgres, refused by
    /feedback/correction with `422 unknown head`, and skipped here -- it taught
    nothing by either route, and nothing said so.

    It asks `admission`'s question in different words, so it is mapped onto it.
    """
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        {"head": "suppression", "target_key": "s1", "verdict": "should_have_been_kept",
         "text": "Remote hands technician, 8 hours, Building B704"},
        {"head": "suppression", "target_key": "s2", "verdict": "correctly_dropped",
         "text": "Page 3 of 14 -- CONFIDENTIAL"},
    ]})
    by = {(r["relation"], r["label"]) for r in rows}
    # should_have_been_kept means the suppression was WRONG, so the atom should
    # have been admitted. Backwards, this would teach the head to drop exactly
    # what a person rescued.
    assert ("admission", "keep") in by, by
    assert ("admission", "drop") in by, by


def test_an_unregistered_judgment_head_is_reported_as_a_dead_surface():
    """Three different failures shared one message and the worst was invisible.

    An unregistered head is not a malformed row -- it is a whole labelling
    surface whose verdicts reach no head. The corpus report has to be able to
    say which one.
    """
    from app.learning.human_labels import IngestReport

    report = IngestReport()
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        {"head": "rule", "target_key": "r1", "verdict": "should_not_fire",
         "text": "a line the list_item_under_label rule fired on"},
    ]}, report=report)
    assert rows == [], "an unregistered head must not produce gold under a guessed relation"
    # `skipped` is keyed by the reason, so the reason IS the report.
    reasons = list(report.skipped)
    assert len(reasons) == 1, reasons
    assert "rule" in reasons[0], reasons
    # `rule` is not a dead surface any more: nothing decides a rule at compile
    # time, so it has no relation and no registry row by design, and
    # `rule_feedback.rows_from_judgments` reads its verdicts directly.
    assert "rule_feedback" in reasons[0], reasons
    assert report.skipped[reasons[0]] == 1

    # A head that genuinely reaches nothing still says so.
    dead = IngestReport()
    rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        {"head": "not_a_head", "target_key": "x", "verdict": "yes", "text": "some text here"},
    ]}, report=dead)
    assert any("HEAD_REGISTRY" in r for r in dead.skipped), dead.skipped

    # And the message must not be the old one, which said the same thing for a
    # malformed row as for a dead surface.
    empty = IngestReport()
    rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        {"head": "gap", "target_key": "g1", "verdict": "", "text": "a real question here"},
    ]}, report=empty)
    assert list(empty.skipped) != reasons, "two different failures still share one message"


def test_evidence_links_become_human_edges():
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "links": [
        {"from_head": "gap", "from_key": "g1", "from_text": "Who provides the lift?", "to_kind": "atom",
         "to_text": "Customer provides the lift", "relation": "answers"},
        {"from_head": "conflict", "from_key": "e1", "from_text": "6 racks", "to_kind": "text",
         "to_text": "704 racks in building B", "relation": "contradicts"},
        {"from_head": "type", "from_key": "k", "from_text": "Scope", "to_kind": "atom",
         "to_text": "Section 3 intro", "relation": "context"},
    ]})
    assert [(r["relation"], r["label"]) for r in rows] == [
        ("edge_relation", "answers"),
        ("edge_relation", "answered_by"),
        ("edge_relation", "contradicts"),
        ("edge_relation", "context"),
    ]
    # Drawn on the Questions card (from = the question), written answer-first
    # like an `answers` link drawn on the answer's own atom card.
    assert rows[0]["raw_text"] == "Customer provides the lift || Who provides the lift?"
    assert rows[1]["raw_text"] == "Who provides the lift? || Customer provides the lift"


def test_an_answer_and_a_deferral_are_different_edges():
    """A reply that closes a question and a reply that does not are the whole
    point of drawing the link. `answers` used to be folded into `supports` and
    `context` was dropped, so both replies below produced the same row -- or no
    row -- and a head could only learn that some atom relates to a question."""
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "links": [
        {"from_head": "type", "from_key": "a", "from_text": "They are intending to use a maglock.",
         "to_kind": "atom", "to_text": "Do we know the type of lock?", "relation": "answers"},
        {"from_head": "type", "from_key": "b", "from_text": "I am not sure if it is already installed",
         "to_kind": "atom", "to_text": "Has the door been installed with the lock?", "relation": "context"},
    ]})
    assert [r["label"] for r in rows] == ["answers", "answered_by", "context"]


def _summary_rows(rows):
    return [(r["raw_text"].split(" [")[0], r["label"]) for r in rows
            if r["relation"] == "reads:deal_summary"]


#: Readings deal threads saved before they were registered. A key outside the
#: registry reaches no backbone task, so each one here must be in
#: app/core/atom_types.json `reads` (and Platform-infra's atom-types.json).
SAVED_READS = ("deal_summary", "removes_cost")


def test_readings_threads_already_saved_are_registered_backbone_tasks():
    from app.learning.multitask_table import DEFAULT_TASKS

    for key in SAVED_READS:
        assert f"reads:{key}" in DEFAULT_TASKS, key


def test_removes_cost_trains_presence_and_carries_which_cost():
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label(
        text="Customer provides wall mounts and parking",
        reads_set={"removes_cost": "the wall-mount cost and the parking cost"})]})
    by = {r["relation"]: r["label"] for r in rows}
    assert by["reads:removes_cost"] == "present"
    assert by["reads_value:removes_cost"] == "the wall-mount cost and the parking cost"


def test_deal_summary_trains_one_positive_against_the_labelers_other_lines():
    rows = rows_for_deal({"deal_id": "d1", "labels": [
        _label(label_key="a", text="4 TVs install in CheckOut New York office.",
               reads_set={"deal_summary": True}),
        _label(label_key="b", text="Mount 110 TVs in the lobby"),
        # Somebody else labeled this deal and marked no summary: no opinion.
        _label(label_key="c", text="Bring a ladder", labeler="b@purtera-it.com"),
    ]})
    got = sorted(_summary_rows(rows))
    assert ("4 TVs install in CheckOut New York office.", "true") in got
    assert ("Mount 110 TVs in the lobby", "false") in got
    assert not any(t.startswith("Bring a ladder") for t, _ in got)
    assert len(got) == 2


def test_a_second_deal_summary_mark_moves_it_and_the_latest_wins():
    rows = rows_for_deal({"deal_id": "d1", "labels": [
        _label(label_key="new", text="Install 4 TVs at CheckOut NYC", labeled_at="2026-10-02T10:00:00Z",
               reads_set={"deal_summary": True}),
        _label(label_key="old", text="Mount 110 TVs in the lobby", labeled_at="2026-10-01T10:00:00Z",
               reads_set={"deal_summary": True}),
    ]})
    assert sorted(_summary_rows(rows)) == [
        ("Install 4 TVs at CheckOut NYC", "true"), ("Mount 110 TVs in the lobby", "false")]


def test_no_deal_summary_mark_means_no_deal_summary_rows():
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label(), _label(label_key="b")]})
    assert _summary_rows(rows) == []


# ---- the Questions card: intake_gap / needed_by / deal_stage, answered_by ----

def _q_rows(rows, field_name):
    return sorted((r["raw_text"], r["label"]) for r in rows if r["relation"] == f"question:{field_name}")


_QUESTION = "Is there power at the TV location?"


def _gap_judgment(fields, **extra):
    return {"head": "gap", "target_key": "gap:1", "verdict": "valid", "text": _QUESTION,
            "target": {"source": {"atomId": "q1"}}, "labeler": "a@b.com", "fields": fields, **extra}


def test_questions_card_fields_become_question_heads():
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        _gap_judgment({"intake_gap": True, "needed_by": ["project_manager", "atlas"], "deal_stage": "planning"}),
    ]})
    assert _q_rows(rows, "intake_gap") == [(_QUESTION, "true")]
    # Multi-label: one row per consumer, same text.
    assert _q_rows(rows, "needed_by") == [(_QUESTION, "atlas"), (_QUESTION, "project_manager")]
    assert _q_rows(rows, "deal_stage") == [(_QUESTION, "planning")]
    # The verdict still trains gap_valid as before.
    assert [r["label"] for r in rows if r["relation"] == "gap_valid"] == ["valid"]


def test_atom_readings_are_the_fallback_only_where_the_card_is_silent():
    label = {"label_key": "k1", "atom_id": "q1", "label_type": "question", "text": _QUESTION,
             "labeler": "a@b.com",
             "reads_set": {"intake_gap": "false", "needed_by": "portal", "deal_stage": "quoting"}}
    other = {"label_key": "k2", "atom_id": "q2", "label_type": "question",
             "text": "Who holds the key to the riser closet?", "labeler": "a@b.com",
             "reads_set": {"needed_by": ["project_manager", "portal"], "deal_stage": "delivery"}}
    report = IngestReport()
    rows = rows_for_deal({"deal_id": "d1", "labels": [label, other], "judgments": [
        # The card answered intake_gap and deal_stage for q1, not needed_by.
        _gap_judgment({"intake_gap": True, "deal_stage": "planning"}),
    ]}, report=report)
    assert _q_rows(rows, "intake_gap") == [(_QUESTION, "true")], "the card wins over the atom"
    assert _q_rows(rows, "deal_stage") == [
        (_QUESTION, "planning"),
        ("Who holds the key to the riser closet?", "delivery"),
    ]
    assert _q_rows(rows, "needed_by") == [
        (_QUESTION, "portal"),
        ("Who holds the key to the riser closet?", "portal"),
        ("Who holds the key to the riser closet?", "project_manager"),
    ]
    assert report.skipped.get("question intake_gap: the Questions card answered it") == 1
    # The atom readings still train their own reads:* heads.
    assert any(r["relation"] == "reads:intake_gap" for r in rows)


def test_the_card_covers_an_atom_by_its_words_after_a_reparse():
    label = {"label_key": "k1", "atom_id": "q1-recompiled", "label_type": "question", "text": _QUESTION,
             "labeler": "a@b.com", "reads_set": {"deal_stage": "quoting"}}
    rows = rows_for_deal({"deal_id": "d1", "labels": [label], "judgments": [
        _gap_judgment({"deal_stage": "planning"}),
    ]})
    assert _q_rows(rows, "deal_stage") == [(_QUESTION, "planning")]


def test_question_values_outside_their_sets_are_skipped_and_machines_teach_nothing():
    report = IngestReport()
    rows = rows_for_deal({"deal_id": "d1", "labels": [], "judgments": [
        _gap_judgment({"deal_stage": "signed", "needed_by": ["accounting"]}),
        _gap_judgment({"intake_gap": True}, labeler="Proposer (assistant)"),
    ]}, report=report)
    assert not [r for r in rows if r["relation"].startswith("question:")]
    assert report.skipped.get("question deal_stage outside its values") == 1
    assert report.skipped.get("question needed_by outside its values") == 1


def test_question_heads_are_backbone_tasks():
    from app.learning.multitask_table import DEFAULT_TASKS, tasks_for

    # The stage is universal; intake_gap and needed_by are Purtera's standard
    # (labeling/portable-labels.md b) and train only its profile.
    assert "question:deal_stage" in DEFAULT_TASKS
    for f in ("intake_gap", "needed_by"):
        assert f"question:{f}" not in DEFAULT_TASKS
    for f in ("intake_gap", "needed_by", "deal_stage"):
        assert f"question:{f}" in tasks_for("purtera")


def test_answers_trains_both_ways_whichever_card_drew_it():
    q, a = "Do we know the type of lock?", "They are intending to use a maglock."
    on_question = rows_for_deal({"deal_id": "d1", "labels": [], "links": [
        {"from_head": "gap", "from_key": "gap:1", "from_text": q, "to_kind": "atom", "to_text": a,
         "relation": "answers", "labeler": "a@b.com"},
    ]})
    on_answer = rows_for_deal({"deal_id": "d1", "labels": [], "links": [
        {"from_head": "type", "from_key": "k", "from_text": a, "to_kind": "atom", "to_text": q,
         "relation": "answers", "labeler": "a@b.com"},
    ]})
    pairs = lambda rows: [(r["raw_text"], r["label"]) for r in rows if r["relation"] == "edge_relation"]
    assert pairs(on_question) == pairs(on_answer) == [
        (f"{a} || {q}", "answers"),
        (f"{q} || {a}", "answered_by"),
    ]


def _reads_rows(reads_set, report=None):
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label(reads_set=reads_set)]}, report=report)
    return sorted((r["relation"], r["label"]) for r in rows if r["relation"].startswith("reads:"))


def test_a_multi_reading_trains_one_class_per_item_in_either_shape():
    want = [("reads:train_for", "delivery_parser"), ("reads:train_for", "quote_parser")]
    assert _reads_rows({"train_for": ["quote_parser", "delivery_parser"]}) == want
    assert _reads_rows({"train_for": "quote_parser,delivery_parser"}) == want


def test_registered_thread_keys_train_as_classes():
    assert _reads_rows({"sow_coverage": "missing_from_sow", "superseded": True,
                        "hours_stated": False, "location_tier": "rural"}) == [
        ("reads:hours_stated", "false"), ("reads:location_tier", "rural"),
        ("reads:sow_coverage", "missing_from_sow"), ("reads:superseded", "true")]


def test_the_renamed_sow_section_value_still_trains():
    assert _reads_rows({"sow_section": "purtera_responsibilities"}) == [
        ("reads:sow_section", "provider_responsibilities")]


def test_meta_and_staging_readings_are_stored_never_trained():
    report = IngestReport()
    assert _reads_rows({"co_company": "purtera", "deal_outcome": "won",
                        "universal_type": "commitment"}, report) == []
    assert report.skipped["reading deal_outcome is stored, never trained"] == 1
    from app.learning.multitask_table import DEFAULT_TASKS

    for key in ("co_company", "deal_outcome", "universal_type"):
        assert f"reads:{key}" not in DEFAULT_TASKS


def test_values_deal_threads_already_write_are_accepted():
    # labeling/portable-labels.md: noise classes and co_reason codes the
    # threads write. A value outside the registry would be skipped at ingest.
    assert _reads_rows({"noise_class": "speech_filler"}) == [("reads:noise_class", "speech_filler")]
    for code in ("seller_status_update", "partner_internal_process", "parser_derived",
                 "internal_housekeeping"):
        assert _reads_rows({"co_reason": code}) == [("reads:co_reason", code)]
    for cls in ("page_chrome", "template_instruction", "cross_reference"):
        assert _reads_rows({"noise_class": cls}) == [("reads:noise_class", cls)]


def test_the_parser_line_never_reaches_training():
    from app.learning.human_labels import split_note, strip_parser_lines

    note = "Four racks set the crew.\n[purtera] keep: two techs.\n[parser] SHOULD SPLIT: two sites in one atom"
    assert split_note(note) == ("Four racks set the crew.", "keep: two techs.")
    assert strip_parser_lines(note) == "Four racks set the crew.\n[purtera] keep: two techs."
    assert strip_parser_lines("Plain WHY.") == "Plain WHY."
    assert split_note("WHY only\n[parser] page 3 cut") == ("WHY only", "")
