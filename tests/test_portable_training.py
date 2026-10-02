"""The universal base and the company layer train apart (labeling/portable-labels.md e, f)."""
from __future__ import annotations

import json
import random

import pytest

from app.core.atom_type_registry import load_registry
from app.learning.human_labels import IngestReport, rows_for_deal, split_note
from app.learning.label_context import (
    DROPOUT_RATES, HEAD_CONTEXT, KNOWN_PARTS, context_text, dropout_copy, parts_for,
)
from app.learning.multitask_table import BASE_TASKS, DEFAULT_TASKS, tasks_for

LAYER = {r["key"]: r["layer"] for r in load_registry()["reads"]}

SELLER_NOTE = (
    "A promise from the seller (seller side, ours) to send a revised quote by end of day. "
    "It owns no delivery work and changes no scope.\n"
    "[purtera] reject: sellers don't own tasks at Purtera; only PM/PC/SA/tech commitments "
    "become planned work."
)


def _label(**kw):
    base = {
        "label_key": "lbl_x", "text": "I'll get the revised quote over by EOD.",
        "label_type": "commitment", "section": ["Re: quote"], "lead_in": [],
        "neighbors_above": ["Thanks Zach"], "neighbors_below": ["Best,"],
        "hints": ["own_words"], "doc_type": "email", "filename": "thread.eml", "page": 1,
        "said_by": {"company": "purtera-it.com", "role": "seller", "side": "ours", "name": "Zach"},
        "said_to": [{"company": "cdw.com", "role": "reseller", "side": "theirs", "name": "Alec"}],
        "labeler": "a@purtera-it.com",
    }
    base.update(kw)
    return base


def _rows(**kw):
    return rows_for_deal({"deal_id": "d1", "labels": [_label(**kw)]})


# --- the note split -----------------------------------------------------------

def test_the_note_splits_at_the_company_line():
    why, policy = split_note(SELLER_NOTE)
    assert why.startswith("A promise from the seller") and "[purtera]" not in why
    assert "Purtera" not in why
    assert policy.startswith("reject: sellers don't own tasks")


def test_the_exclusion_marker_is_not_part_of_either_argument():
    why, policy = split_note(
        "[EXCLUDE_FROM_TRAINING: old manual Deal Kit]\nA labor line from a hand-built kit.\n"
        "[purtera] ignore: old Purtera artifact, nothing to learn.")
    assert why == "A labor line from a hand-built kit."
    assert policy == "ignore: old Purtera artifact, nothing to learn."
    why, _ = split_note("EXCLUDE_FROM_TRAINING: Deal Kit line\nThe universal part.")
    assert why == "The universal part."


def test_a_note_without_a_company_line_is_all_universal():
    assert split_note("Only the WHY, no policy.") == ("Only the WHY, no policy.", "")
    # The marker counts only at the start of a line.
    assert split_note("Says [purtera] mid-sentence.")[1] == ""


def test_rationale_rows_carry_one_argument_each_and_choose_stays_universal():
    rows = _rows(note=SELLER_NOTE, about="deal",
                 reads_set={"co_action": "reject", "co_reason": "seller_promise"})
    by = {r["relation"]: r for r in rows if r["relation"].startswith("rationale:")}
    atom, policy = by["rationale:atom"], by["rationale:policy:purtera"]
    assert "[purtera]" not in atom["label"] and atom["label"].startswith("A promise")
    assert policy["label"].startswith("reject: sellers")
    for r in (atom, policy):
        chose = r["raw_text"].split(" CHOSE: ", 1)[1]
        assert chose == "type=commitment | about=deal"


def test_the_bracketed_exclusion_marker_still_excludes():
    report = IngestReport()
    rows = rows_for_deal({"deal_id": "d1", "labels": [
        _label(note="[EXCLUDE_FROM_TRAINING: old manual Deal Kit]\nA kit line."),
        _label(label_key="lbl_y")]}, report)
    assert {json.loads(r["provenance"]).get("label_key") for r in rows} == {"lbl_y"}


# --- co_action ----------------------------------------------------------------

@pytest.mark.parametrize("action", ["keep", "reject", "ignore"])
def test_co_action_teaches_the_company_filter(action):
    rows = _rows(reads_set={"co_action": action, "co_reason": "seller_promise"})
    policy = [r for r in rows if r["relation"] == "policy:purtera"]
    assert [r["label"] for r in policy] == [action]
    assert json.loads(policy[0]["provenance"])["co_reason"] == "seller_promise"
    # Admission is unchanged: a real type is never an admission drop.
    assert not [r for r in rows if r["relation"] == "admission"]


def test_a_policy_reject_keeps_its_type_and_is_not_an_admission_drop():
    rows = _rows(rejected="true", reads_set={"co_action": "reject"})
    by = {r["relation"]: r["label"] for r in rows}
    assert by["atom_type"] == "commitment" and by["policy:purtera"] == "reject"
    assert "admission" not in by and "rejected" not in by


def test_noise_is_still_an_admission_drop_with_no_policy_row():
    rows = _rows(label_type="small_talk", text="Thanks so much!",
                 reads_set={"noise_class": "greeting_thanks"})
    rels = {r["relation"]: r["label"] for r in rows}
    assert rels["admission"] == "drop" and "policy:purtera" not in rels


# --- profiles -----------------------------------------------------------------

def test_the_base_trains_universal_readings_only():
    reads = {t.split(":", 1)[1] for t in BASE_TASKS if t.startswith("reads:")}
    assert reads == {k for k, v in LAYER.items() if v == "universal"}
    assert not [t for t in BASE_TASKS if t.startswith(("reads:co_", "policy:"))]
    assert not [t for t in BASE_TASKS if "intake_gap" in t or "needed_by" in t]
    assert tasks_for() == tasks_for("base") == BASE_TASKS == DEFAULT_TASKS


def test_the_purtera_profile_adds_its_layer_and_never_meta_or_staging():
    tasks = tasks_for("purtera")
    assert set(BASE_TASKS) < set(tasks)
    extra = set(tasks) - set(BASE_TASKS)
    assert extra == ({f"reads:{k}" for k, v in LAYER.items() if v == "company"}
                     | {"policy:purtera", "question:intake_gap", "question:needed_by"})
    for k, v in LAYER.items():
        if v in ("meta", "staging"):
            assert f"reads:{k}" not in tasks
    with pytest.raises(ValueError):
        tasks_for("acme")


# --- no leakage into context --------------------------------------------------

LEAK = {"co_action": "reject_zz", "co_reason": "seller_promise_zz", "co_company": "purtera_zz",
        "co_stage_raw": "Closed Won zz", "deal_outcome": "won_zz", "universal_type": "commitment_zz",
        "stage_std": "quoted_zz", "deal_stage": "planning_zz"}


def test_no_head_is_served_a_label_policy_or_outcome():
    lb = _label(reads_set=dict(LEAK), **LEAK)
    relations = set(HEAD_CONTEXT) | set(tasks_for("purtera")) | {"policy:purtera"}
    for rel in relations:
        assert set(parts_for(rel)) <= KNOWN_PARTS, rel
        rendered = context_text(rel, lb)
        for v in LEAK.values():
            assert v not in rendered, (rel, v)
    for r in rows_for_deal({"deal_id": "d1", "labels": [lb]}, dropout_seed=1):
        for v in LEAK.values():
            assert v not in r["raw_text"], (r["relation"], v)


def test_known_parts_are_inference_inputs_only():
    assert KNOWN_PARTS == {"table", "section", "lead_in", "doc", "from", "above", "below"}
    assert not [p for p in KNOWN_PARTS if p.startswith("co_") or "stage" in p or "outcome" in p]


# --- context dropout ----------------------------------------------------------

def test_dropout_adds_copies_and_leaves_every_original_byte_identical():
    labels = [_label(label_key=f"k{i}", text=f"Mount {i} displays in the lobby",
                     about="deal", reads_set={"commitment": "mount them"}) for i in range(30)]
    deal = "d-train"
    plain = rows_for_deal({"deal_id": deal, "labels": labels})
    assert plain[0]["split"] == "train"
    aug = rows_for_deal({"deal_id": deal, "labels": labels}, dropout_seed=7)
    originals = [r for r in aug if "augmentation" not in json.loads(r["provenance"])]
    assert originals == plain
    copies = [r for r in aug if "augmentation" in json.loads(r["provenance"])]
    assert copies
    assert {r["relation"] for r in copies} <= {"atom_type", "atom_type_coarse", "facet", "about",
                                               "reads:commitment"}
    for c in copies:
        dropped = json.loads(c["provenance"])["dropped"]
        assert dropped and c["raw_text"] == context_text(c["relation"], next(
            lb for lb in labels if lb["label_key"] == json.loads(c["provenance"])["label_key"]),
            drop=dropped)
    # Deterministic per seed.
    again = rows_for_deal({"deal_id": deal, "labels": labels}, dropout_seed=7)
    assert [r["raw_text"] for r in again] == [r["raw_text"] for r in aug]


def test_dropout_rates_and_the_domain_drop():
    assert DROPOUT_RATES == {"from": 0.2, "doc": 0.2, "neighbors": 0.3,
                             "section": 0.1, "lead_in": 0.1, "from_company": 0.5}
    lb = _label()
    assert "cdw.com" in context_text("about", lb)
    no_domain = context_text("about", lb, drop=["from_company"])
    assert "cdw.com" not in no_domain and "(reseller, theirs)" in no_domain
    assert "Zach" not in context_text("about", lb) and "Alec" not in context_text("about", lb)
    rng = random.Random(0)
    seen = [dropout_copy("about", lb, rng) for _ in range(2000)]
    rate = sum(1 for s in seen if s and "from_company" in s[1]) / len(seen)
    assert 0.3 < rate < 0.5  # 50% of the copies that keep `from`


def test_holdout_rows_get_no_copies():
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label(purpose="eval")]}, dropout_seed=3)
    assert not [r for r in rows if "augmentation" in json.loads(r["provenance"])]


def test_a_deal_kit_row_is_excluded_by_its_column_or_marker_without_train_for():
    # The backfilled shape: weight_tier and consumer are columns, reads_set has
    # no train_for, and the note ends with the company's ignore line.
    note = ("A labor line copied from a hand-built kit for this site.\n"
            "[purtera] ignore: old manual Deal Kit row, nothing to learn.")
    for row in (
        _label(label_key="kit1", weight_tier="exclude", consumer="ignore", note=note,
               reads_set={"co_action": "ignore", "co_reason": "old_manual_deal_kit"}),
        _label(label_key="kit2", note="EXCLUDE_FROM_TRAINING: old kit\n" + note,
               reads_set={"co_action": "ignore"}),
    ):
        report = IngestReport()
        assert rows_for_deal({"deal_id": "d1", "labels": [row]}, report) == []
        assert report.skipped["excluded from training"] == 1
    # Not excluded and no train_for: trains, and the ignore is the policy class.
    rows = _rows(note=note, reads_set={"co_action": "ignore"})
    by = {r["relation"]: r["label"] for r in rows}
    assert by["policy:purtera"] == "ignore"
    assert by["rationale:policy:purtera"].startswith("ignore: old manual Deal Kit row")


def test_provenance_lines_go_through_the_same_generic_split():
    # No special case for review provenance or a policy line that restates the
    # WHY: before the marker is the universal argument, after it the policy.
    note = ("RULING: a reviewer settled this line as a quoted fee basis. Accepted from claude-code "
            "draft with edits.\n[purtera] keep: the fee basis is a quoted fee basis.")
    why, policy = split_note(note)
    assert why.startswith("RULING: a reviewer settled") and "Accepted from claude-code" in why
    assert policy == "keep: the fee basis is a quoted fee basis."
    one_line = "RULING: noise. Accepted from claude-code."
    assert split_note(one_line) == (one_line, "")
