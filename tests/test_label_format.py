"""The head-format transform moves values and loses none; the checks name what
a person still has to decide."""
from __future__ import annotations

import json

from app.learning.label_format import format_checks, transform_link, transform_row
from scripts.labels_to_heads import label_sql, run


def _row(**kw):
    base = {"deal_id": "d1", "label_key": "lbl_00000000000000000001", "labeler": "a@b.c",
            "label_type": "pricing_assumption", "about": None, "note": None, "rejected": None,
            "reads_set": {"co_company": "purtera", "co_action": "keep"}}
    base.update(kw)
    return base


def _checks(row):
    return {c["check"] for c in format_checks(row)}


def test_about_moves_from_reading_to_column():
    new, moved = transform_row(_row(reads_set={"about": "deal", "co_action": "keep"}))
    assert new["about"] == "deal" and "about" not in new["reads_set"]
    assert moved == ["about: reading -> column"]


def test_about_reading_that_disagrees_with_the_column_stays_put():
    row = _row(about="account", reads_set={"about": "deal"})
    new, moved = transform_row(row)
    assert new == row and moved == []


def test_aliases_lists_and_markers():
    row = _row(reads_set={"billing_type": "time_and_materials", "sow_section": "purtera_responsibilities",
                          "train_for": "quote_parser, delivery_parser"},
               note="Hardware price from the partner's quote. [purtera] reject: the partner prices hardware.")
    new, moved = transform_row(row)
    assert new["reads_set"] == {"billing_type": "t_and_m", "sow_section": "provider_responsibilities",
                                "train_for": ["quote_parser", "delivery_parser"]}
    assert new["note"] == "Hardware price from the partner's quote.\n[purtera] reject: the partner prices hardware."
    assert len(moved) == 4


def test_transform_is_idempotent_and_never_touches_type_or_flag():
    row = _row(label_type="_keep", rejected="true",
               reads_set={"universal_type": "commitment", "co_action": "reject", "co_reason": "seller_promise",
                          "about": "deal", "train_for": "quote_parser"},
               note="A promise from the seller to send a quote.\n[purtera] reject: sellers own no tasks.")
    once, _ = transform_row(row)
    twice, moved = transform_row(once)
    assert twice == once and moved == []
    assert once["label_type"] == "_keep" and once["rejected"] == "true"
    assert once["note"] == row["note"]


def test_checks():
    assert _checks(_row(label_type="small_talk", reads_set={})) == {"noise_without_class"}
    assert "noise_with_action" in _checks(_row(label_type="_keep", reads_set={"noise_class": "legal_footer",
                                                                            "co_action": "reject"}))
    # a policy reject parked on _keep during the backfill is a fact, not noise
    parked = _row(label_type="_keep", reads_set={"universal_type": "bom_line", "co_action": "reject",
                                                 "co_reason": "hw_price_not_ours"},
                  note="Unit price of the display.\n[purtera] reject: the partner prices hardware.")
    assert _checks(parked) == set()
    assert {"reason_missing", "policy_line_missing"} <= _checks(_row(reads_set={"co_action": "reject"}))
    assert "policy_words_in_why" in _checks(_row(note="Matches the Deal Kit billing."))
    assert "policy_words_in_why" in _checks(_row(note="Same rate as the internal pricing workbook."))
    assert "policy_words_in_why" not in _checks(_row(note="The customer's pricing workbook lists 4 sites."))
    assert "policy_words_in_why" not in _checks(_row(note='The heading reads "Rejected items".'))
    assert "policy_words_in_why" not in _checks(_row(
        note="[EXCLUDE_FROM_TRAINING: old manual Deal Kit]\nA rate row.\n[purtera] ignore: old kit."))
    assert "value_outside_vocab" in _checks(_row(reads_set={"co_action": "maybe"}))
    assert "unregistered_read" in _checks(_row(reads_set={"made_up": "x"}))
    assert "marker_inline" in _checks(_row(note="Why. [purtera] keep: feeds crew."))


def test_derived_from_stand_in_links_become_the_relation():
    new, moved = transform_link({"id": "u1", "relation": "context", "note": "[derived_from] total from qty x rate"})
    assert new["relation"] == "derived_from" and new["note"] == "total from qty x rate" and moved
    same, moved = transform_link({"id": "u2", "relation": "context", "note": "background"})
    assert same["relation"] == "context" and not moved


def test_sql_is_guarded_on_the_rows_current_values():
    row = _row(reads_set={"about": "deal"})
    new, _ = transform_row(row)
    sql = label_sql(row, new)
    assert "about = 'deal'" in sql
    assert "reads_set = '" + json.dumps({"about": "deal"}) + "'::jsonb" in sql
    assert "note IS NOT DISTINCT FROM NULL" in sql and "label_type" not in sql and "rejected" not in sql


def test_report_counts_per_deal():
    report, sql = run([_row(reads_set={"about": "deal"}), _row(label_key="lbl_00000000000000000002")], [])
    r = report["d1"]
    assert r["rows"] == 2 and r["rows_changed"] == 1 and len(sql) == 1
    assert r["rows_per_head"]["content.frame"] == 1
