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
                  note="Unit price of the display; it would be a labor rate if it priced the mounting.\n[purtera] reject: the partner prices hardware.")
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
    ok = "[EXCLUDE_FROM_TRAINING: old manual Deal Kit]\nA rate row.\n[purtera] ignore: old kit."
    assert "exclude_marker_misplaced" not in _checks(_row(note=ok))
    assert "exclude_marker_misplaced" in _checks(_row(note=ok + "\nEXCLUDE_FROM_TRAINING: old manual internal pricing workbook"))
    assert "exclude_marker_misplaced" in _checks(_row(note="A rate row. EXCLUDE_FROM_TRAINING: old kit"))


def test_derived_from_stand_in_links_become_the_relation():
    new, moved = transform_link({"id": "u1", "relation": "context", "note": "[derived_from] total from qty x rate"})
    assert new["relation"] == "derived_from" and new["note"] == "total from qty x rate" and moved
    same, moved = transform_link({"id": "u2", "relation": "context", "note": "background"})
    assert same["relation"] == "context" and not moved
    new, moved = transform_link({"id": "u3", "relation": "supports", "note": "[derived_from] SOW total"})
    assert new["relation"] == "derived_from" and new["note"] == "SOW total" and moved == ["supports -> derived_from"]
    same, moved = transform_link({"id": "u4", "relation": "supports", "note": "same rate"})
    assert same["relation"] == "supports" and not moved
    same, moved = transform_link({"id": "u5", "relation": "governs", "note": "[derived_from] x"})
    assert same["relation"] == "governs" and not moved


def test_reads_in_real_use_are_registered():
    """Readings deal threads already write (the four dry runs) have a head and
    a vocab, so they neither trip unregistered_read nor drop out of training."""
    reads = {"skip": True, "exclude_from_training": True, "requirement_kind": "access",
             "list_header": True, "governs_count": "4", "derived_from_count": "2", "tech_level": "L2",
             "display_size": "75", "equipment_qty": "4", "labor_hours": "6", "material": "cable",
             "scope_side": "customer", "site": "HQ", "location": "Chicago, IL", "rate": "$95/hr",
             "rate_for": "lead tech", "region": "Chicago area", "placeholder": True, "cadence": "monthly",
             "lead_time": "2 weeks", "tech_coverage": "local", "sow_available_note": "x",
             "scope_category_note": "x", "derivation_note": "x", "sow_coverage_note": "x",
             "address_note": "x", "needed_by": ["quoting", "sow", "delivery"],
             "address_level": "street_no_city", "location_tier": ["major_metro", "rural"]}
    checks = _checks(_row(reads_set=reads))
    assert "unregistered_read" not in checks and "value_outside_vocab" not in checks
    for lvl in ("street_only", "name_only", "state_only", "region_only", "street_no_zip"):
        assert "value_outside_vocab" not in _checks(_row(reads_set={"address_level": lvl}))


def test_older_names_move_to_the_registered_reading():
    new, moved = transform_row(_row(text="2 techs on site for the mount",
                                    reads_set={"qty": "4", "tech_qty": "2", "visit_frequency": "monthly",
                                               "loe_hours": "12", "co_action": "keep"}))
    r = new["reads_set"]
    assert r["equipment_qty"] == "4" and r["crew_size"] == "2" and r["cadence"] == "monthly"
    assert r["labor_hours"] == "12" and not {"qty", "tech_qty", "visit_frequency", "loe_hours"} & set(r)
    # a remote or office role is not field crew: tech_qty stays and is reported
    for text in ("Site: PC | PS-PROJMGMT-REMOTE", "Project manager, 1 tech hour", "Fee row"):
        new, moved = transform_row(_row(text=text, reads_set={"tech_qty": "1"}))
        assert new["reads_set"] == {"tech_qty": "1"} and "crew_size" not in new["reads_set"], text
        assert "tech_qty: kept, the line is not a field tech role" in moved
    # an estimate is ours, a flag says whether hours are stated
    new, _ = transform_row(_row(reads_set={"hours": "8", "hours_stated": "false"}))
    assert new["reads_set"]["co_hours_estimate"] == "8" and "hours" not in new["reads_set"]
    new, _ = transform_row(_row(reads_set={"hours": True}))
    assert new["reads_set"]["hours_stated"] is True
    # a registered name that already says something else wins; nothing is lost
    new, moved = transform_row(_row(reads_set={"qty": "4", "equipment_qty": "6"}))
    assert new["reads_set"]["qty"] == "4" and new["reads_set"]["equipment_qty"] == "6"
    assert any("kept" in m for m in moved)


def test_multi_readings_split_from_any_older_join():
    def after(v, key="needed_by"):
        return transform_row(_row(reads_set={key: v}))[0]["reads_set"][key]
    assert after("project_manager|atlas|portal") == ["project_manager", "atlas", "portal"]
    assert after("quote|sow|delivery") == ["quoting", "sow", "delivery"]
    assert after("quote, SOW and dispatch") == ["quoting", "sow", "delivery"]
    assert after("small_town/rural", "location_tier") == "small_town/rural"
    assert "value_outside_vocab" in _checks(_row(reads_set={"location_tier": "small_town/rural"}))
    assert after("major_metro, rural", "location_tier") == ["major_metro", "rural"]
    mixed = "mixed: 4 major_metro, 2 rural"
    assert after(mixed, "location_tier") == mixed
    row = _row(reads_set={"needed_by": ["quoting"]})
    assert transform_row(row)[1] == []


def test_column_readings_move_into_their_column():
    new, moved = transform_row(_row(supplier=None, entity_keys=["site:a"],
                                    reads_set={"entity_keys": "site:a, site:b", "co_action": "keep"}))
    assert new["entity_keys"] == ["site:a", "site:b"] and "entity_keys" not in new["reads_set"]
    new, _ = transform_row(_row(about="customer", reads_set={"about": "deal"}))
    assert new["reads_set"]["about"] == "deal" and new["about"] == "customer"


def test_sql_is_guarded_on_the_rows_current_values():
    row = _row(reads_set={"about": "deal"})
    new, _ = transform_row(row)
    sql = label_sql(row, new)
    assert "about = 'deal'" in sql
    assert "reads_set = '" + json.dumps({"about": "deal"}) + "'::jsonb" in sql
    assert "note IS NOT DISTINCT FROM NULL" in sql and "label_type" not in sql and "rejected" not in sql


def test_sql_moves_entity_keys_under_a_guard():
    row = _row(entity_keys=["site:a"], supplier=None, reads_set={"entity_keys": "site:b"})
    sql = label_sql(row, transform_row(row)[0])
    assert "entity_keys = '[\"site:a\", \"site:b\"]'::jsonb" in sql
    assert "AND entity_keys = '[\"site:a\"]'::jsonb" in sql and "supplier IS NOT DISTINCT FROM NULL" in sql


def test_report_counts_per_deal():
    report, sql = run([_row(reads_set={"about": "deal"}), _row(label_key="lbl_00000000000000000002")], [])
    r = report["d1"]
    assert r["rows"] == 2 and r["rows_changed"] == 1 and len(sql) == 1
    assert r["rows_per_head"]["content.frame"] == 1


def test_a_parser_remark_belongs_on_the_last_parser_line():
    ok = "Four racks set the crew.\n[purtera] keep: two techs.\n[parser] SHOULD SPLIT: two sites in one atom"
    assert not _checks(_row(note=ok)) & {"parser_line_not_last", "parser_remark_outside_line"}
    assert "parser_remark_outside_line" in _checks(_row(note="Four racks. SHOULD MERGE with the next line."))
    assert "parser_remark_outside_line" in _checks(_row(note="The parser cut this sentence."))
    assert "parser_remark_outside_line" not in _checks(_row(note='Quoted: "the parser" is their word.'))
    misplaced = "[parser] SHOULD MERGE\nFour racks.\n[purtera] keep: two techs."
    assert "parser_line_not_last" in _checks(_row(note=misplaced))
    twice = "Four racks.\n[parser] SHOULD MERGE\n[parser] page 2"
    assert "parser_line_not_last" in _checks(_row(note=twice))


def test_the_duplicate_marker_opens_the_note():
    ok = "DUPLICATE of parser atom lbl_1: same line.\nFour racks."
    assert not _checks(_row(note=ok)) & {"duplicate_marker_misplaced", "parser_remark_outside_line"}
    after_exclude = "[EXCLUDE_FROM_TRAINING: old manual Deal Kit]\nDUPLICATE of parser atom lbl_1\nWHY"
    assert "duplicate_marker_misplaced" not in _checks(_row(note=after_exclude))
    # 010087: the marker follows the EXCLUDE reason on the same line.
    same_line = "EXCLUDE_FROM_TRAINING: copy of a parser row. DUPLICATE of parser atom: lbl_1, same text.\nFour racks."
    for c in ("duplicate_marker_misplaced", "duplicate_marker_unmatched", "parser_remark_outside_line"):
        assert c not in _checks(_row(note=same_line)), c
    assert "parser_remark_outside_line" not in _checks(_row(note=after_exclude))
    assert "parser_remark_outside_line" in _checks(_row(note="EXCLUDE_FROM_TRAINING: x.\nThe parser cut this."))
    twice = "EXCLUDE_FROM_TRAINING: DUPLICATE of parser atom lbl_1\nDUPLICATE of parser atom lbl_1"
    assert "duplicate_marker_misplaced" in _checks(_row(note=twice))
    assert "duplicate_marker_misplaced" in _checks(_row(note="Four racks.\nDUPLICATE of parser atom lbl_1"))
    assert "duplicate_marker_misplaced" in _checks(_row(note="Four racks.\n[parser] DUPLICATE of parser atom lbl_1"))


def test_a_duplicate_marker_in_other_words_is_flagged():
    other = "DUPLICATE: this line is now an atom from the parser.\nFour racks."
    assert "duplicate_marker_unmatched" in _checks(_row(note=other))
    ok = "EXCLUDE_FROM_TRAINING: old manual Deal Kit.\nDUPLICATE of parser atom lbl_1"
    assert not _checks(_row(note=ok)) & {"duplicate_marker_unmatched", "duplicate_marker_misplaced"}


def test_a_kept_fact_whose_why_has_no_flip_is_flagged():
    flat = "Four racks on the second floor set the crew at two techs."
    assert "why_without_flip" in _checks(_row(note=flat))
    flip = flat + " If the racks were already mounted, this would be a cabling line, not an install."
    assert "why_without_flip" not in _checks(_row(note=flip))
    # A flip only inside a quote is the line's words, not the labeler's.
    assert "why_without_flip" in _checks(_row(note='They wrote "if needed". Four racks set the crew.'))
    # The rule line does not count, and noise, excluded and duplicate rows take no flip.
    assert "why_without_flip" in _checks(_row(note=flat + "\n[purtera] keep: if two techs, quote two."))
    assert "why_without_flip" not in _checks(_row(label_type="_keep", note=flat,
                                                   reads_set={"noise_class": "greeting_thanks"}))
    assert "why_without_flip" not in _checks(_row(note="EXCLUDE_FROM_TRAINING: old manual Deal Kit\n" + flat))
    assert "why_without_flip" not in _checks(_row(note="DUPLICATE of parser atom lbl_1\n" + flat))
    assert "why_without_flip" not in _checks(_row(note=None))
