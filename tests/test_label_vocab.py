"""Entity keys and scope_category are closed lists (app/core/label_vocab.json).

Invented rows only. Off-list values are kept and counted, never dropped; only
a one-to-one alias is renamed; a person's row is never changed by the dry-run
script's --apply.
"""
from __future__ import annotations

import gzip
import json
import re

from app.core import label_vocab as lv
from app.learning.human_labels import IngestReport, rows_for_deal
from ml.c3 import vocab as c3_vocab
from scripts import label_vocab_dryrun as dry

SAMPLE_KEYS = [
    "party:acme", "org:acme", "customer:acme", "quantity:12", "qty:12_sites", "amount:500",
    "money:500", "billing:t_and_m", "billing:time_and_materials", "billing:tandm",
    "term:time_and_materials", "billing:hourly_ish", "term:no_slo", "term:after_hours_150pct",
    "term:rate_lock_6_months", "service:onsite_it_support", "service:onsite", "delivery:onsite",
    "part_number:ps_onsite_labor", "distance:40mi", "hours:6_per_visit", "hours:6",
    "site_count:3", "exclusion:remote_support", "exclusion:travel", "price:10", "deal:7",
    "stakeholder:pat_doe", "person:pat_doe", "nocolon", "rate:l2_4hr:80",
]


def test_registry_shape():
    vocab = lv.load_vocab()
    prefixes = set(lv.entity_prefixes())
    assert lv.closed_values("billing") == ("fixed", "t_and_m", "milestone", "per_site")
    assert {"party", "person", "stakeholder", "qty", "money", "site", "billing", "sla"} <= prefixes
    tags = vocab["entity_tags"]
    # Old spellings are not prefixes; every rename lands on the list.
    assert not set(tags["prefix_aliases"]) & prefixes
    assert set(tags["prefix_aliases"].values()) <= prefixes
    for old, new in tags["key_aliases"].items():
        assert lv.entity_tag(new).status == lv.OK, (old, new)
    for review in tags["review_aliases"].values():
        assert review in prefixes
    assert len(lv.scope_categories()) == len(set(lv.scope_categories())) == 25
    assert all(re.fullmatch(r"[a-z_]+", v) and d for v, d in vocab["scope_category"]["values"].items())


def test_entity_tag_statuses():
    t = lv.entity_tag
    assert t("party:acme").status == lv.OK
    assert t("org:acme") == lv.Check("org:acme", "party:acme", lv.ALIAS, "org:* -> party:*")
    assert t("customer:acme").canonical == "party:acme"
    assert t("quantity:12").canonical == "qty:12"
    assert t("amount:500").canonical == "money:500"
    assert t("billing:time_and_materials").canonical == "billing:t_and_m"
    assert t("billing:tandm").canonical == "billing:t_and_m"
    assert t("term:time_and_materials").canonical == "billing:t_and_m"
    assert t("term:no_slo").canonical == "sla:none"
    assert t("term:after_hours_150pct").canonical == "rate:after_hours_150pct"
    assert t("term:rate_lock_6_months").status == lv.OK
    assert t("service:onsite_it_support").canonical == "service:on_site_it_support"
    assert t("service:onsite").canonical == "service:on_site"
    assert t("delivery:onsite").canonical == "delivery:on_site"
    # A part number keeps its own spelling.
    assert t("part_number:ps_onsite_labor").status == lv.OK
    assert t("distance:40mi").canonical == "distance_miles:40"
    assert t("hours:6_per_visit").canonical == "visit_hours:6"
    assert t("hours:6").status == lv.OK
    assert t("site_count:3").canonical == "qty:3_sites"
    assert t("exclusion:remote_support").canonical == "service:remote_support"


def test_off_list_is_kept_and_never_names_an_open_value():
    c = lv.entity_tag("exclusion:travel")
    assert (c.status, c.canonical, c.label, c.proposal) == (lv.OFF_LIST, "exclusion:travel",
                                                            "exclusion:*", "service:* (review)")
    assert lv.entity_tag("deal:7").label == "deal:*"
    assert lv.entity_tag("price:10").proposal == "money:* (review)"
    # A closed prefix spells out its off-list slug; free text is not printed.
    assert lv.entity_tag("billing:hourly_ish").label == "billing:hourly_ish"
    assert lv.entity_tag("billing:Some Typed Words").label == "billing:<free text>"
    assert lv.entity_tag("nocolon").label == "(no prefix)"
    assert lv.canonical_keys(["org:acme", "party:acme", "deal:7"]) == ["party:acme", "deal:7"]


def test_scope_category():
    assert lv.scope_category("services_fees").status == lv.OK
    c = lv.scope_category("Service Model")
    assert (c.status, c.canonical) == (lv.ALIAS, "service_model")
    c = lv.scope_category("summary")
    assert (c.status, c.canonical, c.label) == (lv.OFF_LIST, "summary", "summary")
    assert lv.scope_category("Our display-install group").label == "<free text>"


def test_c3_mirror_agrees_with_the_app_registry():
    for k in SAMPLE_KEYS:
        assert c3_vocab.canonical_key(k) == lv.canonical_key(k), k
    assert c3_vocab.canonical_keys(SAMPLE_KEYS) == lv.canonical_keys(SAMPLE_KEYS)


def _label(**kw):
    base = {"label_key": "lbl_v", "text": "Weekly visit to each site, billed hourly",
            "label_type": "work_scope_item", "labeler": "a@example.com",
            "entity_keys": ["org:acme", "quantity:3", "deal:7", "party:acme"],
            "reads_set": {"scope_category": "summary"}}
    base.update(kw)
    return base


def test_training_counts_off_list_and_renames_one_to_one():
    rep = IngestReport()
    rows = rows_for_deal({"deal_id": "d1", "labels": [_label()]}, rep)
    prov = json.loads(next(r for r in rows if r["relation"] == "atom_type")["provenance"])
    assert prov["entity_keys"] == ["party:acme", "qty:3", "deal:7"]   # off-list kept
    assert rep.off_list == {
        "entity key renamed: org:* -> party:*": 1,
        "entity key renamed: quantity:* -> qty:*": 1,
        "entity key off the list: deal:*": 1,
        "scope_category off the list: summary": 1,
    }
    # The reading still trains (presence), as before.
    assert any(r["relation"] == "reads:scope_category" for r in rows)
    assert not any("scope_category" in k or "entity" in k for k in rep.skipped)


def test_dry_run_counts_rows_per_value(tmp_path, capsys):
    snap = {"labels": [
        _label(),
        _label(label_key="b", entity_keys=["org:beta", "billing:time_and_materials"],
               reads_set=json.dumps({"scope_category": "network"})),
        _label(label_key="c", entity_keys=json.dumps(["party:beta"]), reads_set={}),
    ]}
    res = dry.count(snap["labels"])
    assert res["totals"]["rows"] == 3
    assert res["totals"]["rows_with_alias_key"] == 2
    assert res["totals"]["rows_with_off_list_key"] == 1
    assert res["entity_keys"]["org:* -> party:*"]["rows"] == 2
    assert res["entity_keys"]["billing:time_and_materials -> billing:t_and_m"]["proposed"] == "billing:t_and_m"
    assert res["scope_category"] == {"summary": {"rows": 1, "status": lv.OFF_LIST,
                                                 "proposed": "(none: keep, a person decides)",
                                                 "person_rows": 1}}
    path = tmp_path / "snap.json.gz"
    path.write_bytes(gzip.compress(json.dumps(snap).encode()))
    assert dry.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out and "acme" not in out and "beta" not in out
    assert list(tmp_path.iterdir()) == [path]


def test_apply_never_changes_a_person_row_and_removes_nothing(tmp_path):
    person = _label(entity_keys=["org:acme", "deal:7"], reads_set={"scope_category": "Service Model"})
    machine = _label(label_key="m", labeler="helper (assistant)",
                     entity_keys=["org:acme", "deal:7", "party:acme"],
                     reads_set={"scope_category": "Service Model"})
    snap = {"labels": [person], "proposedLabels": [machine]}
    src, out = tmp_path / "snap.json", tmp_path / "out.json"
    src.write_text(json.dumps(snap))
    assert dry.main([str(src), "--apply", "--out", str(out)]) == 0
    assert json.loads(src.read_text()) == snap                      # input untouched
    new = json.loads(out.read_text())
    assert new["labels"] == [person]                                 # a person's picks stay
    assert new["proposedLabels"][0]["entity_keys"] == ["party:acme", "deal:7"]
    assert new["proposedLabels"][0]["reads_set"]["scope_category"] == "service_model"
    assert len(new["labels"]) + len(new["proposedLabels"]) == 2


def test_apply_refuses_to_overwrite_its_input(tmp_path):
    import pytest

    src = tmp_path / "snap.json"
    src.write_text("[]")
    with pytest.raises(SystemExit):
        dry.main([str(src), "--apply", "--out", str(src)])
    with pytest.raises(SystemExit):
        dry.main([str(src), "--apply"])
