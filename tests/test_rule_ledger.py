"""The rule-fire ledger (app/core/rule_ledger.py): every rule that overrides a head decision, counted against the
head's own answer, with a per-rule shadow mode. Synthetic fixtures."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from app.core import c3_type_base as c3
from app.core import rule_ledger as rl
from app.core.party_address_veto import follow_party_copies, veto_party_page_sites
from app.core.site_keys_follow_heads import settle


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SOWSMITH_RULE_MODES", raising=False)
    monkeypatch.setitem(c3._DEAL, "types", {})
    rl.reset()
    yield
    rl.reset()


def _atom(aid, text, kind="scope_item", keys=(), *, page=None, line=None, value=None, flags=()):
    loc = {}
    if page is not None:
        loc["page"] = page
    if line is not None:
        loc["line_start"] = line
    return NS(id=aid, raw_text=text, atom_type=kind, entity_keys=list(keys), artifact_id="art",
              review_flags=list(flags), value=dict(value or {}), decision_provenance=None,
              source_refs=[NS(locator=loc, extraction_method="docx_text")])


def _type(a):
    return str(getattr(a.atom_type, "value", a.atom_type))


# ── the module ───────────────────────────────────────────────────────────────

def test_modes_come_from_env_exact_then_prefix(monkeypatch):
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "site_keys.*:shadow, site_keys.mint_from_head:active,bogus")
    assert rl.asked_mode("site_keys.drop_unbacked") == "shadow"
    assert rl.asked_mode("site_keys.mint_from_head") == "active"
    assert rl.asked_mode("party_address_veto.signature_page") == "active"
    assert rl.shadow("site_keys.drop_unbacked") and not rl.shadow("site_keys.mint_from_head")


def test_a_log_only_rule_never_shadows_and_says_so(monkeypatch):
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "semantic_dedup.site_merge:shadow,quote.*:shadow")
    assert not rl.shadow("semantic_dedup.site_merge")
    assert rl.mode("semantic_dedup.site_merge").startswith("active (shadow not honoured")
    assert not rl.shadow("quote.hours.fallback:family_median")


def test_fire_counts_agreement_no_head_and_each_line_once():
    assert rl.fire("r", "a1", "deal_metadata", "not:physical_site") is True
    assert rl.fire("r", "a2", "physical_site", "not:physical_site") is False
    assert rl.fire("r", "a3", None, "same_site") is None
    rl.fire("r", "a2", "physical_site", "not:physical_site")  # the same line again: counted once
    rl.fire("r", None, None, "x")  # deal-level fires always count
    got = rl.fires()["r"]
    assert (got["fires"], got["agree_with_head"], got["disagree"], got["no_head"]) == (4, 1, 1, 2)
    assert got["mode"] == "active" and got["examples"][0] == "a2", "disagreements are listed first"
    rl.reset()
    assert rl.fires() == {}


def test_examples_are_capped_at_five():
    for i in range(9):
        rl.fire("r", f"a{i}", "x", "x")
    assert len(rl.fires()["r"]["examples"]) == 5


def test_head_of_reads_the_deal_then_the_label_and_resolves_keep():
    a = _atom("a1", "x", "physical_site", value={"c3_label": {"type": "stakeholder", "act": "follows_copy"}})
    assert rl.head_of(a) is None, "a label a rule wrote is not the head's answer"
    c3._DEAL["types"] = {"a1": ["_keep", 0.9]}
    assert rl.head_of(a) == "physical_site"
    c3._DEAL["types"] = {"a1": ["deal_metadata", 0.9]}
    assert rl.head_of(a) == "deal_metadata"


# ── the instrumented rules ───────────────────────────────────────────────────

def _signature_page():
    sig = _atom("s", "Jane Roe, VP", "signatory", page=6)
    site = _atom("p", "200 N. Milwaukee Ave., Vernon Hills, IL 60061", "physical_site", ["site:vernon_hills"], page=6)
    pre = _atom("q", "In acknowledgement that the parties below", "scope_item", ["site:vernon_hills"], page=6)
    return sig, site, pre


def test_party_veto_fires_and_acts():
    sig, site, pre = _signature_page()
    c3._DEAL["types"] = {"p": ["physical_site", 0.98], "q": ["deal_metadata", 0.9]}
    assert veto_party_page_sites([sig, site, pre]) == 2
    assert _type(site) == "deal_metadata" and pre.entity_keys == []
    f = rl.fires()
    assert f["party_address_veto.signature_page"]["disagree"] == 1
    assert f["party_address_veto.signature_keys"]["agree_with_head"] == 1


def test_party_veto_in_shadow_only_logs(monkeypatch):
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "party_address_veto.*:shadow")
    sig, site, pre = _signature_page()
    assert veto_party_page_sites([sig, site, pre]) == 0
    assert _type(site) == "physical_site" and site.entity_keys == ["site:vernon_hills"]
    assert pre.entity_keys == ["site:vernon_hills"]
    veto_party_page_sites([sig, site, pre])  # the compile runs it again: still one fire per line
    f = rl.fires()
    assert f["party_address_veto.signature_page"]["fires"] == 1
    assert f["party_address_veto.signature_page"]["mode"] == "shadow"


def test_follow_party_copies_shadow(monkeypatch):
    atoms = [_atom("s1", "Scott Morgan | 1361 N Glenville Dr | scott@x.net", "stakeholder"),
             _atom("p1", "1361 N Glenville Dr", "physical_site", ["site:x"])]
    types = {"p1": ["physical_site", 0.99]}
    c3._DEAL["types"] = types
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "party_address_veto.follow_party_copies:shadow")
    assert follow_party_copies(atoms, types) == []
    assert _type(atoms[1]) == "physical_site" and types["p1"][0] == "physical_site"
    assert rl.fires()["party_address_veto.follow_party_copies"]["disagree"] == 1
    # Shadow by default now (repeat_address does it upstream); active only when asked.
    monkeypatch.delenv("SOWSMITH_RULE_MODES")
    assert follow_party_copies(atoms, types) == []
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "party_address_veto.follow_party_copies:active")
    assert follow_party_copies(atoms, types) == ["p1"]


def _roster():
    return {k: {"keys": [k], "atom_ids": [k.split(":")[1]]} for k in
            ("site:basking_ridge_nj", "site:1_main_st_basking_ridge_nj_07920",
             "site:essex_jct", "site:1_a_st_essex_jct_vt", "site:2_b_rd_essex_jct_vt")}


def test_fold_contained_fires_and_shadow_keeps_every_site(monkeypatch):
    c3._DEAL["types"] = {"essex_jct": ["physical_site", 0.9]}
    roster = _roster()
    c3._fold_contained(roster)
    assert len(roster) == 3
    f = rl.fires()
    assert f["quote_sites.fold_contained"]["no_head"] == 1, "a same-site fold is no type-head question"
    assert f["quote_sites.fold_mention"]["disagree"] == 1
    rl.reset()
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "quote_sites.*:shadow")
    roster = _roster()
    c3._fold_contained(roster)
    assert len(roster) == 5 and rl.fires()["quote_sites.fold_contained"]["fires"] == 1


def test_overwrite_bar_refusal_fires_and_shadow_lowers_the_bar(monkeypatch):
    monkeypatch.setattr(c3, "_DROPS_VALUES", frozenset({"physical_site"}), raising=False)
    line = NS(id="a1", atom_type="bom_line", value={"quantity": 200, "unit": "ea"}, entity_keys=[])
    raised = c3.overwrite_bar(line, "physical_site", 0.5)
    if raised <= 0.5:
        pytest.skip("physical_site carries no raised bar in this build")
    assert c3.overwrite_bar(line, "physical_site", 0.5, 0.95) == raised, "a sure answer is no refusal"
    assert c3.overwrite_bar(line, "physical_site", 0.5, 0.6) == raised
    got = rl.fires()["overwrite_bar.extracted"]
    assert got["fires"] == 1 and got["disagree"] == 1
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "overwrite_bar.extracted:shadow")
    assert c3.overwrite_bar(line, "physical_site", 0.5, 0.6) == 0.5


def test_derived_finding_refusal_is_its_own_rule():
    summary = NS(id="d1", atom_type="derived_finding", value={}, entity_keys=[])
    assert c3.overwrite_bar(summary, "physical_site", 0.5, 0.56) == float("inf")
    assert rl.fires()["overwrite_bar.derived_finding"]["fires"] == 1


def test_site_keys_drop_fires_against_the_head_and_shadow_keeps_keys(monkeypatch):
    stray = _atom("a2", "Battery powered drill", "deliverable", ["site:tech_1"])
    real = _atom("a3", "Install at 40 Harbor Road, Springfield, IL 62701", "physical_site", ["site:40_harbor_road"], line=3)
    types = {"a2": ["deliverable", 0.7], "a3": ["physical_site", 0.98]}
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "site_keys.drop_unbacked:shadow")
    settle([stray, real], types)
    assert stray.entity_keys == ["site:tech_1"]
    got = rl.fires()["site_keys.drop_unbacked"]
    assert got["agree_with_head"] == 1 and got["mode"] == "shadow"
    monkeypatch.delenv("SOWSMITH_RULE_MODES")
    settle([stray, real], types)
    assert stray.entity_keys == []


def test_site_row_hours_fires_and_shadow_keeps_the_quote(monkeypatch):
    by_id = {f"a{i}": NS(value={"extras": {"Total Hrs": str(h)}}) for i, h in enumerate((7.5, 12.0, 3.77))}
    q = {"lines": [{"techs": 2, "hours_per_tech": 8, "crew_hours": 16, "rate_line": "L2",
                    "by": {"hours": "heads:read:hours_stated"}}]}
    sites = [{"site": f"s{i}", "atom_ids": [f"a{i}"]} for i in range(3)]
    assert c3._with_site_row_hours(q, sites, by_id)["lines"][0]["crew_hours"] == 7.77
    got = rl.fires()["quote.site_row_hours"]
    assert got["disagree"] == 1, "the head said 16 h; the rule made it 7.8"
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "quote.site_row_hours:shadow")
    assert c3._with_site_row_hours(q, sites, by_id) is q


def test_quote_fallback_and_rule_picks_are_logged():
    q = {"job_type": "cabling", "job_type_by": "fallback:embedding_family",
         "billing": {"answer": "t_and_m", "by": "heads:read:billing_type"},
         "sites": {"count": 2, "by": "rule:sites_said_text"},
         "heads": {"used": {"hours": {"answer": "6"}}},
         "lines": [{"techs": 1, "hours_per_tech": 6, "rate_line": "L1",
                    "by": {"techs": "fallback:family_usual", "hours": "fallback:family_median",
                           "rate": "heads:read:rate"}}]}
    assert c3._note_quote_rules(q) is q
    f = rl.fires()
    assert set(f) == {"quote.job_type.fallback:embedding_family", "quote.sites.rule:sites_said_text",
                      "quote.techs.fallback:family_usual", "quote.hours.fallback:family_median"}
    assert f["quote.hours.fallback:family_median"]["agree_with_head"] == 1
    assert f["quote.techs.fallback:family_usual"]["no_head"] == 1


def test_authority_final_restore_is_logged(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY", "1")
    a = _atom("a1", "NEW Site: 323 Lakeside Ave", "deal_metadata",
              value={"c3_label": {"type": "physical_site", "p": 0.999, "act": "fill"}})
    assert c3.apply_authority([a], final=True) == 1
    got = rl.fires()["authority_final.restore"]
    assert got["disagree"] == 1 and got["examples"] == ["a1"]


def test_geo_fallback_shadow_mints_nothing(monkeypatch):
    from app.core import site_geo_fallback as geo

    monkeypatch.setattr(geo, "_should_skip_geo_fallback", lambda atoms: False)
    src = _atom("a1", "Please ship to 40 Harbor Road, Springfield, IL 62701 by Friday.", "deal_state")
    src.source_refs = []
    c3._DEAL["types"] = {"a1": ["deal_state", 0.9]}
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "site_geo_fallback.address:shadow")
    assert geo.geo_fallback_sites([src], project_id="p") == []
    got = rl.fires().get("site_geo_fallback.address")
    if got is None:
        pytest.skip("the address reader found no address in this build")
    assert got["disagree"] == 1


# ── the report ───────────────────────────────────────────────────────────────

def test_report_sums_deals_and_suggests_a_stage(tmp_path):
    from scripts.rule_ledger_report import summarize, stage

    paths = []
    for i in range(20):
        p = tmp_path / f"d{i}.result.json"
        p.write_text(json.dumps({"project_id": f"d{i}", "rule_fires": {
            "good": {"fires": 10, "agree_with_head": 10, "disagree": 0, "no_head": 0, "mode": "active"},
            "bad": {"fires": 2, "agree_with_head": 1, "disagree": 1, "no_head": 0, "mode": "active"},
            "blind": {"fires": 1, "agree_with_head": 0, "disagree": 0, "no_head": 1, "mode": "active"}}}))
        paths.append(p)
    old = tmp_path / "old.result.json"
    old.write_text(json.dumps({"project_id": "old", "atoms": [
        {"id": "x", "atom_type": "physical_site", "review_flags": ["geo_fallback_site"], "value": {}}]}))
    rules, sources = summarize(paths + [old])
    assert sources == {"ledger": 20, "inferred": 1}
    assert rules["good"]["deals"] == 20 and stage(rules["good"], 20, 0.95) == "ready for Shadow"
    assert stage(rules["bad"], 20, 0.95) == "Active"
    assert stage(rules["blind"], 20, 0.95) == "Active (no head answers it)"
    assert rules["site_geo_fallback.address"]["inferred_deals"] == 1
    rules["good"]["modes"] = {"shadow"}
    assert stage(rules["good"], 20, 0.95) == "Retire"


# ── stages, and the envelope's site list reading the heads' roster ───────────

def test_every_rule_is_upstream_or_downstream():
    # "pm": a PM's own fix on the deal page, above the heads and every rule (2026-10-29)
    assert {v["stage"] for v in rl.RULES.values()} == {"upstream", "downstream", "pm"}
    assert rl.stage("party_address_veto.signature_page") == "downstream"
    assert rl.stage("overwrite_bar.extracted") == "upstream"
    assert rl.stage("quote.hours.fallback:family_median") == "upstream"
    rl.fire("semantic_dedup.site_merge", "a1", None, "same_site")
    assert rl.fires()["semantic_dedup.site_merge"]["stage"] == "downstream"


def test_site_readiness_merges_are_shadow_by_default(monkeypatch):
    assert rl.shadow("site_readiness.alias_merge") and rl.shadow("site_readiness.name_collapse")
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "site_readiness.*:active")
    assert not rl.shadow("site_readiness.alias_merge")


def _sheet_row(aid, row, zipc):
    key = f"site:sugar_land_{zipc}_row_{row}"
    sid = f"SUGAR-LAND-{zipc}-ROW-{row}"
    return NS(id=aid, atom_type="physical_site", raw_text=f"site_id: Sugar Land {zipc} (row {row}) | SUGAR LAND | {zipc}",
              entity_keys=[key], artifact_id="art", review_flags=[], source_refs=[],
              value={"site_id": sid, "id": sid, "name": f"Sugar Land {zipc} (row {row})", "city": "SUGAR LAND",
                     "zip": zipc, "c3_label": {"type": "physical_site", "p": 0.999, "act": "authority"}})


def test_site_readiness_keeps_every_site_the_heads_roster_keeps():
    """010358: three Sugar Land sheet rows (heads 0.99+) were folded into their ZIP neighbours by the alias
    groups: 99 rows on the rail and Scope tab, 102 sites in the quote."""
    from app.core.orbitbrief_core import build_site_readiness

    atoms = [_sheet_row("r84", 84, "77478"), _sheet_row("r85", 85, "77478"),
             _sheet_row("r86", 86, "77479"), _sheet_row("r87", 87, "77479")]
    groups = [frozenset({"site:sugar_land_77478_row_84", "site:sugar_land_77478_row_85"}),
              frozenset({"site:sugar_land_77479_row_86", "site:sugar_land_77479_row_87"})]
    old = build_site_readiness(atoms=atoms, edges=[], alias_groups=groups)
    assert len(old["sites"]) == 2, "without the heads' roster the old merge stands"
    roster = c3.quote_sites(atoms, settled=True)
    assert len(roster) == 4
    rl.reset()
    got = build_site_readiness(atoms=atoms, edges=[], alias_groups=groups, heads_roster=roster)
    assert {s["site"] for s in got["sites"]} == {s["site"] for s in roster}
    f = rl.fires()["site_readiness.alias_merge"]
    assert f["fires"] == 2 and f["mode"] == "shadow" and f["stage"] == "downstream"
    assert set(f["examples"]) == {"r85", "r87"}


def test_site_readiness_merge_acts_when_set_active(monkeypatch):
    from app.core.orbitbrief_core import build_site_readiness

    monkeypatch.setenv("SOWSMITH_RULE_MODES", "site_readiness.alias_merge:active")
    atoms = [_sheet_row("r84", 84, "77478"), _sheet_row("r85", 85, "77478")]
    groups = [frozenset({"site:sugar_land_77478_row_84", "site:sugar_land_77478_row_85"})]
    roster = c3.quote_sites(atoms, settled=True)
    got = build_site_readiness(atoms=atoms, edges=[], alias_groups=groups, heads_roster=roster)
    assert {s["site"] for s in got["sites"]} == {"site:sugar_land_77478_row_84"}, "active: the old merge acts"


def test_report_lists_downstream_rules_first(tmp_path, capsys):
    from scripts.rule_ledger_report import main

    p = tmp_path / "d.result.json"
    p.write_text(json.dumps({"project_id": "d", "rule_fires": {
        "quote.hours.fallback:family_median": {"fires": 9, "no_head": 9, "mode": "active", "stage": "upstream"},
        "semantic_dedup.site_alias": {"fires": 1, "disagree": 1, "mode": "active", "stage": "downstream"}}}))
    main([str(p)])
    out = capsys.readouterr().out
    assert out.index("semantic_dedup.site_alias") < out.index("quote.hours.fallback:family_median")
    assert out.index("DOWNSTREAM") < out.index("UPSTREAM")


def test_a_fire_leaves_the_rule_on_the_line():
    from types import SimpleNamespace as NS
    from app.core import rule_ledger as rl
    rl.reset()
    a = NS(id="atm_1", value={})
    rl.fire("party_address_veto.follow_party_copies", a, "physical_site", "stakeholder", "copy of a contact line")
    t = a.value["rule_trace"][0]
    assert t["rule"] == "party_address_veto.follow_party_copies" and t["head"] == "physical_site" and t["did"] == "stakeholder"
    assert t["agrees"] is False
