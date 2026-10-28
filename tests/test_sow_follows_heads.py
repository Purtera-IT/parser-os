"""SowSmith follows the promoted heads (AUDIT_sowsmith_heads.md, 2026-10-27).

Label hygiene: only a PM's own correction trains. Keep, supplier, owner,
family and sites: the promoted head decides; a rule stands in only where it
abstained. Template lines say so. A failed LLM call is visible.
"""
import json

import pytest

from app.core import c3_sow as m
from ml.c3 import labels_from_sow as lfs
from scripts import main_heads_queue as q
from scripts import sow_fix_triage as tri

A1, A2, A3, A4 = "atm_00000000000000a1", "atm_00000000000000a2", "atm_00000000000000a3", "atm_00000000000000a4"
DEAL = "11111111-2222-3333-4444-555555555555"


# ── 1. Label hygiene ─────────────────────────────────────────────────────────

def _lenv(tmp_path, decisions=()):
    env = {"crm": {"deal_name": "010999 - test"},
           "documents": [{"artifact_id": "d1", "artifact_type": "eml"}],
           "atoms": [{"id": a, "text": f"line {a}", "artifact_id": "d1"} for a in (A1, A2, A3, A4)],
           "c3_sow": {"rule_decisions": list(decisions)}}
    p = tmp_path / "final_010999_env.json"
    p.write_text(json.dumps(env))
    return str(p)


def _labels(tmp_path):
    f = tmp_path / "out" / "010999.json"
    return {lb["atom_id"]: lb for lb in json.loads(f.read_text())["labels"]} if f.exists() else {}


def test_unreviewed_rule_decisions_never_become_labels(tmp_path):
    env = _lenv(tmp_path, [{"rule": "grounding", "head": "sow_keep", "atom_ids": [A1], "value": "drop"},
                           {"rule": "one_supplier", "head": "sow_supplier", "atom_ids": [A2], "value": "customer"},
                           {"rule": "duty_party", "head": "sow_party", "atom_ids": [A3], "value": "customer"}])
    counts = lfs.build([env], [], str(tmp_path / "out"))
    assert _labels(tmp_path) == {}
    assert counts["rule_decisions_report_only"] == 3 and "silver" not in counts


def test_writer_outcome_and_agent_drafts_never_train(tmp_path):
    env = _lenv(tmp_path)
    reviews = [{"deal": "010999", "atom_id": A1, "head": "sow_section", "value": "assumptions", "verdict": "outcome"},
               {"deal": "010999", "atom_id": A2, "head": "sow_keep", "value": "keep", "verdict": "agent_draft"}]
    counts = lfs.build([env], reviews, str(tmp_path / "out"))
    assert _labels(tmp_path) == {} and counts["skipped_not_pm"] == 2


def test_a_pm_delete_is_not_a_co_action_reject_unless_the_pm_says_not_a_job_fact(tmp_path):
    env = _lenv(tmp_path)
    reviews = [{"deal": "010999", "atom_id": A1, "head": "sow_keep", "value": "drop", "verdict": "pm_edit"},
               {"deal": "010999", "atom_id": A2, "head": "sow_keep", "value": "drop", "verdict": "pm_edit",
                "reason": "head", "note": "duplicate of the line above"},
               {"deal": "010999", "atom_id": A3, "head": "sow_keep", "value": "drop", "verdict": "pm_edit",
                "reason": "head", "note": "this is small talk, not a job fact"},
               {"deal": "010999", "atom_id": A4, "head": "sow_section", "value": "assumptions", "verdict": "pm_edit"}]
    counts = lfs.build([env], reviews, str(tmp_path / "out"))
    got = _labels(tmp_path)
    assert A1 not in got and A2 not in got and counts["skipped_keep_drop"] == 2
    assert got[A3]["reads_set"] == {"co_action": "reject"} and got[A3]["weight_tier"] == "ordinary"
    assert got[A4]["reads_set"] == {"sow_section": "assumptions"}


def test_a_pm_rejecting_a_rules_keep_is_a_drop_not_a_keep(tmp_path):
    env = _lenv(tmp_path)
    reviews = [{"deal": "010999", "atom_id": A1, "head": "sow_keep", "value": "keep", "verdict": "rejected"},
               {"deal": "010999", "atom_id": A2, "head": "sow_keep", "value": "drop", "verdict": "rejected"}]
    lfs.build([env], reviews, str(tmp_path / "out"))
    got = _labels(tmp_path)
    assert A1 not in got  # a drop with no not-a-fact reason
    assert got[A2]["reads_set"] == {"co_action": "keep"}


def test_sow_heads_are_main_heads_and_a_bare_delete_is_skipped():
    for h, key in (("sow_section", "read:sow_section"), ("sow_party", "read:task_owner_role"),
                   ("sow_supplier", "col:supplier"), ("sow_keep", "read:co_action")):
        assert q.MAIN_HEADS[h][0] == key
    env = {"crm": {"deal_name": "010999 - W"}, "documents": [],
           "atoms": [{"id": A1, "artifact_id": "f", "text": "Customer ships the TVs."}], "c3_brief": {"sections": {}}}
    envs = {DEAL: q.Envelope(env)}

    def corr(rel, verdict, why=""):
        return {"id": f"c_{rel}_{verdict}", "relation": rel, "verdict": verdict, "scope": "deal", "scope_key": DEAL,
                "exemplars": ["Customer ships the TVs."], "status": "active", "created_by": "pm@purtera-it.com",
                "relations": {"pm_seen_deals": [DEAL], "atom_id": A1},
                "instruction": f"PM thing: keep → {verdict}" + (f" — {why}" if why else "")}
    bare = q.queue_rows(corr("sow_keep", "drop"), envs)[0]
    assert bare["skip"] == "delete_not_a_fact_call" and bare["pm_verdict"] == "drop"
    fact = q.queue_rows(corr("sow_keep", "drop", "not a job fact, it is marketing"), envs)[0]
    assert fact["skip"] is None and fact["verdict"] == "reject" and fact["key"] == "read:co_action"
    sup = q.queue_rows(corr("sow_supplier", "customer", "they ship it"), envs)[0]
    assert sup["skip"] is None and sup["key"] == "col:supplier" and sup["reason"] == "they ship it"
    assert sup["line_id"] == A1 and sup["by_person"]


def _tenv():
    return {"c3_sow": {"job_family": "av_conference_room", "rule_decisions": [], "writer_draft": {},
                       "provider_materials": [{"text": "Wall mounts", "cites": [A1], "kind": "fact"}],
                       "scope": [{"text": "Customer will rack the switch", "cites": [A2], "kind": "fact"}],
                       "atom_trace": {A1: {"text": "Wall mounts", "supplier": ["us", 0.62]},
                                      A2: {"text": "They rack the switch", "owner": ["customer", 0.9]}}},
            "atoms": [{"id": A1, "text": "Wall mounts"}, {"id": A2, "text": "They rack the switch"}]}


def test_studio_supply_move_is_a_supplier_correction_with_the_head_answer_and_reason():
    note = {"note": "customer bought the mounts", "reason": "head", "by": "pm@purtera-it.com"}
    r = tri.classify({"id": "e1", "kind": "move", "fromList": "provider_materials", "toList": "customer_materials",
                      "text": "Wall mounts", "cites": [A1]}, note, _tenv())
    assert r["bucket"] == "head" and r["label"]["head"] == "sow_supplier"
    assert r["label"]["old_value"] == "us" and r["label"]["new_value"] == "customer"
    assert "customer bought the mounts" in r["label"]["why"]
    r.update(deal="010999", deal_id=DEAL, note=note)
    body = tri.correction_body(r)
    assert body["candidates"] == ["us", "customer", "partner"] and "edit e1" in body["rationale"]
    assert tri.review_verdict(r, r["label"]) == "pm_edit"


def test_studio_party_move_carries_party_and_section_and_defends_a_right_head():
    r = tri.classify({"id": "e2", "kind": "move", "fromList": "scope", "toList": "customer_responsibilities",
                      "text": "Customer will rack the switch", "cites": [A2]}, None, _tenv())
    # The owner head already said customer: the writer misplaced it, no head label.
    assert r["bucket"] == "writer" and r["label"] is None and not tri.labels_of(r)
    env = _tenv()
    env["c3_sow"]["atom_trace"][A2]["owner"] = ["technician", 0.8]
    r = tri.classify({"id": "e3", "kind": "move", "fromList": "scope", "toList": "customer_responsibilities",
                      "text": "Customer will rack the switch", "cites": [A2]}, None, env)
    assert r["bucket"] == "head" and r["label"]["head"] == "sow_party" and r["label"]["new_value"] == "customer"
    assert [x["head"] for x in tri.labels_of(r)] == ["sow_party", "sow_section"]


def test_agent_inferred_delete_is_a_draft():
    assert tri.review_verdict({"kind": "delete", "note": None}, {}) == "agent_draft"
    assert tri.review_verdict({"kind": "add"}, {}) == "agent_draft"


# ── 2. Keep: facts() never resurrects a head-dropped line ────────────────────

def _fenv(dropped=()):
    return {"documents": [{"artifact_id": "d", "filename": "request.pdf"}],
            "atoms": [{"id": A1, "artifact_id": "d", "text": "Install the new access points in every classroom.",
                       "atom_type": "task", "structured": {"c3_label": {"type": "task", "p": 0.9}}},
                      {"id": A2, "artifact_id": "d", "text": "The old cabling should be removed from the ceiling.",
                       "atom_type": "assumption", "structured": {"c3_label": {"type": "_keep", "p": 0.2}}},
                      {"id": A3, "artifact_id": "d", "text": "Can you also add a router with cellular backup for us?",
                       "structured": {"c3_label": {"type": "answered_question", "p": 0.9}}}],
            "c3_brief": {"sections": {}, "left_out": {"count": 3, "head_dropped": list(dropped)},
                         "label_next": []}}


def test_typed_directive_and_answered_lines_stay_out_when_the_keep_head_dropped_them():
    ids = {f["id"] for f in m.facts(_fenv())}
    assert {A1, A2, A3} <= ids  # no head read them: the bridges stand in
    assert {f["id"] for f in m.facts(_fenv([A1, A2, A3]))} == set()


def test_coin_flip_rejects_stay_out_and_the_switch_restores_them(monkeypatch):
    env = _fenv()
    env["c3_brief"]["label_next"] = [{"id": A1, "head": "read:co_action", "base_says": "reject", "p": 0.52,
                                      "text": "Install the new access points in every classroom."}]
    assert A1 not in {f["id"] for f in m.facts(env)}
    monkeypatch.setenv("SOWSMITH_C3_SOW_HEADS_DECIDE", "off")
    assert A1 in {f["id"] for f in m.facts(env)}


# ── 3/4. Supplier and owner heads place lines; rules cannot move them ────────

def test_supplier_head_decides_a_non_bom_line_and_no_rule_moves_it():
    fs = [{"id": A1, "text": "Display mounts", "type": "material", "section": "scope_of_work", "supplier": ["customer", 0.9]},
          {"id": A2, "text": "Patch cables", "type": "material", "section": "scope_of_work", "supplier": ["us", 0.5]}]
    content = {"provider_materials": [{"text": "Display mounts", "cites": [A1], "kind": "fact"},
                                      {"text": "Patch cables", "cites": [A2], "kind": "fact"}],
               "customer_materials": [], "provider_tools": []}
    assert m._follow_supplier(content, fs) == 1
    item = content["customer_materials"][0]
    assert item["by"] == "col:supplier" and item["supplier"] == ["customer", 0.9]
    assert "by" not in content["provider_materials"][0]  # below the bar: the rules still decide it

    def mover(c):  # a wording rule that would hand it back to Provider
        c["provider_materials"].append(c["customer_materials"].pop())
    m._traced(content, "some_rule", mover, content)
    assert content["customer_materials"] == [item]
    lock = [d for d in content["rule_decisions"] if d.get("ledger") == "head_lock"]
    assert lock and lock[0]["rule"] == "some_rule" and lock[0]["head"] == "sow_supplier"


def test_party_check_never_judges_head_placed_items_and_a_failed_call_moves_nothing(monkeypatch):
    monkeypatch.setattr(m, "_complete", lambda *a, **k: "")
    content = {"provider_materials": [{"text": "Mounts", "cites": [A1], "kind": "fact", "by": "col:supplier"},
                                      {"text": "Printer", "cites": [A2], "kind": "fact"}],
               "customer_materials": []}
    assert m._party_check(content, {A1: "mounts", A2: "printer"}) == 0
    assert len(content["provider_materials"]) == 2 and not content.get("equipment_in_scope")
    assert content["llm_failures"][0]["call"] == "party_check"


def test_owner_head_places_duties_and_locks_them():
    fs = [{"id": A1, "text": "x", "owner": ["customer", 0.9]}, {"id": A2, "text": "y", "owner": ["technician", 0.6]}]
    content = {"scope": [{"text": "Customer racks the switch", "cites": [A1], "kind": "fact"},
                         {"text": "Label the ports", "cites": [A2], "kind": "fact"}],
               "customer_responsibilities": []}
    assert m._follow_owner(content, fs) == 1
    duty = content["customer_responsibilities"][0]
    assert duty["owner_by"] == "read:task_owner_role" and duty["owner"] == ["customer", 0.9]
    m._traced(content, "first_person", lambda c: c["scope"].append(c["customer_responsibilities"].pop()), content)
    assert content["customer_responsibilities"] == [duty]
    m._traced(content, "assumption_voice", lambda c: c.setdefault("assumptions", []).append(c["customer_responsibilities"].pop()), content)
    assert content["assumptions"] == [duty]  # a section move is not a party call: the rule stands in for sow_section


# ── 5. Family ─────────────────────────────────────────────────────────────────

def test_family_follows_the_service_head_and_marks_the_router_fallback():
    fs = [{"text": "Install 4 cameras and mount them", "section": "scope_of_work"}]
    sure = {"service": {"value": "low_voltage_cabling", "p": 0.81, "by": "base:deal:primary_service"}}
    assert m._family(sure, "cameras_security", fs) == ("low_voltage_cabling", "base:deal:primary_service")
    unsure = {"service": {"value": "low_voltage_cabling", "p": 0.4, "by": "base:deal:primary_service", "unsure": True}}
    assert m._family(unsure, "cameras_security", fs)[1] == "rule:route_family"
    rule = {"service": {"value": "low_voltage_cabling", "by": "rule:domain_pack"}}
    assert m._family(rule, "cameras_security", fs) == ("cameras_security", "rule:route_family")


# ── 6. Sites ─────────────────────────────────────────────────────────────────

def test_the_roster_is_the_site_table_and_the_writer_only_flags(monkeypatch):
    monkeypatch.setattr(m, "_best_of", lambda e, pm: {"overview": [], "locations": [
        {"text": "HQ (100 Main Street, Austin, TX 78701)", "cites": [A1], "kind": "fact"},
        {"text": "Lab (200 Oak Avenue, Dallas, TX 75201)", "cites": [A1], "kind": "fact"}]})
    env = {"atoms": [{"id": A1, "text": "HQ at 100 Main Street, Austin"}],
           "site_readiness": {"sites": [{"display_name": "HQ", "address": "100 Main Street", "city": "Austin",
                                         "state": "TX", "postal_code": "78701"}]}}
    c = m.for_envelope(env)
    assert [s["name"] for s in c["sites"]] == ["HQ"] and c["site_source"] == "heads:site_readiness"
    flags = [f["flag"] for f in c["site_flags"]]
    assert "writer_location_not_in_roster" in flags


def test_an_empty_roster_falls_back_marked(monkeypatch):
    monkeypatch.setattr(m, "_best_of", lambda e, pm: {"overview": []})
    env = {"atoms": [{"id": A1, "text": "The customer's site address will be 530 Fifth Avenue, New York, NY 10036"}],
           "site_readiness": {"sites": []}}
    c = m.for_envelope(env)
    assert c["site_source"] == "rule:customer_site_address" and c["sites"]


# ── 7. Template ──────────────────────────────────────────────────────────────

def test_template_lines_are_tagged_and_deal_lines_are_not():
    c = {"scope": [{"text": "Mount each display", "cites": [], "kind": "method", "standard": True},
                   {"text": "Mount two (2) 85-inch TVs", "cites": [A1], "kind": "fact"},
                   {"text": "Clean up the work area", "cites": [], "kind": "method"},
                   {"text": "Displays", "cites": [], "kind": "heading"}]}
    m._tag_origin(c)
    assert [x.get("origin") for x in c["scope"]] == ["TEMPLATE", "DEAL", "TEMPLATE", None]
    assert c["scope"][0]["template_source"] == "standards" and c["scope"][2]["template_source"] == "writer"
    assert c["origin_counts"] == {"DEAL": 1, "TEMPLATE": 2}


def test_a_standard_never_replaces_a_cited_deal_line(monkeypatch):
    monkeypatch.setattr(m, "purtera_standards", lambda fam: {"assumptions": ["All cable runs are under 300 feet."]})
    sec = next(k for k, v in m._STD_LIST.items() if v == "assumptions")
    monkeypatch.setattr(m, "purtera_standards", lambda fam: {sec: ["All cable runs are under 300 feet."]})
    deal = {"text": "All cable runs are under 300 feet.", "cites": [A1], "kind": "fact"}
    content = {"assumptions": [deal]}
    m._apply_standards(content, "low_voltage_cabling", ["All cable runs are under 300 feet."], [], [])
    assert content["assumptions"] == [deal]
    assert any("deal line stays" in s["reason"] for s in content["standards_suppressed"])


# ── 8. Silent failures ───────────────────────────────────────────────────────

def test_draft_status_names_the_failed_call():
    assert m._draft_status({"llm_failures": []}) == "ok"
    assert m._draft_status({"llm_failures": [{"call": "cover", "tries": 4}]}) == "degraded"
    c = {"llm_failures": [{"call": "scope", "tries": 4}]}
    assert m._draft_status(c) == "failed" and "scope" in c["failed_because"]


def test_a_failed_verify_batch_keeps_its_lines_and_batches_run_in_one_round(monkeypatch):
    import threading
    import time

    calls, live, peak = [], [0], [0]
    lock = threading.Lock()

    def fake(prompt, *a, **k):
        with lock:
            calls.append(1)
            first = len(calls) == 1
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return "" if first else '{"supported": []}'
    monkeypatch.setattr(m, "_complete", fake)
    items = [{"text": f"Zebra quartz {i}", "cites": [A1], "kind": "fact"} for i in range(85)]
    fails: list = []
    out = m._verified(items, {A1: "unrelated words"}, fails)
    assert len(calls) == 3 and peak[0] == 3  # same call count, one round
    assert len(fails) == 1 and fails[0]["call"] == "verify"
    assert len(out) in (40, 5)  # the failed batch keeps its lines; the judged ones go
