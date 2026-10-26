"""The main heads' corrections queue: PM corrections -> super head + atom path, drafts marked."""
import json

from app.core import c3_sow_trace
from app.core.pm_feedback import HEAD_REGISTRY
from scripts import main_heads_queue as q

A1, A2, A3, S1 = "atm_00000000000000a1", "atm_00000000000000a2", "atm_00000000000000a3", "atm_00000000000000s1"
DEAL = "11111111-2222-3333-4444-555555555555"


def _env():
    return {
        "compile_id": "cmp1",
        "crm": {"deal_name": "010999 - Widgets"},
        "documents": [{"artifact_id": "f", "filename": "scope.pdf"}],
        "atoms": [
            {"id": A1, "artifact_id": "f", "atom_type": "task", "text": "Mount four widgets in room A.",
             "locator": {"page": 2}, "structured": {"c3_label": {"type": "task", "p": 0.8}}},
            {"id": A2, "artifact_id": "f", "atom_type": "physical_site", "text": "7 Penn Plaza, New York"},
            {"id": A3, "artifact_id": "f", "atom_type": "physical_site", "text": "7 Penn Plaza floor 12"},
        ],
        "suppressed": [{"id": S1, "artifact_id": "f", "atom_type": "scope_item", "text": "Hope all is well!",
                        "stage": "noise", "reason": "small talk", "filename": "scope.pdf", "label_key": "lbl_s1"}],
        "entities": [{"entity_type": "site", "canonical_name": "7 Penn Plaza", "aliases": ["7 penn plaza"],
                      "source_atom_ids": [A2]},
                     {"entity_type": "site", "canonical_name": "7 Penn Plaza floor 12", "source_atom_ids": [A3]}],
        "c3_brief": {"sections": {"scope_of_work": [{"id": A1, "keep_p": 0.9, "supplier": ["us", 0.7]}]}},
    }


def _corr(relation, verdict, exemplars, created_by="pm", **rel):
    return {"id": f"pm_{relation}_x", "relation": relation, "verdict": verdict, "scope": "deal", "scope_key": DEAL,
            "exemplars": exemplars, "relations": {"pm_seen_deals": [DEAL], **rel}, "status": "active",
            "instruction": f"PM thing: old → {verdict}", "created_by": created_by, "updated_at": 5.0}


def _envs():
    return {DEAL: q.Envelope(_env())}


def test_every_main_head_is_a_registry_head():
    for head in q.MAIN_HEADS:
        assert head in HEAD_REGISTRY
    assert q.RELATION_HEAD["atom_type"] == "type"
    assert q.RELATION_HEAD["same_physical_site"] == "site"
    assert "read:address_level" in q.NO_ROUTE


def test_person_or_draft():
    assert q.by_person("developer@purtera-it.com")
    assert q.by_person("pm")
    assert q.by_person("4031a8b8-0839-4a11-b587-0fd4df000000")
    for draft in ("claude-agent (envelope review, reviewed by Griffin)", "deal-kit:010147",
                  "dealkit-training (Claude, from kits)", "Deal Kit publish", "", "judge:qwen"):
        assert not q.by_person(draft), draft


def test_exemplar_and_was():
    assert q.exemplar_text("Mount it [section: Scope > A] [intro: Subject]") == "Mount it"
    assert q.exemplar_text("From: Sam\n[ctx] authority_class=quoted") == "From: Sam"
    assert q.was_of("PM Keep / drop: deal_metadata → drop — because") == "deal_metadata"
    assert q.was_of("Site identity: distinct_site -> same_site -- why") == "distinct_site"


def test_type_correction_resolves_by_atom_id_and_carries_the_path():
    rows = q.queue_rows(_corr("atom_type", "scope_item", ["Mount four widgets in room A. [section: S]"],
                              created_by="developer@purtera-it.com", atom_id=A1), _envs())
    (r,) = rows
    assert r["key"] == "col:label_type" and r["status"] == "person" and r["skip"] is None
    assert r["atoms"] == [A1] and r["resolved"] == "atom_id" and r["deal"] == "010999"
    p = r["path"][0]
    assert p["file"] == "scope.pdf" and p["where"] == "p. 2" and p["admitted"] is True
    assert r["said"] == "task"            # what the head said, before the PM's fix


def test_reparsed_atom_found_by_text_and_suppressed_line_by_label_key():
    (r,) = q.queue_rows(_corr("atom_type", "task", ["Mount four widgets in room A."], atom_id="atm_gone"), _envs())
    assert r["atoms"] == [A1] and r["resolved"] == "text"
    (s,) = q.queue_rows(_corr("admission", "drop", ["Hope all is well!"], created_by="claude-agent (x)",
                              label_key="lbl_s1"), _envs())
    assert s["atoms"] == [S1] and s["resolved"] == "suppressed" and s["status"] == "draft"
    assert s["path"][0]["admitted"] is False and s["said"] == "drop"


def test_supplier_and_site_pair():
    (r,) = q.queue_rows(_corr("bom_owner", "customer_furnished", ["Mount four widgets in room A."], atom_id=A1), _envs())
    assert r["key"] == "col:supplier" and r["said"] == "us"
    (p,) = q.queue_rows(_corr("same_physical_site", "same_site", ["7 penn plaza || 7 Penn Plaza floor 12"]), _envs())
    assert p["key"] == "jdg:site" and p["atoms"] == [A2, A3] and p["said"] == "distinct_site"


def test_skips_are_explicit():
    probe = q.queue_rows(_corr("physical_site", "job_site", ["ZZZ routing probe, synthetic, not a real atom"],
                               atom_id="routing_probe"), _envs())
    assert probe[0]["skip"] == "probe"
    off = _corr("atom_type", "task", ["x"])
    off["status"] = "disabled"
    assert q.queue_rows(off, _envs())[0]["skip"] == "disabled"
    assert q.queue_rows(_corr("atom_type", "task", ["Nowhere in the deal"]), _envs())[0]["skip"] == "unresolved"
    assert q.queue_rows(_corr("atom_type", "task", ["x"]), {})[0]["skip"] == "no_envelope"
    assert q.queue_rows(_corr("gap_valid", "valid", ["x"]), _envs()) == []   # not a main head


def test_conflict_supports_also_trains_the_relation():
    env = _env()
    env["atoms"].append({"id": "atm_00000000000000b1", "artifact_id": "f", "text": "Room A gets four widgets."})
    (r,) = q.queue_rows(_corr("edge_relation", "supports",
                              ["Mount four widgets in room A. || Room A gets four widgets."]), {DEAL: q.Envelope(env)})
    assert r["key"] == "jdg:conflict" and r["also"] == ["rel:supports"] and len(r["atoms"]) == 2


def test_trace_atoms_reaches_atoms_no_sow_line_cites():
    t = c3_sow_trace.trace_atoms(_env(), [A1])
    assert t[A1]["section"] == "scope_of_work" and t[A1]["supplier"] == ["us", 0.7]
    assert c3_sow_trace.build_atom_trace(_env()) == {}       # no c3_sow: the SOW trace is unchanged


def test_main_writes_queue_and_fails_loudly_without_inputs(tmp_path):
    envs = tmp_path / "envelopes"
    envs.mkdir()
    (envs / f"{DEAL}.json").write_text(json.dumps(_env()))
    corr = tmp_path / "corrections.jsonl"
    corr.write_text(json.dumps(_corr("atom_type", "scope_item", ["Mount four widgets in room A."], atom_id=A1)) + "\n")
    assert q.main(["--corrections", str(corr), "--envelopes", str(envs), "--out", str(tmp_path / "q")]) == 0
    rows = [json.loads(x) for x in (tmp_path / "q" / "queue.jsonl").read_text().splitlines()]
    assert rows[0]["atoms"] == [A1]
    summary = json.loads((tmp_path / "q" / "summary.json").read_text())
    assert summary["trainable"] == 1 and summary["by_head"]["col:label_type"] == {"person": 1}
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert q.main(["--corrections", str(empty), "--envelopes", str(envs), "--out", str(tmp_path / "q2")]) == 3
    assert q.main(["--corrections", str(corr), "--envelopes", str(tmp_path / "none"), "--out", str(tmp_path / "q3")]) == 3
