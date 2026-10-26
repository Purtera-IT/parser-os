import json

from scripts import sow_fix_triage as t

A1, A2, A3 = "atm_00000000000000a1", "atm_00000000000000a2", "atm_00000000000000a3"


def _env():
    return {
        "compile_id": "cmp1",
        "atoms": [
            {"id": A1, "artifact_id": "f", "atom_type": "task", "text": "We need four widgets mounted per site in room A."},
            {"id": A2, "artifact_id": "f", "atom_type": "dependency", "text": "Customer ships the Acme X200 widgets to each site."},
            {"id": A3, "artifact_id": "f", "atom_type": "open_question", "text": "Can we use the Acme X200 bracket kit in room B?"},
        ],
        "c3_brief": {"sections": {"scope_of_work": [{"id": A1, "keep_p": 0.97}],
                                  "customer_responsibilities": [{"id": A2, "keep_p": 0.9}]}},
        "c3_sow": {
            "job_family": "widget_install",
            "crm": {"account_name": "SHI"},
            "end_customer": "Catalyst Brands",
            "scope": [{"text": "Mount four widgets per site in room A.", "cites": [A1], "kind": "fact"},
                      {"text": "Meet the onsite contact.", "cites": [], "kind": "method", "standard": True},
                      {"text": "Bring a ladder.", "cites": [], "kind": "method"}],
            "customer_responsibilities": [{"text": "Ship the Acme X200 widgets to each site.", "cites": [A2], "kind": "fact"},
                                          {"text": "Install the widgets in room A.", "cites": [A1], "kind": "fact"}],
            "assumptions": [], "out_of_scope": [],
            "writer_draft": {"scope": [{"text": "Mount four widgets per site in room A.", "cites": [f"['{A1}']"]}]},
            "rule_decisions": [{"rule": "aside_not_task", "head": "sow_keep", "atom_ids": [A3], "value": "drop", "was": "scope_of_work", "text": "x"}],
        },
    }


def edit(kind, text, frm=None, to=None, cites=(), **kw):
    return {"id": f"se_{kind}_{len(text)}", "kind": kind, "text": text, "fromList": frm, "toList": to, "cites": list(cites), **kw}


def test_specifics_ignore_sentence_initial_capitals():
    assert t.specifics("Ensure packing materials are sufficient.") == set()
    assert t.specifics("Provide a Cisco Meraki MS120 PoE switch.") == {"cisco", "meraki", "ms120", "poe"}
    assert t.specifics("Reuse existing CAT6 cabling for 12 drops") == {"cat6", "12"}


def test_rubric_first_matching_test():
    assert t.rubric_section("We will not provide disposal.") == "out_of_scope"
    assert t.rubric_section("Work window is 8:00 am to 5 pm.") == "timeline"
    assert t.rubric_section("Install the bracket.") == "scope_of_work"
    assert t.rubric_section("Existing cabling is reused.") == "assumptions"


def test_pm_reason_wins_and_head_reason_carries_the_label():
    env = _env()
    e = edit("move", "Mount four widgets per site in room A.", "scope", "assumptions", [A1])
    r = t.classify(e, {"reason": "head", "note": "it's a condition"}, env)
    assert r["bucket"] == "head" and r["confidence"] == "high"
    assert r["label"] == {**r["label"], "head": "sow_section", "target_id": A1, "old_value": "scope_of_work", "new_value": "assumptions"}
    assert t.classify(e, {"reason": "writer"}, env)["bucket"] == "writer"
    p = t.classify(edit("delete", "Meet the onsite contact.", "scope"), {"reason": "template", "note": "n"}, env)
    assert p["bucket"] == "template" and p["proposal"]["action"] == "remove" and p["proposal"]["job_family"] == "widget_install"


def test_defends_the_heads_on_a_move_to_where_they_already_placed_it():
    env = _env()
    r = t.classify(edit("move", "Install the widgets in room A.", "customer_responsibilities", "scope", [A1]), None, env)
    assert r["bucket"] == "writer" and "already placed" in r["why"]


def test_delete_of_a_line_the_writer_placed_against_the_heads_is_the_writers():
    env = _env()
    r = t.classify(edit("delete", "Install the widgets in room A.", "customer_responsibilities", None, [A1]), None, env)
    assert r["bucket"] == "writer" and "placed its source" in r["why"]


def test_uncited_lines_split_template_and_writer():
    env = _env()
    assert t.classify(edit("delete", "Meet the onsite contact.", "scope"), None, env)["bucket"] == "template"
    assert t.classify(edit("delete", "Bring a ladder.", "scope"), None, env)["bucket"] == "writer"


def test_adds():
    env = _env()
    # A fact an atom states that the heads left out → keep label …
    r = t.classify(edit("add", "Customer ships Acme X200 widgets to every site.", None, "customer_responsibilities"), None, env)
    assert r["bucket"] == "head" and r["label"]["new_value"] == "keep" and r["label"]["target_id"] == A2
    # … unless a writer rule dropped that atom: the rule is the suspect.
    r = t.classify(edit("add", "Use the Acme X200 bracket kit in room B.", None, "scope"), None, env)
    assert r["bucket"] == "writer" and "aside_not_task" in r["why"]
    # Generic words → template; the paper's parties only → that reseller's paper template; a new fact → ask.
    assert t.classify(edit("add", "Leave the work area clean.", None, "scope"), None, env)["bucket"] == "template"
    r = t.classify(edit("add", "SHI will provide a technical point of contact.", None, "customer_responsibilities"), None, env)
    assert r["bucket"] == "template" and r["proposal"]["job_family"] == "paper:shi"
    assert t.classify(edit("add", "Install two Zeta Q9 hubs.", None, "scope"), None, env)["bucket"] == "ask"


def test_misread_atom_routes_related_fixes_to_parser():
    env = _env()
    r = t.classify(edit("delete", "Ship the Acme X200 widgets to each site.", "customer_responsibilities", None, [A2]), None, env, {A2})
    assert r["bucket"] == "parser"


def test_end_to_end_dry_run(tmp_path):
    envs, edits, out = tmp_path / "envs", tmp_path / "edits", tmp_path / "out"
    envs.mkdir(); edits.mkdir()
    (envs / "final_010999_env.json").write_text(json.dumps(_env()))
    move = edit("move", "Mount four widgets per site in room A.", "scope", "assumptions", [A1])
    (edits / "010999.json").write_text(json.dumps({
        "deal": "010999", "deal_id": "uuid-1", "compile_id": "cmp1",
        "sowEditLog": [move, edit("add", "Install two Zeta Q9 hubs.", None, "scope"), edit("add", "Mount the Zeta Q9 rail.", None, "scope")],
        "sowFixNotes": [{"id": "n1", "reason": "head", "kind": "move", "editId": move["id"], "text": move["text"], "by": "pm@x"}],
    }))
    assert t.main(["--edits", str(edits), "--envs", str(envs), "--out", str(out)]) == 0
    posted = [json.loads(l) for l in open(out / "would_post.jsonl")]
    assert posted == [{**posted[0], "head": "sow_section", "deal_id": "uuid-1", "target_id": A1, "new_value": "assumptions", "pm": "pm@x"}]
    qs = [json.loads(l) for l in open(out / "questions.jsonl")]
    assert len(qs) == 1 and len(qs[0]["edit_ids"]) == 2  # two lines naming "Zeta Q9": one question
    assert (out / "report.md").read_text().startswith("# SOW fix triage")


def test_post_refuses_a_non_dev_service():
    import pytest

    with pytest.raises(SystemExit):
        t.post([{"deal_id": "d", "target_id": "a"}], "https://parser-os-service-prod.example", "tok")
