"""The SOW writer's fact list: the keep head (read:co_action) decides which
lines are facts; a type or a regex never vetoes or adds one on its own
(sitefix/audit_out/overrides.md, override points 3 and 9). Made-up atoms;
the 010358 survey wording is the audit's own example."""
from __future__ import annotations

from app.core import c3_sow as m

SURVEY = ("Conducts the on-site Ekahau wireless RF survey (during business hours for the 6 "
          "no-after-hours-access sites) and records AP placement for each floor.")
COMMIT = "Commitment | Full-time field, alternating day-phase and night-phase weeks for 20 weeks."


def _atom(i, text, typ="scope_item", ln=10, label=None, **st):
    s = {"c3_label": {"type": label}} if label else {}
    s.update(st)
    return {"id": i, "artifact_id": "mail", "text": text, "atom_type": typ, "structured": s,
            "locator": {"line_start": ln, "message_index": 2}}


def _survey_env(by="purtera:read:co_action"):
    atoms = [_atom("sv", SURVEY, "small_talk", 20, admission_reject=True),
             _atom("cm", COMMIT, "small_talk", 21),
             _atom("bye", "Thanks so much, talk soon!", "small_talk", 22),
             _atom("sig", "Need help with: ISP Upgrade", "small_talk", 23)]
    item = lambda a, sec: {"id": a["id"], "text": a["text"], "by": by, "keep_p": 0.97}  # noqa: E731
    return {"atoms": atoms, "c3_brief": {
        "sections": {"scope_of_work": [item(atoms[0], "scope_of_work")], "timeline": [item(atoms[1], "timeline")]},
        "also_mentioned": [], "kept_unsure": [], "label_next": []}}


def test_rule_typed_chatter_the_keep_head_kept_is_a_fact():
    env = _survey_env()
    rej = m._rejected_atoms(env)
    assert "sv" not in rej and "cm" not in rej  # 010358: kept 0.96 / 0.98, typed small_talk by the signature regex
    assert {"bye", "sig"} <= rej                   # the keep head kept neither: the rule still guards them
    got = {f["id"] for f in m.facts(env)}
    assert {"sv", "cm"} <= got


def test_a_line_kept_by_a_rule_not_the_head_is_still_guarded():
    env = _survey_env(by="rule:type")
    assert {"sv", "cm"} <= m._rejected_atoms(env)


def test_the_type_heads_small_talk_is_no_veto_on_an_also_mentioned_line():
    a = _atom("x", "But the biggest thing is the conference room displays for all.", label="small_talk")
    env = {"atoms": [a], "c3_brief": {"sections": {}, "also_mentioned": [{"id": "x", "text": a["text"], "keep_p": 0.7}]}}
    assert "x" not in m._rejected_atoms(env)


def _left_out_env(heads_used=("read:co_action",)):
    atoms = [_atom("dis", "If you received this email in error, please contact the sender and delete it from all computers.",
                   "scope_item", 30, c3_label={"type": "_keep", "p": 0.6}),
             _atom("dir", "The cable should be moved behind the wall so it is not visible.", "assumption", 31, c3_label={"type": "_keep", "p": 0.6}),
             _atom("prog", "Each location may opt in, and quantities TBD are billed on actuals once installed.",
                   "scope_item", 32, label="_keep", source="hubspot_note"),
             _atom("kept", "The two TVs should be mounted at 60 inches on the north wall.", "scope_item", 33, label="_keep")]
    return {"atoms": atoms, "documents": [{"artifact_id": "mail", "filename": "mail.eml"}],
            "c3_brief": {"sections": {}, "also_mentioned": [], "label_next": [],
                         "kept_unsure": [{"id": "kept", "text": atoms[3]["text"], "p": 0.4}],
                         "decided_by": {"heads_used": list(heads_used)}}}


def test_the_sow_adds_no_line_the_keep_head_left_out_by_parser_type_or_regex():
    got = {f["id"]: f["section"] for f in m.facts(_left_out_env())}
    assert "dis" not in got   # parser scope_item under an abstaining type head: no longer a fact
    assert "dir" not in got   # _DIRECTIVE matched, the keep head left it out
    assert "prog" not in got  # _PROGRAM matched, the keep head left it out
    assert got.get("kept") in ("typed_only", "unsure")  # the head kept it (below the bar): the writer sees it


def test_without_a_keep_reading_the_regex_routes_still_feed_the_writer():
    got = {f["id"] for f in m.facts(_left_out_env(heads_used=()))}
    assert "dir" in got
    assert "dis" not in got   # the parser-type route is gone either way


def test_the_program_model_fixup_writes_only_from_lines_the_keep_head_kept():
    notes = [_atom("o", "Each location will opt in to the install program over the summer.", "scope_item", 40, source="hubspot_note"),
             _atom("t", "Clock quantities TBD until the survey closes for each district.", "scope_item", 41, source="hubspot_note")]
    env = {"atoms": notes, "documents": [], "c3_brief": {"sections": {}, "also_mentioned": [], "kept_unsure": [],
                                                         "decided_by": {"heads_used": ["read:co_action"]}}}
    c = {"assumptions": [], "customer_responsibilities": []}
    m._program_model(c, env)
    assert not c["assumptions"]
    env["c3_brief"]["also_mentioned"] = [{"id": "o", "text": notes[0]["text"]}, {"id": "t", "text": notes[1]["text"]}]
    m._program_model(c, env)
    assert c["assumptions"]


# ---- step 0 (regex_retirement S16, S23, S27, S68): the re-add paths add only
# lines the keep head kept, whenever there is a keep reading.

def _readd_env(heads_used=("read:co_action",), kept=()):
    atoms = [_atom("typ", "Remove the old wall-mounted clocks and patch the anchor holes.", "assumption", 50,
                   c3_label={"type": "scope_item", "p": 0.9}),
             _atom("ans", "Can you also add a cellular backup router at the main office?", "open_question", 51,
                   c3_label={"type": "answered_question", "p": 0.9})]
    return {"atoms": atoms, "documents": [{"artifact_id": "mail", "filename": "mail.eml"}],
            "c3_brief": {"sections": {}, "label_next": [], "kept_unsure": [],
                         "also_mentioned": [{"id": i, "text": a["text"]} for a in atoms for i in kept if a["id"] == i],
                         "decided_by": {"heads_used": list(heads_used)}}}


def test_typed_only_adds_no_line_the_keep_head_left_out():
    assert "typ" not in {f["id"] for f in m.facts(_readd_env())}
    assert {f["id"]: f["section"] for f in m.facts(_readd_env(heads_used=()))}.get("typ") == "typed_only"


def test_answered_question_readd_needs_the_keep_head():
    assert "ans" not in {f["id"] for f in m.facts(_readd_env())}
    assert {f["id"]: f["section"] for f in m.facts(_readd_env(heads_used=()))}.get("ans") == "answered_question"
    assert "ans" in {f["id"] for f in m.facts(_readd_env(kept=("ans",)))}


def test_list_siblings_add_only_kept_siblings():
    atoms = [{"id": "s1", "artifact_id": "d", "text": "IDF to side door 154ft"},
             {"id": "s2", "artifact_id": "d", "text": "IDF to back door 88ft"}]
    fact = {"id": "s1", "section": "scope_of_work", "when": "", "under": "", "text": "IDF to side door 154ft"}
    out = [dict(fact)]
    m._list_siblings(out, atoms, {"s1"})
    assert [f["id"] for f in out] == ["s1"]
    out = [dict(fact)]
    m._list_siblings(out, atoms, {"s1", "s2"})
    assert [f["id"] for f in out] == ["s1", "s2"]
    out = [dict(fact)]
    m._list_siblings(out, atoms)  # no keep reading: the rule alone
    assert [f["id"] for f in out] == ["s1", "s2"]


def test_schedule_from_facts_reads_only_kept_dated_atoms():
    dated = _atom("dl", "Work can start 11/3 and continue through 11/21 for all stores.", "scope_item", 60,
                  c3_label={"type": "deadline", "p": 0.9})

    def run(heads_used, kept=()):
        env = {"atoms": [dated], "c3_brief": {"sections": {}, "kept_unsure": [],
                                              "also_mentioned": [{"id": i, "text": dated["text"]} for i in kept],
                                              "decided_by": {"heads_used": list(heads_used)}}}
        c = {"schedule": []}
        m._schedule_from_facts(c, [], env)
        return [x["cites"] for x in c["schedule"]]
    assert run(["read:co_action"]) == []
    assert run(["read:co_action"], kept=("dl",)) == [["dl"]]
    assert run([]) == [["dl"]]
