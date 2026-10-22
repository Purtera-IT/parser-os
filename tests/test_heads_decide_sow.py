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
