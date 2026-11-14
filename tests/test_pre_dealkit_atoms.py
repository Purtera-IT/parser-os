"""Before Deal Kit answers become PM-confirmed atoms the C3 heads read (pre_dealkit_atoms)."""
from __future__ import annotations

import json

import pytest

from app.core import pre_dealkit_atoms as pdk
from app.core import rule_ledger
from app.core.schemas import AtomType, AuthorityClass, ReviewStatus

DOC = {
    "schema": "purpulse.pre-dealkit-answers.v1",
    "deal_id": "d1",
    "total": 8,
    "answered": 3,
    "answers": [
        {"key": "known_scope", "question": "Have we done this exact scope before?", "answer": True, "why": "",
         "statement": None, "answered_by": "pm@x", "answered_at": "2026-11-13T20:10:00.000Z"},
        {"key": "regulated_labor", "question": "Is any site union, prevailing wage, or a regulated environment?",
         "answer": True, "why": "Chicago union hall",
         "statement": "PM confirmed: at least one site is union, prevailing wage, or a regulated environment. "
                      "Why: Chicago union hall.",
         "answered_by": "pm@x", "answered_at": "2026-11-13T20:10:00.000Z"},
        {"key": "after_hours", "question": "Will any of the work be nights or weekends?", "answer": False, "why": "",
         "statement": "PM confirmed: all of the work is during normal business hours.",
         "answered_by": "pm@x", "answered_at": "2026-11-14T01:00:00.000Z"},
    ],
}


def test_one_atom_per_statement_verbatim_pm_confirmed_and_flagged():
    atoms = pdk.answers_to_atoms(pdk.parse_document(json.dumps(DOC)), project_id="d1")
    assert [a.raw_text for a in atoms] == [DOC["answers"][1]["statement"], DOC["answers"][2]["statement"]]
    a = atoms[0]
    assert a.id.startswith(pdk.ATOM_PREFIX)
    assert a.atom_type == AtomType.compliance_rule  # starting type by question; the C3 base retypes where sure
    assert atoms[1].atom_type == AtomType.site_access_window
    assert a.authority_class == AuthorityClass.pm_confirmed
    assert a.review_status == ReviewStatus.approved
    assert "pm_pre_dealkit" in a.review_flags and "pm_authored" in a.review_flags
    assert a.value["key"] == "regulated_labor" and a.value["answer"] == "yes"
    assert a.value["said_by"] == {"side": "ours", "role": "pm"}
    loc = a.source_refs[0].locator
    assert loc["sent_at"] == "2026-11-13T20:10:00.000Z" and loc["question"].startswith("Is any site union")
    assert pdk.is_pre_dealkit_atom(a)


def test_a_question_about_us_makes_no_atom():
    atoms = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1")
    assert all(a.value["key"] != "known_scope" for a in atoms)


def test_ids_are_stable_and_a_changed_answer_is_a_new_atom():
    a1 = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1")
    a2 = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1")
    assert [a.id for a in a1] == [a.id for a in a2]
    changed = json.loads(json.dumps(DOC))
    changed["answers"][2]["statement"] = "PM confirmed: some of the work will be done nights or weekends."
    changed["answers"][2]["answer"] = True
    a3 = pdk.answers_to_atoms(pdk.parse_document(changed), project_id="d1")
    assert a3[1].id != a1[1].id and a3[0].id == a1[0].id


def test_a_cut_run_admits_only_answers_given_by_then():
    atoms = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1", as_of="2026-11-13T23:59:59")
    assert [a.value["key"] for a in atoms] == ["regulated_labor"]


def test_not_a_document_or_wrong_schema_is_nothing():
    assert pdk.parse_document(b"not json") == []
    assert pdk.parse_document({"schema": "other", "answers": DOC["answers"]}) == []
    assert pdk.answers_to_atoms([{"key": "after_hours", "answer": "yes", "statement": "x"}], project_id="d1") == []


def test_off_unless_the_pm_answer_switch_is_on(monkeypatch):
    monkeypatch.delenv("SOWSMITH_FEEDBACK_BLOB", raising=False)
    assert pdk.enabled() is False
    assert pdk.load_pre_dealkit_atoms(project_id="d1") == []
    monkeypatch.setenv("SOWSMITH_FEEDBACK_BLOB", "1")
    assert pdk.enabled() is True
    monkeypatch.setenv("SOWSMITH_PRE_DEALKIT_ATOMS", "0")
    assert pdk.enabled() is False


def test_loads_from_blob_and_never_raises(monkeypatch):
    monkeypatch.setenv("SOWSMITH_FEEDBACK_BLOB", "1")
    monkeypatch.setattr(pdk, "load_document", lambda deal_id: json.dumps(DOC).encode())
    atoms = pdk.load_pre_dealkit_atoms(project_id="d1", deal_id="d1")
    assert len(atoms) == 2

    def boom(deal_id):
        raise RuntimeError("blob down")

    monkeypatch.setattr(pdk, "load_document", boom)
    assert pdk.load_pre_dealkit_atoms(project_id="d1", deal_id="d1") == []


@pytest.fixture
def fresh_ledger(monkeypatch):
    rule_ledger.reset()
    yield
    rule_ledger.reset()


def test_quoting_view_keeps_the_answers_and_ledgers_it(fresh_ledger, monkeypatch):
    from app.core.c3_type_base import _with_pre_dealkit

    monkeypatch.delenv("SOWSMITH_RULE_MODES", raising=False)
    lines = [{"atom_id": "a1"}, {"atom_id": "atm_pdk_0001"}, {"atom_id": "a2"}, {"atom_id": "atm_pdk_0002"}]
    assert _with_pre_dealkit([0], lines) == [0, 1, 3]
    assert _with_pre_dealkit([0, 1, 2, 3], lines) == [0, 1, 2, 3]
    fires = rule_ledger.fires().get("view.pm_pre_dealkit_in") or {}
    assert fires.get("fires") == 2 and fires.get("stage") == "upstream"


def test_shadow_mode_logs_and_leaves_the_view(fresh_ledger, monkeypatch):
    from app.core.c3_type_base import _with_pre_dealkit

    monkeypatch.setenv("SOWSMITH_RULE_MODES", "view.pm_pre_dealkit_in:shadow")
    lines = [{"atom_id": "a1"}, {"atom_id": "atm_pdk_0001"}]
    assert _with_pre_dealkit([0], lines) == [0]
    assert (rule_ledger.fires().get("view.pm_pre_dealkit_in") or {}).get("fires") == 1


def test_document_job_scope_never_judges_or_drops_the_answers(monkeypatch):
    from app.core import document_job_scope as djs

    atoms = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1")
    seen = {}

    def fake_bundles(by_doc, index):
        seen["docs"] = list(by_doc)
        return {}

    monkeypatch.setattr(djs, "bundle_documents", fake_bundles)
    monkeypatch.setattr(djs, "c3_mode", lambda: "off")
    monkeypatch.setattr(djs, "manifest_index", lambda project_dir: {})
    kept, dropped, _ = djs.judge_documents(atoms, deal_name="Deal", project_id="d1")
    assert seen["docs"] == []
    assert kept == atoms and dropped == []


def test_every_answerable_question_has_a_starting_type_per_answer_and_each_is_ledgered():
    rule_ledger.reset()
    for key, (yes, no) in pdk.TYPE_BY_QUESTION.items():
        assert yes and no, key
    atoms = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1")
    rec = rule_ledger.fires().get("pre_dealkit.type_by_question") or {}
    assert rec.get("fires") == len(atoms) == 2 and rec.get("no_head") == 2 and rec.get("stage") == "upstream"
    no = json.loads(json.dumps(DOC))
    no["answers"][1]["answer"] = False
    no["answers"][1]["statement"] = "PM confirmed: no site is union, prevailing wage, or a regulated environment."
    assert pdk.answers_to_atoms(pdk.parse_document(no), project_id="d1")[0].atom_type == AtomType.pricing_assumption
    rule_ledger.reset()


def test_the_answers_are_never_read_as_our_deal_kit():
    """The base never reads a Deal Kit line and the brief drops it (old_manual_deal_kit): the
    answers' source must not look like one by name, or no head ever reads them."""
    from c3_record import file_kind, is_deal_kit
    from app.core.c3_type_base import doc_kind, set_aside, skipped

    for a in pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1", deal_id="d1"):
        ref = a.source_refs[0]
        assert not is_deal_kit(file_kind(ref.filename)) and file_kind(ref.filename) == "file"
        assert "dealkit" not in "".join(ch for ch in (a.artifact_id + ref.filename + str(ref.path)).lower() if ch.isalpha())
        assert not skipped(a, {}) and doc_kind(a, {}) == "file" and set_aside(a, {}) == ""


def test_the_nights_weekends_answer_is_its_lines_work_window_in_the_quotes_read(fresh_ledger, monkeypatch):
    """Owner 2026-11-14: the PM's nights/weekends answer sets the work window on its own line (ledgered), until
    the work-window head reads it itself (it read business_hours for every after-hours wording on dev)."""
    from app.core import c3_type_base

    seen = {}
    monkeypatch.delenv("SOWSMITH_RULE_MODES", raising=False)
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    yes = json.loads(json.dumps(DOC))
    yes["answers"][2].update(answer=True, statement="PM confirmed: work will be performed after hours, nights and weekends.")
    atoms = pdk.answers_to_atoms(pdk.parse_document(yes), project_id="d1")
    after = next(a for a in atoms if a.value["key"] == "after_hours")
    union = next(a for a in atoms if a.value["key"] == "regulated_labor")
    reads = {after.id: {"read:work_window": ["business_hours", 0.87]}}
    monkeypatch.setattr(c3_type_base, "_DEAL", {"reads": reads, "deal": {}, "view": "quoting"})

    def fake(path, body, timeout):
        seen.update(body=body)
        return {"job_type": "cabling", "lines": []}

    monkeypatch.setattr(c3_type_base, "_request", fake)
    c3_type_base.quote(atoms, {"deal": {}, "sections": {}}, deal_id="d1")
    rows = {r["id"]: r for r in seen["body"]["deal"]["atoms"]}
    assert rows[after.id]["reads"]["read:work_window"] == ["after_hours", 0.99]
    assert "reads" not in rows[union.id]  # only the nights/weekends answer
    rec = rule_ledger.fires()["pre_dealkit.work_window"]
    assert rec["fires"] == 1 and rec["disagree"] == 1 and rec["stage"] == "upstream"

    # Shadow: logged, the head's own read is what the quote gets.
    rule_ledger.reset()
    monkeypatch.setenv("SOWSMITH_RULE_MODES", "pre_dealkit.work_window:shadow")
    c3_type_base.quote(atoms, {"deal": {}, "sections": {}}, deal_id="d1")
    rows = {r["id"]: r for r in seen["body"]["deal"]["atoms"]}
    assert rows[after.id]["reads"]["read:work_window"] == ["business_hours", 0.87]
    assert rule_ledger.fires()["pre_dealkit.work_window"]["fires"] == 1


def test_work_window_by_answer():
    no = pdk.answers_to_atoms(pdk.parse_document(DOC), project_id="d1")
    assert [pdk.work_window(a) for a in no] == [None, "business_hours"]
    assert pdk.work_window({"id": "x", "value": {"key": "after_hours", "answer": "yes"}}) is None  # not ours
