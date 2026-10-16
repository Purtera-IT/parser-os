"""Brief v2: heads decide where the registry trusts them, rules otherwise."""
from types import SimpleNamespace as NS

from app.core import c3_brief
from c3_record import heads


def _atom(i, t, text="x"):
    return NS(id=f"a{i}", atom_type=t, raw_text=text, review_status="approved")


def test_registry_layers_and_trust():
    reg = heads.registry()
    assert reg, "c3_record/heads.json ships with the package"
    assert heads.layer("read:co_action") == "purtera"
    assert heads.layer("read:sow_section") == "base"
    assert heads.trusted("read:co_action")
    assert not heads.trusted("col:wants")  # one-sided labels: not trusted


def test_no_read_falls_back_to_rules():
    atoms = [_atom(1, "scope_item", "Install 4 cameras"), _atom(2, "small_talk", "thanks!"),
             _atom(3, "open_question", "Who owns the lift?")]
    b = c3_brief.build(atoms, service_fallback="security_camera")
    assert [x["id"] for x in b["sections"]["scope_of_work"]] == ["a1"]
    assert b["service"] == {"value": "security_camera", "by": "rule:domain_pack", "base_says": None,
                            "decided_by_layer": "rule"}
    assert [q["id"] for q in b["questions"]] == ["a3"]
    assert b["decided_by"]["heads_used"] == []


def test_trusted_heads_decide_and_say_so():
    atoms = [_atom(1, "scope_item"), _atom(2, "scope_item"), _atom(3, "task")]
    reads = {"a1": {"read:co_action": ["keep", 0.9], "read:sow_section": ["deliverables", 0.8]},
             "a2": {"read:co_action": ["reject", 0.7], "read:co_reason": ["hw_price_not_ours", 0.6]},
             "a3": {"read:co_action": ["keep", 0.7]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert b["sections"]["deliverables"][0]["by"] == "purtera:read:co_action"
    assert b["sections"]["deliverables"][0]["section_by"] == "base:read:sow_section"
    assert [x["id"] for x in b["also_mentioned"]] == ["a3"]
    assert b["left_out"]["by_reason"] == {"hw_price_not_ours": 1}
    assert {t["id"] for t in b["label_next"]} == {"a3"}


def test_brief_survives_result_round_trip():
    from app.core.schemas import CompileResult

    r = CompileResult(project_id="p", atoms=[], entities=[], edges=[], packets=[], project_dir="/x",
                      c3_brief={"version": 2})
    assert CompileResult.model_validate_json(r.model_dump_json()).c3_brief == {"version": 2}


def test_catalog_questions_ranked_by_the_question_head(monkeypatch):
    import c3_record.questions_catalog as qc

    monkeypatch.setattr(qc, "by_text", lambda: {
        "Who approves site access?": {"id": "site_access_approver", "family": "site_access", "layer": "base"},
        "What VMS do they run?": {"id": "vms", "family": "network_it", "layer": "base",
                                  "applies_to": ["security_camera"]}})
    asked = [{"text": "Who approves site access?", "p_valid": 0.6},
             {"text": "What VMS do they run?", "p_valid": 0.7},
             {"text": "Is there parking?", "p_valid": 0.1}]
    b = c3_brief.build([_atom(1, "scope_item")], service_fallback="low_voltage_cabling",
                       readings={"reads": {"a1": {"read:co_action": ["keep", 0.95]}}, "links": [],
                                 "deal": {}, "questions": asked})
    got = [q["id"] for q in b["questions"]]
    assert got == ["q:site_access_approver"]  # VMS is for cameras; parking is under the bar
    assert b["questions"][0]["by"] == "base:jdg:gap"


def test_envelope_questions_carry_what_the_head_reads():
    brief = {"questions": [{"id": "q:indemnification", "text": "What indemnification terms apply?",
                            "asked_as": "Missing: Indemnification clause", "family": "commercial_terms",
                            "layer": "base", "p": 0.9, "by": "base:jdg:gap"},
                           {"id": "q:payment_terms", "text": "What are the payment terms?",
                            "asked_as": "What are the payment terms?", "p": 0.8, "by": "base:jdg:gap"}],
             "written": {"catalog_questions": [{"id": "q:indemnification", "question": "What indemnification terms apply?",
                                                "p": 0.9, "cites": ["a1"]}],
                         "answered_in_deal": [{"id": "q:payment_terms", "question": "What are the payment terms?",
                                               "answered_by": ["a7"]}]}}
    got = c3_brief.envelope_questions(brief)
    assert [(q["question_id"], q["review_status"]) for q in got] == [
        ("q:indemnification", "needs_review"), ("q:payment_terms", "answered")]
    assert got[0]["query"] == "Missing: Indemnification clause" and got[0]["kind"] == "catalog"
    assert got[1]["atom_id"] is None and got[1]["cites"] == ["a7"]


def test_cut_off_prose_keeps_its_complete_sentences():
    from app.core.c3_brief import _json_of, _salvage

    raw = ('```json\n{"summary": "Install \\"three\\" cameras [atm_a].", "sections": {"scope": ['
           '{"text": "Mount cameras.", "cites": ["atm_a"]}, {"text": "Run cab')
    assert _json_of(raw) is None
    doc = _salvage(raw)
    assert doc["summary"].startswith('Install "three"')
    assert doc["sections"]["scope"] == [{"text": "Mount cameras.", "cites": ["atm_a"]}]
    assert _salvage("no json here") is None


def test_the_latest_sow_governs_the_contract_sections():
    from types import SimpleNamespace as N

    from app.core.c3_brief import _drop_superseded

    atoms = [N(id="d1", artifact_id="draft"), N(id="c1", artifact_id="co"), N(id="s1", artifact_id="signed"),
             N(id="x1", artifact_id="co"), N(id="o1", artifact_id="draft")]
    docs = [{"doc_id": "draft", "kind": "sow_draft", "filename": "Install 06.22.26 v1.docx"},
            {"doc_id": "co", "kind": "change_order", "filename": "Change Order (7.29.26).docx"},
            {"doc_id": "signed", "kind": "file", "filename": "SOW_207586 ready for signature.pdf"}]
    brief = {"sections": {"fees": [{"id": "d1", "text": "$7,890"}, {"id": "c1", "text": "$1,260"},
                                   {"id": "s1", "text": "$9,150"}],
                          "overview": [{"id": "x1", "text": "Documents disagree"}],
                          "out_of_scope": [{"id": "o1", "text": "Painting is excluded"}]}}
    brief["sections"]["overview"][0]["id"] = "xdoc_conflict_1"
    atoms.append(N(id="xdoc_conflict_1", artifact_id="co"))
    assert _drop_superseded(brief, atoms, docs) == 3
    assert [i["id"] for i in brief["sections"]["fees"]] == ["s1"]
    assert brief["sections"]["overview"] == []
    # The signed SOW says nothing on exclusions: the draft's stays.
    assert [i["id"] for i in brief["sections"]["out_of_scope"]] == ["o1"]
