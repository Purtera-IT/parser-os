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


def test_no_trusted_head_lost_to_live_on_the_grade():
    """Test A's registry: a head the held-out grade measured worse than the service's never decides."""
    for key, h in heads.registry().items():
        ev = h.get("holdout") or {}
        if h["status"] == "ready" and ev.get("live") is not None:
            assert ev["exact"] >= ev["live"] and ev.get("verdict") != "worse", key
    assert not heads.trusted("read:sow_section")  # test A 0.376 vs live 0.629 on heldout7


def test_no_read_falls_back_to_rules():
    atoms = [_atom(1, "scope_item", "Install 4 cameras"), _atom(2, "small_talk", "thanks!"),
             _atom(3, "open_question", "Who owns the lift?")]
    b = c3_brief.build(atoms, service_fallback="security_camera")
    assert [x["id"] for x in b["sections"]["scope_of_work"]] == ["a1"]
    assert b["service"] == {"value": "security_camera", "by": "rule:domain_pack", "base_says": None,
                            "decided_by_layer": "rule"}
    assert [q["id"] for q in b["questions"]] == ["a3"]
    assert b["decided_by"]["heads_used"] == []


def test_trusted_heads_decide_and_say_so(monkeypatch):
    # The mechanism, not the shipped registry: test A's read:sow_section is not trusted (worse than live).
    monkeypatch.setenv("SOWSMITH_C3_BRIEF_TRUST", "read:co_action,read:co_reason,read:sow_section")
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


def test_a_signed_sow_attached_to_nothing_still_governs_by_its_date():
    from types import SimpleNamespace as N

    from app.core.c3_brief import _drop_superseded

    atoms = [N(id="c1", artifact_id="co"), N(id="s1", artifact_id="signed")]
    docs = [{"doc_id": "signed", "kind": "file", "filename": "SOW_207586 is ready for signature.pdf",
             "authored_at": "2026-09-30"},
            {"doc_id": "co", "kind": "change_order", "filename": "Change Order (7.29.26).docx",
             "authored_at": "2026-07-29"}]
    brief = {"sections": {"fees": [{"id": "c1", "text": "$1,260"}, {"id": "s1", "text": "$9,150"}]}}
    assert _drop_superseded(brief, atoms, docs) == 1
    assert [i["id"] for i in brief["sections"]["fees"]] == ["s1"]
    assert atoms[0].value["superseded_by"].startswith("SOW_207586") and not getattr(atoms[1], "value", None)


def test_contacts_are_people_with_a_role_email_or_phone():
    from types import SimpleNamespace as N

    from app.core.c3_brief import contacts

    st = "stakeholder"
    atoms = [N(id="1", atom_type=st, raw_text="Nicole Tomasik 312.705.9059",
               value={"name": "Nicole Tomasik", "role": "Project Manager", "email": "n@cdw.com"}),
             N(id="2", atom_type=st, raw_text="Nicole Tomasik", value={"name": "Nicole Tomasik"}),
             N(id="3", atom_type=st, raw_text="Hi Tanner", value={"name": "Hi Tanner", "role": "x"}),
             N(id="4", atom_type=st, raw_text="Brooksource Cleveland OH site",
               value={"name": "Brooksource Cleveland OH site", "role": "y"}),
             N(id="5", atom_type=st, raw_text="Sep 30", value={"name": "B Chase Smith", "role": "filler"}),
             N(id="6", atom_type=st, raw_text="Rob Cole office manager 419-376-8767", value={"name": "Rob Cole"})]
    got = contacts(atoms)
    assert [c["name"] for c in got] == ["Nicole Tomasik", "Rob Cole"]
    assert got[0]["line"] == "Nicole Tomasik (Project Manager, n@cdw.com, 312.705.9059)" or "Project Manager" in got[0]["line"]
    assert "419-376-8767" in got[1]["line"]


def test_job_details_keep_the_lines_a_pm_prices_from():
    from types import SimpleNamespace as N

    from app.core.c3_brief import job_details

    def at(i, t, text, **v):
        return N(id=i, atom_type=t, raw_text=text, value=v, source_refs=[], review_flags=[])

    atoms = [at("1", "quantity", "Side door 154 ft cable run from the IDF"),
             at("2", "site_attribute", "All three cameras mount on brick above the doors"),
             at("3", "quantity", "Side door 154 ft cable run from the IDF"),
             at("4", "task", "Is there a lift on site?"),
             at("5", "bom_line", "Old draft line", superseded_by="SOW.pdf"),
             at("6", "task", "Hi team, thanks"),
             at("7", "commercial_total", "SALES TAX: $329.23")]
    got = job_details(atoms)
    assert [d["cites"] for d in got] == [["1"], ["2"]]


def test_a_cut_run_admits_only_pm_answers_given_by_the_cutoff(monkeypatch):
    import app.core.pm_answer_blob as B

    monkeypatch.setattr(B, "load_answer_events", lambda d: [
        {"created_at": "2026-06-20T10:00:00Z", "x": 1}, {"created_at": "2026-10-16T10:00:00Z", "x": 2}])
    seen = {}

    def fake(events, project_id):
        seen["n"] = len(list(events))
        return []
    import app.core.pm_answer_atoms as A
    monkeypatch.setattr(A, "pm_answers_to_atoms", fake)
    B.load_pm_answer_atoms(project_id="p", as_of="2026-06-23T14:48:50Z")
    assert seen["n"] == 1


def test_a_deal_question_the_deal_answers_is_dropped(monkeypatch):
    import app.core.c3_brief as B

    brief = {"service": {"value": "camera"}, "written": {"open_questions": [
        {"question": "Can you confirm the cable lengths? [atm_a]", "cites": ["atm_a"]},
        {"question": "Who supplies the mounts?", "cites": []}]}}

    def fake_ev(qs, atoms):
        for q in qs:
            q["evidence"] = [{"id": "atm_b", "text": "Side door 154 ft, back door 88 ft"}]
    monkeypatch.setattr(B, "_attach_evidence", fake_ev)
    monkeypatch.setattr(B, "_complete", lambda p, m, t: '{"answered": [{"id": "d0", "by": "atm_b"}]}')
    assert B._drop_answered_open_questions(brief, []) == 1
    w = brief["written"]
    assert [q["question"] for q in w["open_questions"]] == ["Who supplies the mounts?"]
    assert w["answered_open_questions"][0]["answered_by"] == ["atm_b"]


def _trusted_all(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_BRIEF_TRUST", "read:co_action,read:sow_section,read:co_reason")


def test_a_line_the_base_reads_as_answered_or_a_heading_stays_out(monkeypatch):
    from app.core import c3_type_base as T

    _trusted_all(monkeypatch)
    monkeypatch.setitem(T._DEAL, "types", {"a1": ["answered_question", 0.9985], "a2": ["list_header", 0.997]})
    atoms = [_atom(1, "open_question", "Do you happen to have the model of the camera?"),
             _atom(2, "deal_metadata", "Site Contacts:"), _atom(3, "scope_item", "Mount 3 cameras")]
    reads = {a.id: {"read:co_action": ["keep", 0.99], "read:sow_section": ["customer_responsibilities", 0.6]}
             for a in atoms}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    kept = [x["id"] for v in b["sections"].values() for x in v]
    assert kept == ["a3"]
    assert b["left_out"]["by_reason"] == {"base type answered_question": 1, "base type list_header": 1}
    assert "a1" not in [q["id"] for q in b["questions"]]


def test_the_owner_head_wins_when_surer_than_the_section_head(monkeypatch):
    _trusted_all(monkeypatch)
    atoms = [_atom(1, "dependency", "Mount the Aruba APs at the NEW site")]
    reads = {"a1": {"read:co_action": ["keep", 0.99], "read:sow_section": ["customer_responsibilities", 0.65],
                    "read:task_owner_role": ["technician", 0.70]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert [x["id"] for x in b["sections"]["scope_of_work"]] == ["a1"]
    assert "customer_responsibilities" not in b["sections"]


def test_a_document_the_customer_furnishes_is_not_our_fees(monkeypatch):
    _trusted_all(monkeypatch)
    atoms = [NS(id=f"a{i}", atom_type=t, raw_text=x, review_status="approved", artifact_id="cdw_order")
             for i, (t, x) in enumerate([("vendor_line_item", "ITEM: CF83-E camera | QTY: 1 | $1,325.27"),
                                         ("vendor_line_item", "ITEM: CD53-E dome | QTY: 2 | $2,797.86"),
                                         ("vendor_line_item", "ITEM: license | QTY: 3 | $1,313.79"),
                                         ("commercial_total", "GRAND TOTAL: $6,184.28")], 1)]
    sup = {"a1": ["customer", 0.76], "a2": ["customer", 0.73], "a3": ["us", 0.88], "a4": ["third_party", 0.66]}
    reads = {a.id: {"read:co_action": ["keep", 0.99], "read:sow_section": ["fees", 0.8],
                    "col:supplier": sup[a.id]} for a in atoms}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert not b["sections"].get("fees")
    assert {x["id"] for x in b["sections"]["customer_furnished"]} == {"a1", "a2", "a3", "a4"}


def test_the_writer_keeps_each_sentence_in_its_heads_section(monkeypatch):
    B = c3_brief
    brief = {"service": {"value": "low_voltage_cabling"}, "conflicts": [], "questions": [],
             "sections": {"scope_of_work": [{"id": "a1", "text": "Mount 3 APs"}],
                          "overview": [{"id": "a2", "text": "Ladders on site; no backboard, please install one"}]}}
    out = ('{"summary": "Three APs. [a1]", "sections": {"scope": [{"text": "Mount three APs.", "cites": ["a1"]}],'
           ' "out_of_scope": [{"text": "Painting the backboard is not included.", "cites": ["a2"]}]}}')
    monkeypatch.setattr(B, "_complete", lambda p, m, t: out)
    w = B.write(brief)
    assert list(w["sections"]) == ["scope"]
    assert w["moved_dropped"] == 1


def test_job_details_come_only_from_lines_the_heads_kept():
    atoms = [_atom(1, "scope_item", "IDF to side door 154ft"), _atom(2, "scope_item", "concrete ceiling")]
    got = c3_brief.job_details(atoms, {"a1"})
    assert [d["cites"] for d in got] == [["a1"]]


def test_ids_written_only_in_the_sentence_still_count_as_its_cites(monkeypatch):
    B = c3_brief
    brief = {"service": {"value": "security_camera"}, "conflicts": [], "questions": [],
             "sections": {"scope_of_work": [{"id": "atm_a1", "text": "3 cameras"}]}}
    monkeypatch.setattr(B, "_complete", lambda p, m, t: '{"summary": "x", "sections": {"scope": '
                        '[{"text": "Install three cameras [atm_a1]."}]}}')
    w = B.write(brief)
    assert w["sections"]["scope"][0]["cites"] == ["atm_a1"] and w["uncited_dropped"] == 0


def test_a_summary_written_as_sentence_objects_is_text(monkeypatch):
    B = c3_brief
    brief = {"service": {"value": "x"}, "conflicts": [], "questions": [],
             "sections": {"scope_of_work": [{"id": "atm_a1", "text": "45 desks"}]}}
    monkeypatch.setattr(B, "_complete", lambda p, m, t: '{"summary": [{"text": "45 desks, two monitors each.", '
                        '"cites": ["atm_a1"]}, {"text": "Day work."}], "sections": {}}')
    assert B.write(brief)["summary"] == "45 desks, two monitors each. Day work."


def test_a_kept_line_typed_small_talk_is_not_dropped(monkeypatch):
    _trusted_all(monkeypatch)
    atoms = [_atom(1, "small_talk", "They are thinking the three weeks after Thanksgiving if all stays on track.")]
    reads = {"a1": {"read:co_action": ["keep", 0.64]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert [x["id"] for x in b["also_mentioned"]] == ["a1"]
    assert c3_brief.job_details(atoms, {"a1"})[0]["cites"] == ["a1"]


def test_a_named_contact_person_is_the_on_site_contact():
    atoms = [NS(id="a1", atom_type="stakeholder", raw_text="Contact Person is Rafael Fernandez", value={})]
    got = c3_brief.contacts(atoms)
    assert got[0]["name"] == "Rafael Fernandez" and got[0]["role"] == "on-site contact"


def test_a_phone_before_a_name_is_not_that_persons():
    loc = lambda n: [NS(locator={"line_start": n})]
    atoms = [NS(id="a1", atom_type="stakeholder", raw_text="Contact Person is Rafael Fernandez", value={}, artifact_id="d",
                source_refs=loc(205)),
             NS(id="a2", atom_type="deal_metadata", raw_text="Cell Number:  904-386-5066", value={}, artifact_id="d",
                source_refs=loc(207)),
             NS(id="a3", atom_type="stakeholder", raw_text="Cell Number: 904-386-5066, Bill Ort, AVP | ortw@vystarcu.org",
                value={"name": "Bill Ort", "role": "AVP", "email": "ortw@vystarcu.org"}, artifact_id="n")]
    got = {c["name"]: c for c in c3_brief.contacts(atoms)}
    assert got["Rafael Fernandez"]["phone"] == "904-386-5066" and got["Bill Ort"]["phone"] == ""


def test_our_task_filed_out_of_scope_goes_back_to_scope(monkeypatch):
    _trusted_all(monkeypatch)
    atoms = [_atom(1, "scope_item", "Remove the Printer from the OLD site")]
    reads = {"a1": {"read:co_action": ["keep", 0.95], "read:sow_section": ["out_of_scope", 0.54],
                    "read:task_owner_role": ["technician", 0.84]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert [x["id"] for x in b["sections"]["scope_of_work"]] == ["a1"]


def test_question_lists_nested_in_sections_are_kept(monkeypatch):
    B = c3_brief
    brief = {"service": {"value": "x"}, "conflicts": [], "questions": [],
             "sections": {"scope_of_work": [{"id": "atm_a1", "text": "Move 3 APs"}]}}
    monkeypatch.setattr(B, "_complete", lambda p, m, t: '{"summary": "s", "sections": {"scope": [{"text": "Move 3 APs.", '
                        '"cites": ["atm_a1"]}], "pricing_questions": [{"question": "What is the ceiling height?", '
                        '"cites": ["atm_a1"]}]}}')
    w = B.write(brief)
    assert [q["question"] for q in w["open_questions"]] == ["What is the ceiling height?"] and w["uncited_dropped"] == 0


def test_the_coverage_pass_adds_cited_missing_facts(monkeypatch):
    B = c3_brief
    brief = {"sections": {"scope_of_work": [{"id": "atm_a1", "text": "Move the APs"}]},
             "figures": [{"id": "atm_f1", "text": "3 Aruba APs are marked."}],
             "written": {"summary": "Move APs.", "sections": {"scope": [{"text": "Move the APs.", "cites": ["atm_a1"]}]}}}
    monkeypatch.setattr(B, "_complete", lambda p, m, t: '{"supported": [0]}' if "numbered sentence" in p else
                        '{"add": [{"text": "There are 3 Aruba APs.", "cites": ["atm_f1"]},'
                        ' {"text": "Invented.", "cites": ["atm_zz"]}]}')
    assert B._cover_missing_facts(brief) == 1
    assert brief["written"]["sections"]["job_details"] == [{"text": "There are 3 Aruba APs.", "cites": ["atm_f1"]}]


def _judge_says(monkeypatch, answers):
    from app.core import sow_rubric_judge as J

    def fake_judge(text, **_):
        sec, own = answers.get(text, (None, None))
        return J.Verdict(sec, 0.9 if sec else 0.0, own, 0.9 if own else 0.0, "llm" if sec else "fallback")

    monkeypatch.setattr(J, "judge", fake_judge)


def test_the_rubric_judge_places_facts_when_on(monkeypatch):
    _trusted_all(monkeypatch)
    monkeypatch.setenv("SOWSMITH_SOW_JUDGE", "on")
    _judge_says(monkeypatch, {"Customer ships the TVs": ("customer_responsibilities", "customer"),
                              "Thanks all": ("_absent", "none")})
    atoms = [_atom(1, "scope_item", "Customer ships the TVs"), _atom(2, "scope_item", "Thanks all"),
             _atom(3, "scope_item", "Mount 3 TVs")]
    reads = {a.id: {"read:co_action": ["keep", 0.99], "read:sow_section": ["scope_of_work", 0.6]} for a in atoms}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    cr = b["sections"]["customer_responsibilities"]
    assert [x["id"] for x in cr] == ["a1"] and cr[0]["section_by"] == "judge:jdg:sow_section"
    assert cr[0]["judge"] == {"section": "customer_responsibilities", "owner": "customer", "p": 0.9}
    # _absent is not a drop (keep/drop stays co_action's) and an undecided judge changes nothing
    sow = {x["id"]: x for x in b["sections"]["scope_of_work"]}
    assert set(sow) == {"a2", "a3"}
    assert sow["a2"]["section_by"] == "base:read:sow_section" and sow["a2"]["judge"]["section"] == "_absent"
    assert "judge" not in sow["a3"]


def test_the_rubric_judge_in_shadow_changes_nothing(monkeypatch):
    _trusted_all(monkeypatch)
    monkeypatch.setenv("SOWSMITH_SOW_JUDGE", "shadow")
    _judge_says(monkeypatch, {"Customer ships the TVs": ("customer_responsibilities", "customer")})
    atoms = [_atom(1, "scope_item", "Customer ships the TVs")]
    reads = {"a1": {"read:co_action": ["keep", 0.99], "read:sow_section": ["scope_of_work", 0.6]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    item = b["sections"]["scope_of_work"][0]
    assert item["section_by"] == "base:read:sow_section"
    assert item["judge"]["section"] == "customer_responsibilities"


def test_the_rubric_judge_is_off_by_default(monkeypatch):
    _trusted_all(monkeypatch)
    monkeypatch.delenv("SOWSMITH_SOW_JUDGE", raising=False)
    _judge_says(monkeypatch, {"Customer ships the TVs": ("customer_responsibilities", "customer")})
    atoms = [_atom(1, "scope_item", "Customer ships the TVs")]
    reads = {"a1": {"read:co_action": ["keep", 0.99], "read:sow_section": ["scope_of_work", 0.6]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert "judge" not in b["sections"]["scope_of_work"][0]


def test_the_writer_checks_run_side_by_side_and_end_as_one_after_another(monkeypatch):
    """verify -> cover beside facts-answer -> deal-answered (compile speed, 2026-10-28):
    each chain keeps its order and neither reads what the other writes, so the brief
    ends as it did when the four ran one after another."""
    import threading

    B = c3_brief
    brief = {"service": {"value": "x"},
             "sections": {"scope_of_work": [{"id": "atm_a1", "text": "Move the APs, ceiling is 12 ft"}]},
             "figures": [{"id": "atm_f1", "text": "3 Aruba APs are marked."}],
             "written": {"summary": "Move APs.",
                         "sections": {"scope": [{"text": "Move the APs.", "cites": ["atm_a1"]},
                                                {"text": "Invented.", "cites": ["atm_a1"]}]},
                         "open_questions": [{"question": "What is the ceiling height?", "kind": "pricing"},
                                            {"question": "Who has the lift?", "kind": "deal"}]}}
    order: list[str] = []
    threads: set[str] = set()

    def fake(p, m, t):
        threads.add(threading.current_thread().name)
        if "numbered sentence" in p:
            order.append("verify")
            return '{"supported": [0]}'
        if "does NOT already state" in p:
            order.append("cover")
            return '{"add": [{"text": "There are 3 Aruba APs.", "cites": ["atm_f1"]}]}'
        if '"p0"' in p:
            order.append("facts")
            return '{"answered": ["p0"]}'
        order.append("ask")
        return '{"answered": []}'

    monkeypatch.setattr(B, "_complete", fake)
    monkeypatch.setattr(B, "_attach_evidence", lambda probe, atoms: None)
    B._check_written(brief, [])
    w = brief["written"]
    assert [x["text"] for x in w["sections"]["scope"]] == ["Move the APs."]
    assert [x["text"] for x in w["sections"]["job_details"]] == ["There are 3 Aruba APs."]
    assert [q["question"] for q in w["pricing_answered_in_deal"]] == ["What is the ceiling height?"]
    assert [q["question"] for q in w["open_questions"]] == ["Who has the lift?"]
    assert order.index("verify") < order.index("cover") and order.index("facts") < order.index("ask")
    assert len(threads) >= 2, "the two chains ran on one thread"


def _typed(i, t, text, how="label"):
    v = ({"c3_label": {"type": t, "p": 0.97, "act": "authority", "was": "task"}} if how == "label"
         else {"c3_confirm": {"type": t, "p": 0.9, "said": "_keep"}} if how == "confirm" else {})
    return NS(id=f"a{i}", atom_type=t, raw_text=text, review_status="approved", value=v)


def test_a_head_typed_exclusion_keeps_its_section_against_the_owner_head(monkeypatch):
    """010362: an exclusion (type head 0.96) and a pricing term (0.99) were moved to
    scope_of_work by read:task_owner_role, because against a rule placement the
    override compared the owner's p with 0."""
    _trusted_all(monkeypatch)
    atoms = [_typed(1, "exclusion", "Firewall programming performed independently is out of scope"),
             _typed(2, "pricing_assumption", "After hours work will be billed at a 50% uplift", "confirm"),
             _typed(3, "dependency", "Customer ensures the circuit is active before the visit")]
    reads = {f"a{i}": {"read:co_action": ["keep", 0.95], "read:task_owner_role": ["technician", 0.88]}
             for i in (1, 2, 3)}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    secs = {x["id"]: s for s, xs in b["sections"].items() for x in xs}
    assert secs["a1"] == c3_brief.SECTION_BY_TYPE["exclusion"]
    assert secs["a2"] == c3_brief.SECTION_BY_TYPE["pricing_assumption"]
    assert secs["a3"] == c3_brief.SECTION_BY_TYPE["dependency"]
    assert all(x["type_by"] == "head" and x["section_by"] == "rule:type"
               for xs in b["sections"].values() for x in xs)


def test_the_owner_head_fills_only_where_no_head_typed_the_line_and_says_so(monkeypatch):
    _trusted_all(monkeypatch)
    atoms = [_typed(1, "dependency", "Mount the APs at the new site", how=None),
             _typed(2, "dependency", "Patch the new switch", how=None)]
    reads = {"a1": {"read:co_action": ["keep", 0.95], "read:task_owner_role": ["technician", 0.86]},
             "a2": {"read:co_action": ["keep", 0.95], "read:task_owner_role": ["technician", 0.51]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    filled = b["sections"]["scope_of_work"]
    assert [x["id"] for x in filled] == ["a1"]
    assert filled[0]["section_by"] == "base:read:task_owner_role" and filled[0]["type_by"] == "rule"
    assert "owner head filled" in filled[0]["section_fill"]
    # Below the owner bar the rule's placement stands.
    assert [x["id"] for x in b["sections"][c3_brief.SECTION_BY_TYPE["dependency"]]] == ["a2"]


def test_a_parser_typed_exclusion_is_no_duty_for_the_owner_head(monkeypatch):
    """010355: "Lifts ... are excluded" (no type-head mark) read technician 0.82
    by the owner head and was filed as our scope."""
    _trusted_all(monkeypatch)
    atoms = [_typed(1, "exclusion", "Lifts and scaffolding are excluded", how=None)]
    reads = {"a1": {"read:co_action": ["keep", 0.95], "read:task_owner_role": ["technician", 0.9]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert [x["id"] for x in b["sections"]["out_of_scope"]] == ["a1"]


def test_a_kept_line_moved_after_the_keep_loop_says_where_it_went():
    """010362: 92 lines read "kept" in line_decisions but were moved to
    ``superseded`` after the keep loop, so the ledger and the brief disagreed."""
    from types import SimpleNamespace as N

    from app.core.c3_brief import _drop_superseded, reconcile_line_decisions

    atoms = [N(id="d1", artifact_id="draft"), N(id="s1", artifact_id="signed"), N(id="m1", artifact_id="signed")]
    docs = [{"doc_id": "draft", "kind": "sow_draft", "filename": "Install v1.docx"},
            {"doc_id": "signed", "kind": "signed_sow", "filename": "SOW signed.pdf"}]
    brief = {"sections": {"fees": [{"id": "d1", "text": "$7,890"}, {"id": "s1", "text": "$9,150"}]},
             "also_mentioned": [{"id": "m1"}],
             "line_decisions": {"cols": ["id", "outcome", "by", "p", "col:admission", "read:co_action"],
                                "rows": [["d1", "kept", "base:col:admission", 0.9, None, None],
                                         ["s1", "kept", "base:col:admission", 0.9, None, None],
                                         ["m1", "also", "base:col:admission", 0.6, None, None],
                                         ["z1", "kept", "base:col:admission", 0.9, None, None],
                                         ["n1", "not_admitted", "base:col:admission", 0.2, None, None]]}}
    assert _drop_superseded(brief, atoms, docs) == 1
    assert reconcile_line_decisions(brief) == 2
    got = {r[0]: (r[1], r[2]) for r in brief["line_decisions"]["rows"]}
    assert got["d1"] == ("superseded", "rule:superseded")
    assert got["s1"][0] == "kept" and got["m1"][0] == "also" and got["n1"][0] == "not_admitted"
    assert got["z1"] == ("kept_unplaced", "none:base:col:admission")
    placed = {x["id"] for v in brief["sections"].values() for x in v} | {"m1"}
    assert all(r[0] in placed for r in brief["line_decisions"]["rows"] if r[1] in ("kept", "also"))


def test_an_untrusted_section_head_is_recorded_where_the_rule_places_the_line(monkeypatch):
    """Run O answers read:sow_section in the planning view; the registry does not trust it, so the rule
    places the line, and the line still carries the head's answer (the evidence a promotion needs)."""
    monkeypatch.setenv("SOWSMITH_C3_BRIEF_TRUST", "read:co_action")
    atoms = [_atom(1, "scope_item", "Install 4 cameras"), _atom(2, "exclusion", "Lifts are not included")]
    reads = {"a1": {"read:co_action": ["keep", 0.99], "read:sow_section": ["scope_of_work", 0.9]},
             "a2": {"read:co_action": ["keep", 0.99], "read:sow_section": ["assumptions", 0.7]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    items = {x["id"]: x for v in b["sections"].values() for x in v}
    assert {i["section_by"] for i in items.values()} == {"rule:type"}
    assert items["a1"]["section_head"] == ["scope_of_work", 0.9]
    assert items["a2"]["section_head"] == ["assumptions", 0.7]
    part = b["decided_by"]["parts"]["read:sow_section"]
    assert part["by"] == "rule" and part["head_answered"] == 2 and part["head_agrees"] == 1
