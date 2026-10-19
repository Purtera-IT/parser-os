"""SOW writer rules, batch 7 (pass V grades: 010373 B, 010116 B-, 010083 B, 010147 B, 010264 C-)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


# --- 010373 B: a capability sentence is never a supply line ------------------

def test_a_capability_reply_is_not_a_provider_material():
    c = _content()
    c["provider_materials"] = [{"text": "Sixteen (16) monitor arms for the installation.", "cites": ["f1", "f2"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Provide all monitor arms, monitors, mounting hardware and cables.", "cites": [], "standard": True}]
    fs = [_fact("f1", "Each desk/arm will have 3 new 27\" monitors mounted to them for a total of 48 monitors."),
          _fact("f2", "We can handle the monitor arms and monitor installation.")]
    m._capability_not_supply(c, fs)
    assert c["provider_materials"] == []
    assert _texts(c, "customer_materials") == ["Sixteen (16) monitor arms for the installation."]


def test_a_capability_reply_with_no_customer_side_goes_to_equipment():
    c = _content()
    c["provider_materials"] = [{"text": "Four (4) access points.", "cites": ["f1"], "kind": "fact"}]
    m._capability_not_supply(c, [_fact("f1", "We can install the four access points next week.")])
    assert c["provider_materials"] == [] and _texts(c, "equipment_in_scope") == ["Four (4) access points."]


def test_a_line_saying_we_supply_stays_ours():
    c = _content()
    c["provider_materials"] = [{"text": "Cat6 cable for the four drops.", "cites": ["f1", "f2"], "kind": "fact"}]
    fs = [_fact("f1", "We can handle the cabling."), _fact("f2", "Our tech will provide the Cat6 cable.")]
    m._capability_not_supply(c, fs)
    assert _texts(c, "provider_materials") == ["Cat6 cable for the four drops."]


def test_one_supplier_reads_a_spelled_count_and_a_procurement_exclusion():
    c = _content()
    c["provider_materials"] = [{"text": "Sixteen (16) monitor arms for the installation.", "cites": ["f1"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Provide all monitor arms, monitors, mounting hardware and cables.", "cites": [], "standard": True}]
    c["out_of_scope"] = [{"text": "Procurement of monitors, arms or accessories.", "cites": [], "standard": True}]
    m._one_supplier(c, [_fact("f1", "16 monitor arms on new desks")])
    assert c["provider_materials"] == [] and _texts(c, "customer_materials") == ["Sixteen (16) monitor arms for the installation."]


def test_a_counted_step_in_a_standard_slot_keeps_the_steps_later_work(monkeypatch):
    monkeypatch.setattr(m, "purtera_standards", lambda fam: {"method_steps": ["Mount the monitors on each arm, then align and secure them."]})
    c = _content()
    c["scope"] = [{"text": "Mount forty-eight (48) new 27-inch monitors on the installed arms, three (3) per desk/arm.", "cites": ["f1"], "kind": "fact"}]
    m._apply_standards(c, "smart_hands_general", ["48 monitors mounted, 3 per arm"])
    assert "Mount forty-eight (48) new 27-inch monitors on the installed arms, three (3) per desk/arm, then align and secure them." in _texts(c, "scope")


# --- 010279 B+: who a split RFP sentence names, reuse, the report ------------

def _atom(i, text, line, sp, art="a1"):
    return {"id": i, "text": text, "artifact_id": art, "section_path": sp, "locator": {"line_start": line}}


def test_a_split_rfp_sentence_keeps_its_subject():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide a resource at the corporate office for assistance and escalation as needed.", "cites": ["f1"]}]
    f = _fact("f1", "This work will require no more than 2 tech POS Installer team. Harris Teeter will provide a resource at the "
                    "HT Corporate Office for assistance and escalation as needed.")
    f["supplier"] = ("us", 0.95)
    m._duty_party(c, [f], {"atoms": []})
    assert _texts(c, "customer_responsibilities") == ["Provide a resource at the corporate office for assistance and escalation as needed."]
    assert c["scope"] == []


def test_a_supplier_head_duty_with_no_named_party_is_ours():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide proof of delivery for each site.", "cites": ["f1"]}]
    f = _fact("f1", "Proof of delivery for each site.")
    f["supplier"] = ("us", 0.95)
    m._duty_party(c, [f], {"atoms": []})
    assert _texts(c, "scope") == ["Provide proof of delivery for each site."]


def test_a_duty_under_a_technician_lead_line_is_the_technicians():
    sp = ["2.3 Post-Installation"]
    atoms = [_atom("h", "2.3 Post-Installation", 59, sp), _atom("l", "Technician to provide post installation support", 60, sp),
             _atom("f1", "Verifies all tasks complete and store ready for business", 61, sp)]
    c = _content()
    c["scope"] = [{"text": "Closeout", "kind": "heading", "cites": []}, {"text": "Confirm completion.", "cites": [], "standard": True}]
    c["customer_responsibilities"] = [{"text": "Verify all tasks complete and store ready for business.", "cites": ["f1"]}]
    m._duty_party(c, [_fact("f1", "Verifies all tasks complete and store ready for business")], {"atoms": atoms})
    assert c["customer_responsibilities"] == []
    assert _texts(c, "scope")[-1] == "Verify all tasks complete and store ready for business."


def test_a_connector_pin_count_is_not_a_secret_and_the_report_never_shows_one():
    assert "[withheld]" not in m._withhold("Toshiba 12V Powered USB to 6 Pin | 42M5630 | 8")
    assert "[withheld]" in m._withhold("Door PIN: 4471")
    c = {"deliverables": [{"item": "Work Completion Report", "description": ["8 cables installed.", "Gate code PIN: 4471 used [withheld]"]}]}
    m._scrub(c)
    assert c["deliverables"][0]["description"] == ["8 cables installed."]


def test_a_removal_never_takes_what_the_job_reuses():
    c = _content()
    c["scope"] = [{"text": "Disconnect and remove the existing POS terminals and peripherals as specified.", "cites": ["f1"]}]
    c["assumptions"] = [{"text": "Existing power, network, and peripheral connections are operational and can be reused.", "cites": []}]
    m._reused_not_removed(c, [])
    assert _texts(c, "scope") == ["Disconnect and remove the existing POS terminals as specified."]


def test_the_report_states_outcomes_not_a_bill_of_materials():
    c = _content()
    c["scope"] = [{"text": "Install 242 Toshiba Garage Units and 242 cash drawer cables.", "cites": ["f1"]}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["242 Toshiba Garage Units", "8 Toshiba 2X20 TCx Displays",
                                                                             "Completion photos of each installed location."]}]
    m._report_counts_are_outcomes(c)
    assert c["deliverables"][0]["description"] == ["242 Toshiba Garage Units installed.", "Completion photos of each installed location."]


def test_a_long_dropped_line_does_not_strike_an_outcome_it_only_mentions():
    c = _content()
    c["job_family"] = "pos_install_refresh"
    c["rule_decisions"] = [{"value": "drop", "text": "Provider will de-install the existing terminals, apply asset tags, record terminal MAC "
                                                     "addresses by store, palletize old hardware for return/disposal, and complete "
                                                     "installation and testing checklists."}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Completed Customer installation and testing checklists."]}]
    m._deliverables_hygiene(c)
    assert c["deliverables"][0]["description"] == ["Completed Customer installation and testing checklists."]


def test_an_unsupported_copy_of_a_live_step_does_not_strike_its_outcome():
    c = _content()
    c["job_family"] = "pos_install_refresh"
    c["scope"] = [{"text": "Complete the Customer-provided installation and testing checklists.", "cites": [], "standard": True}]
    c["dropped_unsupported"] = [{"text": "Complete the Customer-provided installation and testing checklists."}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Completed Customer installation and testing checklists."]}]
    m._deliverables_hygiene(c)
    assert c["deliverables"][0]["description"] == ["Completed Customer installation and testing checklists."]


# --- 010116 B-: rooms from the sheet, BOM lists whole and in one form --------

def test_a_row_under_another_areas_heading_moves_to_its_own_area():
    atoms = [{"id": "b1", "text": "1: QM75C Samsung 75IN 4K UHD LED Display", "locator": {"section_path": ["Sheet1", "Boardroom"]}},
             {"id": "c1", "text": "2: QM65C Samsung 65IN 4K UHD LED Display", "locator": {"section_path": ["Sheet1", "Common Area Displays"]}}]
    c = _content()
    c["scope"] = [{"text": "Boardroom", "kind": "heading", "cites": []},
                  {"text": "Install one (1) Samsung 75-inch QM75C display.", "cites": ["b1"]},
                  {"text": "Install two (2) Samsung 65-inch QM65C displays.", "cites": ["c1"]},
                  {"text": "Mount each display at the approved location and height.", "cites": [], "standard": True},
                  {"text": "Closeout", "kind": "heading", "cites": []},
                  {"text": "Confirm completion.", "cites": [], "standard": True}]
    m._room_from_section(c, {"atoms": atoms})
    assert _texts(c, "scope") == ["Boardroom", "Install one (1) Samsung 75-inch QM75C display.",
                                  "Common Area", "Install two (2) Samsung 65-inch QM65C displays.",
                                  "All work areas", "Mount each display at the approved location and height.",
                                  "Closeout", "Confirm completion."]


def test_bom_rows_read_fused_sheet_rows():
    assert m._bom_rows("2: 2 | QM65C: TS525TU | Samsung 65IN Display: Chief Wall Mount") == [
        (2, "QM65C", "Samsung 65IN Display"), (2, "TS525TU", "Chief Wall Mount")]
    assert m._bom_rows("1: CS-BAR-T-C-K9 Cisco Room Bar in Carbon Black w/Navigator -36mo") == [
        (1, "CS-BAR-T-C-K9", "Cisco Room Bar in Carbon Black w/Navigator")]
    assert m._bom_rows("Each desk/arm will have 3 new 27 inch monitors") == []


def test_materials_hold_every_cited_bom_row_in_one_form():
    atoms = [{"id": "r1", "text": "33 C9350-48U with Stack module and cable"},
             {"id": "r2", "text": "69 C9350-48U with Stack module, stack cable, and 4 SFP uplinks"},
             {"id": "r3", "text": "82 C9350-49U no stack and no SFP"},
             {"id": "n1", "text": "CDW will acquire, config and ship all equipment."}]
    c = _content()
    c["scope"] = [{"text": "Install thirty-three (33) C9350-48U switches.", "cites": ["r1"]},
                  {"text": "Install sixty-nine (69) C9350-48U switches.", "cites": ["r2"]},
                  {"text": "Install eighty-two (82) C9350-49U switches.", "cites": ["r3"]}]
    c["customer_materials"] = [{"text": "All replacement switch equipment and accessories.", "cites": ["n1", "r1", "r2"]},
                               {"text": "33 C9350-48U with Stack module and cable", "cites": ["r1"]}]
    m._bom_lists(c, atoms)
    assert _texts(c, "customer_materials") == ["All replacement switch equipment and accessories.",
                                               "33 x C9350-48U with Stack module and cable",
                                               "69 x C9350-48U with Stack module, stack cable, and 4 SFP uplinks",
                                               "82 x C9350-49U no stack and no SFP"]


def test_no_bom_row_line_means_no_bom_list():
    atoms = [{"id": "r1", "text": "33 C9350-48U with Stack module and cable"}]
    c = _content()
    c["scope"] = [{"text": "Install thirty-three (33) C9350-48U switches.", "cites": ["r1"]}]
    m._bom_lists(c, atoms)
    assert c["customer_materials"] == []


def test_the_report_lists_every_installed_row_in_the_same_form():
    atoms = [{"id": "r1", "text": "1: QM75C Samsung 75IN Display"}, {"id": "r2", "text": "1: SW4-TPS-TX240 Lightware TPS transmitter"},
             {"id": "r3", "text": "2: QM65C Samsung 65IN Display"}]
    c = _content()
    c["scope"] = [{"text": "Install the displays and the transmitter.", "cites": ["r1", "r2", "r3"]}]
    c["customer_materials"] = [{"text": "1: QM75C Samsung 75IN Display", "cites": ["r1"]}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": [
        "Two (2) Samsung 65-inch QM65C displays installed", "Completion confirmed with the onsite contact."]}]
    m._bom_lists(c, atoms)
    assert c["deliverables"][0]["description"] == ["1 x QM75C Samsung 75IN Display installed.", "1 x SW4-TPS-TX240 Lightware TPS transmitter installed.",
                                                   "2 x QM65C Samsung 65IN Display installed.", "Completion confirmed with the onsite contact."]


# --- 010083 B: a merge keeps the deal's nouns; site mix; items not tasks -----

def test_a_guarded_step_whose_nouns_the_standard_lacks_comes_back_instead_of_merging():
    c = _content()
    deal = {"text": "Migrate trunks and AP ports according to the Customer-provided minimal matrix.", "cites": ["f1"], "kind": "fact"}
    std = {"text": "Migrate patch cable connections according to the Customer-provided port and cable matrix.", "cites": [], "standard": True}
    before = m._snap({**c, "scope": [deal, std]})
    lists_before = {k: [] for k in m._LIST_SECTION}
    lists_before["scope"] = [deal, std]
    c["scope"] = [std]
    out = m._guard_drops(c, "standards", before, lists_before)
    assert [o["guard"] for o in out] == ["restored"]
    assert deal["text"] in _texts(c, "scope")


def test_a_per_site_quantity_mix_is_written_whole():
    c = _content()
    c["assumptions"] = [{"text": "Treat per-site switch quantities as variable across the approximately ninety-two (92) locations.", "cites": ["f1"]}]
    f = _fact("f1", "Are the switches generally two (2) per site, or will quantities vary by location? [Betty] Apx 83 sites will be singles.  "
                    "Memphis Main will be apx. 51 switches on multiple floors.  The remainder will be 2-3 switches each.", section="assumptions")
    m._planning_basis(c, [f])
    assert _texts(c, "assumptions")[-1] == ("The Customer has indicated that approximately 83 sites will be singles; Memphis Main will be "
                                            "approximately 51 switches on multiple floors; the remainder will be 2-3 switches each.")


def test_a_task_or_a_document_is_never_equipment():
    c = _content()
    c["scope"] = [{"text": "Support completion of the Customer-provided post-install testing checklist.", "cites": [], "standard": True}]
    c["customer_materials"] = [{"text": "Support the Customer-provided post-install testing checklist.", "cites": ["f1"]},
                               {"text": "The site list with the site address and the quantity of time clocks at the site.", "cites": ["f2"]},
                               {"text": "Forty-eight (48) monitors.", "cites": ["f3"]}]
    m._items_not_tasks(c)
    assert _texts(c, "customer_materials") == ["Forty-eight (48) monitors."]
    assert _texts(c, "customer_responsibilities") == ["Provide the site list with the site address and the quantity of time clocks at the site."]
    assert len(c["scope"]) == 1


def test_a_duration_is_not_a_completion_date():
    c = _content()
    c["schedule"] = [{"text": "Requested Completion Date: not to exceed five (5) months.", "cites": ["f1"]},
                     {"text": "Requested Completion Date: January 31, 2027", "cites": ["f2"]}]
    m._duration_not_date(c)
    assert _texts(c, "schedule") == ["Project Duration: not to exceed five (5) months.", "Requested Completion Date: January 31, 2027"]
