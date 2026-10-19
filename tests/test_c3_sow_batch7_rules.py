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


# --- 010147 B: the customer's verbs, the packed inventory, the transport mode -

def _move_job():
    c = _content()
    c["job_family"] = "relocation_move"
    c["scope"] = [{"text": "Network and servers", "kind": "heading", "cites": []},
                  {"text": "Pack five (5) network switches.", "cites": ["f1"]},
                  {"text": "Unrack and ready three (3) individual 1U-4U servers/devices.", "cites": ["f2"]},
                  {"text": "Pack twenty-five (25) to thirty (30) Yealink phones.", "cites": ["f3"]},
                  {"text": "Coordinate transportation of the packed equipment, including pickup at the origin location.", "cites": [], "standard": True}]
    c["customer_responsibilities"] = [{"text": "Unrack all applicable devices before packing begins.", "cites": [], "standard": True},
                                      {"text": "Unrack and ready approximately twenty (20) cables and related equipment.", "cites": ["f4"]}]
    c["out_of_scope"] = [{"text": "Powering down, disconnecting or unracking equipment at the origin location.", "cites": [], "standard": True}]
    c["equipment_in_scope"] = [{"text": "Five (5) network switches.", "cites": ["f1"]}]
    return c


def test_a_provider_step_takes_its_siblings_verb_when_the_customer_claims_its_own():
    c = _move_job()
    m._claimed_verbs(c)
    assert "Pack three (3) individual 1U-4U servers/devices." in _texts(c, "scope")


def test_a_claimed_step_with_no_sibling_verb_is_the_customers():
    c = _content()
    c["scope"] = [{"text": "Disconnect all workstations at the origin.", "cites": ["f1"]}]
    c["customer_responsibilities"] = [{"text": "Disconnect all equipment before the technicians arrive.", "cites": [], "standard": True}]
    m._claimed_verbs(c)
    assert c["scope"] == [] and "Disconnect all workstations at the origin." in _texts(c, "customer_responsibilities")


def test_a_generic_customer_verb_claims_nothing():
    c = _content()
    c["scope"] = [{"text": "Mount one (1) 9U network cabinet onto the backer board.", "cites": ["f1"]},
                  {"text": "Install the switch.", "cites": ["f2"]}]
    c["customer_responsibilities"] = [{"text": "Mount nothing until the walls are ready.", "cites": []}]
    m._claimed_verbs(c)
    assert _texts(c, "scope")[0] == "Mount one (1) 9U network cabinet onto the backer board."


def test_equipment_in_scope_names_everything_the_move_packs():
    c = _move_job()
    m._claimed_verbs(c)
    m._moved_equipment(c)
    assert _texts(c, "equipment_in_scope") == ["Five (5) network switches.", "Three (3) individual 1U-4U servers/devices.",
                                               "Twenty-five (25) to thirty (30) Yealink phones.",
                                               "Approximately twenty (20) cables and related equipment."]


def test_the_transport_step_names_the_requested_mode():
    c = _move_job()
    atoms = [{"id": "t1", "text": "Primary Wave Transportation Method: Secondary Wave Transportation Method | Shared Truck/LTL Shipment (5-7 days): Select transportation method"},
             {"id": "t2", "text": "States: Alabama | Select One Method: Dedicated Direct Truck/Climate Controlled"}]
    m._transport_mode(c, atoms)
    step = c["scope"][-1]
    assert step["text"] == ("Coordinate transportation of the packed equipment by Shared Truck/LTL Shipment (5-7 days), "
                            "including pickup at the origin location.")
    assert step["cites"] == ["t1"]


# --- 010264 C-: the program model, the sheet's count, prices, kit parts -------

def _program_env():
    docs = [{"artifact_id": "n1", "filename": "010264-hs-note-116624101737-pricing.txt"},
            {"artifact_id": "e1", "filename": "010264-hs-email-1.eml"}]
    texts = ["Pricing would be displayed as unit rates with total quantities TBD, reflecting client opt-in decisions",
             "Chase analyzed the location list and identified about 600 duplicate addresses out of 1500 clocks , indicating multiple devices at some sites",
             "Billing to be done monthly in arrears based on actual deployed clocks per location ( 09:24 )",
             "The team planned to define reasonable outreach limits, such as 2-3 calls per site, to control scope and avoid endless follow-ups",
             "Quinton preferred an email campaign combined with limited calls to reduce effort and cost, proposing a best-effort outreach model"]
    atoms = [{"id": f"p{i}", "artifact_id": "n1", "text": t, "structured": {"source": "hubspot_note", "c3_label": {"type": "_keep", "p": 0.3}}}
             for i, t in enumerate(texts)]
    atoms.append({"id": "e9", "artifact_id": "e1", "text": "We can opt in to anything you like, opt-in is easy.", "structured": {}})
    return {"documents": docs, "atoms": atoms}


def test_program_model_lines_come_only_from_our_own_notes():
    got = m._program_atoms(_program_env())
    assert [a["id"] for a in got] == ["p0", "p2", "p3", "p4", "p1"] or {a["id"] for a in got} == {"p0", "p1", "p2", "p3", "p4"}
    assert "e9" not in {a["id"] for a in got}


def test_the_program_model_is_written_from_its_facts():
    c = _content()
    c["scope"] = [{"text": "Installation", "kind": "heading", "cites": []},
                  {"text": "Install one clock at each address/location.", "cites": ["x"]},
                  {"text": "Closeout", "kind": "heading", "cites": []}]
    m._program_model(c, _program_env())
    sc, asm = _texts(c, "scope"), _texts(c, "assumptions")
    assert "Install one clock at each address/location." not in sc
    assert sc[:4] == ["Optional Site Outreach",
                      "When purchased, perform best-effort outreach to each location by email and limited calls to determine whether it opts in, "
                      "opts out or self-installs, or does not respond.",
                      "Limit outreach to 2-3 calls per site plus email.",
                      "Record each location as opted in, opted out (self-install) or no response on the central location list."]
    assert "Multi Clock Locations" in sc and sc.index("Multi Clock Locations") < sc.index("Closeout")
    assert "Some locations have more than one clock." in asm
    assert "Provider will invoice for the actual locations and clocks completed." in asm
    assert "The final quantities of locations and clocks are to be determined." in asm
    assert all(x["cites"] for x in c["assumptions"] + [y for y in c["scope"] if y.get("kind") != "heading"])


def test_one_passing_mention_is_no_program_model():
    env = _program_env()
    env["atoms"] = env["atoms"][:1]
    c = _content()
    m._program_model(c, env)
    assert c["assumptions"] == [] and c["scope"] == []


def test_pricing_talk_is_never_an_assumption():
    c = _content()
    c["assumptions"] = [{"text": "Pricing model discussed: about $350 for the first clock at a site.", "cites": ["f1"]},
                        {"text": "The additional clock rate applies only when installed in the same visit.", "cites": [], "standard": True},
                        {"text": "AC power is available at each site.", "cites": ["f2"]}]
    m._pricing_not_assumption(c)
    assert _texts(c, "assumptions") == ["The additional clock rate applies only when installed in the same visit.", "AC power is available at each site."]


def test_a_part_that_ships_with_the_customers_gear_is_theirs():
    c = _content()
    c["provider_materials"] = [{"text": "A power supply unit for each clock installed on standard AC power.", "cites": []}]
    m._ships_with_gear(c, [{"id": "a1", "text": "If Standard ac power is to be used a power supply will be provided with your time clock."}])
    assert c["provider_materials"] == [] and c["customer_materials"][0]["cites"] == ["a1"]


def test_a_dropped_exclusion_comes_back_citing_the_boundary_it_states():
    c = _content()
    c["dropped"] = [{"list": "out_of_scope", "text": "Onsite device configuration or provisioning beyond pre-shipment setup.", "cites": ["t1"],
                     "reason": "check: the cited lines do not state it"},
                    {"list": "out_of_scope", "text": "Painting of walls.", "cites": ["t2"], "reason": "check: the cited lines do not state it"}]
    atoms = [{"id": "a1", "text": "Standardize configuration and provisioning to ensure each clock is configured and ready before shipment."}]
    m._recite_exclusions(c, atoms)
    assert c["out_of_scope"] == [{"text": "Onsite device configuration or provisioning beyond pre-shipment setup.", "cites": ["a1"], "kind": "fact"}]
    assert [d["text"] for d in c["dropped"]] == ["Painting of walls."]


def test_a_pivot_tally_row_is_never_a_fact():
    env = {"c3_brief": {"sections": {"scope_of_work": [{"id": "a", "text": "Detail: Test Clock | Count: 1"},
                                                       {"id": "b", "text": "Install the replacement clocks at each site."}]}}, "atoms": []}
    assert [f["id"] for f in m.facts(env)] == ["b"]


# --- 010120 B-: every display stays, observations, the doubled address -------

def test_when_every_display_stays_no_line_places_a_display(monkeypatch):
    monkeypatch.setattr(m, "purtera_standards", lambda fam: {
        "method_steps": ["Review each intended display mounting location with the onsite contact before installation.",
                         "Mount each video bar at its designated location and connect it to its display."],
        "customer_responsibilities": ["Provide the final display locations and mounting heights before drilling or permanent installation."],
        "assumptions": ["Power receptacles and cabling are operational and long enough for the installed display positions."],
        "deliverable_outcomes": ["Displays mounted at the approved locations and heights.", "Completion confirmed with the onsite contact."]})
    c = _content()
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Displays mounted at the approved locations and heights."]}]
    facts = ["The two TVs in the front of the room will stay mounted in Current Position - No Change is needed for them.",
             "mounted on wall between both TV's"]
    m._apply_standards(c, "av_conference_room", facts)
    every = [t for t in _texts(c, "scope") if t != "Closeout"] + _texts(c, "customer_responsibilities") + _texts(c, "assumptions")
    assert every == ["Mount each video bar at its designated location and connect it to its display."]
    assert c["deliverables"][0]["description"] == ["Completion confirmed with the onsite contact."]


def test_displays_the_job_mounts_keep_their_placement_lines(monkeypatch):
    monkeypatch.setattr(m, "purtera_standards", lambda fam: {
        "method_steps": ["Review each intended display mounting location with the onsite contact before installation."]})
    c = _content()
    m._apply_standards(c, "av_conference_room", ["The old TVs will stay in the lobby.", "Mount two (2) new 65-inch displays in Room 4."])
    assert "Review each intended display mounting location with the onsite contact before installation." in _texts(c, "scope")


def test_a_survey_observation_is_not_a_task_or_an_assumption():
    c = _content()
    c["scope"] = [{"text": "There are two large TVs mounted on the front wall of the room.", "cites": ["a"]},
                  {"text": "One TV is wall-mounted with an articulating arm mount.", "cites": ["b"]},
                  {"text": "Site survey and assessment has already been completed by SHI's internal team.", "cites": ["c"]},
                  {"text": "Existing speakers are dropping down from the ceiling and are to be decommissioned.", "cites": ["d"]}]
    c["assumptions"] = [{"text": "One of the Neat devices is currently on a tripod.", "cites": ["e"]},
                        {"text": "There are four TVs that will remain in place on the floor.", "cites": ["f"]},
                        {"text": "There is no impact to production during the installation.", "cites": ["g"]}]
    m._observation_not_line(c)
    assert _texts(c, "scope") == ["Existing speakers are dropping down from the ceiling and are to be decommissioned."]
    assert _texts(c, "assumptions") == ["There are four TVs that will remain in place on the floor.",
                                        "There is no impact to production during the installation."]


def test_a_street_carrying_its_city_is_not_given_it_twice():
    got = m._sites({}, {"sites": [{"name": "6501 Legacy Dr, Plano, TX 75024, Plano, TX 75024", "address": "6501 Legacy Dr, Plano, TX 75024",
                                   "city": "Plano", "state": "TX", "postal_code": "75024"}]})
    assert got[0]["name"] == "6501 Legacy Dr, Plano, TX 75024" and got[0]["address"] == "6501 Legacy Dr, Plano, TX 75024"


# --- 010012 A- / 010115 A-: said once, every room, plain wording -------------

def test_a_standard_step_every_rooms_block_says_goes():
    c = _content()
    c["scope"] = [{"text": "Conference Room", "kind": "heading", "cites": []},
                  {"text": "Connect the Conference Room bar and controller to power and to the network or Wi-Fi the Customer provides.", "cites": ["a"]},
                  {"text": "Shared Office", "kind": "heading", "cites": []},
                  {"text": "Connect the Shared Office bar and controller to power and to the network or Wi-Fi the Customer provides.", "cites": ["b"]},
                  {"text": "All Work Areas", "kind": "heading", "cites": []},
                  {"text": "Connect each device to power and to the network or Wi-Fi the customer provides.", "cites": [], "standard": True}]
    m._one_statement(c)
    assert "Connect each device to power and to the network or Wi-Fi the customer provides." not in _texts(c, "scope")


def test_a_report_line_names_every_room_with_the_same_install():
    c = _content()
    c["scope"] = [{"text": "Conference Room", "kind": "heading", "cites": []},
                  {"text": "Install one (1) Poly X72 bar and one (1) TC10 controller in the Conference Room.", "cites": ["a"]},
                  {"text": "Shared Office", "kind": "heading", "cites": []},
                  {"text": "Install one (1) Poly X72 bar and one (1) TC10 controller in the Shared Office.", "cites": ["b"]}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": [
        "One (1) Poly X72 bar and one (1) TC10 controller installed and tested in the Conference Room"]}]
    m._report_every_room(c)
    assert c["deliverables"][0]["description"] == [
        "One (1) Poly X72 bar and one (1) TC10 controller installed and tested in the Conference Room and Shared Office"]


def test_plain_wording():
    c = _content()
    c["assumptions"] = [{"text": "All two (2) Poly X72 bars are on site.", "cites": []}]
    c["customer_responsibilities"] = [{"text": "Customer inform Provider in advance of any security protocols.", "cites": []},
                                      {"text": "Customer escort is required at eight (8) substations.", "cites": []}]
    m._plain_wording(c)
    assert _texts(c, "assumptions") == ["Both Poly X72 bars are on site."]
    assert _texts(c, "customer_responsibilities") == ["Customer will inform Provider in advance of any security protocols.",
                                                      "Customer escort is required at eight (8) substations."]


def test_an_uncited_line_a_standard_says_goes():
    c = _content()
    c["assumptions"] = [{"text": "All everyday POS hardware is powered down and released for removal before arrival.", "cites": []},
                        {"text": "Equipment is powered down and released for removal before the technician arrives.", "cites": [], "standard": True},
                        {"text": "Provide a pathway for the Tap location, or approve surface raceway.", "cites": ["t"]}]
    m._standard_says_it(c)
    assert _texts(c, "assumptions") == ["Equipment is powered down and released for removal before the technician arrives.",
                                        "Provide a pathway for the Tap location, or approve surface raceway."]


def test_an_exclusion_most_of_whose_items_are_excluded_elsewhere_keeps_the_rest():
    c = _content()
    c["out_of_scope"] = [{"text": "Transport, shipping or freight of packed equipment.", "standard": True},
                         {"text": "Inventory audit, asset reconciliation, disposal, recycling, data destruction, or certified IT asset disposition.", "standard": True},
                         {"text": "Packing materials, transportation, shipping or disposal of equipment.", "standard": True},
                         {"text": "Removal or disposal of existing monitors, arms, computers, furniture or unrelated site waste.", "standard": True}]
    m._exclusion_items_once(c)
    assert _texts(c, "out_of_scope")[2:] == ["Packing materials.",
                                             "Removal or disposal of existing monitors, arms, computers, furniture or unrelated site waste."]


def test_an_assumption_its_own_cite_assumes_comes_back():
    c = _content()
    c["dropped"] = [{"list": "assumptions", "text": "Packing materials are available onsite for boxing.", "cites": ["a1"],
                     "reason": "check: the cited lines do not state it"}]
    m._recite_exclusions(c, [{"id": "a1", "text": "Im so going to assume they will have all the packing materials onsite."}])
    assert c["assumptions"] == [{"text": "Packing materials are available onsite for boxing.", "cites": ["a1"], "kind": "fact"}]
    assert c["dropped"] == []
