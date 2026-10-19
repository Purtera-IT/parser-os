"""SOW writer rules, batch 3: the drop guard's holes (010217), exclusions that
are not requests (010284, 010147), third-party supply and terms (010376),
cited counts and the after-hours basis (010083), pack-and-move jobs (010147)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def test_guard_restores_a_line_whose_other_cite_is_a_quoted_copy():
    """010217: tidy dropped the 86" install; one of its cites was the email's
    quoted copy (noise), so the guard let the whole line go."""
    line = {"text": "Install two (2) new 86 inch displays in training room 250.", "cites": ["note", "quoted"], "kind": "fact"}
    content = {"_noise_atoms": ["quoted"],
               "scope": [{"text": "Training Room 250", "cites": [], "kind": "heading"}, line,
                         {"text": "Room 247", "cites": [], "kind": "heading"},
                         {"text": "Install the 98 inch display.", "cites": ["r247"], "kind": "fact"},
                         {"text": "Closeout", "cites": [], "kind": "heading"}]}
    m._traced(content, "tidy", lambda: content.__setitem__("scope", [x for x in content["scope"] if x is not line]))
    texts = [x["text"] for x in content["scope"]]
    # back under its own heading, not before Closeout
    assert texts.index(line["text"]) == texts.index("Training Room 250") + 1
    assert content["rule_decisions"][0]["guard"] == "restored"


def test_a_fact_only_the_summary_cites_is_not_in_the_task_list():
    """010217: the 98" install's atom lived on in the overview only."""
    line = {"text": "Install one (1) new 98 inch display in room 247 with its wall mount.", "cites": ["a98"], "kind": "fact"}
    content = {"overview": [{"text": "Provider will install a 98-inch display in room 247.", "cites": ["a98"], "kind": "fact"}],
               "scope": [line, {"text": "Closeout", "cites": [], "kind": "heading"}]}
    m._traced(content, "tidy", lambda: content.__setitem__("scope", content["scope"][1:]))
    assert content["scope"][0] is line


def test_a_summary_line_leaning_on_a_quoted_line_is_still_droppable():
    content = {"_noise_atoms": ["q"],
               "overview": [{"text": "Provider will manage receipt, bench and disposal.", "cites": ["q", "r"], "kind": "fact"}]}
    m._traced(content, "grounding", lambda: content.__setitem__("overview", []))
    assert content["overview"] == []


def test_an_exclusion_citing_unrelated_chatter_stays_an_exclusion():
    """010284: "Lift rental, boom-lift services ..." cited pole talk and became
    "Provide lift rental ..." in the task list."""
    content = {"overview": [], "scope": [],
               "out_of_scope": [{"text": "Lift rental, boom-lift services, and other specialty access equipment.",
                                 "cites": ["p"], "kind": "fact"}]}
    m._grounding(content, [_fact("p", "That could be on a 60 foot pole and we have to bring out a boom lift.")])
    assert not any("lift" in x["text"].lower() for x in content["scope"])
    assert content["out_of_scope"][0]["text"].startswith("Lift rental")
    assert content["out_of_scope"][0]["cites"] == []


def test_a_requested_service_flipped_into_an_exclusion_is_still_scope():
    content = {"overview": [], "scope": [],
               "out_of_scope": [{"text": "End-user support for O365 clients is excluded from the service scope.",
                                 "cites": ["s"], "kind": "fact"}]}
    m._grounding(content, [_fact("s", "End-user support for O365 clients")])
    assert content["scope"] and content["scope"][0]["text"].startswith("Provide end-user support")


def test_a_zero_quantity_row_is_an_exclusion_never_a_task():
    """010147: "6U-10U servers x 0" became "Provide 6U-10U servers are not included"."""
    content = {"overview": [], "scope": [],
               "out_of_scope": [{"text": "6U-10U servers are not included.", "cites": ["z"], "kind": "fact"}]}
    m._grounding(content, [_fact("z", "6U-10U servers x 0")])
    assert not content["scope"]
    assert content["out_of_scope"][0]["text"].startswith("6U-10U servers")


def _av_scope():
    return {"job_family": "av_conference_room",
            "scope": [{"text": "Training Room 250", "cites": [], "kind": "heading"},
                      {"text": "Remove the two (2) existing 55 inch displays for relocation to the NOC.", "cites": ["r"], "kind": "fact"},
                      {"text": "Room 247", "cites": [], "kind": "heading"},
                      {"text": "Install one (1) new 98 inch display with its wall mount.", "cites": ["n"], "kind": "fact"},
                      {"text": "Closeout", "cites": [], "kind": "heading"}]}


_FACTS_217 = ["They have (2) 55 inch Philips displays mounted on a backwall. They want to replace these with (2) larger displays.",
              "Those (2) 55 inch Philips displays would then need to be installed in their NOC (same location).",
              "The existing mounts should be to be reused as they support 55-100 inch displays.",
              "Lastly, they are buying (1) new 98 inch Philips display and need it installed in room 247. This will also need a backing board installed."]


def test_reused_mounts_do_not_rule_out_mounting_the_new_displays():
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", _FACTS_217)
    texts = [x["text"] for x in content["scope"]]
    assert "Mount each display at the approved location and height." in texts
    assert "Adjust the mounts for alignment and spacing." in texts


def test_relocated_displays_are_not_decommissioned():
    """010217: "replace these" fired the removal/staging lines while the
    displays move to the NOC."""
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", _FACTS_217)
    every = " ".join(x["text"] for k in ("scope", "customer_responsibilities") for x in content.get(k) or [])
    assert "designated for removal" not in every and "Stage removed equipment" not in every
    assert "approved for removal" not in every
    assert "Transport the relocated displays" in every
    # a real removal still decommissions
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", ["Remove the existing projector and install a new 75 inch display."])
    assert any("designated for removal" in x["text"] for x in content["scope"])


def test_steps_for_every_room_get_their_own_heading():
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", _FACTS_217)
    texts = [x["text"] for x in content["scope"]]
    room247 = texts.index("Room 247")
    dress = texts.index("Dress and secure accessible cabling associated with each display.")
    assert "All Work Areas" in texts[room247:dress]


def _env_376():
    atoms = [{"id": "t1", "artifact_id": "prop", "text": "Box Truck w/ Liftgate", "locator": {}},
             {"id": "t2", "artifact_id": "prop", "text": "ThinkTLS will provide dedicated vehicles for the secure transportation of equipment.", "locator": {}},
             {"id": "t3", "artifact_id": "prop", "text": "Recycling, certificates and ESG reporting are done offsite.", "locator": {}},
             {"id": "k1", "artifact_id": "mail1", "text": "The commodity value for UPS's is only $.15/pound, so your client would have to cut a PO.",
              "locator": {"sender": "kgrove@thinktls.com"}},
             {"id": "q1", "artifact_id": "quote", "text": "*Tax is estimated. Invoice will include the full and final tax due.", "locator": {}},
             {"id": "c1", "artifact_id": "mail2", "text": "Replace a 5K UPS and eWaste the old UPS.", "locator": {"sender": "fhasan@guidewire.com"}},
             {"id": "p1", "artifact_id": "mail3", "text": "Bring a step ladder.", "locator": {"sender": "aj@purtera-it.com"}}]
    docs = [{"artifact_id": "prop", "filename": "Exton PA - UPS Pickup Proposal Final.pdf"},
            {"artifact_id": "quote", "filename": "SHI Quote-28013789.pdf"}]
    return {"atoms": atoms, "documents": docs, "crm": {"account_name": "SHI", "deal_name": "Guidewire UPS"}}


def test_a_third_partys_supply_terms_and_work_are_never_providers():
    """010376: the recycler's truck, vehicles and packing supplies were
    Technician Provided Tools; its $/lb note and the hardware quote's tax line
    were PurTera assumptions; the overview had Provider eWaste the units."""
    env = _env_376()
    fs = [_fact(a["id"], a["text"], supplier=["us", 0.9]) for a in env["atoms"]]
    fs[0]["supplier"] = None
    content = {"end_customer": "Guidewire",
               "overview": [{"text": "Provider will replace the UPS and eWaste the removed UPS after the upgrade.", "cites": ["c1"], "kind": "fact"}],
               "scope": [{"text": "Remove the existing UPS for eWaste.", "cites": ["c1"], "kind": "fact"}],
               "provider_tools": [{"text": "A box truck with a liftgate for transport.", "cites": ["t1"], "kind": "fact"},
                                  {"text": "Dedicated vehicles for secure transportation.", "cites": ["t2"], "kind": "fact"},
                                  {"text": "A step ladder.", "cites": ["p1"], "kind": "fact"}],
               "provider_materials": [{"text": "Packing materials.", "cites": ["zz"], "kind": "fact"}],
               "assumptions": [{"text": "The UPS units have a commodity value of $0.15/lb.", "cites": ["k1"], "kind": "fact"},
                               {"text": "Tax is estimated; the invoice will include the full and final tax due.", "cites": ["q1"], "kind": "fact"}],
               "out_of_scope": []}
    m._traced(content, "third_party", m._third_party, content, env, fs)
    assert [x["text"] for x in content["provider_tools"]] == ["A step ladder."]
    assert content["provider_materials"] == []  # no supplier answer at all
    asm = [x["text"] for x in content["assumptions"]]
    assert "ThinkTLS provides a box truck with a liftgate for transport." in asm
    assert not any("commodity" in t or "Tax" in t for t in asm)
    assert any(t.startswith("ThinkTLS performs the offsite transport and recycling") for t in asm)
    assert "eWaste" not in content["overview"][0]["text"] and "eWaste" not in content["scope"][0]["text"]
    assert any("recycling" in x["text"] for x in content["out_of_scope"])
    assert any(d["rule"] == "third_party" for d in content["rule_decisions"])


def test_no_third_party_no_change():
    env = {"atoms": [{"id": "a", "artifact_id": "m", "text": "We will provide the cable.", "locator": {"sender": "x@cdw.com"}}],
           "crm": {"account_name": "CDW"}}
    content = {"provider_materials": [{"text": "Cat6 cable.", "cites": ["a"], "kind": "fact"}], "assumptions": []}
    m._third_party(content, env, [_fact("a", "We will provide the cable.", supplier=["us", 0.7])])
    assert content["provider_materials"][0]["text"] == "Cat6 cable."


def test_an_acronym_fact_the_writer_left_out_goes_to_the_must_place_pass(monkeypatch):
    """010376: "they will need PurTera's COI" was kept by the brief, silently dropped."""
    seen = {}
    monkeypatch.setattr(m, "_complete", lambda prompt, *a, **k: seen.setdefault("p", prompt) and '{"add": []}')
    content = {k: [] for k in m.LISTS}
    content["scope"] = [{"text": "Install the UPS.", "cites": ["u"], "kind": "fact"}]
    m._must_include(content, [_fact("coi", "Also, they will need PurTera's COI (certificate of insurance) if you are able to provide that.",
                                    section="customer_responsibilities")], None)
    assert "COI" in seen.get("p", "")


def test_ups_replacement_standards_gate_on_a_ups_swap_only():
    fixed = m.purtera_standards("smart_hands_general")
    ups = [t for t in fixed["method_steps"] if "UPS" in t]
    assert ups and all(m._standard_applies(t, ["Replace a 5K UPS (Power input connector type L6-30P)"])[0] for t in ups)
    assert not any(m._standard_applies(t, ["Action items: follow-ups by end of week"])[0] for t in ups)


_SINGLES = "[Betty] Apx 83 sites will be singles.  Memphis Main will be apx. 51 switches on multiple floors.  The remainder will be 2-3 switches each."


def test_a_count_matches_its_cited_line():
    """010083: "About ninety-two (92) sites will be single-switch locations"
    cited "Apx 83 sites will be singles"; 92 is the site total."""
    c = {"overview": [{"text": "About ninety-two (92) sites will be single-switch locations.", "cites": ["a"], "kind": "fact"},
                      {"text": "Install 92 switches at the sites.", "cites": ["a"], "kind": "fact"}]}
    m._cited_counts(c, {"atoms": [{"id": "a", "text": _SINGLES}]})
    assert c["overview"][0]["text"] == "About eighty-three (83) sites will be single-switch locations."
    assert c["overview"][1]["text"] == "Install 92 switches at the sites."  # a total is not a sub-count


def test_the_site_count_snap_never_overrides_a_cited_count(monkeypatch):
    monkeypatch.setattr(m, "_best_of", lambda e, pm: {"project_name": "Switch refresh", "overview": [
        {"text": "About eighty-three (83) sites will be single-switch locations.", "cites": ["a"], "kind": "fact"},
        {"text": "Provider will refresh switches at approximately 90 locations.", "cites": ["b"], "kind": "fact"}]})
    monkeypatch.setattr(m, "requested_by", lambda e: {})
    env = {"atoms": [{"id": "a", "text": _SINGLES}, {"id": "b", "text": "a switch refresh at about 90 locations"}],
           "site_readiness": {"sites": []}, "site_count_reconciliation": {"stated": 92}}
    c = m.for_envelope(env)
    texts = [x["text"] for x in c["overview"]]
    assert "About eighty-three (83) sites will be single-switch locations." in texts


def test_after_hours_only_replaces_the_business_hours_basis():
    content = {"schedule": [{"text": "Work Window: after hours, no weekends, no overnights.", "cites": ["w"], "kind": "fact"}],
               "assumptions": [{"text": "All Services will be performed during normal Business Hours.", "cites": [], "kind": "method"}],
               "customer_responsibilities": [{"text": "Confirm which areas may be completed during business hours and which require after-hours windows.",
                                              "cites": [], "kind": "method", "standard": True}]}
    m._hours_basis(content, [])
    assert [x["text"] for x in content["assumptions"]] == ["All Services will be performed After Hours during the week."]
    assert content["customer_responsibilities"] == []


def test_switch_standards_follow_the_after_hours_and_serial_facts():
    fixed = m.purtera_standards("network_switch_refresh")
    facts = ["Will installs be performed during normal business hours, after hours, overnight, or weekends? [Betty] After hours, no weekends, no overnights.",
             "What closeout documentation is required per site? - before/after pictures and serial numbers of the new switches."]
    cr = [t for ok, t in (m._standard_applies(x, facts) for x in fixed["customer_responsibilities"]) if ok]
    assert not any("during business hours" in t for t in cr)
    out = [t for ok, t in (m._standard_applies(x, facts) for x in fixed["deliverable_outcomes"]) if ok]
    assert "Serial numbers of each new switch installed." in out
    assert not any("if requested" in t for t in out)


def test_what_the_technician_does_is_not_a_customer_duty():
    content = {"customer_responsibilities": [{"text": "Customer remote resource will provide verbal validation at completion, and the "
                                                      "technician may leave once the posted completion time is reached.", "cites": ["v"], "kind": "fact"}]}
    m._tidy(content)
    assert content["customer_responsibilities"][0]["text"] == "Customer remote resource will provide verbal validation at completion."


def test_pace_and_arrival_lead_time_reach_the_schedule():
    content = {"schedule": []}
    fs = [_fact("s", "Hoping for late Aug or Sept start with 8 scheduled a week not to exceed 5 months.", section="timeline"),
          _fact("r", "Techs will be requested to arrive no less than 30 minutes before closing time.")]
    m._schedule_from_facts(content, fs, {"atoms": []})
    texts = [x["text"] for x in content["schedule"]]
    assert "Pace: eight (8) sites per week" in texts
    assert "Arrival Time: no less than 30 minutes before closing time" in texts


def test_a_documentation_duty_naming_serials_stays_the_customers():
    content = {"customer_responsibilities": [{"text": "Provide required documentation for the site, including cable diagrams and serial numbers.",
                                              "cites": ["d"], "kind": "fact"}]}
    m._tidy(content)
    assert content["customer_responsibilities"] and not content.get("deliverables")


def test_a_cited_step_is_covered_only_by_a_standard_with_its_verb():
    content = {"scope": [{"text": "Install", "cites": [], "kind": "heading"},
                         {"text": "Label all cables before removing the old switches.", "cites": ["l"], "kind": "fact"},
                         {"text": "Closeout", "cites": [], "kind": "heading"}]}
    m._apply_standards(content, "network_switch_refresh", ["Replace the old switches."])
    assert any(x["text"].startswith("Label all cables") for x in content["scope"])


def test_zero_quantity_rows_make_no_line():
    env = {"atoms": [{"id": "z1", "text": "6U-10U Server Quantity: 0"}, {"id": "z2", "text": "# of Complete SAN Frame/Cabinets: 0"},
                     {"id": "z3", "text": "# of Complete Rack Tape Libraries: 0"}, {"id": "h", "text": "If moving in Complete Cabinets"},
                     {"id": "s", "text": "1U-4U Server Quantity: 3"}]}
    content = {"scope": [{"text": "Leave complete cabinets out of scope.", "cites": ["z2", "z3", "h"], "kind": "fact"},
                         {"text": "Pack the three (3) 1U-4U servers.", "cites": ["s"], "kind": "fact"}],
               "out_of_scope": [{"text": "6U-10U servers are not included.", "cites": ["z1"], "kind": "fact"}]}
    m._traced(content, "zero_rows", m._zero_rows, content, env)
    assert [x["text"] for x in content["scope"]] == ["Pack the three (3) 1U-4U servers."]
    assert content["out_of_scope"] == []


def test_what_the_customer_says_it_will_do_is_its_duty():
    """010147: the questionnaire's "We will also unrack and ready any cables
    and equipment" became a Provider task."""
    env = {"crm": {"account_name": "CDW"},
           "atoms": [{"id": "q", "text": "No: There will not be a need for a cable map. We will manage things at the destination. "
                                         "We will also unrack and ready any cables and equipment.", "locator": {}},
                     {"id": "r", "text": "We will mount the NUC under the table.", "locator": {"sender": "rep@cdw.com"}}]}
    content = {"scope": [{"text": "Unrack and ready any cables and equipment as part of the pack-and-deliver work.", "cites": ["q"], "kind": "fact"},
                         {"text": "Mount the NUC under the table.", "cites": ["r"], "kind": "fact"}],
               "customer_responsibilities": []}
    m._traced(content, "first_person", m._first_person_duties, content, env)
    assert [x["text"] for x in content["scope"]] == ["Mount the NUC under the table."]  # the reseller's "we" is the delivery team
    assert content["customer_responsibilities"][0]["text"] == "Unrack and ready any cables and equipment."


def test_assumptions_are_conditions_not_orders():
    content = {"assumptions": [{"text": "Direct what is packed, staged, and delivered during the move.", "cites": [], "kind": "method"},
                               {"text": "Avoid any lift rental or specialty rigging for this job.", "cites": [], "kind": "method"},
                               {"text": "Require no lift rental, hoisting, or other specialized moving equipment for this job.", "cites": [], "kind": "method"},
                               {"text": "Existing walls are suitable.", "cites": [], "kind": "method"}],
               "out_of_scope": [{"text": "Lift rental or specialty rigging.", "cites": [], "kind": "method"}],
               "customer_responsibilities": []}
    m._assumption_voice(content)
    assert [x["text"] for x in content["assumptions"]] == [
        "No lift rental, hoisting, or other specialized moving equipment is required.", "Existing walls are suitable."]
    assert content["customer_responsibilities"][0]["text"].startswith("Direct what is packed")


def test_moved_gear_is_not_customer_material_and_the_destination_stays_the_customers():
    content = {"customer_materials": [{"text": "Twenty-five (25) to thirty (30) curved monitors.", "cites": ["m"], "kind": "fact"},
                                      {"text": "Packing tape and pallets.", "cites": ["p"], "kind": "fact"},
                                      {"text": "The networking rack.", "cites": ["r"], "kind": "fact"}],
               "scope": [{"text": "Unload and stage the equipment at the destination.", "cites": [], "kind": "method"},
                         {"text": "Deliver the packed equipment to the destination dock.", "cites": ["d"], "kind": "fact"}]}
    m._moved_gear(content, "relocation_move")
    assert [x["text"] for x in content["equipment_in_scope"]] == ["Twenty-five (25) to thirty (30) curved monitors."]
    assert len(content["customer_materials"]) == 2
    m._destination_owner(content, [_fact("q", "We will manage things at the destination.")])
    assert [x["text"] for x in content["scope"]] == ["Deliver the packed equipment to the destination dock."]


def test_a_contact_whose_designate_is_unnamed_is_not_the_check_in():
    content = {"scope": [{"text": "Check in with Bill Tripp at 1875 Lawrence Street.", "cites": ["o"], "kind": "fact"}],
               "customer_responsibilities": []}
    m._onsite_designate(content, [_fact("o", "Origin Address: Point of Contact | 1875 Lawrence Street, Suite 750: Bill Tripp - Onsite designate to be named.")])
    assert content["scope"][0]["text"] == "Check in with the onsite designate named by Bill Tripp at 1875 Lawrence Street."
    assert "onsite designate" in content["customer_responsibilities"][0]["text"]


def test_a_number_inside_a_date_is_not_the_count_written():
    seen = {}
    import json as _j
    def fake(prompt, *a, **k):
        seen["p"] = prompt
        return _j.dumps({"add": []})
    m._complete, old = fake, m._complete
    try:
        content = {k: [] for k in m.LISTS}
        content["schedule"] = [{"text": "Requested Start Date: 2026-08-25", "cites": ["d"], "kind": "fact"}]
        m._must_include(content, [_fact("c", "Estimated # of Cables: 20", section="scope_of_work")], None)
    finally:
        m._complete = old
    assert "Estimated # of Cables: 20" in seen.get("p", "")


def test_relocation_tools_are_move_tools_unless_the_move_installs():
    fixed = m.purtera_standards("relocation_move")
    t = fixed["tools"][0]
    assert m._standard_applies(t, ["We will also unrack and ready any cables and equipment."])[0]
    assert not m._standard_applies(t, ["Mount the networking hardware from the OLD site to the rack in the NEW site"])[0]
