"""The SOW writer's deterministic rules: a rule may not drop a cited deal
line that the summary or a standard step only mentions in passing, and the
customer's ship-out list stays the customer's (000123, 010198, 010279, 010085)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": None, "pictured": False, "quoted": False, "asked": False}


def test_grounding_keeps_a_task_the_summary_only_mentions():
    content = {"overview": [{"text": "Provider will install one (1) Square register and one (1) kitchen printer at "
                                     "7021 Roosevelt Rd, Berwyn, IL 60402.", "cites": ["a1", "a2"], "kind": "fact"}],
               "scope": [{"text": "Install one (1) Square register at 7021 Roosevelt Rd, Berwyn, IL 60402.",
                          "cites": ["a1"], "kind": "fact"},
                         {"text": "Confirm completion with the onsite contact.", "cites": [], "kind": "method", "standard": True}]}
    m._grounding(content, [_fact("a1", "We will be setting just 1 Square register and 1 kitchen printer.")])
    assert any("Square register" in x["text"] for x in content["scope"])


def test_grounding_drops_a_line_that_restates_the_summary_whole():
    content = {"overview": [{"text": "Provider will install POS systems at 161 locations.", "cites": ["a1"], "kind": "fact"}],
               "scope": [{"text": "Install POS systems at 161 locations.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Confirm completion with the onsite contact.", "cites": [], "kind": "method", "standard": True}]}
    m._grounding(content, [_fact("a1", "Installation of POS systems at 161 locations")])
    assert not any("POS systems" in x["text"] for x in content["scope"])


def test_tidy_keeps_a_cited_install_the_pooled_standard_words_cover():
    content = {"job_family": "pos_install_refresh",
               "scope": [{"text": "Install the printer at each location.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Install the tablet and stand at each register lane.", "cites": [], "kind": "method", "standard": True},
                         {"text": "Connect each printer to power and perform a test print.", "cites": [], "kind": "method", "standard": True}]}
    m._tidy(content)
    assert any(x["text"] == "Install the printer at each location." for x in content["scope"])


def test_ship_out_list_siblings_follow_their_anchor():
    atoms = [{"id": "t", "artifact_id": "doc", "text": "Samsung Tablet + Stand", "locator": {"line_start": 15}},
             {"id": "p", "artifact_id": "doc", "text": "Printer", "locator": {"line_start": 16}},
             {"id": "r", "artifact_id": "doc", "text": "Reuse barcode scanner and cash drawers for each site.", "locator": {"line_start": 17}},
             {"id": "far", "artifact_id": "doc", "text": "Spare printer", "locator": {"line_start": 60}}]
    content = {"customer_materials": [{"text": "Samsung Tablet + Stand", "cites": ["t"], "kind": "fact"}]}
    m._ship_list_siblings(content, {"atoms": atoms}, [])
    got = [x["text"] for x in content["customer_materials"]]
    assert "Printer" in got
    assert not any("Reuse" in t for t in got)  # a task line beside the list is not an item
    assert "Spare printer" not in got           # far from the list


def test_lead_in_joins_a_list_label():
    atoms = [{"id": "h", "artifact_id": "d", "text": "Overall we'd ship out:", "locator": {"line_start": 31}},
             {"id": "c", "artifact_id": "d", "text": "Cords for the above", "locator": {"line_start": 36}}]
    fs = [{"id": "c", "under": "Customer notes:", "text": "Cords for the above"}]
    m._lead_ins(fs, atoms)
    assert "ship out" in fs[0]["under"]


def test_labor_only_summary_keeps_its_labor_verbs():
    import re
    bad = re.compile(r"(?i)^(stage|tag|transport)$")
    assert m._strip_verbs("Provider will stage, tag, transport, de-install, install, label, and test the equipment.", bad) \
        == "Provider will de-install, install, label, and test the equipment."
    assert m._strip_verbs("Provider will install the equipment.", bad) == "Provider will install the equipment."


def test_a_duty_handing_something_to_the_customer_is_provider_work():
    content = {"end_customer": "Harris Teeter", "overview": [], "scope": [],
               "customer_responsibilities": [
                   {"text": "Provide all terminal MAC addresses to Harris Teeter for entry into TUNA.", "cites": ["a1"], "kind": "fact"},
                   {"text": "Provide access to the Customer's site.", "cites": ["a2"], "kind": "fact"}]}
    fs = [_fact("a1", "Supply all terminal MAC addresses to Harris Teeter to be entered into TUNA."),
          _fact("a2", "Site access will be provided.")]
    m._grounding(content, fs)
    assert any("MAC addresses" in x["text"] for x in content["scope"])
    assert any("site" in x["text"] for x in content["customer_responsibilities"])


def test_customer_we_will_stays_the_customers_in_the_summary():
    content = {"overview": [{"text": "Provider will preconfigure the equipment before shipping, then set it up, wire it, "
                                     "and test it.", "cites": ["a1"], "kind": "fact"}],
               "out_of_scope": [{"text": "Preconfiguration or imaging of hardware.", "cites": [], "kind": "method"}]}
    fs = {"a1": _fact("a1", "We will preconfigure all items before shipping them. They just need to be set up.",
                      under="Customer notes:")}
    m._speaker_in_overview(content, fs)
    t = content["overview"][0]["text"]
    assert t.startswith("Customer will preconfigure") and "Provider will set it up" in t


def test_provider_we_will_is_untouched():
    content = {"overview": [{"text": "Provider will configure the router and test failover.", "cites": ["a1"], "kind": "fact"}],
               "out_of_scope": [{"text": "Firewall configuration beyond basic setup.", "cites": [], "kind": "method"}]}
    m._speaker_in_overview(content, {"a1": _fact("a1", "Can you configure the router for us?")})
    assert content["overview"][0]["text"].startswith("Provider will configure")


def test_mixed_hours_replace_the_business_hours_basis():
    content = {"schedule": [], "assumptions": [{"text": "All work is performed during normal business hours.", "cites": [], "kind": "method"}],
               "customer_responsibilities": []}
    m._hours_basis(content, [_fact("h", "Sounds like the work would ideally be done during a mix of business hours and after hours.")])
    assert content["schedule"][0]["text"].startswith("Work Window: A mix of Business Hours")
    assert not any("normal business hours" in x["text"] for x in content["assumptions"])
    assert any("after-hours windows" in x["text"] for x in content["customer_responsibilities"])


def test_a_daytime_window_redefines_business_hours():
    content = {"schedule": [{"text": "Work Window: Monday-Thursday, 6:00 AM-5:00 PM", "cites": ["w"], "kind": "fact"}],
               "assumptions": []}
    m._hours_basis(content, [])
    assert content["assumptions"][0]["text"].startswith("Business Hours for this SOW are 6:00 AM to 5:00 PM")


def test_access_is_the_sites_and_the_work_stays_providers():
    content = {"scope": [{"text": "Provide POS access for terminal identification and configuration tasks.", "cites": ["a"], "kind": "fact"}],
               "customer_responsibilities": []}
    m._tidy(content)
    assert content["scope"][0]["text"] == "Perform terminal identification and configuration tasks."
    assert content["customer_responsibilities"][0]["text"].startswith("Provide POS access for")


def test_location_table_rows_are_not_cut_to_a_stated_count(monkeypatch):
    rows = [{"id": f"r{i}", "artifact_id": "sheet", "text": f"Store: {i} | Region: South | City: Town{i} | State: NC"}
            for i in range(1, 112)]
    env = {"atoms": rows + [{"id": "c", "text": "113 sites in total"}],
           "site_count_reconciliation": {"stated": 6, "all_counts_seen": {"6": 5, "113": 4}},
           "site_readiness": {"sites": []}}
    monkeypatch.setattr(m, "_best_of", lambda e, pm: {"overview": [{"text": "Provider will install terminals.", "cites": [], "kind": "fact"}],
                                                     "deliverables": [{"description": ["PDF for the 110-location rollout."]}]})
    c = m.for_envelope(env)
    assert c["site_summary"] == {"stated": 113, "resolved": 111, "from_table": True}
    assert len(c["sites"]) == 111
    assert any("113" in x["text"] for x in c["overview"])
    assert c["deliverables"][0]["description"][0] == "PDF for the 113-location rollout."


def test_standards_conditions_compile_and_gate():
    import json
    import re
    from pathlib import Path
    data = json.loads((Path(m.__file__).parent / "purtera_standards.json").read_text())
    for fam, secs in data.items():
        if fam.startswith("_"):
            continue
        for items in secs.values():
            for t in items if isinstance(items, list) else []:
                for c in re.findall(r"^\{(?:when|unless):(.*?)\}\s+(?=[A-Z{])", t):
                    re.compile(c)
    park = next(t for t in data["pos_install_refresh"]["assumptions"] if "parking" in t)
    assert not m._standard_applies(park, ["Pricing by store to include all travel and expenses."])[0]
    assert m._standard_applies(park, ["Install two registers."])[0]
    cfg = next(t for t in data["network_switch_refresh"]["customer_responsibilities"] if "fully configured" in t)
    assert m._standard_applies(cfg, [])[0]


def _drop_rule(content, text):
    content["scope"] = [x for x in content["scope"] if x.get("text") != text]


def test_guard_restores_the_only_line_citing_a_fact():
    content = {"scope": [{"text": "Route cabling through the wall passthroughs.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Closeout", "cites": [], "kind": "heading"}]}
    m._traced(content, "tidy", _drop_rule, content, "Route cabling through the wall passthroughs.")
    assert content["scope"][0]["text"].startswith("Route cabling")
    d = content["rule_decisions"]
    assert [(x["value"], x["guard"]) for x in d] == [("keep", "restored")]


def test_guard_moves_the_fact_onto_a_surviving_duplicate():
    content = {"scope": [{"text": "Power on the installed devices.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Power on the installed devices.", "cites": [], "kind": "method", "standard": True}]}
    m._traced(content, "standards", lambda: content.__setitem__("scope", content["scope"][1:]))
    assert content["scope"] == [{"text": "Power on the installed devices.", "cites": ["a1"], "kind": "method", "standard": True}]
    assert content["rule_decisions"][0]["guard"] == "merged"


def test_guard_lets_noise_and_covered_facts_go():
    content = {"_noise_atoms": ["n1"],
               "scope": [{"text": "We pride ourselves on white-glove service.", "cites": ["p1"], "kind": "fact"},
                         {"text": "Mount the camera.", "cites": ["n1"], "kind": "fact"},
                         {"text": "Install the camera.", "cites": ["c1"], "kind": "fact"},
                         {"text": "Install the camera on the north wall.", "cites": ["c1", "c2"], "kind": "fact"}]}
    m._traced(content, "grounding", lambda: content.__setitem__("scope", content["scope"][3:]))
    assert [x["text"] for x in content["scope"]] == ["Install the camera on the north wall."]
    assert all(d["value"] == "drop" for d in content["rule_decisions"])


def test_no_verbless_caption_in_scope():
    content = {"overview": [], "scope": [{"text": "Conference Room 2102 contains a large rectangular table with ten chairs.",
                                          "cites": ["c"], "kind": "fact"}]}
    m._grounding(content, [_fact("c", "The conference room contains a large table")])
    assert not content["scope"]


def test_a_scope_line_never_says_excluded():
    assert m._as_scope("End-user support, including support for O365/M365 clients, is excluded from the service scope.") \
        == "Provide end-user support, including support for O365/M365 clients."
    content = {"scope": [{"text": "Provide end-user support is excluded from the service scope.", "cites": ["a"], "kind": "fact"},
                         {"text": "Out of scope.", "cites": ["b"], "kind": "fact"}]}
    m._scope_polarity(content)
    assert [x["text"] for x in content["scope"]] == ["Provide end-user support."]


def test_summary_names_what_the_tasks_count():
    content = {"overview": [{"text": "Provider will install sixteen (16) Humanscale monitor arms.", "cites": [], "kind": "fact"}],
               "scope": [{"text": "Mount three (3) new 27-inch monitors on each arm, forty-eight (48) total.", "cites": ["m"], "kind": "fact"},
                         {"text": "Mount each clock with the top screws 45 3/8 inches above the floor.", "cites": ["h"], "kind": "fact"},
                         {"text": "Remove and haul away packaging debris.", "cites": [], "kind": "method", "standard": True}]}
    m._overview_covers_scope(content, [{"text": "We found someone that can haul away the trash."}])
    got = [x["text"] for x in content["overview"]]
    assert any("48" in t for t in got) and any("haul" in t for t in got)
    assert not any("45 3/8" in t for t in got)


def test_a_merge_keeps_the_facts_brand():
    content = {"scope": [{"text": "Install sixteen (16) Humanscale monitor arms on new desks.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Install each of the sixteen (16) monitor arms on new desks.", "cites": [], "kind": "method", "standard": True}]}
    m._traced(content, "standards", lambda: content.__setitem__("scope", content["scope"][1:]))
    assert content["scope"][0]["text"] == "Install each of the sixteen (16) Humanscale monitor arms on new desks."


def test_an_explicit_date_beats_the_writers_normalised_one():
    content = {"schedule": [{"text": "Requested Start Date: 2026-08-09", "cites": ["d"], "kind": "fact"}],
               "overview": [{"text": "Provider will perform the work on Sunday, 2026-08-09.", "cites": ["d"], "kind": "fact"}]}
    fs = [_fact("d", "We actually rescheduled this installation for next Sunday, August 16th.")]
    m._fix_dates(content, fs, "2026-08-09")
    assert content["schedule"][0]["text"] == "Requested Start Date: Sunday, August 16, 2026"
    assert content["overview"][0]["text"] == "Provider will perform the work on Sunday, August 16, 2026."
    m._weekend(content, fs)
    assert any(x["text"].startswith("Work Window: Weekend") for x in content["schedule"])
    assert not m._dates_ok("Requested Start Date: 2026-08-09", fs[0]["text"])


def test_an_accepted_proposed_time_is_the_arrival():
    atoms = [{"id": "q", "text": "How would 8am Sunday work for you?", "locator": {"sender": "sam@cust.io", "sent_at": "Fri, 07 Aug 2026 14:31:19 +0000"}},
             {"id": "y", "text": "Yes, we can hit that date!", "locator": {"sender": "aj@purtera-it.com", "sent_at": "Fri, 07 Aug 2026 16:33:23 +0000"}}]
    content = {"schedule": []}
    m._agreed_arrival(content, {"atoms": atoms})
    assert content["schedule"][0]["text"] == "Arrival Time: 8:00 AM"


def test_schedule_terms_reach_the_schedule():
    fs = [_fact("a", "All work 7-10am Monday, Tuesday, Thursday"), _fact("b", "Only 1 store per time zone"),
          _fact("c", "Project installs will start 10/13 and continue through 11/19"),
          _fact("d", "Days of Operation | Monday - Friday | Hours of Operation | 8:00am - 5:00pm")]
    content = {"schedule": []}
    m._schedule_from_facts(content, fs, {"atoms": []})
    got = [x["text"] for x in content["schedule"]]
    assert "Work Hours: All work 7-10am Monday, Tuesday, Thursday" in got
    assert "Scheduling Constraint: Only 1 store per time zone" in got
    assert "Work Window: 10/13 through 11/19" in got
    assert not any("Hours of Operation" in t for t in got)


def test_roles_and_a_full_day_become_lead_and_duration():
    fs = [_fact("r", "Typically, we need one network technician and one help desk technician per community."),
          _fact("d", "This is usually a full day of work.")]
    content = {"schedule": []}
    m._roles_and_hours(content, fs)
    assert content["scope_lead"].startswith("Provider will dispatch one (1) network technician and one (1) help desk technician")
    assert "sixteen (16) combined hours" in content["schedule"][0]["text"]


def test_deliverables_drop_what_the_sow_excludes_or_dropped():
    content = {"assumptions": [{"text": "Provider does not ship or dispose of removed equipment; removed switches are staged onsite."}],
               "out_of_scope": [], "rule_decisions": [{"value": "drop", "text": "Provide serial numbers for the benched equipment."}],
               "deliverables": [{"description": ["Included cabling transfer, labeling, before and after photographs, switch validation, "
                                                 "and removed legacy switch shipment for wipe and disposal.",
                                                 "serial numbers for the benched equipment."]}]}
    m._deliverables_hygiene(content)
    assert content["deliverables"][0]["description"] == ["Included cabling transfer, labeling, before and after photographs, and switch validation."]


def test_a_hedged_follow_up_is_never_guarded_back():
    content = {"overview": [{"text": "Plan for a full day of work, with follow-up the next day.", "cites": ["f"], "kind": "fact"}]}
    m._traced(content, "grounding", lambda: content.__setitem__("overview", []))
    assert content["overview"] == []


def test_info_is_not_equipment():
    content = {"equipment_in_scope": [{"text": "The RL standard naming convention for equipment labels.", "cites": ["n"], "kind": "fact"},
                                      {"text": "New Cisco power cords.", "cites": ["p"], "kind": "fact"}],
               "customer_responsibilities": []}
    m._tidy(content)
    assert [x["text"] for x in content["equipment_in_scope"]] == ["New Cisco power cords."]
    assert content["customer_responsibilities"][0]["text"].startswith("Provide the RL standard naming convention")


def test_project_name_drops_a_persons_name(monkeypatch):
    monkeypatch.setattr(m, "_best_of", lambda e, pm: {"project_name": "Ralph Lauren Switch Replacement Dan Tighe", "overview": []})
    monkeypatch.setattr(m, "requested_by", lambda e: {"name": "Daniel Tighe"})
    c = m.for_envelope({"atoms": [], "site_readiness": {"sites": []}})
    assert c["project_name"] == "Ralph Lauren Switch Replacement"
