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
