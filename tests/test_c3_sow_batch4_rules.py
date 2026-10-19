"""SOW writer rules, batch 4 (000132 pass T, C+): a line's meaning keeps its
source's polarity; the summary names the sites and every service; standard
steps sit under the service they support; the template's generic contact
line does not swallow a standard that says more."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _atom(i, text, atom_type="deal_metadata", was="scope_item"):
    return {"id": i, "text": text, "atom_type": atom_type,
            "structured": {"alt_atom_types": [was], "c3_label": {"type": atom_type, "was": was}}}


def test_an_exclusion_its_source_does_not_state_turns_into_the_requested_service():
    """000132: the requested "End-user support including O365/M365 clients"
    (a type-head relabel to deal_metadata, placed under assumptions) was
    written "End-user support, including O365/M365 clients, is excluded."."""
    src = "End-user support including O365/M365 clients"
    content = {"overview": [],
               "scope": [{"text": "Network Support", "cites": [], "kind": "heading"},
                         {"text": "Support physical network components.", "cites": ["n"], "kind": "fact"},
                         {"text": "Closeout", "cites": [], "kind": "heading"},
                         {"text": "Confirm completion with the onsite contact before leaving site.", "cites": [], "kind": "method",
                          "standard": True}],
               "assumptions": [{"text": "End-user support, including O365/M365 clients, is excluded.", "cites": ["e"], "kind": "fact"}]}
    fs = [_fact("e", src, section="assumptions", type="deal_metadata"), _fact("n", "Support for physical network components (LAN, WLAN)")]
    m._traced(content, "source_polarity", m._source_polarity, content, fs, {"atoms": [_atom("e", src)]})
    texts = [x["text"] for x in content["scope"]]
    assert content["assumptions"] == []
    assert "Provide end-user support, including O365/M365 clients." in texts
    assert texts.index("End User Support") + 1 == texts.index("Provide end-user support, including O365/M365 clients.")
    assert texts.index("End User Support") < texts.index("Closeout")
    d = content["rule_decisions"][0]
    assert (d["rule"], d["head"], d["was"], d["value"]) == ("source_polarity", "sow_section", "assumptions", "scope_of_work")


def test_an_exclusion_with_a_negating_or_limiting_source_stays():
    content = {"scope": [], "assumptions": [
        {"text": "Replacement parts are not included.", "cites": ["p"], "kind": "fact"},
        {"text": "Cabling is not included; Customer's team will handle it.", "cites": ["c"], "kind": "fact"}]}
    fs = [_fact("p", "we dont provide parts, engineering support only"), _fact("c", "Their IT team will handle the cabling")]
    m._source_polarity(content, fs, {"atoms": []})
    assert len(content["assumptions"]) == 2 and content["scope"] == []


def test_a_reversed_line_whose_source_is_not_work_goes():
    content = {"scope": [], "customer_responsibilities": [
        {"text": "Parking will not be provided by Customer.", "cites": ["g"], "kind": "fact"}]}
    m._source_polarity(content, [_fact("g", "Parking garage on Elm Street", section="job_details", type="site_condition")],
                       {"atoms": []})
    assert content["customer_responsibilities"] == [] and content["scope"] == []


def test_a_bound_or_a_site_condition_is_not_an_exclusion():
    content = {"scope": [], "assumptions": [
        {"text": "The longest cable run does not exceed two hundred (200) feet.", "cites": ["r"], "kind": "fact"},
        {"text": "No concealed utilities or structural obstructions are present in the work areas.", "cites": ["w"], "kind": "fact"}]}
    m._source_polarity(content, [_fact("r", "Run to IDF: 200 ft"), _fact("w", "Open ceiling in the work areas")], {"atoms": []})
    assert len(content["assumptions"]) == 2


def test_standards_and_uncited_lines_are_the_templates():
    content = {"scope": [], "assumptions": [
        {"text": "All work is performed on site; remote IT support is not included.", "cites": [], "kind": "method", "standard": True}]}
    m._source_polarity(content, [], {"atoms": []})
    assert len(content["assumptions"]) == 1


def _std(t):
    return {"text": t, "cites": [], "kind": "method", "standard": True}


def _support_scope():
    h = lambda t: {"text": t, "cites": [], "kind": "heading"}
    f = lambda t, c: {"text": t, "cites": [c], "kind": "fact"}
    return [h("Onsite Support"),
            _std("Coordinate visit priorities with the customer's designated IT contact before or during each visit."),
            _std("Assist with workstation, peripheral, printer, phone and connectivity issues and desk-side support."),
            _std("Perform hands-on support for network equipment, servers, power connections, rack placement and cabling as directed."),
            _std("Support server and endpoint validation after approved patching or upgrade work."),
            _std("Assist with basic troubleshooting of switches, wireless access points, patching, cabling and network connectivity."),
            _std("Assist with server or NAS troubleshooting while onsite and under Customer direction."),
            _std("Support troubleshooting related to virtual servers, host visibility, storage connectivity and incident response."),
            h("Network Support"), f("Support physical network components, including LAN and WLAN.", "n"),
            h("Server Support"), f("Support physical server and NAS environments.", "s"),
            {"text": "Support remote direction from Customer's internal IT team.", "cites": [], "kind": "method"},
            h("Virtualization Support"), f("Support the virtual server environment based on VMware vSphere.", "v"),
            h("Windows and Linux Patching"), f("Perform operating system patching and upgrades for Windows and Linux.", "p"),
            {"text": "Support server validation after approved Windows and Linux patching and upgrade work.", "cites": [], "kind": "method"},
            _std("Validate reported issues and document observed site conditions, findings and recommended next steps."),
            h("End User Support"), f("Provide end-user support, including O365/M365 clients.", "e"),
            h("Closeout"), _std("Confirm completion with the onsite contact before leaving site.")]


def _groups(scope):
    out, cur = {}, None
    for x in scope:
        if x.get("kind") == "heading":
            cur = x["text"]
            out[cur] = []
        else:
            out[cur].append(x["text"])
    return out


def test_standard_steps_sit_under_the_service_they_support():
    """000132: fifteen standard steps under "Onsite Support", the visit
    validation step under "Windows and Linux Patching"."""
    content = {"services": ["staff_augmentation", "support"], "scope": _support_scope()}
    m._service_groups(content)
    g = _groups(content["scope"])
    assert any(t.startswith("Assist with workstation") for t in g["End User Support"])
    assert any(t.startswith("Assist with basic troubleshooting of switches") for t in g["Network Support"])
    assert any(t.startswith("Assist with server or NAS") for t in g["Server Support"])
    assert any(t.startswith("Support troubleshooting related to virtual servers") for t in g["Virtualization Support"])
    assert any(t.startswith("Support server and endpoint validation") for t in g["Windows and Linux Patching"])
    # a step naming two services, and the visit steps, stay general
    assert any(t.startswith("Perform hands-on support") for t in g["Onsite Support"])
    assert any(t.startswith("Validate reported issues") for t in g["Onsite Support"])
    assert any(t.startswith("Coordinate visit priorities") for t in g["Onsite Support"])
    # patch validation said once; remote direction is coordination, in the general group
    assert not any(t.startswith("Support server validation after approved Windows") for t in g["Windows and Linux Patching"])
    assert "Coordinate with Customer's designated remote IT contact during each visit." in g["Onsite Support"]
    assert not any("remote direction" in t for ts in g.values() for t in ts)
    assert list(g)[-1] == "Closeout" and g["Closeout"] == ["Confirm completion with the onsite contact before leaving site."]


def test_service_groups_leave_install_jobs_alone():
    content = {"services": ["implementation"], "scope": _support_scope()}
    before = [x["text"] for x in content["scope"]]
    m._service_groups(content)
    assert [x["text"] for x in content["scope"]] == before


def test_a_standard_contact_duty_that_says_more_than_the_template_stays():
    """000132: the standards' "point of contact available by phone during
    scheduled visits" was deduped against the template's generic line."""
    content = {"customer_responsibilities": [
        _std("Provide a Customer point of contact available by phone during scheduled visits."),
        _std("Provide a point of contact for the project and onsite visit."),
        {"text": "Provide an onsite point of contact.", "cites": ["x"], "kind": "fact"}]}
    m._tidy(content)
    assert [x["text"] for x in content["customer_responsibilities"]] == [
        "Provide a Customer point of contact available by phone during scheduled visits."]


def test_the_support_summary_names_the_sites_and_every_service():
    """000132 pass T: the summary shrank to one line of four technical areas."""
    sites = [{"city": c, "state": s} for c, s in (("Delphos", "OH"), ("Hudson", "WI"), ("Troy", "MI"))]
    c = {"services": ["staff_augmentation", "support"], "sites": sites,
         "overview": [{"text": "Provider will support physical network components and physical server environments.", "cites": [], "kind": "method"}],
         "scope": [{"text": "Recurring Support Visits", "cites": [], "kind": "heading"},
                   {"text": "Perform weekly on-site visits as needed at Delphos, OH, Hudson, WI, and Troy, MI.", "cites": ["w"], "kind": "fact"},
                   {"text": "Coordinate with local stakeholders at each visit.", "cites": ["k"], "kind": "fact"},
                   {"text": "Support physical network components.", "cites": ["n"], "kind": "fact"},
                   {"text": "Provide end-user support, including O365/M365 clients.", "cites": ["e"], "kind": "fact"}]}
    m._summary_names_job(c)
    ov = " ".join(x["text"] for x in c["overview"])
    assert "Delphos" in ov and "Troy" in ov and "weekly on-site visits" in ov
    assert "coordination with local stakeholders" in ov and "end-user support, including O365/M365 clients" in ov
    assert ov.count("physical network components") == 1


def test_a_few_site_job_names_its_cities_once():
    c = {"services": ["implementation"], "sites": [{"city": "Gallup", "state": "NM"}, {"city": "Harlan", "state": "IA"}],
         "overview": [{"text": "Provider will install the antennas.", "cites": [], "kind": "fact"}], "scope": []}
    m._summary_names_job(c)
    assert c["overview"][1]["text"] == "Provider will perform the work at two (2) Customer locations: Gallup, NM and Harlan, IA."
    m._summary_names_job(c)
    assert len(c["overview"]) == 2


def test_an_uncited_date_the_deal_does_not_state_never_renders():
    """010198 live (pass T): "Requested Start Date: 2026-08-09" (the as-of
    Sunday) with no cites, beside "next Sunday, August 16th"."""
    fs = [_fact("d", "We actually rescheduled this installation for next Sunday, August 16th.", section="timeline",
                type="milestone_phase")]
    content = {"schedule": [{"text": "Requested Start Date: 2026-08-09", "cites": [], "kind": "fact"},
                            {"text": "Arrival Time: 8:00 AM", "cites": [], "kind": "fact"}]}
    m._fix_dates(content, fs, "2026-08-09")
    assert [x["text"] for x in content["schedule"]] == ["Requested Start Date: Sunday, August 16, 2026", "Arrival Time: 8:00 AM"]
    assert content["schedule"][0]["cites"] == ["d"]
    # an undated cite does not vouch for the date either; with the deal's
    # start already written, the stray line just goes
    content = {"schedule": [{"text": "Requested Start Date: Sunday, August 16, 2026", "cites": ["d"], "kind": "fact"},
                            {"text": "Requested Start Date: 2026-08-09", "cites": ["x"], "kind": "fact"}]}
    m._fix_dates(content, fs + [_fact("x", "Install the register.")], "2026-08-09")
    assert [x["text"] for x in content["schedule"]] == ["Requested Start Date: Sunday, August 16, 2026"]


def test_a_date_the_deal_states_stays_uncited():
    fs = [_fact("d", "Install window 10/13 - 11/19", section="timeline")]
    content = {"schedule": [{"text": "Work Window: 10/13 through 11/19", "cites": [], "kind": "fact"}]}
    m._fix_dates(content, fs, "2026-09-01")
    assert content["schedule"][0]["text"] == "Work Window: 10/13 through 11/19"


def test_an_exclusion_is_never_a_provider_task():
    """010419 pass T: library exclusions citing an email subject became
    "Provide managed IT services ...", "Provide lift rental ..." Closeout tasks."""
    fs = [_fact("subj", "Subject: FW: Scope of Support for Techs during onboarding"),
          _fact("day", "This is usually a full day of work, with some follow-up needed the next day.")]
    content = {"scope": [{"text": "Closeout", "cites": [], "kind": "heading"},
                         {"text": "Provide managed IT services or ongoing help desk coverage.", "cites": ["subj", "day"], "kind": "method"},
                         {"text": "Provide lift rental or other specialized trade services.", "cites": ["subj"], "kind": "method"},
                         {"text": "Lift rental is excluded.", "cites": [], "kind": "method"}],
               "out_of_scope": [],
               "writer_draft": {"out_of_scope": [{"text": "Lift rental or other specialized trade services.", "cites": ["subj"]}]}}
    m._scope_polarity(content, fs)
    m._exclusions_never_tasks(content, fs)
    assert [x["text"] for x in content["scope"]] == ["Closeout"]
    oos = [x["text"] for x in content["out_of_scope"]]
    assert "Managed IT services or ongoing help desk coverage." in oos and "Lift rental." in oos
    assert "Lift rental or other specialized trade services." in oos


def test_a_negated_assumption_never_returns_as_a_task():
    """010085 pass T: "No rack modification or cabinet modification." moved
    into the task list as "Provide rack modification ..."."""
    fs = [_fact("r", "they stated that the racks should have enough space for the new equipment regardless.")]
    content = {"scope": [{"text": "Provide rack modification or cabinet modification.", "cites": ["r"], "kind": "method"}],
               "out_of_scope": [],
               "writer_draft": {"assumptions": [{"text": "No rack modification or cabinet modification.", "cites": ["r"]}]}}
    m._exclusions_never_tasks(content, fs)
    assert content["scope"] == []
    assert [x["text"] for x in content["out_of_scope"]] == ["No rack modification or cabinet modification."]


def test_a_requested_service_that_reads_like_an_exclusion_stays_a_task():
    fs = [_fact("r", "Customer wants managed IT services or ongoing help desk coverage at the site")]
    content = {"scope": [{"text": "Provide managed IT services or ongoing help desk coverage.", "cites": ["r"], "kind": "fact"}],
               "out_of_scope": []}
    m._exclusions_never_tasks(content, fs)
    assert len(content["scope"]) == 1 and content["out_of_scope"] == []


def test_a_model_number_in_a_file_name_is_reworded_not_dropped():
    """010279 pass T: "replace 785 existing POS terminals" (785 is the model,
    from "785 POS Termainal Refresh RFP.doc") was dropped whole."""
    c = {"overview": [{"text": "Perform installation labor to replace 785 existing POS terminals at 113 sites.", "cites": []}],
         "scope": [{"text": "Remove the existing seven hundred eighty-five (785) POS terminals.", "cites": []},
                   {"text": "Replace all 785 terminals with 913 terminals.", "cites": []}],
         "deliverables": [{"item": "Work Completion Report", "description": ["Installed 200 wireless access points."]}]}
    m._model_numbers(c, {"documents": [{"filename": "785 POS Termainal Refresh RFP.doc"}, {"filename": "200 AP_s.txt"}]})
    assert [x["text"] for k in ("overview", "scope") for x in c[k]] == [
        "Perform installation labor to replace existing 785 POS terminals at 113 sites.",
        "Remove the existing 785 POS terminals.", "Replace all existing 785 terminals with 913 terminals."]
    assert c["deliverables"][0]["description"] == ["Installed 200 wireless access points."]


def test_the_store_table_comes_from_the_document_whole():
    """010279 pass T: row atoms whose cells repeated another row's were cut
    ("Store: 419 |"), so the exhibit listed 111 of 113 stores."""
    rows = [{"Store": str(n), "Region": "Northern", "City": c, "State": "VA"} for n, c in
            ((6, "Ashburn"), (310, "Ashburn"), (419, "Virginia Beach"), (7, "Virginia Beach"))]
    env = {"atoms": [{"artifact_id": "x", "text": "Store: 6 | Region: Northern | City: Ashburn | State: VA"},
                     {"artifact_id": "x", "text": "Store: 419 |"}],
           "documents": [{"artifact_id": "x", "structured": {"pages": [{"sections": [{"blocks": [
               {"kind": "table", "columns": ["Store", "Region", "City", "State"], "rows": rows}]}]}]}}]}
    assert [r["name"] for r in m.location_table(env)] == ["Store 6", "Store 310", "Store 419", "Store 7"]


def test_every_check_drop_says_why(monkeypatch):
    monkeypatch.setattr(m, "_verified", lambda items, lines: [])
    c = {k: [] for k in m.LISTS}
    c["scope"] = [{"text": "Install the printer.", "cites": ["a"], "kind": "fact"},
                  {"text": "Install 14 Zebra ZT411 printers.", "cites": [], "kind": "method"}]
    m.check(c, {"a": "a printer"})
    assert c["dropped"][0]["reason"].startswith("check:")
    assert c["dropped_unsupported"][0]["reason"]
