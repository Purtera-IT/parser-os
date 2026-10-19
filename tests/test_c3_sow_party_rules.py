"""SOW writer rules from the 010326 phone-swap grade: a Provider task never
names work the SOW gives the customer or excludes, a question is never a
commitment, transition standards stay off a phone swap, and a "reuse the
existing mounts" line does not cancel every phone standard."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": "", "type": kw.get("type", ""), "when": kw.get("when", ""),
            "supplier": None, "pictured": False, "quoted": kw.get("quoted", False), "asked": kw.get("asked", False)}


def _rack_content():
    return {"scope": [{"text": "Phone replacements", "cites": [], "kind": "heading"},
                      {"text": "Install three hundred seventeen (317) phones.", "cites": ["p"], "kind": "fact"},
                      {"text": "Provider will also help with racking, stacking, and wall mounting the hardware.", "cites": ["early"], "kind": "fact"},
                      {"text": "Handle rack ears and brackets for equipment installed in racks.", "cites": ["ears"], "kind": "fact"},
                      {"text": "Remove each existing wall mount and install the replacement mount and phone.", "cites": ["wm"], "kind": "fact"}],
            "customer_responsibilities": [{"text": "Handle rack hardware and rack & stack gear, including the gateways.",
                                           "cites": ["split"], "kind": "fact"}],
            "out_of_scope": [{"text": "Rack hardware, rack ears, and gateway installation.", "cites": [], "kind": "method"},
                             {"text": "Core drilling, wall repair, painting, or structural modification.", "cites": [], "kind": "method"}]}


def _rack_facts():
    return [_fact("p", "317 physical phones"),
            _fact("early", "they will need help with racking/stacking/wall mounting the hardware.", when="2026-09-14", quoted=True),
            _fact("ears", "you guys are gonna be handling the gear that goes in the racks. Like putting the rack ears on the brackets.", "job_details"),
            _fact("wm", "Existing wall mounts will be reused by swapping old mounts for new ones."),
            _fact("split", "They will handle rack & stack gear", "customer_responsibilities")]


def test_a_provider_task_never_names_a_customer_duty_or_a_backed_exclusion():
    c = _rack_content()
    m._traced(c, "contradiction", m._scope_contradictions, c, _rack_facts())
    texts = [x["text"] for x in c["scope"]]
    assert not any("racking" in t or "rack ears" in t for t in texts)
    assert any("317" in t for t in texts) and any("wall mount" in t for t in texts)
    assert len(c["out_of_scope"]) == 2  # the exclusions stand
    drops = [d for d in c["rule_decisions"] if d["rule"] == "contradiction"]
    assert {tuple(d["atom_ids"]) for d in drops} == {("early",), ("ears",)}
    assert all(d["head"] == "sow_keep" and d["value"] == "drop" for d in drops)


def test_a_later_fact_keeps_the_provider_task():
    c = _rack_content()
    fs = _rack_facts()
    for f in fs:
        if f["id"] == "early":
            f.update(quoted=False, when="2026-10-01")
        if f["id"] == "split":
            f["when"] = "2026-09-17"
    m._scope_contradictions(c, fs)
    assert any("racking" in x["text"] for x in c["scope"])


def test_an_exclusion_nothing_backs_gives_way_to_a_cited_task():
    c = {"scope": [{"text": "Mount each display bracket on the wall.", "cites": ["d"], "kind": "fact"}],
         "customer_responsibilities": [],
         "out_of_scope": [{"text": "Display bracket mounting.", "cites": [], "kind": "method"}]}
    m._scope_contradictions(c, [_fact("d", "Please mount the display brackets on the wall")])
    assert c["scope"] and not c["out_of_scope"]


def test_a_question_is_never_a_commitment():
    c = {"scope": [{"text": "After each installation, Provider will complete phone registration, E911 verification, and test calls.",
                    "cites": ["q1", "q2"], "kind": "fact"},
                   {"text": "Install one (1) Ubiquiti router with cellular backup.", "cites": ["ask"], "kind": "fact"}]}
    fs = [_fact("q1", "once phones are installed, is there any type of registering, any type of E911?", "answered_question", asked=True),
          _fact("q2", "Are you talking about the E911 stuff?", "answered_question", asked=True),
          _fact("ask", "I would also like to add a Ubiquiti router with cellular backup - is that something you can help with?",
                "answered_question", asked=True)]
    m._traced(c, "question_not_commitment", m._question_not_commitment, c, fs)
    assert [x["text"] for x in c["scope"]] == ["Install one (1) Ubiquiti router with cellular backup."]
    assert c["rule_decisions"][0]["atom_ids"] == ["q1", "q2"]


def test_a_question_answered_on_the_same_line_stays():
    c = {"scope": [{"text": "Work after hours with no weekends.", "cites": ["qa"], "kind": "fact"}]}
    m._question_not_commitment(c, [_fact("qa", "What hours? [Betty] After hours, no weekends", "answered_question", asked=True)])
    assert c["scope"]


PHONE = ["We will swap 317 phones one for one", "They will be converting from intrado to redsky; cutover per site",
         "Existing wall mounts will be reused by swapping old mounts for new ones with new phones."]


def test_transition_standards_stay_off_a_phone_swap_and_phone_standards_are_written():
    c = {"job_family": "smart_hands_general", "scope": [{"text": "Phone replacements", "cites": [], "kind": "heading"}],
         "customer_responsibilities": [], "assumptions": [], "out_of_scope": []}
    m._apply_standards(c, "smart_hands_general", PHONE)
    every = " ".join(x["text"] for k in ("scope", "customer_responsibilities", "assumptions", "out_of_scope") for x in c[k])
    assert "outgoing provider" not in every and "runbook" not in every and "nurse call" not in every
    assert "estimates and are not guaranteed" not in every
    assert "staging area for removed phones" in every and "final deployment list" in every
    assert "may require a Change Order" in every and "Electrical installation" in every
    assert "Coordinate technician scheduling" in every


def test_transition_standards_still_fire_on_an_it_transition():
    c = {"job_family": "smart_hands_general", "scope": [], "customer_responsibilities": [], "assumptions": [], "out_of_scope": []}
    m._apply_standards(c, "smart_hands_general", ["IT Transition & Onboarding support for the clinic"])
    every = " ".join(x["text"] for k in ("scope", "customer_responsibilities", "assumptions") for x in c[k])
    assert "runbook" in every and "outgoing provider" in every


def test_a_cited_reuse_step_is_not_written_twice_by_its_standard():
    c = {"job_family": "smart_hands_general",
         "scope": [{"text": "Phone replacements", "cites": [], "kind": "heading"},
                   {"text": "Remove each existing phone and install its replacement using the existing cabling.", "cites": ["r"], "kind": "fact"}]}
    m._apply_standards(c, "smart_hands_general", PHONE)
    removes = [x["text"] for x in c["scope"] if x["text"].lower().startswith(("remove each existing phone", "disconnect and remove"))]
    assert len(removes) == 1 and "using the existing cabling" in removes[0]


def _row(i, text, art="mail"):
    return {"id": i, "artifact_id": art, "text": text}


def test_the_per_site_count_list_is_the_site_table():
    rows = [_row("h", "Physical Site / location | Phone Count")]
    names = [("City Hall (900 Quay St, Manitowoc, WI 54220)", 60), ("Fire Station 1 (911 Franklin Street Manitowoc, WI 54220)", 8),
             ("10th St Bridge", 1), ("Police Department (910 Jay St, Manitowoc, WI 54220)", 40),
             ("Zoo (1215 N 8th St, Manitowoc, WI 54220)", 1), ("Senior Center (3330 Custer St, Manitowoc, WI 54220)", 6)]
    rows += [_row(f"r{i}", f"{n} | {k}") for i, (n, k) in enumerate(names)]
    rows += [_row("t", "TOTAL | 116"), _row("x", "Primary Switch | 96", "sheet"),  # another document's cells
             _row("s1", "OK Status | 50", "status"), _row("s2", "Failed | 5", "status")]
    got = m.count_table({"atoms": rows})
    assert [r["name"] for r in got][:2] == ["City Hall", "Fire Station 1"] and len(got) == 6
    fs1 = got[1]
    assert (fs1["street"], fs1["city"], fs1["state"], fs1["postal_code"], fs1["count"]) == ("911 Franklin Street", "Manitowoc", "WI", "54220", 8)
    assert fs1["extra"] == {"phones": "8"} and got[2]["address"] == ""


def test_a_count_list_that_does_not_add_up_is_not_the_site_table():
    rows = [_row("h", "Site | Phone Count")] + [_row(f"r{i}", f"Site {c} (1{i}0 Main St, Town, WI 54220) | {i + 1}") for i, c in enumerate("ABCDEF")]
    rows.append(_row("t", "TOTAL | 99"))
    assert m.count_table({"atoms": rows}) == []


def test_a_status_summary_with_a_total_is_not_a_site_list():
    rows = [_row(f"r{i}", f"{t} | {n}", "s") for i, (t, n) in enumerate([("OK Status in Open CC", 10), ("Failed Status", 2), ("Test Clock", 1),
                                                                          ("OK Closed", 3), ("Failed Closed", 4), ("Missing", 5)])]
    rows.append(_row("t", "Total | 25", "s"))
    assert m.count_table({"atoms": rows}) == []


def test_a_crm_title_naming_the_customers_work_yields_to_the_writers_name():
    c = {"scope": [{"text": "Install three hundred seventeen (317) phones.", "cites": ["p"], "kind": "fact"},
                   {"text": "Remove each existing wall mount and install the replacement mount.", "cites": ["w"], "kind": "fact"}],
         "overview": [], "customer_responsibilities": [{"text": "Handle rack hardware and rack & stack gear.", "cites": ["s"], "kind": "fact"}]}
    assert m._name_contradicted("Manitowoc - Racking/Wall mount", c)
    assert not m._name_contradicted("Manitowoc Phone Deployment", c)
    assert not m._name_contradicted("Manitowoc Wall Mount Install", c)


def _atom(i, text, ln, typ="scope_item", label=None, lead=None, **st):
    s = {"c3_label": {"type": label}} if label else {}
    s.update(st)
    return {"id": i, "artifact_id": "mail", "text": text, "atom_type": typ, "structured": s,
            "locator": {"line_start": ln, "message_index": 4, **({"lead_in": lead} if lead else {})}}


SIG = [_atom("s1", "Firewall Request", 115, "small_talk"), _atom("s2", "ISP Upgrade", 116, "small_talk"),
       _atom("s3", "Network Consultation/New Installation/Upgrade/Questions", 117, label="_keep", lead=["Need help with:"]),
       _atom("s4", "New Unit Opening", 118, "small_talk"), _atom("s5", "Wastewatch Connectivity", 119, "small_talk"),
       _atom("ok", "Commercial site with kitchen and office areas (Sodexo location)", 40, label="scope_item"),
       _atom("ab", "Photo requirements mentioned: None mentioned", 41, "deal_metadata"),
       _atom("hq", "Hardware Quantity: Not provided", 42, "deal_metadata")]


def test_a_signature_menu_and_small_talk_are_never_facts():
    rej = m._rejected_atoms({"atoms": SIG})
    assert {"s1", "s2", "s3", "s4", "s5"} <= rej and "ok" not in rej
    env = {"atoms": SIG, "c3_brief": {"sections": {"scope_of_work": [{"id": a["id"], "text": a["text"]} for a in SIG]}}}
    got = {f["id"] for f in m.facts(env)}
    assert got == {"ok", "hq"}  # an absence note goes; a count the request says is unknown stays


def test_lines_built_on_rejected_atoms_or_absence_notes_go():
    env = {"atoms": SIG, "deal_roster": {"people": [{"name": "AJ Deuel"}, {"name": "Chase Smith"}, {"name": "Patrick Kelly"}]}}
    c = {"overview": [{"text": "This work is for a new unit opening at a commercial Sodexo site with kitchen and office areas.",
                       "cites": ["s4", "ok"], "kind": "fact"},
                      {"text": "Provider will work at a commercial Sodexo site with kitchen and office areas.", "cites": ["s4", "ok"], "kind": "fact"},
                      {"text": "The one-day install is led by AJ Deuel, with Chase and Kelly in delivery.", "cites": ["ok"], "kind": "fact"}],
         "scope": [{"text": "Include network consultation and ISP upgrade support.", "cites": ["s3", "s2"], "kind": "fact"},
                   {"text": "Provide do not install a patch cable at the far-end biscuit box.", "cites": ["ok"], "kind": "fact"}],
         "provider_tools": [{"text": "No testing or certification tools were specified for this job.", "cites": ["ab"], "kind": "fact"}],
         "assumptions": [{"text": "The paying contact is unknown.", "cites": ["ab"], "kind": "fact"}]}
    m._traced(c, "rejected_source", m._rejected_source, c, env)
    assert [x["text"] for x in c["overview"]] == ["Provider will work at a commercial Sodexo site with kitchen and office areas."]
    assert [x["text"] for x in c["scope"]] == ["Do not install a patch cable at the far-end biscuit box."]
    assert not c["provider_tools"] and not c["assumptions"]


def test_a_task_the_customer_is_to_decide_on_is_not_provider_work():
    c = {"scope": [{"text": "Perform a project survey.", "cites": ["s"], "kind": "fact"},
                   {"text": "Mount each clock at the approved height.", "cites": ["c"], "kind": "fact"}],
         "customer_responsibilities": [{"text": "Decide whether to proceed with the optional site survey.", "cites": ["d"], "kind": "fact"},
                                       {"text": "On-site staff may help mount the back of the clock if needed.", "cites": ["h"], "kind": "fact"}],
         "out_of_scope": []}
    m._scope_contradictions(c, [_fact(i, "x") for i in "scdh"])
    assert [x["text"] for x in c["scope"]] == ["Mount each clock at the approved height."]


def test_the_provider_half_of_a_split_supply_line_is_written():
    c = {"provider_materials": [{"text": "Screws, anchors, washers and standard mounting hardware.", "cites": [], "kind": "method"}],
         "provider_tools": [], "customer_materials": [{"text": "One (1) 9U rack.", "cites": ["r"], "kind": "fact"}]}
    m._provider_half(c, [_fact("r", "Provider will supply backing board; AJ/CDW will provide/order the 9U rack"),
                         _fact("t", "Toggle bolts / heavy-duty drywall anchors: Provider will bring toggle bolts and stud finder"),
                         _fact("x", "The service provider will provide support as needed")])
    assert [x["text"] for x in c["provider_materials"]][1:] == ["Backing board.", "Toggle bolts."]
    assert [x["text"] for x in c["provider_tools"]] == ["Stud finder."]
