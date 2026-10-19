"""SOW writer rules, batch 6 (pass V grades: 000132 C+)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _content():
    return {k: [] for k in m.LISTS}


SITES = [{"name": f"{c}, {s}", "city": c, "state": s, "address": f"{c}, {s}"}
         for c, s in (("Delphos", "OH"), ("Hudson", "WI"), ("Plymouth", "MI"), ("Troy", "MI"), ("Tupelo", "MS"), ("Wilmington", "DE"))]


# --- 000132 C+: a spelled 24/7 is still the hedged promise -------------------

def test_a_spelled_round_the_clock_promise_goes_whole_and_becomes_an_exclusion():
    c = _content()
    c["overview"] = [{"text": "Provider will perform end-user support including O365/M365 clients.", "cites": [], "kind": "method"},
                     {"text": "Provider will provide twenty-four (24) by seven (7) on-call availability.", "cites": [], "kind": "method"}]
    c["scope"] = [{"text": "Provide 24/7 on-call availability.", "cites": ["f1"], "kind": "fact"}]
    c["_clock_hedges"] = [{"id": "h1", "text": "The 24/7 availability can be tough because you're paying for someone to be on call."}]
    m._grounding(c, [_fact("f1", "24/7 on-call availability")])
    txt = [x["text"] for x in c["overview"] + c["scope"]]
    assert not any("seven" in t or "24" in t for t in txt), txt
    assert "Provider will perform end-user support including O365/M365 clients." in txt
    oos = [x for x in c["out_of_scope"] if "24/7" in x["text"]]
    assert len(oos) == 1 and oos[0]["cites"] == ["h1"]
    assert "_clock_hedges" not in c


def test_an_unhedged_on_call_line_is_left_alone():
    c = _content()
    c["overview"] = [{"text": "Provider will provide twenty-four (24) by seven (7) on-call availability.", "cites": [], "kind": "method"}]
    m._grounding(c, [_fact("f1", "24/7 on-call availability")])
    assert len(c["overview"]) == 1 and not c["out_of_scope"]


def test_the_strip_never_leaves_the_bare_verb():
    c = _content()
    c["overview"] = [{"text": "Provider will provide weekly site visits with 24/7 on-call support.", "cites": [], "kind": "method"}]
    c["assumptions"] = [{"text": "Services are performed during standard business hours.", "cites": [], "kind": "method"}]
    m._grounding(c, [])
    assert [x["text"] for x in c["overview"]] == ["Provider will provide weekly site visits."]


# --- 000132 C+: one per-site line per resolved site --------------------------

def test_a_per_site_family_gets_the_site_it_left_out():
    c = _content()
    c["sites"] = [dict(s) for s in SITES]
    tail = " once per week for four (4) to eight (8) hours as needed."
    c["scope"] = [{"text": "Weekly Site Visits", "cites": [], "kind": "heading"}] + \
        [{"text": f"Visit {s['city']}, {s['state']}{tail}", "cites": ["visit", f"site_{s['city']}"], "kind": "fact"} for s in SITES[1:]] + \
        [{"text": "Infrastructure Support", "cites": [], "kind": "heading"}]
    atoms = [{"id": "atm_delphos", "atom_type": "physical_site", "text": "Delphos, OH"}]
    m._per_site_lines(c, atoms)
    visits = [x for x in c["scope"] if x["text"].startswith("Visit ")]
    assert len(visits) == len(SITES)
    new = next(x for x in visits if "Delphos" in x["text"])
    assert new["text"] == f"Visit Delphos, OH{tail}" and new["cites"] == ["visit", "atm_delphos"]
    assert c["scope"][-1]["text"] == "Infrastructure Support"


def test_a_line_naming_a_few_sites_is_not_a_family():
    c = _content()
    c["sites"] = [dict(s) for s in SITES]
    c["scope"] = [{"text": "Visit Troy, MI once per week.", "cites": ["a"], "kind": "fact"},
                  {"text": "Replace the switch at Tupelo, MS.", "cites": ["b"], "kind": "fact"}]
    before = [x["text"] for x in c["scope"]]
    m._per_site_lines(c, [])
    assert [x["text"] for x in c["scope"]] == before


# --- 000132 C+: a service the summary promises is a task ---------------------

def test_a_summary_service_is_also_a_task():
    c = _content()
    c["services"] = ["staff_augmentation", "support"]
    c["overview"] = [{"text": "Provider will perform Windows and Linux operating system patching and upgrades, troubleshooting and "
                              "incident response, and end-user support including O365/M365 clients.", "cites": [], "kind": "method"}]
    c["scope"] = [{"text": "Infrastructure Support", "cites": [], "kind": "heading"},
                  {"text": "Support physical server and NAS environments.", "cites": ["s1"], "kind": "fact"},
                  {"text": "Perform operating system patching and upgrades for Windows and Linux.", "cites": ["s2"], "kind": "fact"},
                  {"text": "Closeout", "cites": [], "kind": "heading"}]
    atoms = [{"id": "e1", "atom_type": "deal_metadata", "text": "End-user support including O365/M365 clients"},
             {"id": "e2", "atom_type": "deal_metadata", "text": "Support for operating system patches and upgrades (Windows + Linux)"},
             {"id": "e3", "atom_type": "commitment", "text": "24/7 on-call support"}]
    m._summary_services_tasked(c, atoms)
    t = [x["text"] for x in c["scope"]]
    assert t.count("Provide end-user support, including O365/M365 clients.") == 1
    assert t.index("Provide end-user support, including O365/M365 clients.") == 3
    assert not any("24/7" in x for x in t)
    added = next(x for x in c["scope"] if x["text"].startswith("Provide end-user"))
    assert added["cites"] == ["e1"]


def test_a_project_job_gets_no_summary_service_task():
    c = _content()
    c["services"] = ["installation"]
    c["overview"] = [{"text": "Provider will provide end-user support including O365/M365 clients.", "cites": [], "kind": "method"}]
    m._summary_services_tasked(c, [{"id": "e1", "atom_type": "scope_item", "text": "End-user support including O365/M365 clients"}])
    assert not c["scope"]
