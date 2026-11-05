"""A store list's rows are sites, not parts: the customer-BOM rule skips lines the heads type as a place
(010035, 2026-11-05: 168 Goodyear store rows were listed as customer-supplied materials)."""
from __future__ import annotations

from app.core import c3_sow as m


def _f(i, text, type_):
    return {"id": i, "section": "scope_of_work", "text": text, "under": "", "type": type_, "when": "", "supplier": None}


def test_store_rows_are_not_customer_materials():
    fs = [_f("s1", "1139 | 1663 Penfield Road | Rochester | NY | 14625-2549", "physical_site"),
          _f("s2", "4226 | 3209 Kirby Drive | Pearland | TX | 77584-8466", "physical_site"),
          _f("b1", "12 | PA-440 firewall", "bom_line")]
    c = {"customer_materials": []}
    m._customer_bom_rows(c, fs, lambda i: True)
    assert [x["cites"] for x in c["customer_materials"]] == [["b1"]]


def test_parts_rows_still_go_to_the_customer():
    fs = [_f("b1", "12 | PA-440 firewall", "bom_line"), _f("b2", "4: SFP-10G-SR", "")]
    c = {}
    m._customer_bom_rows(c, fs, lambda i: i == "b1")
    assert [x["text"] for x in c["customer_materials"]] == ["12 x PA-440 firewall", "4 x SFP-10G-SR"]


def test_a_schedule_line_the_facts_state_is_cited_not_dropped():
    lines = {"f1": "Assume regular business hours.", "f2": "Site No.: dc 5837 | State: TX"}
    c = {k: [] for k in m.LISTS}
    c["schedule"] = [{"text": "Work Window: regular business hours", "cites": [], "kind": "method"},
                     {"text": "Work Window: overnight weekends only", "cites": [], "kind": "method"}]
    m.check(c, lines)
    kept = {x["text"]: x.get("cites") for x in c["schedule"]}
    assert kept.get("Work Window: regular business hours") == ["f1"]
    assert "Work Window: overnight weekends only" not in kept


def test_cited_schedule_lines_say_what_their_cites_say():
    lines = {"f1": "Start date Feb 1st , 2027, and have a 6-month deadline.", "f2": "Assume regular business hours."}
    c = {k: [] for k in m.LISTS}
    c["schedule"] = [{"text": "Requested Start Date: February 1, 2027", "cites": ["f1"], "kind": "fact"},
                     {"text": "Requested Completion Date: Monday, February 1, 2027", "cites": ["f1"], "kind": "fact"},
                     {"text": "Work Window: Weekend (Sunday)", "cites": ["f1"], "kind": "fact"},
                     {"text": "Work Window: regular business hours", "cites": ["f1"], "kind": "fact"}]
    m.check(c, lines)
    kept = {x["text"]: x["cites"] for x in c["schedule"]}
    assert kept == {"Requested Start Date: February 1, 2027": ["f1"], "Work Window: regular business hours": ["f2"]}


def test_a_completion_date_from_start_plus_span_is_kept():
    fs = [{"id": "f1", "text": "Start date Feb 1st , 2027, and have a 6-month deadline.", "section": "timeline", "type": "milestone_phase"}]
    c = {"schedule": [{"text": "Requested Start Date: Feb 1, 2027", "cites": ["f1"], "kind": "fact"},
                      {"text": "Requested Completion Date: Aug 1, 2027", "cites": ["f1"], "kind": "fact"}],
         "overview": []}
    m._fix_dates(c, fs, "2026-10-20")
    assert [x["text"] for x in c["schedule"]][1] == "Requested Completion Date: Aug 1, 2027"


def test_a_completion_date_no_fact_gives_is_still_redated():
    fs = [{"id": "f1", "text": "Start date Feb 1st , 2027.", "section": "timeline", "type": "milestone_phase"}]
    c = {"schedule": [{"text": "Requested Completion Date: Aug 1, 2027", "cites": ["f1"], "kind": "fact"}], "overview": []}
    m._fix_dates(c, fs, "2026-10-20")
    assert "Aug" not in c["schedule"][0]["text"]


def test_a_sunday_deadline_is_not_weekend_work():
    fs = [{"id": "f2", "text": "Assume regular business hours.", "section": "assumptions"}]
    c = {"schedule": [{"text": "Requested Start Date: February 1, 2027", "cites": ["f1"], "kind": "fact"},
                      {"text": "Requested Completion Date: August 1, 2027", "cites": ["f1"], "kind": "fact"},
                      {"text": "Work Window: regular business hours", "cites": ["f2"], "kind": "fact"}],
         "assumptions": []}
    m._weekend(c, fs)
    assert [x["text"] for x in c["schedule"]][-1] == "Work Window: regular business hours"


def test_a_sunday_start_is_still_weekend_work():
    c = {"schedule": [{"text": "Requested Start Date: August 16, 2026", "cites": ["f1"], "kind": "fact"}], "assumptions": []}
    m._weekend(c, [])
    assert any("Weekend (Sunday)" in x["text"] for x in c["schedule"])


def test_site_rows_do_not_crowd_out_the_deals_lines(monkeypatch):
    monkeypatch.setattr(m, "MAX_FACTS", 5)
    atoms = [{"id": f"s{i}", "atom_type": "physical_site", "text": f"{1000+i} | {i} Main St | Town | NY"} for i in range(8)]
    atoms.append({"id": "d1", "atom_type": "scope_item", "text": "All sites completed by Thanksgiving 2026"})
    env = {"atoms": atoms, "c3_brief": {"sections": {"scope_of_work": [{"id": a["id"], "text": a["text"]} for a in atoms]}}}
    ids = [f["id"] for f in m.facts(env)]
    assert "d1" in ids and len(ids) == 5


def test_one_sites_row_is_not_the_jobs_schedule():
    env = {"atoms": [{"id": "s1", "atom_type": "physical_site", "text": "Site No.: dc 8615 | Day 1 ... 2:00 PM to 2:00 AM"},
                     {"id": "f1", "atom_type": "milestone_phase", "text": "Start date Feb 1st , 2027"}]}
    c = {"schedule": [{"text": "Requested Start Date: February 1, 2027", "cites": ["f1"], "kind": "fact"},
                      {"text": "Work Hours: Fri, December 1, 2026, 2:00 PM to 2:00 AM", "cites": ["s1"], "kind": "fact"}]}
    m._site_rows_not_schedule(c, env)
    assert [x["text"] for x in c["schedule"]] == ["Requested Start Date: February 1, 2027"]


def test_counts_and_paraphrased_windows_stand_and_uncited_dates_take_their_line():
    lines = {"f1": "Start date Feb 1st , 2027, and have a 6-month deadline.",
             "f2": "Two techs will survey business hours at the six sites with no after-hours access.",
             "f3": "We will staff 2 resources."}
    c = {k: [] for k in m.LISTS}
    c["schedule"] = [{"text": "Number of Resources: two (2)", "cites": ["f3"], "kind": "fact"},
                     {"text": "Work Window: business hours for six (6) no-after-hours-access sites", "cites": ["f2"], "kind": "fact"},
                     {"text": "Requested Start Date: February 1, 2027", "cites": [], "kind": "method"},
                     {"text": "Requested Completion Date: August 1, 2027", "cites": [], "kind": "method"}]
    m.check(c, lines)
    kept = {x["text"]: x["cites"] for x in c["schedule"]}
    assert kept == {"Number of Resources: two (2)": ["f3"],
                    "Work Window: business hours for six (6) no-after-hours-access sites": ["f2"],
                    "Requested Start Date: February 1, 2027": ["f1"], "Requested Completion Date: August 1, 2027": ["f1"]}


def test_an_effective_date_is_not_the_start_date():
    lines = {"f1": "Effective Date: June 17, 2026"}
    c = {k: [] for k in m.LISTS}
    c["schedule"] = [{"text": "Requested Start Date: June 17, 2026", "cites": ["f1"], "kind": "fact"}]
    m.check(c, lines)
    assert c["schedule"] == []
