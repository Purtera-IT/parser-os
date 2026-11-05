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
