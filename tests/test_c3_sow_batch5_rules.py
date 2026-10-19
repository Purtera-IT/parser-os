"""SOW writer rules, batch 5 (pass U grades: 010373 A-, 000132 B-, 010209 B-)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


# --- 010373 A-: a spelled count is the count a fact gives -------------------

def test_a_spelled_count_a_fact_gives_is_supported():
    pool = "16 humanscale monitor arms on new desks. total of 48 monitors, 3 per arm. 530 fifth avenue"
    lead = "Provider will install sixteen (16) Humanscale monitor arms and forty-eight (48) 27-inch monitors at 530 Fifth Avenue."
    assert not m._unsupported_specifics(lead, pool + " 27\"")
    assert not m._unsupported_specifics("Assume all forty-eight monitors are on site.", pool)
    # A count no fact gives still goes, spelled or not.
    assert m._unsupported_specifics("Install forty-nine (49) monitors.", pool)
    assert m._unsupported_specifics("Install twenty monitors.", pool)


def test_the_check_keeps_counted_method_lines_and_says_why_it_drops():
    lines = {"a": "16 Humanscale monitor arms on the new desks", "b": "3 new 27\" monitors per arm, total of 48 monitors"}
    content = {k: [] for k in m.LISTS}
    content["overview"] = [{"text": "Provider will install sixteen (16) Humanscale monitor arms and forty-eight (48) 27-inch monitors.",
                            "cites": [], "kind": "method"}]
    content["assumptions"] = [{"text": "Assume all sixteen (16) monitor arms and forty-eight (48) monitors are on site.",
                               "cites": [], "kind": "method"},
                              {"text": "Assume seventy (70) cables are on site.", "cites": [], "kind": "method"}]
    m.check(content, lines)
    assert len(content["overview"]) == 1 and len(content["assumptions"]) == 1
    assert [x["text"] for x in content["dropped_unsupported"]] == ["Assume seventy (70) cables are on site."]


def test_a_counted_supply_line_is_grounded_on_its_digits():
    x = {"text": "Supply sixteen (16) monitor arms and forty-eight (48) monitors, plus mounting hardware and cables.",
         "cites": ["a", "b"]}
    assert m._grounded(x, {"a": "16 humanscale monitor arms", "b": "total of 48 monitors"})


def test_a_counted_cited_step_takes_the_standard_steps_slot():
    """010373: the standard "Install each monitor arm" stayed in its slot and
    the cited "Install sixteen (16) ..." was appended after cable management."""
    content = {"scope": [
        {"text": "Monitor Installation", "cites": [], "kind": "heading"},
        {"text": "Install sixteen (16) Humanscale monitor arms on the new desks.", "cites": ["a"], "kind": "fact"},
        {"text": "Mount forty-eight (48) new 27-inch monitors on the arms, three (3) per arm, and secure them.",
         "cites": ["b"], "kind": "fact"}],
        "customer_materials": [{"text": "Sixteen (16) monitor arms and forty-eight (48) monitors, plus mounting hardware and cables.",
                                "cites": ["a", "b"], "kind": "fact"}]}
    facts = ["16 humanscale monitor arms on the new desks", "3 monitors per arm, 48 monitors"]
    m._apply_standards(content, "smart_hands_general", facts)
    steps = [x["text"] for x in content["scope"] if x.get("kind") != "heading"]
    i_chk = next(i for i, t in enumerate(steps) if t.startswith("Check in"))
    i_ins = steps.index("Install sixteen (16) Humanscale monitor arms on the new desks.")
    i_mnt = next(i for i, t in enumerate(steps) if t.startswith("Mount forty-eight"))
    i_cab = next(i for i, t in enumerate(steps) if t.startswith("Perform basic cable management"))
    assert i_chk < i_ins < i_mnt < i_cab
    assert not any(t.startswith("Install each") or t.startswith("Mount the monitors") for t in steps)
    assert "align and secure them" in steps[i_mnt]
    # The counted supply line, not the uncounted standard duty.
    assert not any("Provide all monitor arms" in str(x.get("text")) for x in content.get("customer_responsibilities") or [])
    why = {s["text"]: s["reason"] for s in content["standards_suppressed"]}
    assert "takes its place" in why["Install each monitor arm on its desk at the location the customer designates."]


def test_same_action_needs_the_same_verb_and_object():
    assert m._same_action("Install each monitor arm on its desk.", "Install sixteen (16) Humanscale monitor arms on desks.")
    assert not m._same_action("Install each monitor arm on its desk.", "Install forty-eight (48) monitors on the arms.")
    assert not m._same_action("Mount the monitors on each arm.", "Install forty-eight (48) monitors.")


def test_the_summary_leads_with_the_work_and_closes_with_cleanup():
    content = {"overview": [{"text": "Provider will haul away installation trash and packing materials.", "cites": ["t"], "kind": "fact"},
                            {"text": "Provider will install sixteen (16) monitor arms.", "cites": ["a"], "kind": "fact"}]}
    m._traced(content, "summary_order", m._summary_order, content)
    assert content["overview"][0]["text"].startswith("Provider will install")


# --- 000132 B-: a clause that loses its object goes whole ----------------------

def test_a_withheld_commitment_takes_its_whole_clause():
    rx = m.re.compile(r"(?i)24\s*/\s*7|on-?call")
    t = "Provider will perform weekly on-site visits for four (4) to eight (8) hours as needed and remain available for 24/7 on-call support."
    assert m._drop_clause(t, rx) == "Provider will perform weekly on-site visits for four (4) to eight (8) hours as needed."
    assert m._drop_clause("Provide 24/7 on-call support.", rx) == ""


def test_no_line_ends_on_a_preposition():
    c = {"overview": [{"text": "Provider will perform weekly visits as needed and remain available for.", "cites": [], "kind": "fact"},
                      {"text": "Provide support for.", "cites": [], "kind": "fact"}],
         "arrival_protocol": [{"text": "Provide the visitor badge and parking pass at check-in.", "cites": [], "kind": "fact"},
                              {"text": "Reconnect everything like for like.", "cites": [], "kind": "fact"},
                              {"text": "Technicians check in.", "cites": [], "kind": "fact"}]}
    m._sentence_integrity(c)
    assert [x["text"] for x in c["overview"]] == ["Provider will perform weekly visits as needed."]
    assert len(c["arrival_protocol"]) == 3


def test_the_summary_never_names_visit_logistics_as_a_service():
    c = {"services": ["support"], "overview": [{"text": "Provider will support network and servers.", "cites": ["a"], "kind": "fact"}],
         "sites": [], "scope": [
             {"text": "Coordinate arrival and site access with the designated onsite contact.", "cites": ["b"], "kind": "fact"},
             {"text": "Provide end-user support, including O365/M365 clients.", "cites": ["c"], "kind": "fact"}]}
    m._summary_names_job(c)
    added = c["overview"][-1]["text"]
    assert added == "Provider will also provide end-user support, including O365/M365 clients."


def test_locations_name_every_site_of_a_city_list():
    c = {"locations": [{"text": "Hudson, WI.", "cites": ["h"], "kind": "fact"}],
         "sites": [{"city": "Delphos", "state": "OH"}, {"city": "Hudson", "state": "WI"}]}
    m._locations_cover_sites(c)
    assert [x["text"] for x in c["locations"]] == ["Hudson, WI.", "Delphos, OH."]


def test_a_counted_task_the_summary_repeats_stays_a_task():
    t = "Perform weekly on-site visits lasting approximately four (4) to eight (8) hours as needed."
    c = {"overview": [{"text": "Provider will perform weekly on-site visits for four (4) to eight (8) hours as needed.", "cites": ["v"], "kind": "fact"}],
         "scope": [{"text": "Onsite Support", "cites": [], "kind": "heading"}, {"text": t, "cites": ["v"], "kind": "fact"}]}
    m._grounding(c, [_fact("v", "On-site visit once per week, 4-8 hours as needed")])
    assert any(x["text"] == t for x in c["scope"])
