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


# --- 010209 B-: another party's phase, schedule voice, placed terms ------------

_LIMIT = {"text": "Provider's scope is limited to physical onsite services; Provider does not procure, receive, stage, configure, "
                  "package or ship equipment.", "cites": [], "kind": "method", "standard": True}


def test_another_partys_phase_is_neither_a_task_nor_a_duty():
    fs = [_fact("r", "Validate receipt with store manager"),
          _fact("b", "Provide serial numbers of benched equipment relating to the defined store", under="48 port switch components:"),
          _fact("n", "Label all equipment with RL standard naming convention", under="48 port switch components:"),
          _fact("l", "Label every cable before removing the old switch")]
    c = {"assumptions": [_LIMIT],
         "scope": [{"text": "Validate receipt with the store manager before beginning work.", "cites": ["r"], "kind": "fact"},
                   {"text": "Label every cable before removing any old switch.", "cites": ["l"], "kind": "fact"},
                   {"text": "Pack removed switches for shipment for data wipe and disposal.", "cites": ["l"], "kind": "fact"},
                   {"text": "Stage the removed legacy switches onsite in a customer-designated area.", "cites": [], "kind": "method", "standard": True},
                   {"text": "Capture and provide the serial number of each new switch installed.", "cites": [], "kind": "method", "standard": True}],
         "customer_responsibilities": [{"text": "Provide the RL standard naming convention for each store", "cites": ["n"], "kind": "fact"}],
         "schedule": [{"text": "If equipment is received earlier, the project may start earlier.", "cites": ["r"], "kind": "fact"}]}
    m._traced(c, "partner_phase", m._partner_phase, c, fs)
    assert [x["text"] for x in c["scope"]] == ["Label every cable before removing any old switch.",
                                               "Stage the removed legacy switches onsite in a customer-designated area."]
    assert c["customer_responsibilities"] == [] and len(c["schedule"]) == 1


def test_schedule_lines_read_in_sow_voice():
    c = {"schedule": [{"text": "Work Hours: All work 7-10am Monday, Tuesday, Thursday", "cites": ["h"], "kind": "fact"},
                      {"text": "Scheduling Constraint: Any site not complete by 11/19 will be scheduled once we are out of holiday "
                               "blackout starting mid-January", "cites": ["h"], "kind": "fact"},
                      {"text": "Work Hours: tomorrow (Wednesday 7/8) 2 - 3:00pm EST", "cites": ["h"], "kind": "fact"},
                      {"text": "Requested Completion Date: week of July 6th", "cites": [], "kind": "fact"}]}
    m._schedule_voice(c, "2026-08-16T00:00:00Z", {"h"})
    assert [x["text"] for x in c["schedule"]] == [
        "Work Hours: 7:00 AM to 10:00 AM local site time, Monday, Tuesday, Thursday",
        "Scheduling Constraint: Any site not complete by November 19, 2026 will be scheduled once the Customer is out of "
        "holiday blackout starting mid-January 2027",
        "Requested Completion Date: week of July 6, 2026"]


def test_purtera_we_is_not_the_customer():
    c = {"schedule": [{"text": "Scheduling Constraint: We will ask about blackout dates", "cites": ["p"], "kind": "fact"}]}
    m._schedule_voice(c, "2026-09-01", set())
    assert c["schedule"][0]["text"] == "Scheduling Constraint: We will ask about blackout dates"


def test_one_summary_sentence_per_claim():
    c = {"overview": [
        {"text": "Provider will replace store switches at about thirty (30) sites, with each store receiving two (2) to five (5) switches.",
         "cites": ["a"], "kind": "fact"},
        {"text": "Provider will label all cables before removal.", "cites": ["b"], "kind": "fact"},
        {"text": "Provider will replace two (2) to five (5) switches per store across approximately thirty (30) sites, with "
                 "approximately fifty (50) 24-port switches.", "cites": ["c"], "kind": "fact"}]}
    m._summary_order(c)
    assert [x["cites"] for x in c["overview"]] == [["c"], ["b"]]


def test_a_cited_fact_whose_term_no_line_carries_goes_to_the_must_place_pass(monkeypatch):
    seen = []
    monkeypatch.setattr(m, "_complete", lambda prompt, *a, **k: (seen.append(prompt), '{"add": []}')[1])
    fs = [_fact("n", "We have had issues where the tech did not label EVERY cable (this extra time must be spent at no charge to RL)"),
          _fact("r", "Not all stores have a rack but still need rack mount hardware installed")]
    c = {k: [] for k in m.LISTS}
    c["scope"] = [{"text": "Label every cable before removing any old switch.", "cites": ["n"], "kind": "fact"},
                  {"text": "Install rack mount hardware where required.", "cites": ["r"], "kind": "fact"}]
    m._must_include(c, fs, None)
    assert seen and "[n]" in seen[0] and "[r]" in seen[0]
    c["assumptions"] = [{"text": "Time to correct Provider labeling errors is at no charge.", "cites": ["n"], "kind": "fact"},
                        {"text": "Rack mount hardware is installed even where a store has no rack.", "cites": ["r"], "kind": "fact"}]
    seen.clear()
    m._must_include(c, fs, None)
    assert not seen



# --- 010082 B-: exclusions stay exclusions; hours; re-cite; onsite contact -----

def test_grounding_never_moves_an_exclusion_into_scope():
    c = {"overview": [], "scope": [],
         "out_of_scope": [{"text": "Testing, certification, labeling, or as-built documentation.", "cites": ["q"], "kind": "fact"}]}
    m._grounding(c, [_fact("q", "Any testing, labeling, certification, or as-built documentation requirements")])
    assert c["scope"] == [] and c["out_of_scope"][0]["text"].startswith("Testing, certification")


def test_an_open_checklist_item_is_never_a_provider_task():
    c = {"scope": [{"text": "Provide testing, certification, labeling, or as-built documentation.", "cites": ["q"], "kind": "fact"}]}
    m._question_not_commitment(c, [_fact("q", "Any testing, labeling, certification, or as-built documentation requirements")])
    assert c["scope"] == []


def test_the_site_hours_line_is_kept_as_work_hours():
    c = {"scope": [], "schedule": [
        {"text": "Requested Start Date: July 13th 2026", "cites": ["s"], "kind": "fact"},
        {"text": "Requested Completion Date: Aug 28th 2026", "cites": ["s"], "kind": "fact"},
        {"text": "Site hours were stated as Monday through Friday, 8:00am to 3:00pm, for planning on-site work.", "cites": ["h"], "kind": "fact"}]}
    m._tidy(c)
    assert "Work Hours: Monday through Friday, 8:00am to 3:00pm" in [x["text"] for x in c["schedule"]]


def test_a_check_drop_is_re_cited_to_the_line_that_states_it(monkeypatch):
    lines = {"t": "It may have metal studs.", "p": "Walls may have metal studs", "x": "Black Cat6 preferred"}
    c = {k: [] for k in m.LISTS}
    c["assumptions"] = [{"text": "Assume the wall may have metal studs; use appropriate anchoring if wood studs are not available.",
                         "cites": ["t"], "kind": "fact"}]
    monkeypatch.setattr(m, "_complete", lambda *a, **k: '{"supported": []}')
    m.check(c, lines)
    assert c["assumptions"] and c["assumptions"][0]["cites"] == ["p"]


def test_only_the_named_onsite_contact_is_called_onsite():
    c = {"customer_responsibilities": [{"text": "Provide AJ Deuel as the main onsite technical contact.", "cites": ["a"], "kind": "fact"},
                                       {"text": "Provide Henry Hankins as the onsite contact.", "cites": ["h"], "kind": "fact"}]}
    m._contact_roles(c, [_fact("h", "On Site Contact Name | Henry Hankins (SDX GM)")])
    assert [x["text"] for x in c["customer_responsibilities"]] == ["Provide AJ Deuel as the main technical contact.",
                                                                  "Provide Henry Hankins as the onsite contact."]


# --- 010012 B: lines naming different rooms are never merged ------------------

def test_lines_naming_different_rooms_are_two_lines():
    a = "Install one (1) Poly X72 bar with one (1) TC10 controller in Conference Room 2102."
    b = "Install one (1) Poly X72 bar with one (1) TC10 controller in Shared Office 2105."
    c = {k: [] for k in m.LISTS}
    c["scope"] = [{"text": a, "cites": ["x"], "kind": "fact"}, {"text": b, "cites": ["x"], "kind": "fact"},
                  {"text": a, "cites": ["x"], "kind": "fact"}]
    m._dedupe_items(c)
    assert [x["text"] for x in c["scope"]] == [a, b]
    c["overview"] = [{"text": "Provider will install two (2) Poly X72 bars with TC10s.", "cites": ["x"], "kind": "fact"}]
    m._overview_covers_scope(c, [])
    assert len(c["overview"]) == 1  # one room's share is not the job's count
