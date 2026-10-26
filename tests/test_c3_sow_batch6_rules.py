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


# --- 010419 pass V: the job family is pinned per deal ------------------------

import json as _json
import os as _os

import pytest as _pytest

from tests.test_c3_sow_batch4_rules import _SAVED

#: Each deal's job family from pass U (wave_U logs), except 010376, which
#: 0cfb740 deliberately moved from decommission_packing to smart_hands_general
#: (a replacement job is not dismantle-and-pack; pass S also had it there).
PINNED_FAMILY = {
    "000123": "pos_install_refresh", "000132": "smart_hands_general", "010012": "av_conference_room",
    "010043": "cameras_security", "010082": "low_voltage_cabling", "010083": "network_switch_refresh",
    "010085": "network_switch_refresh", "010115": "decommission_packing", "010116": "av_conference_room",
    "010120": "av_conference_room", "010147": "relocation_move", "010171": "av_conference_room",
    "010198": "pos_install_refresh", "010209": "network_switch_refresh", "010217": "av_conference_room",
    "010264": "time_clock_kiosk", "010279": "pos_install_refresh", "010326": "smart_hands_general",
    "010373": "smart_hands_general", "010376": "smart_hands_general", "010419": "smart_hands_general",
}


def _saved_env(deal):
    path = f"{_SAVED}/final_{deal}_env.json"
    if not _os.path.exists(path):
        _pytest.skip("saved envelope not on this machine")
    env = _json.load(open(path))
    if not env.get("c3_sow"):
        _pytest.skip("saved envelope has no c3_sow")
    return env


@_pytest.mark.parametrize("deal", sorted(PINNED_FAMILY))
def test_the_routed_family_is_the_pinned_one(deal):
    """The writer's own call (as the envelope recorded it) routes to the
    deal's family; a smart-hands deal also gets there from a generic "other"
    call, the run-to-run wobble that cost 010419 its 97 standards."""
    env = _saved_env(deal)
    fs = m.facts(env)
    want = PINNED_FAMILY[deal]
    assert m._route_family(str(env["c3_sow"].get("job_family") or "other"), fs) == want
    assert m._route_family(want, fs) == want
    # 010376 is smart hands by its own call (a replacement); "other" there
    # is not a wobble the facts can settle, so only the recurring, staffing
    # and phone-swap deals are held to it.
    if want == "smart_hands_general" and deal != "010376":
        assert m._route_family("other", fs) == want


def test_technician_staffing_for_a_transition_is_smart_hands():
    fs = [_fact("a", "Typically, we need one network technician and one help desk technician per community."),
          _fact("b", "They will ensure that all systems are functioning as they were before the transition.")]
    assert m._route_family("other", fs) == "smart_hands_general"
    # A question about what the onsite technician does is not staffing.
    assert m._route_family("other", [_fact("a", "Will the onsite technician need to handle device configuration?")]) == "other"


def test_a_cutover_or_takeover_call_of_other_is_smart_hands():
    assert m._route_family("other", [_fact("a", "We are handling the IT cutover from the outgoing MSP at the clinic.")]) == "smart_hands_general"
    assert m._route_family("other", [_fact("a", "PurTera is taking over the IT support for the site next month.")]) == "smart_hands_general"


def test_a_family_without_standards_takes_the_smart_hands_block_only():
    shg = m.purtera_standards("smart_hands_general")
    assert shg and m.purtera_standards("other") == shg
    # A specific trade's block is never borrowed (wireless_ap -> cabling).
    assert m.purtera_standards("wireless_ap") == {}
    assert m.purtera_standards("av_conference_room") != shg


# --- 010419 pass V: a hedged follow-up is never a commitment -----------------

def test_a_hedged_follow_up_leaves_the_assumption_and_becomes_the_exclusion():
    c = _content()
    c["assumptions"] = [{"text": "The visit can be completed in one full day, with any remaining items handled the next day if needed.",
                         "cites": [], "kind": "method"}]
    c["scope"] = [{"text": "Return the next day to complete any remaining items.", "cites": ["f"], "kind": "fact"},
                  {"text": "Validate that systems function as before.", "cites": ["g"], "kind": "fact"}]
    atoms = [{"id": "h", "atom_type": "scope_item",
              "text": "This is usually a full day of work, with some follow-up needed the next day to complete any remaining items."}]
    m._followup_not_promised(c, atoms)
    assert [x["text"] for x in c["assumptions"]] == ["The visit can be completed in one full day."]
    assert [x["text"] for x in c["scope"]] == ["Validate that systems function as before."]
    assert c["out_of_scope"] == [{"text": "Follow-up visits unless authorized in writing or by Change Order.", "cites": ["h"], "kind": "fact"}]


def test_a_firm_follow_up_stays():
    c = _content()
    c["schedule"] = [{"text": "Day 2: follow-up visit to finish testing.", "cites": ["f"], "kind": "fact"}]
    m._followup_not_promised(c, [{"id": "f", "atom_type": "scope_item", "text": "Day 2 is a follow-up visit to finish testing."}])
    assert len(c["schedule"]) == 1 and not c["out_of_scope"]


# --- 010419 pass V: a job that installs nothing reads in its own words -------

def test_install_wording_on_a_non_install_job_is_reworded():
    c = _content()
    c["scope"] = [{"text": "Provide local onsite technician support.", "cites": ["a"], "kind": "fact"},
                  {"text": "Document any installation exceptions, access issues, damaged fittings, missing equipment, or blockers.", "cites": ["a"], "kind": "method"}]
    c["provider_tools"] = [{"text": "Standard hand tools suitable for device installation activities.", "cites": [], "kind": "method"}]
    c["customer_responsibilities"] = [
        {"text": "Ensure technicians have safe and authorized access to the installation location.", "cites": [], "kind": "method"},
        {"text": "Provide any required equipment, cables, labels, mounting materials, or tools not normally carried by the technician.", "cites": [], "kind": "method"},
        {"text": "Confirm the mounting height of each display.", "cites": [], "kind": "method"}]
    atoms = [{"id": "a", "atom_type": "scope_item", "text": "Typically, we need one network technician and one help desk technician per community."}]
    m._no_install_wording(c, atoms)
    assert c["scope"][1]["text"] == "Document any exceptions, access issues, missing equipment, or blockers."
    assert c["provider_tools"][0]["text"] == "Standard hand tools suitable for onsite support activities."
    assert [x["text"] for x in c["customer_responsibilities"]] == [
        "Ensure technicians have safe and authorized access to the work location.",
        "Provide any required equipment, cables, labels, or tools not normally carried by the technician."]


def test_install_wording_stays_on_an_install_job():
    c = _content()
    c["scope"] = [{"text": "Install two (2) displays in Room 4.", "cites": ["a"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Ensure access to the installation location.", "cites": [], "kind": "method"}]
    m._no_install_wording(c, [{"id": "a", "atom_type": "scope_item", "text": "Need help with 2 displays in room 4"}])
    assert c["customer_responsibilities"][0]["text"] == "Ensure access to the installation location."


# --- 010209 pass V: schedule and step hygiene --------------------------------

def test_customer_hour_estimates_are_not_schedule_lines():
    c = _content()
    c["schedule"] = [{"text": "2 switches are estimated at 2.0 hours per store.", "cites": ["a"], "kind": "fact"},
                     {"text": "Estimated Duration: one (1) onsite workday of eight (8) hours per technician", "cites": [], "kind": "method"},
                     {"text": "Work Hours: 7:00 AM to 10:00 AM local site time", "cites": ["b"], "kind": "fact"}]
    m._schedule_not_estimates(c)
    assert [x["text"][:14] for x in c["schedule"]] == ["Estimated Dura", "Work Hours: 7:"]


def test_a_work_window_the_deal_extends_says_so():
    c = _content()
    c["schedule"] = [{"text": "Work Window: October 13, 2026 through November 19, 2026", "cites": ["w"], "kind": "fact"},
                     {"text": "Scheduling Constraint: Any site not complete by November 19, 2026 will be scheduled once the Customer is "
                              "out of holiday blackout starting mid-January 2027", "cites": ["r"], "kind": "fact"}]
    m._window_rollover(c)
    assert c["schedule"][0]["text"] == ("Work Window: October 13, 2026 through November 19, 2026; any site not complete by then "
                                        "is scheduled from mid-January 2027")
    assert c["schedule"][0]["cites"] == ["w", "r"]


def test_the_universal_copy_of_a_qualified_step_goes():
    c = _content()
    c["scope"] = [{"text": "Install rack mount hardware where required, including stores without an existing rack.", "cites": ["a"], "kind": "fact"},
                  {"text": "Install rack mount hardware at every store, including stores without an existing rack.", "cites": ["a"], "kind": "fact"},
                  {"text": "Label every cable at every store.", "cites": ["b"], "kind": "fact"}]
    m._qualified_over_universal(c)
    assert [x["text"][:20] for x in c["scope"]] == ["Install rack mount h", "Label every cable at"]
    assert "where required" in c["scope"][0]["text"]


def test_nothing_new_goes_in_before_the_old_gear_comes_out():
    c = _content()
    t = lambda s_, std=False: {"text": s_, "cites": [] if std else ["x"], "kind": "fact", **({"standard": True} if std else {})}
    c["scope"] = [{"text": "Switch replacement", "cites": [], "kind": "heading"},
                  t("Arrive onsite and check in.", True),
                  t("Label every cable before removing the existing switches."),
                  t("Install the new power cords with the replacement switches."),
                  t("Manage all cables with Velcro and no zip ties."),
                  t("Take after photos of the completed switch replacement and cabling."),
                  t("Confirm the switches and patch cables are onsite before starting.", True),
                  t("Remove the existing switches and install the replacement switches."),
                  t("Install the new switches in the existing rack.", True),
                  t("Connect power to each new switch.", True),
                  {"text": "Closeout", "cites": [], "kind": "heading"},
                  t("Take prework photographs of the existing switch locations.")]
    sc = c["scope"]
    # The before photos leave Closeout for the first work group, after arrival.
    pre = sc.pop()
    m._field_sequence(c)
    got = [x["text"][:22] for x in c["scope"]]
    assert got == ["Switch replacement", "Arrive onsite and chec", "Confirm the switches a", "Label every cable befo",
                   "Remove the existing sw", "Install the new power ", "Install the new switch", "Connect power to each ",
                   "Manage all cables with", "Take after photos of t", "Closeout"]
    assert pre["text"].startswith("Take prework")


def test_a_removal_the_deal_puts_after_installation_is_no_anchor():
    body = [{"text": "Install the new rack.", "cites": ["a"], "kind": "fact"},
            {"text": "Remove existing infrastructure, including APs, after installation.", "cites": ["b"], "kind": "fact"}]
    assert m._replacement_order(list(body)) == body


def test_prework_photos_never_go_under_closeout():
    c = _content()
    c["scope"] = [{"text": "Switch replacement", "cites": [], "kind": "heading"},
                  {"text": "Arrive onsite and check in.", "cites": [], "kind": "method", "standard": True},
                  {"text": "Label every cable before removing the existing switches.", "cites": ["a"], "kind": "fact"},
                  {"text": "Take prework photographs of the existing switch locations and connected cabling.", "cites": ["p"], "kind": "fact"},
                  {"text": "Closeout", "cites": [], "kind": "heading"},
                  {"text": "Confirm completion with the onsite contact.", "cites": [], "kind": "method"}]
    m._grounding(c, [_fact("a", "Label every cable"), _fact("p", "Take before pictures")])
    got = [x["text"][:20] for x in c["scope"]]
    assert got.index("Take prework photogr") == 2 and got.index("Closeout") == 4, got


# --- 010326 pass V -------------------------------------------------------------

def test_a_helping_verb_is_not_an_object():
    assert m._collides("Provider will assist with racking, stacking, and wall mounting the hardware.",
                       "Customer will handle rack hardware and rack/stack gear, including gateways.")


def test_a_question_is_never_equipment_or_a_summary_claim():
    c = _content()
    c["equipment_in_scope"] = [{"text": "All devices on the BOM are in scope for onsite installation.", "cites": ["q"], "kind": "fact"}]
    c["overview"] = [{"text": "All devices on the BOM are installed onsite.", "cites": ["q"], "kind": "fact"}]
    f = _fact("q", "Are all devices from the BOM are actually in scope for onsite installation?", section="answered_question")
    f["asked"] = True
    m._question_not_commitment(c, [f])
    assert not c["equipment_in_scope"] and not c["overview"]


def test_a_description_cell_is_not_a_task():
    c = _content()
    c["scope"] = [{"text": "Provider will handle the meter dial-up connection for MPU MO Harmonica.", "cites": ["d"], "kind": "fact"},
                  {"text": "Install 242 cash drawer cables.", "cites": ["t"], "kind": "fact"}]
    m._cell_not_task(c, [_fact("d", "Description: Dial up connection for Meter"),
                         _fact("t", "Description | Part # | Quantity Needed | Cash Drawer Cables | 242")])
    assert [x["text"] for x in c["scope"]] == ["Install 242 cash drawer cables."]


def test_licenses_and_self_installed_gear_are_not_customer_materials():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Customer will handle rack hardware and rack/stack gear, including gateways.", "cites": ["r"], "kind": "fact"},
                                      {"text": "Provide the 9U rack.", "cites": ["k"], "kind": "fact"}]
    c["customer_materials"] = [{"text": "Customer furnishes Cisco VG400 Analog Voice Gateways, quantity 2.", "cites": ["a"], "kind": "fact"},
                               {"text": "Customer furnishes Cisco Unified Communications Security licenses, quantity 2.", "cites": ["b"], "kind": "fact"},
                               {"text": "Customer furnishes Cisco Smart Net Total Care service, quantity 2.", "cites": ["c"], "kind": "fact"},
                               {"text": "Customer furnishes eighteen (18) Cisco ATA191 phone adapters.", "cites": ["d"], "kind": "fact"}]
    m._materials_onsite_only(c)
    assert [x["text"] for x in c["customer_materials"]] == ["Customer furnishes eighteen (18) Cisco ATA191 phone adapters."]


def test_a_supplied_rack_stays_material():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide the 9U rack and ensure it is onsite.", "cites": ["k"], "kind": "fact"}]
    c["customer_materials"] = [{"text": "One (1) 9U rack.", "cites": ["k"], "kind": "fact"}]
    m._materials_onsite_only(c)
    assert len(c["customer_materials"]) == 1


def test_a_standard_exclusion_the_check_dropped_comes_back():
    c = _content()
    c["scope"] = [{"text": "Disconnect and remove each existing phone and install its replacement.", "cites": ["s"], "kind": "fact"}]
    c["dropped"] = [{"list": "out_of_scope", "text": "Disposal, shipping, data destruction, or asset disposition of removed phones.",
                     "cites": ["a"], "kind": "fact", "reason": "check: the cited lines do not state it"},
                    {"list": "out_of_scope", "text": "Exclude the quarterly vendor meeting.", "cites": ["b"], "kind": "fact",
                     "reason": "check: the cited lines do not state it"}]
    m._standard_exclusion_survives(c)
    assert [x["text"] for x in c["out_of_scope"]] == ["Disposal, shipping, data destruction, or asset disposition of removed phones."]
    assert [d["text"] for d in c["dropped"]] == ["Exclude the quarterly vendor meeting."]


# --- pass V cross-deal: one statement per step, never broader than the source -

def test_a_standard_a_deal_step_already_says_collapses_into_it():
    c = _content()
    c["scope"] = [{"text": "Arrive onsite during regular business hours and check in through the middle gate.", "cites": ["a"], "kind": "fact"},
                  {"text": "Arrive onsite during the Customer-approved regular business hours.", "cites": [], "kind": "method", "standard": True},
                  {"text": "Perform functional testing of the Square register.", "cites": ["b"], "kind": "fact"},
                  {"text": "Perform functional testing of each installed device before leaving the site.", "cites": [], "kind": "method", "standard": True},
                  {"text": "Clean the work area and confirm completion with the onsite contact.", "cites": ["c"], "kind": "fact"},
                  {"text": "Clean the immediate work area and remove Provider-generated installation debris.", "cites": [], "kind": "method", "standard": True}]
    m._one_statement(c)
    got = [x["text"][:30] for x in c["scope"]]
    assert "Arrive onsite during the Custo" not in got
    # Said of each device, broader than the one register: stays.
    assert "Perform functional testing of " in [g for g in got if g.startswith("Perform")][-1] and len([g for g in got if g.startswith("Perform")]) == 2
    # A clause of its own (debris): stays.
    assert "Clean the immediate work area " in got
    assert c["standards_suppressed"][0]["reason"].endswith("(one_statement)")


def test_a_line_that_drops_its_sources_limit_goes():
    c = _content()
    c["assumptions"] = [{"text": "The one (1) MDF and thirteen (13) raised IDFs can be reached with boom or scissor lifts as needed.",
                         "cites": ["a", "b"], "kind": "fact"},
                        {"text": "A forty-foot (40-foot) lift is available in warehouse areas.", "cites": ["b"], "kind": "fact"},
                        {"text": "Only one (1) store per time zone is worked at a time.", "cites": ["c"], "kind": "fact"}]
    atoms = [{"id": "a", "text": "They have 1 MDF reachable on foot but have 13 IDFs which are raised and would require some sort of lift."},
             {"id": "b", "text": "They said a majority of the IDFs would be reachable with a boom/scissor lift but would recommend a 40ft lift for the warehouse."},
             {"id": "c", "text": "Only 1 store per time zone"}]
    m._no_broader_than_source(c, atoms)
    assert [x["text"][:4] for x in c["assumptions"]] == ["Only"]


# --- 010043 pass V -------------------------------------------------------------

def test_a_standard_suppressed_for_coverage_comes_back_when_nothing_covers_it():
    c = _content()
    c["scope"] = [{"text": "Cabling", "cites": [], "kind": "heading"},
                  {"text": "Run approximately 146 ft of cable from the IDF to the front door camera location.", "cites": ["a"], "kind": "fact"},
                  {"text": "Test each installed cable for continuity and pinout.", "cites": ["b"], "kind": "fact"},
                  {"text": "Mount four (4) cameras at the confirmed door locations.", "cites": ["c"], "kind": "fact"},
                  {"text": "Closeout", "cites": [], "kind": "heading"}]
    c["standards_suppressed"] = [
        {"section": "method_steps", "text": "Terminate and label each cable run at the camera location and the IDF.",
         "reason": "a cited deal step with its own count or exception says it"},
        {"section": "method_steps", "text": "Mount each camera at its confirmed location.",
         "reason": "a cited deal step with its own count does the same work and takes its place"}]
    m._standards_coverage(c)
    got = [x["text"][:20] for x in c["scope"]]
    assert got.index("Terminate and label ") == 2 and got.count("Mount each camera at") == 0
    assert c["standards_suppressed"][0]["reason"].endswith("restored: no final step says it")


def test_an_uncited_repeat_of_a_cited_line_goes():
    c = _content()
    c["assumptions"] = [{"text": "All cable runs are under two hundred (200) feet.", "cites": ["a"], "kind": "fact"},
                        {"text": "All cable runs are under 200 feet.", "cites": [], "kind": "method"},
                        {"text": "Standard ladders reach every camera location.", "cites": [], "kind": "method"}]
    m._no_repeat_lines(c)
    assert [x["text"][:12] for x in c["assumptions"]] == ["All cable ru", "Standard lad"]


# --- 010198 pass V -------------------------------------------------------------

def test_a_request_restating_a_task_goes_and_the_summary_names_the_device():
    c = _content()
    c["overview"] = [{"text": "Provider will install one (1) Square register and one (1) kitchen printer.", "cites": ["a"], "kind": "fact"}]
    c["scope"] = [{"text": "Install one Square register.", "cites": ["a"], "kind": "fact"},
                  {"text": "Install the Ubiquiti router with cellular backup.", "cites": ["r"], "kind": "fact"},
                  {"text": "Closeout", "cites": [], "kind": "heading"},
                  {"text": "Add one (1) Ubiquiti router with cellular backup to the installation scope.", "cites": ["r"], "kind": "fact"}]
    m._summary_names_installs(c)
    m._request_wording(c)
    assert c["overview"][-1]["text"] == "Provider will also install the Ubiquiti router with cellular backup."
    assert not any(x["text"].startswith("Add one") for x in c["scope"])


def test_a_request_with_no_task_reads_as_the_install():
    c = _content()
    c["scope"] = [{"text": "Add a second kitchen printer to the quote.", "cites": ["r"], "kind": "fact"}]
    m._request_wording(c)
    assert c["scope"][0]["text"] == "Install a second kitchen printer."


def test_steps_move_under_the_device_heading_and_a_headless_device_gets_one():
    c = _content()
    h = lambda t: {"text": t, "cites": [], "kind": "heading"}
    st = lambda t: {"text": t, "cites": [], "kind": "method", "standard": True}
    c["scope"] = [h("Kitchen printer"), {"text": "Install one kitchen printer.", "cites": ["p"], "kind": "fact"},
                  h("All Work Areas"), st("Connect each printer to power and perform a test print."),
                  {"text": "Install the Ubiquiti router with cellular backup.", "cites": ["r"], "kind": "fact"},
                  st("Connect the internet service to the router's WAN connection."), st("Perform the basic router configuration."),
                  st("Power on the installed devices."), h("Closeout"), st("Confirm completion.")]
    m._device_headings(c)
    got = [x["text"][:24] for x in c["scope"]]
    assert got == ["Kitchen printer", "Install one kitchen prin", "Connect each printer to ",
                   "Ubiquiti router with cel", "Install the Ubiquiti rou", "Connect the internet ser", "Perform the basic router",
                   "All Work Areas", "Power on the installed d", "Closeout", "Confirm completion."]


def test_a_room_organised_list_gets_no_device_heading():
    c = _content()
    h = lambda t: {"text": t, "cites": [], "kind": "heading"}
    c["scope"] = [h("General"), {"text": "Install the two new 86-inch displays in Room 250.", "cites": ["a"], "kind": "fact"},
                  {"text": "Reconnect the display connections in Room 250.", "cites": ["b"], "kind": "fact"},
                  {"text": "Mount each display at the approved height.", "cites": [], "kind": "method", "standard": True},
                  h("Room 247"), {"text": "Install the 98-inch display in room 247.", "cites": ["c"], "kind": "fact"}]
    before = [x["text"] for x in c["scope"]]
    m._device_headings(c)
    assert [x["text"] for x in c["scope"]] == before


# --- 010171 / 010376 pass V: the guard, side lines, disposal, hours ----------

def test_a_question_is_never_guarded():
    q = _fact("q", "Just to confirm, there are no displays that need to be mounted as part of this project, correct?",
              section="answered_question")
    ids, _ = m._noise({"atoms": [], "documents": []}, [q, _fact("s", "We do need to mount three TVs.")])
    assert "q" in ids and "s" not in ids


def test_the_guard_never_restores_what_an_exclusion_rules_out():
    c = _content()
    c["out_of_scope"] = [{"text": "External mics and mic pods.", "cites": ["x"], "kind": "fact"}]
    c["scope"] = [{"text": "Install three (3) Rally Bars.", "cites": ["b"], "kind": "fact"}]
    before = {("a",): ("scope", "Install three (3) Rally Bars and two (2) mic pods.")}
    m._guard_drops(c, "tidy", before, {"scope": []})
    assert [x["text"] for x in c["scope"]] == ["Install three (3) Rally Bars."]
    assert c["guard_declined"][0]["reason"].startswith("ruled out by:")


def test_the_customers_onsite_gear_assumption_comes_back():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide all displays, hardware, cables and mounts.", "cites": [], "kind": "method", "standard": True}]
    c["customer_materials"] = [{"text": "HDMI cabling if needed.", "cites": ["h"], "kind": "fact"},
                               {"text": "USB cabling if needed.", "cites": ["u"], "kind": "fact"}]
    c["overview"] = [{"text": "Provider will mount three (3) TVs using provided TV mounts.", "cites": ["t"], "kind": "fact"}]
    c["dropped"] = [{"list": "assumptions", "text": "Customer-provided televisions, TV mounts, and any needed HDMI or USB cabling are complete "
                                                    "and onsite before Provider arrival.", "cites": ["t"], "kind": "fact",
                     "reason": "check: the cited lines do not state it"}]
    m._supply_assumption_back(c)
    assert len(c["assumptions"]) == 1 and not c["dropped"]


def test_a_task_never_disposes_of_what_another_party_recycles():
    c = _content()
    c["assumptions"] = [{"text": "ThinkTLS performs the offsite transport and recycling of the removed equipment.", "cites": ["t"], "kind": "fact"}]
    c["overview"] = [{"text": "Provider will remove the old UPS and battery backup for eWaste.", "cites": ["o"], "kind": "fact"}]
    c["scope"] = [{"text": "Removal and eWaste", "cites": [], "kind": "heading"},
                  {"text": "Move the removed equipment to the parking area for loading.", "cites": ["m"], "kind": "fact"},
                  {"text": "EWaste the removed UPS and battery backup.", "cites": ["e"], "kind": "fact"}]
    m._disposal_not_ours(c)
    assert [x["text"] for x in c["scope"]] == ["Removal", "Move the removed equipment to the parking area for loading."]
    assert c["overview"][0]["text"] == "Provider will remove the old UPS and battery backup."


def test_an_aside_never_fills_the_arrival_protocol_or_equipment():
    c = _content()
    c["arrival_protocol"] = [{"text": "Verify the security tag number against shipping records before seal removal.", "cites": ["a"], "kind": "fact"}]
    c["equipment_in_scope"] = [{"text": "Uniquely numbered security tags for trailer sealing.", "cites": ["b"], "kind": "fact"},
                               {"text": "One (1) APC Smart-UPS SRT 5000VA UPS.", "cites": ["u"], "kind": "fact"}]
    m._aside_not_block(c, [_fact("a", "tag verified", section="also_kept"), _fact("b", "sealed with a tag", section="also_kept"),
                           _fact("u", "Replace a 5K UPS")])
    assert not c["arrival_protocol"] and [x["cites"] for x in c["equipment_in_scope"]] == [["u"]]


def test_hours_are_stated_only_when_a_source_gives_them():
    c = {"schedule": []}
    m._roles_and_hours(c, [_fact("d", "Two technicians for one business day to replace the UPS.")])
    assert c["schedule"][-1]["text"] == "Estimated Duration: one (1) onsite business day; two (2) technicians"
    c = {"schedule": []}
    m._roles_and_hours(c, [_fact("d", "Two technicians for one full day, 4 hours per tech.")])
    assert "four (4) hours per technician" in c["schedule"][-1]["text"]
