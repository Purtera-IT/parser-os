"""SOW writer rules, batch 8 (pass W grades: 000132 C+)."""
from __future__ import annotations

import re

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


_NEG = re.compile(r"(?i)\b(not|no|never|exclud\w*)\b")


def _end_user_case():
    c = _content()
    c["scope"] = [{"text": "Network support", "cites": [], "kind": "heading"},
                  {"text": "Support physical network components (LAN, WLAN).", "cites": ["f2"], "kind": "fact"},
                  {"text": "Closeout", "cites": [], "kind": "heading"},
                  {"text": "Confirm completion with the onsite contact before leaving site.", "cites": [], "standard": True}]
    c["assumptions"] = [{"text": "Assume support does not include end-user support, including O365/M365 client support.",
                         "cites": ["f1"], "kind": "fact"}]
    fs = [_fact("f1", "End-user support including O365/M365 clients", type="scope_item"),
          _fact("f2", "Support for network (LAN/WLAN)", type="scope_item")]
    return c, fs


# --- 000132 C+: source_polarity never moves a negated line -------------------

def test_a_negated_line_moves_as_the_sources_positive_wording():
    c, fs = _end_user_case()
    m._traced(c, "source_polarity", m._source_polarity, c, fs, {"atoms": []})
    assert c["assumptions"] == []
    scope = _texts(c, "scope")
    assert "Provide end-user support including O365/M365 clients." in scope
    assert not any(_NEG.search(t) for t in scope), scope
    assert "End User Support" in scope and "Does Not Include End" not in scope


def test_a_negated_line_with_no_positive_wording_goes():
    c = _content()
    c["assumptions"] = [{"text": "Support does not include printer support.", "cites": ["f1"], "kind": "fact"}]
    # The source names work but has no clean positive form short enough.
    fs = [_fact("f1", " ".join(["printer support and"] * 12) + " more", type="scope_item")]
    m._source_polarity(c, fs, {"atoms": []})
    assert c["assumptions"] == [] and not any(_NEG.search(t) for t in _texts(c, "scope"))


def test_the_drop_guard_never_restores_what_source_polarity_rejected():
    c, fs = _end_user_case()
    m._traced(c, "source_polarity", m._source_polarity, c, fs, {"atoms": []})
    gone = "Assume support does not include end-user support, including O365/M365 client support."
    before = {("f9",): ("assumptions", gone)}
    lists_before = {"assumptions": [{"text": gone, "cites": ["f9"], "kind": "fact"}]}
    out = m._guard_drops(c, "tidy", before, lists_before)
    assert out == [] and gone not in _texts(c, "assumptions")
    assert any(d["reason"] == "source_polarity rejected it" for d in c.get("guard_declined") or [])


def test_the_summary_never_names_a_negated_line_as_a_service():
    c = _content()
    c["services"] = ["support"]
    c["overview"] = [{"text": "Provider will support physical network components.", "cites": [], "kind": "method"}]
    c["scope"] = [{"text": "Support does not include end-user support.", "cites": ["f1"], "kind": "fact"},
                  {"text": "Coordinate with local stakeholders.", "cites": ["f2"], "kind": "fact"}]
    m._summary_names_job(c)
    ov = " ".join(_texts(c, "overview"))
    assert "does not include" not in ov and "support for does not" not in ov
    assert "coordination with local stakeholders" in ov


# --- 000132 C+: a merge target later removed gives its merged lines back -----

def test_a_suppressed_merge_target_gives_the_merged_line_back():
    c = _content()
    std = {"text": "Coordinate visit priorities with local stakeholders before or during each visit.", "cites": [], "kind": "method", "standard": True}
    deal = {"text": "Coordinate with local stakeholders during each visit.", "cites": ["f1"], "kind": "fact"}
    other = {"text": "Coordinate visit priorities with the remote IT contact during each visit.", "cites": ["f2"], "kind": "fact"}
    c["scope"] = [{"text": "Onsite Support", "cites": [], "kind": "heading"}, std, deal, other]

    def drop_deal(c):  # a dedupe that takes the deal line as the standard's repeat
        c["scope"] = [x for x in c["scope"] if x is not deal]
    m._traced(c, "standards", drop_deal, c)
    assert deal["text"] not in _texts(c, "scope") and "f1" in std["cites"]  # merged

    def drop_std(c):  # a later rule suppresses the standard
        c["scope"] = [x for x in c["scope"] if x is not std]
    m._traced(c, "one_statement", drop_std, c)
    scope = _texts(c, "scope")
    assert deal["text"] in scope, scope
    assert scope.index(deal["text"]) == 1  # where its target stood
    assert any(d.get("guard") == "merged_back" and d["atom_ids"] == ["f1"] for d in c["rule_decisions"])


def test_a_merged_line_stays_merged_while_its_target_stands_or_is_reworded():
    c = _content()
    std = {"text": "Coordinate visit priorities with local stakeholders before or during each visit.", "cites": [], "kind": "method", "standard": True}
    deal = {"text": "Coordinate with local stakeholders during each visit.", "cites": ["f1"], "kind": "fact"}
    c["scope"] = [std, deal]
    m._traced(c, "standards", lambda c: c.__setitem__("scope", [std]), c)
    std["text"] = "Coordinate visit priorities with local stakeholders before or during each onsite visit."
    m._traced(c, "plain", lambda c: None, c)
    assert _texts(c, "scope") == [std["text"]]


# --- 010217 B: supply lines, counts, IT duty, pre-start step, onsite assumption, requester ---

_ATOMS_217 = [
    {"id": "a1", "text": "Those (2) 55” Philips displays would then need to be installed in their NOC (same location). They are ordering a 3rd 55” Philips display to be installed in a typical “3x1” design using articulating mounts."},
    {"id": "a2", "text": "Lastly, they are buying (1) new 98” Philips display and need it installed in room 247. This will also need a backing board installed to support the display and we’re providing a new wall mount for it as well."},
    {"id": "a3", "text": "(3) new 32-65 inch articulating TV mount"},
    {"id": "a4", "text": "(1) new 55-100 inch VESA TV mount"},
    {"id": "a5", "text": "IT will handle any connections outside of the display install."},
    {"id": "a6", "text": "The existing connections will need to be reconnected to the new displays (most likely HDMI cables)."},
] + [{"id": f"n{i}", "text": f"Filler line {i} about the wall and the project mount schedule."} for i in range(8)]


def test_a_supply_line_is_one_item_with_its_count_and_its_own_room():
    c = _content()
    c["customer_materials"] = [
        {"text": "Training Room 250: Customer will furnish (1) Chief Thinstall Large 25 Inch Extension Monitor Arm Wall Mount "
                 "(32-65 inch articulating TV mount) and (1) Chief Fusion X-Large Tilt Wall Mount XTM1U (55-100 inch VESA TV mount) for the install.",
         "cites": ["b1", "b2"], "kind": "fact"},
        {"text": "One (1) new 55-100 inch VESA TV mount.", "cites": ["a4"], "kind": "fact"}]
    m._supply_items(c, _ATOMS_217)
    assert _texts(c, "customer_materials") == [
        "Three (3) Chief Thinstall Large 25 Inch Extension Monitor Arm Wall Mounts (32-65 inch articulating TV mount) for the NOC.",
        "One (1) Chief Fusion X-Large Tilt Wall Mount XTM1U (55-100 inch VESA TV mount) for Room 247."]
    assert "a3" in c["customer_materials"][0]["cites"]


def test_an_inherited_room_label_with_no_own_sentence_is_left_off():
    c = _content()
    c["customer_materials"] = [{"text": "Training Room 250: (2) Philips 86 inch displays.", "cites": ["b1"], "kind": "fact"}]
    m._supply_items(c, [{"id": "x", "text": "Install the shelf in room 101."}])
    assert _texts(c, "customer_materials") == ["Two (2) Philips 86 inch displays."]


def test_a_cited_cable_type_stays_in_the_step():
    c = _content()
    c["scope"] = [{"text": "Reconnect the existing display cabling to the two new displays.", "cites": ["a6"], "kind": "fact"}]
    m._cited_connector(c, _ATOMS_217)
    assert _texts(c, "scope") == ["Reconnect the existing display cabling (most likely HDMI) to the two new displays."]


def test_the_backing_step_names_its_rooms_and_counts():
    c = _content()
    c["scope"] = [{"text": "Training Room 250", "kind": "heading", "cites": []},
                  {"text": "Install the required backing board and mount the two new 86-inch Philips displays higher on the back wall.", "cites": ["x"], "kind": "fact"},
                  {"text": "NOC", "kind": "heading", "cites": []},
                  {"text": "Install the new 55-inch display in the NOC using articulating mounts.", "cites": ["y"], "kind": "fact"},
                  {"text": "Room 247", "kind": "heading", "cites": []},
                  {"text": "Install the new 98-inch Philips display in room 247 with the required backing board and wall mount.", "cites": ["z"], "kind": "fact"},
                  {"text": "All Work Areas", "kind": "heading", "cites": []},
                  {"text": "Install three quarter (3/4) inch plywood backing, approximately twenty four (24) inches by forty eight (48) inches, as required to securely support each wall mount.",
                   "cites": [], "standard": True}]
    m._backing_counts(c)
    assert c["scope"][-1]["text"].endswith("to securely support the wall mounts: two (2) in Training Room 250 and one (1) in Room 247.")


def test_what_it_will_handle_is_a_customer_duty():
    c = _content()
    c["overview"] = [{"text": "Provider will reinstall the displays in the NOC.", "cites": ["a1", "a5"], "kind": "fact"}]
    m._it_handles_duty(c, _ATOMS_217)
    assert _texts(c, "customer_responsibilities") == ["Have Customer IT handle any connections outside of the display install."]


def test_a_before_start_step_leaves_closeout():
    c = _content()
    line = "If needed, obtain current room photos or conduct a site survey to confirm installation conditions before starting work."
    c["scope"] = [{"text": "Arrival", "kind": "heading", "cites": []}, {"text": "Check in with the onsite contact.", "cites": [], "kind": "method"},
                  {"text": "Room 247", "kind": "heading", "cites": []}, {"text": "Install the display.", "cites": ["z"], "kind": "fact"},
                  {"text": "Closeout", "kind": "heading", "cites": []}, {"text": line, "cites": ["q"], "kind": "fact"}]
    m._before_start_not_closeout(c)
    assert _texts(c, "scope").index(line) == 2


def test_an_onsite_assumption_names_only_customer_supplied_items():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide all displays, hardware, cables and mounts.", "cites": [], "standard": True}]
    c["assumptions"] = [{"text": "All listed displays, mounts, and backing boards are onsite and ready for installation when Provider arrives.", "cites": [], "kind": "method"}]
    m._onsite_assumption_suppliers(c)
    assert _texts(c, "assumptions") == ["All listed displays and mounts are onsite and ready for installation when Provider arrives."]
    c["assumptions"].append({"text": "Customer-provided equipment is complete, compatible, undamaged and ready for installation.", "cites": [], "standard": True})
    c["assumptions"].insert(0, {"text": "All listed displays, mounts, and backing boards are onsite and ready.", "cites": [], "kind": "method"})
    m._onsite_assumption_suppliers(c)
    assert not any("backing" in t for t in _texts(c, "assumptions"))


def test_requested_by_reads_the_requesters_signature_card():
    env = {"crm": {"account_name": "CDW"}, "atoms": [
        {"id": "s1", "atom_type": "stakeholder", "text": "Ralph Smith III | Consulting Solutions Architect | CDW | 813.373.1644"},
        {"id": "e1", "atom_type": "scope_item", "text": "Hey Trent,", "locator": {"sender": "Ralph Smith III <ralpsmi@cdw.com>", "sent_at": "Wednesday, August 12, 2026 1:48 PM", "quoted": True},
         "structured": {"email_thread": {"to": ["ralpsmi@cdw.com"]}}}]}
    rb = m.requested_by(env)
    assert rb["name"] == "Ralph Smith III" and rb["phone"] == "813.373.1644"


# --- 010209 B-: the summary never claims work the SOW's limits deny -----------

def test_the_summary_drops_ruled_out_clauses():
    c = _content()
    c["assumptions"] = [{"text": "Provider does not procure, receive, stage, configure, package or ship equipment.", "cites": [], "standard": True},
                        {"text": "Removed legacy switches will be left onsite in a customer-designated area.", "cites": [], "standard": True}]
    c["overview"] = [
        {"text": "Provider will receive the hardware, bench it for twenty-four (24) to thirty-six (36) hours, install required rack mount hardware "
                 "and new Cisco power cords, and coordinate with RL remote support during each installation window.", "cites": ["a"], "kind": "fact"},
        {"text": "Provider will take before-and-after pictures, and remove the old switches for packing, shipping, data wipe, and disposal.",
         "cites": ["b"], "kind": "fact"}]
    m._summary_denied_work(c)
    assert _texts(c, "overview") == [
        "Provider will install required rack mount hardware and new Cisco power cords and coordinate with RL remote support during each installation window.",
        "Provider will take before-and-after pictures and remove the old switches."]


def test_a_partner_phase_assumption_goes_but_the_start_line_stays():
    c = _content()
    c["assumptions"] = [{"text": "Provider's scope is limited to physical onsite services; Provider does not procure, receive, stage, configure, package or ship equipment.", "cites": [], "standard": True},
                        {"text": "Allow hardware receipt and a twenty-four (24) to thirty-six (36) hour bench period before deployment.", "cites": ["f1"], "kind": "fact"},
                        {"text": "If equipment is received earlier than planned, the project may start earlier.", "cites": ["f2"], "kind": "fact"}]
    m._partner_phase(c, [_fact("f1", "Equipment will be on bench for 24-36 hours"), _fact("f2", "If equipment is received earlier, we can start the project earlier")])
    assert _texts(c, "assumptions")[1:] == ["If equipment is received earlier than planned, the project may start earlier."]


def test_an_answered_na_question_rules_its_work_out():
    c = _content()
    atoms = [{"id": "q", "atom_type": "answered_question", "text": "Testing requirements: What must the technician complete? Will any console work need to be completed? N/a"}]
    m._answered_none(c, atoms)
    assert _texts(c, "assumptions") == ["No console work is anticipated."]


def test_a_dropped_exclusion_holding_a_standard_exclusion_survives(monkeypatch):
    c = _content()
    c["dropped"] = [{"list": "out_of_scope", "text": "Logical switch configuration, programming, firmware changes, licensing, network engineering, routing, firewall, wireless, cloud configuration, and network design.",
                     "cites": ["a"], "kind": "fact", "reason": "check: the cited lines do not state it"}]
    m._standard_exclusion_survives(c)
    assert len(_texts(c, "out_of_scope")) == 1 and c["dropped"] == []


# --- 010419 B-: transition visit -------------------------------------------------

def test_break_fix_filler_goes_on_a_transition_with_no_issue_named():
    c = _content()
    c["scope"] = [{"text": "Validate reported issues, equipment checks, and site readiness tasks.", "cites": [], "kind": "method"},
                  {"text": "Perform basic issue triage.", "cites": [], "kind": "method"},
                  {"text": "Provide local on site technician support at the designated Customer location.", "cites": [], "kind": "method"},
                  {"text": "Provide basic onsite assistance for end user devices.", "cites": [], "kind": "method"}]
    m._work_no_fact_names(c, "one network technician and one help desk technician per community. usually a full day of work.")
    assert _texts(c, "scope") == ["Provide basic onsite assistance for end user devices."]


def test_a_full_day_is_eight_hours_per_technician():
    c = _content()
    fs = [_fact("f1", "We need one network technician and one help desk technician per community."),
          _fact("f2", "This is usually a full day of work, with some follow-up needed the next day.")]
    m._roles_and_hours(c, fs)
    assert any("eight (8) hours per technician; two (2) technicians, sixteen (16) combined hours" in t for t in _texts(c, "schedule"))


def test_a_business_day_still_gives_no_hours():
    c = _content()
    m._roles_and_hours(c, [_fact("f1", "Two technicians for one business day.")])
    assert not any("hours" in t for t in _texts(c, "schedule"))


def test_template_and_standard_duties_are_not_repeated():
    c = _content()
    c["customer_responsibilities"] = [
        {"text": "Provide a remote engineer responsible for network device configuration and technical direction during the onsite visit.", "cites": [], "kind": "method"},
        {"text": "Provide all device configurations and troubleshooting support as required.", "cites": [], "kind": "method"},
        {"text": "Ensure employees are prepared for technician visits on scheduled dates.", "cites": [], "kind": "method"},
        {"text": "Provide approval before any work continues beyond authorized scope.", "cites": [], "kind": "method"},
        {"text": "Provide a remote engineer who owns configuration and cutover during the visit.", "cites": [], "standard": True}]
    m._standard_says_it(c)
    assert _texts(c, "customer_responsibilities") == ["Provide approval before any work continues beyond authorized scope.",
                                                      "Provide a remote engineer who owns configuration and cutover during the visit."]


def test_a_single_site_sow_says_nothing_per_community():
    c = _content()
    c["sites"] = [{"name": "Bradley Creek", "city": "Wilmington", "state": "NC"}]
    c["schedule"] = [{"text": "Number of Resources: one (1) network technician and one (1) help desk technician per community", "cites": ["f"], "kind": "fact"}]
    c["assumptions"] = [{"text": "Staff one (1) network technician and one (1) help desk technician per community.", "cites": ["f"], "kind": "fact"}]
    m._single_site_wording(c)
    assert _texts(c, "schedule") == ["Number of Resources: one (1) network technician and one (1) help desk technician"]
    assert _texts(c, "assumptions") == []


# --- 010326 B- / 010198 B -----------------------------------------------------------

def test_a_derived_atom_alone_supports_no_task():
    c = _content()
    c["scope"] = [{"text": "Perform a site survey to finalize the scope.", "cites": ["d"], "kind": "fact"},
                  {"text": "Replace the phones.", "cites": ["d", "s"], "kind": "fact"}]
    m._derived_not_task(c, [{"id": "d", "text": "survey outstanding", "locator": {"derived": True}}, {"id": "s", "text": "Replace the phones"}])
    assert _texts(c, "scope") == ["Replace the phones."]


def test_a_clipped_line_is_completed_from_its_source():
    c = _content()
    c["assumptions"] = [{"text": "The Customer has indicated that each location has at least 350 watt PoE switches capable of powering the new.", "cites": ["s"], "kind": "fact"},
                        {"text": "No need for new structured cabling or electrical work is required.", "cites": [], "kind": "method"}]
    m._clipped_lines(c, [{"id": "s", "text": "Each location has at least 350 watt PoE switches capable of powering the new phones."}])
    assert _texts(c, "assumptions") == ["The Customer has indicated that each location has at least 350 watt PoE switches capable of powering the new phones.",
                                        "No new structured cabling or electrical work is required."]


def test_a_part_family_task_takes_the_orders_count():
    c = _content()
    c["scope"] = [{"text": "Install and handle ATAs as part of the phone swap effort.", "cites": ["x"], "kind": "fact"}]
    m._bom_count_on_task(c, [{"id": "r", "atom_type": "rate_card", "text": "Cisco - VoIP phone adapter 18 6044227 Mfg. Part#: ATA191-3PW-K9 TAX: WI"}])
    assert _texts(c, "scope") == ["Install and handle eighteen (18) ATAs as part of the phone swap effort."]


def test_no_after_hours_work_is_an_exclusion():
    c = _content()
    m._no_after_hours(c, [{"id": "a", "text": "No after-hours work is required, easing workforce planning."}])
    assert _texts(c, "out_of_scope") == ["After-hours, weekend, or holiday services unless separately authorized in writing."]


def test_counted_sites_answer_the_count_duty_and_widen_the_report():
    c = _content()
    c["sites"] = [{"name": "A", "count": 300, "unit": "phones"}, {"name": "B", "count": 17, "unit": "phones"}]
    c["customer_responsibilities"] = [{"text": "Provide the final phone count by site spreadsheet.", "cites": ["q"], "kind": "fact"}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Document the fewer than ten (10) wall mounted phone installations."]}]
    m._counts_given(c)
    assert c["customer_responsibilities"] == []
    assert c["deliverables"][0]["description"] == ["Document all three hundred seventeen (317) phone installations, including the fewer than ten (10) wall mounted phones."]


def test_a_condition_is_never_cut_by_the_duty_owner_rule():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide all device configurations and troubleshooting support.", "cites": [], "kind": "method"}]
    t = "Perform a test print and pair the kitchen printer with the POS environment where the available Customer credentials and configuration support it."
    c["scope"] = [{"text": t, "cites": [], "kind": "method"}]
    m._duty_owner(c, [])
    assert _texts(c, "scope") == [t]


def test_a_title_naming_work_the_summary_gives_the_customer_is_contradicted():
    c = _content()
    c["overview"] = [{"text": "Provider will install desk phones, while rack-mounted gateways and rack hardware stay with Customer's internal team.", "cites": [], "kind": "fact"}]
    c["out_of_scope"] = [{"text": "Rack-mounted gateway and rack hardware installation.", "cites": [], "kind": "method"}]
    assert m._name_contradicted("Manitowoc - Racking/Wall mount", c)
    assert not m._name_contradicted("Manitowoc Phone Deployment", c)


# --- 010171 C: no silent deletion ------------------------------------------------

import glob as _glob
import json as _json
import os as _os

import pytest as _pytest

from tests.test_c3_sow_batch4_rules import _SAVED, _replay


def _unaccounted(c):
    lines = [x for k in m.LISTS for x in c.get(k) or [] if isinstance(x, dict)]
    shown = {str(x.get("text")).strip() for x in lines}
    cited = {a for x in lines for a in x.get("cites") or []}
    for d in c.get("deliverables") or []:
        if isinstance(d, dict):
            shown |= {str(t).strip() for t in d.get("description") or []}
    gone = {str(x.get("text")).strip() for k in ("dropped", "dropped_unsupported") for x in c.get(k) or [] if isinstance(x, dict)}
    decided = {str(d.get("text")).strip() for d in c.get("rule_decisions") or []}
    decided_ids = {a for d in c.get("rule_decisions") or [] for a in d.get("atom_ids") or []}
    out = []
    for k, xs in (c.get("writer_draft") or {}).items():
        for x in xs:
            t = str(x.get("text")).strip()
            ids = set(x.get("cites") or [])
            if t in shown or t in gone or t in decided or (ids and (ids & cited or ids & decided_ids)):
                continue
            if not ids:
                continue  # an uncited writer line is the template's to keep or lose
            if m._shown_reworded(t, shown):
                continue  # shown in other words (batch 9: the ledger's own test)
            out.append((k, t))
    return out


@_pytest.mark.parametrize("deal", sorted(re.search(r"final_(\d+)_env", p_).group(1) for p_ in _glob.glob(f"{_SAVED}/final_*_env.json"))
                         or ["none"])
def test_every_cited_writer_line_is_shown_dropped_or_decided(deal, monkeypatch):
    """010171 pass W: "Install one (1) Logitech Sight in the requested room."
    left the SOW with nothing in dropped or rule_decisions."""
    path = f"{_SAVED}/final_{deal}_env.json"
    if not _os.path.exists(path) or "c3_sow" not in _json.load(open(path)):
        _pytest.skip("saved envelope not on this machine")
    c = _replay(deal, monkeypatch)
    assert _unaccounted(c) == [], deal


# --- 010085 A- / 010082 B- / 010043 A- / 010171 C -----------------------------------

def test_an_unanswered_question_is_never_an_exclusion():
    c = _content()
    c["out_of_scope"] = [{"text": "Remote access or console work.", "cites": [], "kind": "method"},
                         {"text": "Electrical work.", "cites": [], "kind": "method"}]
    m._question_not_exclusion(c, [{"id": "q", "atom_type": "open_question", "text": "Once installed, move to next? No remote access/console work?"}])
    assert _texts(c, "out_of_scope") == ["Electrical work."]


def test_a_neighbour_familys_library_step_needs_a_deal_line(monkeypatch):
    monkeypatch.setattr(m, "_library", lambda: {"network_switch_refresh": {"method_steps": [{"text": "Install the new switches."}]},
                                                "smart_hands_general": {"method_steps": [{"text": "Pull, remove, organize, or route cables as requested and where safely accessible."}]}})
    c = _content()
    c["job_family"] = "network_switch_refresh"
    c["scope"] = [{"text": "Install the new switches.", "cites": [], "kind": "method"},
                  {"text": "Pull, remove, organize, or route cables as requested and where safely accessible.", "cites": [], "kind": "method"}]
    m._library_other_family(c, [{"id": "a", "text": "Install about 20 switches."}])
    assert _texts(c, "scope") == ["Install the new switches."]


def test_an_introduction_is_not_a_duty():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Coordinate timing with Mike, who leads that team.", "cites": ["a", "b"], "kind": "fact"}]
    m._intro_not_duty(c, [{"id": "a", "text": "And if you can't tell already, Mike. Mike leads that team."}, {"id": "b", "text": "Do we want to wait for Mike, you said?"}])
    assert c["customer_responsibilities"] == []


def test_site_rules_bind_the_provider_and_conditions_are_assumptions():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Follow client drilling rules and ladder use protocols.", "cites": ["a"], "kind": "fact"}]
    c["scope"] = [{"text": "Closeout", "kind": "heading", "cites": []},
                  {"text": "The existing large safe in the space is not to be moved.", "cites": ["b"], "kind": "fact"}]
    m._rules_bind_provider(c)
    m._condition_not_task(c)
    assert _texts(c, "assumptions") == ["Provider technicians follow the Customer's drilling rules and ladder use protocols.",
                                        "The existing large safe in the space is not to be moved."]
    assert c["customer_responsibilities"] == [] and _texts(c, "scope") == ["Closeout"]


def test_one_duty_per_person_and_one_material_per_item():
    c = _content()
    c["customer_responsibilities"] = [
        {"text": "Provide Henry Hankins as the onsite escort for technician access.", "cites": ["a"], "kind": "fact"},
        {"text": "Customer's Henry Hankins greets the technician on arrival and escorts the technician throughout the job.", "cites": ["b"], "kind": "fact"}]
    c["provider_materials"] = [{"text": "Toggle bolts.", "cites": ["c"], "kind": "fact"},
                               {"text": "Toggle bolts / heavy-duty drywall anchors for mounting as needed.", "cites": ["d"], "kind": "fact"}]
    c["customer_materials"] = [{"text": "The TV mounts.", "cites": ["e"], "kind": "fact"},
                               {"text": "The rolling mount for the FiveStar TV at the NEW site.", "cites": ["f"], "kind": "fact"}]
    m._same_line_twice(c)
    assert _texts(c, "customer_responsibilities") == ["Provide Henry Hankins as the onsite escort for technician access."]
    assert _texts(c, "provider_materials") == ["Toggle bolts / heavy-duty drywall anchors for mounting as needed."]
    assert len(c["customer_materials"]) == 2


def test_a_step_several_item_groups_share_goes_to_the_general_group():
    c = _content()
    t = "Terminate and label each cable run at the camera location and the IDF."
    c["scope"] = [{"text": "General", "kind": "heading", "cites": []}, {"text": "Install three cameras.", "cites": ["a"], "kind": "fact"},
                  {"text": "Side door", "kind": "heading", "cites": []}, {"text": "Run 154 ft of cable run from the IDF to the side door.", "cites": ["b"], "kind": "fact"},
                  {"text": "Back door", "kind": "heading", "cites": []}, {"text": "Run 88 ft of cable run from the IDF to the back door.", "cites": ["c"], "kind": "fact"},
                  {"text": "Closeout", "kind": "heading", "cites": []}]
    c["standards_suppressed"] = [{"section": "method_steps", "text": t, "reason": "a deal step already says it (one_statement)"}]
    m._standards_coverage(c)
    tx = _texts(c, "scope")
    assert tx.index(t) < tx.index("Side door")


def test_a_named_place_is_recited_from_the_same_sender():
    c = _content()
    c["scope"] = [{"text": "Mount one Verkada camera above the side door.", "cites": ["a"], "kind": "fact"}]
    fs = [_fact("a", "the cameras are going to be installed above 3 different doors"), _fact("b", "IDF to side door 154ft"), _fact("z", "side door at the other site")]
    atoms = [{"id": "a", "locator": {"sender": "x@cdw.com"}}, {"id": "b", "locator": {"sender": "x@cdw.com"}}, {"id": "z", "locator": {"sender": "y@other.com"}}]
    m._recite_places(c, fs, atoms)
    assert c["scope"][0]["cites"] == ["a", "b"]


def test_what_the_customer_will_do_itself_is_not_a_provider_clause():
    c = _content()
    c["overview"] = [{"text": "Provider will mount three (3) TVs using provided mounts and complete device configuration and testing for the new rooms.",
                      "cites": ["a", "b"], "kind": "fact"}]
    m._customer_says_it_will(c, [{"id": "a", "text": "We do need to mount three TVs."},
                                 {"id": "b", "text": "No, I will do the device configuration and testing.", "locator": {"sender": "karl@shi.com"}}])
    assert _texts(c, "overview") == ["Provider will mount three (3) TVs using provided mounts."]


def test_one_customer_name():
    c = _content()
    c["end_customer"] = "APWireless"
    c["overview"] = [{"text": "Provider will install three setups at the La Jolla site for Radius Global, with two in offices.", "cites": [], "kind": "fact"}]
    m._one_customer_name(c, [{"id": "a", "text": "Below are the answers from the APWireless Team."}])
    assert _texts(c, "overview") == ["Provider will install three setups at the La Jolla site for APWireless, with two in offices."]


def test_an_overview_install_on_weak_cites_becomes_a_task():
    c = _content()
    c["overview"] = [{"text": "Provider will install three setups.", "cites": ["s"], "kind": "fact"},
                     {"text": "Install one (1) Logitech Sight in the requested room.", "cites": ["w"], "kind": "fact"}]
    c["scope"] = [{"text": "Closeout", "kind": "heading", "cites": []}]
    m._weak_commitments(c, {"w"}, [{"id": "w", "atom_type": "scope_item", "text": "We would like to have the Logitech Sight installed."}])
    assert _texts(c, "overview") == ["Provider will install three setups."]
    assert _texts(c, "scope") == ["Install one (1) Logitech Sight in the requested room.", "Closeout"]


def test_the_untraced_ledger_records_a_silent_drop():
    c = _content()
    c["writer_draft"] = {"scope": [{"text": "Install the router.", "cites": ["r"]}, {"text": "Mount the TV.", "cites": ["t"]}]}
    c["scope"] = [{"text": "Mount the TV.", "cites": ["t"], "kind": "fact"}]
    m._untraced_ledger(c)
    assert [d["text"] for d in c["rule_decisions"] if d["rule"] == "untraced"] == ["Install the router."]


# --- 010376 B- / 010083 B ----------------------------------------------------------

def test_a_third_partys_facility_is_its_own_and_a_bare_service_phrase_goes():
    c = _content()
    c["assumptions"] = [{"text": "ThinkTLS provides a dedicated truck to transport the two removed units back to Provider’s facility.", "cites": ["a"], "kind": "fact"}]
    c["scope"] = [{"text": "Provide smart Hand Support to install.", "cites": ["b"], "kind": "fact"}]
    m._speakers_facility(c)
    assert _texts(c, "assumptions") == ["ThinkTLS provides a dedicated truck to transport the two removed units back to ThinkTLS's facility."]
    assert c["scope"] == []


def test_an_uncited_fact_no_source_names_is_rejected_and_counted():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide the visitor badge and parking pass at check-in.", "cites": [], "kind": "fact"}]
    got = m.check(c, {"a": "Replace a 5K UPS (Power input connector type L6-30P)"})
    assert got["uncited_fact"] == 1 and c["customer_responsibilities"] == []


def test_an_arrival_step_is_not_a_repeat_of_pooled_closing_words():
    c = _content()
    c["scope"] = [{"text": "Arrival", "kind": "heading", "cites": []},
                  {"text": "Check in with the onsite contact before beginning work.", "cites": [], "kind": "method"},
                  {"text": "Closeout", "kind": "heading", "cites": []},
                  {"text": "Confirm completion with the onsite contact before leaving site.", "cites": [], "standard": True}]
    m._tidy(c)
    assert "Check in with the onsite contact before beginning work." in _texts(c, "scope")


def test_an_uncited_universal_never_contradicts_a_deal_line():
    c = _content()
    c["assumptions"] = [{"text": "Each site will provide adequate rack or mounting space for the replacement switches.", "cites": [], "kind": "method"}]
    m._no_universal_against_fact(c, [{"id": "a", "text": "Not all stores have a rack but still need rack mount hardware installed."}])
    assert c["assumptions"] == []


def test_a_line_resting_on_an_unanswered_question_goes():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Customer remote resource verbal validation is required.", "cites": ["q"], "kind": "fact"}]
    m._unanswered_cites(c, [{"id": "q", "text": "What closeout documentation is required? Verbal validation from customer remote resource?"}])
    assert c["customer_responsibilities"] == []


def test_source_shorthand_and_the_after_hours_slot():
    c = _content()
    c["job_family"] = "network_switch_refresh"
    c["overview"] = [{"text": "Provider will refresh Customer from C3650s to C9350s at approximately 92 sites.", "cites": ["a"], "kind": "fact"},
                     {"text": "Apx 83 sites will be singles, Memphis Main will have apx. 51 switches.", "cites": ["b"], "kind": "fact"}]
    c["schedule"] = [{"text": "Work Window: after hours, no weekends, no overnights.", "cites": ["c"], "kind": "fact"}]
    c["assumptions"] = [{"text": "All Services will be performed After Hours during the week.", "cites": ["c"], "kind": "fact"}]
    m._source_shorthand(c)
    assert _texts(c, "overview") == ["Provider will refresh the Customer's switches from C3650s to C9350s at approximately 92 sites.",
                                     "Approximately 83 sites will each have a single switch, Memphis Main will have approximately 51 switches."]
    assert c["assumptions"] == []
