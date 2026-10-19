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


# --- 010115 A-: standards nothing else carries stay; form chrome ---------------

def test_a_standard_assumption_goes_only_for_a_line_that_says_it_back():
    c = {"scope": [], "schedule": [], "assumptions": [
        {"text": "Onsite packing materials are available for boxing the removed hardware.", "cites": [], "kind": "method"},
        {"text": "Onsite parking is available; any parking fee is reimbursable.", "cites": [], "kind": "method", "standard": True}]}
    m._tidy(c)
    assert any("parking" in x["text"] for x in c["assumptions"])


def test_a_shipment_area_is_not_shipping_work():
    c = {"job_family": "decommission_packing", "rule_decisions": [], "dropped": [],
         "out_of_scope": [{"text": "Transport, shipping or freight.", "cites": [], "kind": "method", "standard": True}],
         "deliverables": [{"item": "Work Completion Report", "description": [
             "Packed equipment staged in the designated shipment area.", "Equipment shipped to the depot."]}]}
    m._deliverables_hygiene(c)
    assert c["deliverables"][0]["description"] == ["Packed equipment staged in the designated shipment area."]


def test_empty_form_fields_and_blank_field_footnotes_are_not_terms():
    fs = [_fact("n", "Hardware to be Installed (By Category) | N/A"),
          _fact("r", "Rush?* | *A rush fee is added if a location requires installation within 10 days."),
          _fact("d", "Days of Operation | Monday – Friday")]
    c = {"assumptions": [{"text": "No hardware is to be installed; the category is N/A.", "cites": ["n"], "kind": "fact"},
                         {"text": "A rush fee of 20% applies.", "cites": ["r"], "kind": "fact"},
                         {"text": "Work is performed Monday through Friday.", "cites": ["d"], "kind": "fact"}]}
    m._form_chrome(c, fs)
    assert [x["cites"] for x in c["assumptions"]] == [["d"]]


# --- 010279 B: labor only gates the RFP's staging section ----------------------

def test_labor_only_gates_the_staging_section_and_keeps_the_customers_standard_duties():
    fs = [_fact("l", "We are pricing physical installation labor only"),
          _fact("m", "Supply all terminal MAC addresses to Harris Teeter the prior week", under="2.1\tStorage / Staging of Equipment"),
          _fact("t", "Apply Asset tag for each POS terminal", under="2.1\tStorage / Staging of Equipment")]
    c = {"overview": [], "scope": [{"text": "Staging", "cites": [], "kind": "heading"},
                                   {"text": "Provide all terminal MAC addresses to Harris Teeter the prior week.", "cites": ["m"], "kind": "fact"},
                                   {"text": "Install", "cites": [], "kind": "heading"},
                                   {"text": "Install the terminals.", "cites": ["l"], "kind": "fact"}],
         "customer_responsibilities": [{"text": "Provide all hardware, cables, adapters, accessories, labels and asset tags required for installation.",
                                        "cites": [], "kind": "method", "standard": True}],
         "customer_materials": [{"text": "Apply Asset tag for each POS terminal", "cites": ["t"], "kind": "fact"}]}
    m._grounding(c, fs)
    assert not any("MAC" in x["text"] for x in c["scope"])
    assert c["customer_responsibilities"] and c["customer_materials"][0]["text"] == "Asset tags for each POS terminal."


# --- 010376 C-: replacement family, exclusions, vendor's summary, slots, COI ----

def test_an_exclusion_item_never_takes_back_a_sold_task():
    c = {"scope": [{"text": "Install one (1) APC Smart-UPS SRT 5000VA UPS.", "cites": ["a"], "kind": "fact"}],
         "out_of_scope": [{"text": "Hardware installation, configuration, programming, or system testing.", "cites": [], "kind": "method", "standard": True},
                          {"text": "Installation of new in-wall cabling.", "cites": [], "kind": "method"}]}
    m._no_exclusion_negates_task(c)
    assert [x["text"] for x in c["out_of_scope"]] == ["Configuration, programming, or system testing.", "Installation of new in-wall cabling."]


def test_staffing_and_one_business_day_fill_the_slots():
    c = {"schedule": []}
    m._roles_and_hours(c, [_fact("d", "two technicians for one business day to remove the existing equipment and install the new UPS")])
    texts = [x["text"] for x in c["schedule"]]
    assert "Number of Resources: two (2) technicians" in texts and any(t.startswith("Estimated Duration") for t in texts)


def test_provider_supplies_its_own_coi():
    c = {"customer_responsibilities": [{"text": "Provide PurTera's certificate of insurance (COI) before site work begins.", "cites": ["c"], "kind": "fact"}]}
    m._own_documents(c)
    assert c["customer_responsibilities"][0]["text"] == "Provide the site's certificate of insurance (COI) requirements before site work begins."
    assert c["assumptions"][0]["text"].startswith("Provider will provide its certificate of insurance (COI)")


# --- 010326 C+: an explanation of data is never an exclusion -------------------

def test_an_explanation_of_sheet_rows_is_never_an_exclusion():
    fs = [_fact("e", "The other unaddressed phones within the sheet are virtual extensions, fax lines, or phones connected via ATA or RJ11."),
          _fact("n", "So, the 317 with addresses are RJ45 connected phones.")]
    c = {"scope": [{"text": "Install the three hundred seventeen (317) physical phones at the listed sites, and exclude the virtual, fax, "
                            "ATA, and RJ11 entries from onsite installation.", "cites": ["n", "e"], "kind": "fact"}],
         "out_of_scope": [{"text": "ATA and RJ11 entries are excluded.", "cites": ["e"], "kind": "fact"}]}
    m._explanation_not_exclusion(c, fs)
    assert c["scope"][0]["text"] == "Install the three hundred seventeen (317) physical phones at the listed sites."
    assert c["out_of_scope"] == []


def test_an_uncited_counted_line_a_fact_states_is_cited_not_dropped(monkeypatch):
    monkeypatch.setattr(m, "_complete", lambda *a, **k: '{"supported": [0, 1, 2, 3]}')
    lines = {"q": "ITEM: Cisco Desk Phone 9851 - VoIP phone | QTY: 259 | UNIT PRICE: $210.00"}
    c = {k: [] for k in m.LISTS}
    c["customer_materials"] = [{"text": "Provide two hundred fifty-nine (259) Cisco Desk Phone 9851 units.", "cites": [], "kind": "method"}]
    m.check(c, lines)
    assert c["customer_materials"] and c["customer_materials"][0]["cites"] == ["q"]


def test_billing_blocks_signatures_and_travel_remarks_leave_summary_and_schedule():
    c = {"overview": [{"text": "Provider will install 317 phones.", "cites": [], "kind": "fact"},
                      {"text": "Project billing/shipping details are Accts Payable and Keith Lyons.", "cites": [], "kind": "fact"}],
         "schedule": [{"text": "Quick signature of the main SOW is needed to unlock engineering resources.", "cites": [], "kind": "fact"},
                      {"text": "The project sites are within a 3-5 mile radius, which reduces travel time.", "cites": [], "kind": "fact"},
                      {"text": "Requested Start Date: November 2, 2026", "cites": [], "kind": "fact"}]}
    m._not_job_terms(c)
    assert len(c["overview"]) == 1 and [x["text"] for x in c["schedule"]] == ["Requested Start Date: November 2, 2026"]
    assert c["assumptions"][0]["text"].startswith("The project sites are within")


def test_a_title_naming_excluded_work_is_contradicted():
    c = {"scope": [{"text": "Remove the existing wall mounts and replace the phones.", "cites": ["a"], "kind": "fact"}],
         "out_of_scope": [{"text": "Rack hardware, gateways, or rack-and-stack work.", "cites": [], "kind": "method"}]}
    assert m._name_contradicted("Manitowoc - Racking/Wall mount", c)
    assert not m._name_contradicted("Manitowoc Phone Deployment", c)


# --- 010419 C+: transition jobs ---------------------------------------------------

def test_work_no_fact_names_goes_even_when_cited():
    c = {"scope": [{"text": "Assist with workstation cabling and cable organization.", "cites": ["s"], "kind": "method"},
                   {"text": "Validate reported issues.", "cites": [], "kind": "method"}],
         "assumptions": [{"text": "Cable pathways are accessible for low-voltage review.", "cites": [], "kind": "method"}],
         "provider_tools": [{"text": "Standard IT tools to perform workstation migrations.", "cites": [], "kind": "method"}]}
    m._work_no_fact_names(c, "one network technician and one help desk technician per community. onboarding.")
    assert [x["text"] for x in c["scope"]] == ["Validate reported issues."] and not c["assumptions"] and not c["provider_tools"]


def test_a_task_never_claims_the_work_a_customer_duty_holds():
    c = {"customer_responsibilities": [{"text": "Provide all device configurations and troubleshooting support as required.", "cites": [], "kind": "method"}],
         "scope": [{"text": "Perform basic issue triage and hands on troubleshooting.", "cites": ["s"], "kind": "method"},
                   {"text": "Escalate advanced troubleshooting to the Customer's IT contacts.", "cites": [], "kind": "method"}]}
    m._duty_owner(c, [_fact("s", "Typically, we need one network technician and one help desk technician per community.")])
    assert [x["text"] for x in c["scope"]] == ["Perform basic issue triage.", "Escalate advanced troubleshooting to the Customer's IT contacts."]


def test_a_transition_staffed_by_role_groups_by_role():
    c = {"scope_lead": "Provider will dispatch one (1) network technician and one (1) help desk technician to perform the following:",
         "scope": [{"text": "Installation", "cites": [], "kind": "heading"},
                   {"text": "Trace switch uplinks with the remote engineer.", "cites": [], "kind": "method"},
                   {"text": "Bradley Creek", "cites": [], "kind": "heading"},
                   {"text": "Validate workstation and printer function for users.", "cites": [], "kind": "method"},
                   {"text": "Document findings.", "cites": [], "kind": "method"},
                   {"text": "Closeout", "cites": [], "kind": "heading"},
                   {"text": "Confirm completion.", "cites": [], "kind": "method"}]}
    m._role_groups(c)
    assert [x["text"] for x in c["scope"] if x["kind"] == "heading"] == ["Network", "End User Computing", "Shared", "Closeout"]


# --- 010120 B-: stay-mounted TVs, captions, retained items, action headings ----

def test_tvs_that_stay_mounted_do_not_fire_the_display_mount_standards():
    std = [t for t in m.purtera_standards("av_conference_room")["method_steps"] if "Mount each display" in t][0]
    assert not m._standard_applies(std, ["These 4 TVs to Stay in Place, they will stay mounted"], [])[0]
    assert not m._standard_applies(std, ["Front wall includes two large TVs mounted in place."], [])[0]
    assert m._standard_applies(std, ["Two new 65 inch displays need to be mounted"], [])[0]


def test_the_summary_never_retains_what_the_tasks_remove():
    c = {"scope": [{"text": "Remove twelve (12) existing speakers.", "cites": ["a"], "kind": "fact"}],
         "overview": [{"text": "Provider will remove the cameras and twelve (12) speakers while retaining the existing room displays, "
                                "TV mounts, and ceiling speaker system.", "cites": ["b"], "kind": "fact"}]}
    m._summary_consistent(c)
    assert c["overview"][0]["text"].endswith("while retaining the existing room displays and TV mounts.")


def test_a_cited_step_leaves_the_testing_group_for_its_action_heading():
    c = {"scope": [{"text": "Decommission", "cites": [], "kind": "heading"},
                   {"text": "Remove twelve (12) existing speakers.", "cites": ["a"], "kind": "fact"},
                   {"text": "Ceiling Devices", "cites": [], "kind": "heading"},
                   {"text": "Mount the two (2) Neat cameras.", "cites": ["b"], "kind": "fact"},
                   {"text": "Installation and Testing", "cites": [], "kind": "heading"},
                   {"text": "Mount the center device on the wall between both TVs.", "cites": ["c"], "kind": "fact"},
                   {"text": "Closeout", "cites": [], "kind": "heading"},
                   {"text": "Also remove the circular speakers.", "cites": ["d"], "kind": "fact"}]}
    m._facts_to_action_heading(c)
    t = [x["text"] for x in c["scope"]]
    assert t.index("Remove the circular speakers.") < t.index("Ceiling Devices")
    assert t.index("Mount the center device on the wall between both TVs.") < t.index("Installation and Testing")


# --- 010217 C+: rooms, logs that match the output, counted materials ----------

def test_places_tell_rooms_apart():
    assert m._places("Install one (1) new 98-inch display in room 247.") != m._places("Install two (2) 86-inch displays in training room 250.")
    assert m._places("Install displays.") == frozenset()


def test_decision_logs_match_the_final_sow():
    c = {k: [] for k in m.LISTS}
    c["scope"] = [{"text": "Install two (2) 86-inch displays.", "cites": ["b"], "kind": "fact"}]
    c["rule_decisions"] = [
        {"rule": "grounding", "head": "sow_keep", "atom_ids": ["a"], "value": "keep", "text": "Install one (1) 98-inch display in room 247."},
        {"rule": "tidy", "head": "sow_keep", "atom_ids": ["b"], "value": "drop", "text": "Install two (2) 86-inch displays."}]
    c["dropped_unsupported"] = [{"text": "Install two (2) 86-inch displays.", "list": "scope"}]
    m._decisions_match_output(c)
    assert [d["value"] for d in c["rule_decisions"]] == ["drop", "keep"] and all(d.get("superseded") for d in c["rule_decisions"])
    assert c["dropped_unsupported"] == []


def test_materials_are_one_counted_line_per_item():
    c = {"customer_materials": [{"text": "The 55-100 inch VESA TV mount.", "cites": ["a"], "kind": "fact"},
                                {"text": "The 32-65 inch articulating TV mount.", "cites": ["b"], "kind": "fact"},
                                {"text": "Three (3) new 32-65 inch articulating TV mounts.", "cites": ["c"], "kind": "fact"},
                                {"text": "(1) new 55-100 inch VESA TV mount", "cites": ["d"], "kind": "fact"}]}
    m._materials_counted(c)
    assert [x["text"] for x in c["customer_materials"]] == ["Three (3) new 32-65 inch articulating TV mounts.",
                                                            "One (1) new 55-100 inch VESA TV mount."]


import glob as _glob
import os as _os
import re as _re

import pytest as _pytest

from tests.test_c3_sow_batch4_rules import _SAVED, _replay


@_pytest.mark.parametrize("deal", sorted(_re.search(r"final_(\d+)_env", p_).group(1) for p_ in _glob.glob(f"{_SAVED}/final_*_env.json"))
                         or ["none"])
def test_every_keep_decision_is_in_the_output(deal, monkeypatch):
    """010217 pass U: a guard "keep" whose line the SOW never showed."""
    import json as _json
    path = f"{_SAVED}/final_{deal}_env.json"
    if not _os.path.exists(path) or "c3_sow" not in _json.load(open(path)):
        _pytest.skip("saved envelope not on this machine")
    c = _replay(deal, monkeypatch)
    lines = [x for k in m.LISTS for x in c.get(k) or [] if isinstance(x, dict)]
    shown = {str(x.get("text")).strip() for x in lines}
    cited = {a for x in lines for a in x.get("cites") or []}
    last = {}
    for d in c.get("rule_decisions") or []:
        if d.get("head") == "sow_keep":
            last[tuple(d.get("atom_ids") or [])] = d
    for ids, d in last.items():
        if d["value"] == "keep":
            assert set(ids) & cited or str(d.get("text")).strip() in shown, (deal, d)
        else:
            assert str(d.get("text")).strip() not in shown, (deal, d)


# --- 010171 C+: rooms from the source, customer self-test, hours sentence ------

def test_this_specific_room_is_the_room_the_sentence_before_names():
    atoms = [{"id": "a", "artifact_id": "e", "text": "And one small conference room with a table with openings at the top.",
              "locator": {"message_index": 0, "line_start": 12, "sentence_index": 0}},
             {"id": "b", "artifact_id": "e", "text": "We would like to have the Logitech Sight installed in this specific room.",
              "locator": {"message_index": 0, "line_start": 12, "sentence_index": 2}}]
    c = {"scope": [{"text": "Conference room", "cites": [], "kind": "heading"},
                   {"text": "Install one (1) Rally Bar in the small conference room.", "cites": ["a"], "kind": "fact"},
                   {"text": "Common work", "cites": [], "kind": "heading"},
                   {"text": "Install one (1) Logitech Sight in the main conference room.", "cites": ["b"], "kind": "fact"}]}
    m._room_from_source(c, [], {"atoms": atoms})
    t = [x["text"] for x in c["scope"]]
    assert "Install one (1) Logitech Sight in the small conference room." in t
    assert t.index("Install one (1) Logitech Sight in the small conference room.") < t.index("Common work")


def test_a_customer_hours_sentence_becomes_the_slot_value():
    c = {"schedule": [{"text": "Work Hours: This can be during office hours Monday and Friday (8am-5PM) or outside of business hours (Mon-Fri)",
                       "cites": [], "kind": "fact"}]}
    m._schedule_voice(c, "2026-08-11")
    assert c["schedule"][0]["text"] == "Work Hours: Monday through Friday, 8:00 AM to 5:00 PM local site time"
