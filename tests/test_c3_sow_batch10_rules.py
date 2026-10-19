"""SOW writer rules, batch 10 (pass X grades: 010082 C+, 000123 B+, 010012 B+, 010043 B+,
010083 B-, 010326 B, 010419 B, 010115 A-, 010085 A-, 010376 B)."""
from __future__ import annotations

from app.core import c3_sow as m


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


def _fact(t, cites=("a",), **kw):
    return {"text": t, "cites": list(cites), "kind": "fact", **kw}


def _std(t):
    return {"text": t, "cites": [], "kind": "method", "standard": True}


# --- 010082 C+: a limit on the Provider's work is not a customer duty ----------

def test_a_provider_limit_stays_an_assumption():
    c = _content()
    c["assumptions"] = [_fact("Keep all Cat6 cable runs at or below one hundred (100) feet."),
                        _fact("Provide any permits or site-specific approvals required for the work.", ())]
    m._assumption_voice(c)
    assert _texts(c, "assumptions") == ["All Cat6 cable runs are at or below one hundred (100) feet."]
    assert _texts(c, "customer_responsibilities") == ["Provide any permits or site-specific approvals required for the work."]


def test_a_provider_limit_another_assumption_sets_goes():
    c = _content()
    c["assumptions"] = [_fact("Keep all Cat6 cable runs at or below one hundred (100) feet."),
                        _fact("All cable runs are no more than 100 feet.")]
    m._assumption_voice(c)
    assert _texts(c, "assumptions") == ["All cable runs are no more than 100 feet."]
    assert not c["customer_responsibilities"]


def test_the_clients_safety_rules_bind_the_provider():
    c = _content()
    c["customer_responsibilities"] = [_fact("Work in the client’s building must follow the client’s safety rules.")]
    m._rules_bind_provider(c)
    assert not c["customer_responsibilities"]
    assert _texts(c, "assumptions") == ["Provider technicians follow the Customer's safety rules."]


def test_a_length_limit_is_no_supply_of_the_cable():
    c = _content()
    c["provider_materials"] = [_fact("The Cat6 cable.", ("p",))]
    c["customer_responsibilities"] = [_fact("Keep all Cat6 cable runs at or below one hundred (100) feet.", ("q",))]
    m._one_supplier(c, [{"id": "p", "when": ""}, {"id": "q", "when": ""}])
    assert _texts(c, "provider_materials") == ["The Cat6 cable."]
    assert not c["customer_materials"]
    # A duty that supplies it still gives it to the customer.
    c["customer_responsibilities"] = [_fact("Provide the Cat6 cable onsite.", ("q",)),
                                      _fact("Ensure the Cat6 cable is onsite before arrival.", ("r",))]
    m._one_supplier(c, [{"id": "p", "when": ""}, {"id": "q", "when": ""}, {"id": "r", "when": ""}])
    assert _texts(c, "customer_materials") == ["The Cat6 cable."]


def test_what_the_papering_reseller_is_to_provide_is_ours():
    cited = ["Detailed Description of Work Requested | Install 4ft x 4ft backboard ...\n"
             "CDW to provide ¾’ backer board, Cat6 network cable and RJ45 ends"]
    assert m._account_supplies("The RJ45 ends.", cited, "CDW")
    assert m._account_supplies("Provide the Cat6 cable.", cited, "CDW")
    assert not m._account_supplies("The 9U rack.", cited, "CDW")
    assert not m._account_supplies("The RJ45 ends.", cited, "Southern Farm Bureau")


def test_a_stricter_standard_check_is_not_said_by_a_weaker_line():
    items = [_fact("Support the cable with J-hooks and Velcro as needed, and test each run for continuity."),
             _std("Test each cable run for 4-pair continuity.")]
    assert m._stricter_than("Test each cable run for 4-pair continuity.", items)
    items[0]["text"] = "Test each cable run for 4-pair continuity at both ends."
    assert not m._stricter_than("Test each cable run for 4-pair continuity.", items)
    assert not m._stricter_than("Test each cable run for continuity.", items)


# --- 000123 B+ -----------------------------------------------------------------

def test_a_planning_basis_is_never_cut_mid_word():
    c = _content()
    fs = [{"id": "x", "section": "scope_of_work", "text": "We'd need to average about a dozen stores a week. I was estimating that "
           "each store would take 2 hours to complete (30 minutes per register) once we figure out the procedure"}]
    m._planning_basis(c, fs)
    assert "The Customer has indicated that each store would take 2 hours to complete (30 minutes per register)." in _texts(c, "assumptions")
    assert not any("figur" in t for t in _texts(c, "assumptions"))


def test_the_customers_hours_estimate_is_the_labor_basis():
    c = _content()
    c["site_summary"] = {"stated": 161}
    c["assumptions"] = [_fact("The Customer has indicated that each store would take 2 hours to complete (30 minutes per register).", ("x",))]
    m._labor_basis(c)
    assert _texts(c, "assumptions")[1] == "Customer-indicated labor basis: 2 hours per site x 161 sites = 322 hours."
    assert c["assumptions"][1]["cites"] == ["x"]
    m._labor_basis(c)
    assert len(c["assumptions"]) == 2


def test_no_overtime_is_a_schedule_line():
    c = _content()
    c["out_of_scope"] = [_fact("Overtime work on nights or weekends is not requested."), _fact("Lift rental.")]
    m._overtime_in_schedule(c)
    assert _texts(c, "out_of_scope") == ["Lift rental."]
    assert _texts(c, "schedule") == ["Work Window: normal business hours; no night or weekend overtime is planned."]


def test_pos_standards_respect_preconfiguration_and_a_dispatched_rollout():
    std = m.purtera_standards("pos_install_refresh")
    pair = next(t for t in std["method_steps"] if "pairing the printer" in t)
    ok, _ = m._standard_applies(pair, ["We will preconfigure all items before shipping them.", "1 receipt printer"])
    assert not ok
    ok, _ = m._standard_applies(pair, ["1 receipt printer per lane"])
    assert ok
    sched = next(t for t in std["exclusions"] if "Store scheduling" in t)
    ok, _ = m._standard_applies(sched, ["161 stores in CA", "We'd need to average about a dozen stores a week."])
    assert not ok
    ok, _ = m._standard_applies(sched, ["Store list: 113 stores in NC"])
    assert ok


# --- 010012 B+ -----------------------------------------------------------------

def test_an_install_written_twice_keeps_the_copy_not_from_captions():
    c = _content()
    c["scope"] = [_fact("Install two (2) Poly X72 bars with TC10 controllers in the Conference Room and Shared Office.", ("n",)),
                  _std("Mount each video bar at its designated location."),
                  _fact("Install two (2) Poly X72 bars with TC10 controllers in Conference Room 2102 and Shared Office 2105.", ("p1", "p2"))]
    atoms = [{"id": "n", "text": "Install X72 with TC10 in the conference room and shared office"},
             {"id": "p1", "text": "Conference Room 2102 is shown with a large table and chairs."},
             {"id": "p2", "text": "The image shows a detailed floor plan of an office space labeled with room names and numbers."}]
    m._same_install_twice(c, atoms)
    assert _texts(c, "scope") == ["Install two (2) Poly X72 bars with TC10 controllers in the Conference Room and Shared Office.",
                                  "Mount each video bar at its designated location."]


def test_the_report_counts_what_the_step_installs():
    c = _content()
    c["scope"] = [_fact("Install two (2) Poly X72 bars with TC10 controllers in the Conference Room and Shared Office.")]
    c["deliverables"] = [{"item": "Work Completion Report", "description": [
        "One (1) Poly X72 bar with TC10 controller installed in the Conference Room.", "Completion confirmed."]}]
    m._report_counts(c)
    assert c["deliverables"][0]["description"][0] == \
        "Two (2) Poly X72 bars with TC10 controllers installed in the Conference Room and Shared Office."


def test_a_writer_step_a_standard_step_does_goes():
    c = _content()
    c["scope"] = [{"text": "Power on each installed system and confirm basic operation.", "cites": [], "kind": "method"},
                  _std("Power on each video bar and confirm it powers up and connects."),
                  {"text": "Place each TC10 controller and route its cable to the video bar.", "cites": [], "kind": "method"}]
    m._standard_step_repeat(c)
    assert _texts(c, "scope") == ["Power on each video bar and confirm it powers up and connects.",
                                  "Place each TC10 controller and route its cable to the video bar."]


# --- 010043 B+ -----------------------------------------------------------------

def test_one_material_per_model():
    c = _content()
    c["customer_materials"] = [_fact("Two (2) Verkada Dome Series CD53-E cameras."), _fact("One (1) Verkada CF83-E camera."),
                               _fact("Two (2) Verkada cameras, Mfg. Part#: CD53-512E-HW.")]
    m._materials_by_model(c)
    assert _texts(c, "customer_materials") == ["Two (2) Verkada Dome Series CD53-E cameras.", "One (1) Verkada CF83-E camera."]


def test_numbers_in_the_cited_line_survive_a_count_strip():
    cited = "installed above 3 different doors | 6125 Tyvola Centre Drive, Charlotte, NC 28217"
    t = "Mount one (1) Verkada camera above the front door at 6125 Tyvola Centre Drive, Charlotte, NC 28217."
    assert m._drop_unstated_counts(t, cited) == t
    assert m._drop_unstated_counts("Install two (2) suspended Neat camera devices.", "Neat camera") == "Install suspended Neat camera devices."


def test_one_order_is_one_supplier(monkeypatch):
    c = _content()
    c["customer_materials"] = [_fact("Two (2) Verkada CD53-E cameras.", ("o1",)), _fact("Verkada camera junction box - circular.", ("o2",))]
    monkeypatch.setattr(m, "_complete", lambda *a, **k: '{"stated": [0]}')
    moved = m._party_check(c, {"o1": "CD53-E QTY 2", "o2": "ACC-MNT-CJBOX-1 junction box"}, {"o1": "art_po", "o2": "art_po"})
    assert moved == 0
    assert len(c["customer_materials"]) == 2 and not c["equipment_in_scope"]


def test_above_door_is_not_the_above_list():
    c = {"provider_tools": [_fact("Ladders for the above-door mounting work.")]}
    import re
    for x in c["provider_tools"]:
        x["text"] = re.sub(r"(?i)\b(for|with|to|of|on) the above(?:[- ](?:gear|items?|equipment|devices?|list|hardware))?\b(?!-)",
                           r"\1 the installed equipment", x["text"])
    assert _texts(c, "provider_tools") == ["Ladders for the above-door mounting work."]


# --- 010083 B- -----------------------------------------------------------------

def test_removed_gear_staged_onsite_is_not_shipped():
    c = _content()
    c["overview"] = [_fact("Provider will remove legacy hardware for shipment to a single First Horizon site.")]
    c["assumptions"] = [_std("Removed legacy switches will be left onsite in a customer-designated area.")]
    m._removed_gear_onsite(c)
    assert _texts(c, "overview") == ["Provider will remove legacy hardware and stage it onsite in the Customer-designated area."]
    assert _texts(c, "out_of_scope") == ["Shipment of removed equipment to a single First Horizon site (available by change order)."]


def test_one_count_per_item():
    c = _content()
    c["overview"] = [_fact("Provider will replace 184 Cisco C9350 switches for First Horizon Bank at approximately 92 sites.")]
    c["scope"] = [_fact("Install 194 Cisco C9350 48-port 60W UPoE switches.")]
    c["customer_materials"] = [_fact("33 x C9350-48U with Stack module and cable"), _fact("69 x C9350-48U with Stack module"),
                               _fact("82 x C9350-49U no stack and no SFP")]
    m._one_item_count(c)
    assert _texts(c, "scope") == ["Install 184 Cisco C9350 48-port 60W UPoE switches."]
    assert any("also states 194" in t for t in _texts(c, "assumptions"))


def test_a_counted_step_keeps_the_standard_place():
    assert m._keep_std_place("Install the new switches in the existing rack, cabinet or customer-designated network area.",
                             "Install 194 Cisco C9350 48-port 60W UPoE switches.") == \
        "Install 194 Cisco C9350 48-port 60W UPoE switches in the existing rack, cabinet or customer-designated network area."
    assert m._keep_std_place("Install the new switches in the existing rack.", "Install 20 switches in the MDF.") == \
        "Install 20 switches in the MDF."


def test_a_single_visit_claim_needs_a_line():
    c = _content()
    c["assumptions"] = [{"text": "Each site can be completed in a single onsite visit.", "cites": [], "kind": "method"}]
    m._single_visit_unsupported(c, [{"text": "Memphis Main will be apx. 51 switches on multiple floors."}])
    assert not c["assumptions"]
    c["assumptions"] = [{"text": "Each site can be completed in a single onsite visit.", "cites": [], "kind": "method"}]
    m._single_visit_unsupported(c, [{"text": "Each store is a one-day visit."}])
    assert c["assumptions"]


def test_a_tech_step_is_never_an_aside():
    c = _content()
    c["arrival_protocol"] = [_fact("Provider calls into the bridge before turn up.", ("b",))]
    m._aside_not_block(c, [{"id": "b", "section": "also_kept", "text": "Tech will call in to a bridge (# tbd & provided by FH)"}])
    assert _texts(c, "arrival_protocol") == ["Provider calls into the bridge before turn up."]
    c["arrival_protocol"] = [_fact("Security tag numbers are verified.", ("t",))]
    m._aside_not_block(c, [{"id": "t", "section": "also_kept", "text": "the security tag number will be verified against records"}])
    assert not c["arrival_protocol"]


# --- 010326 B ------------------------------------------------------------------

def test_no_per_site_equipment_beside_the_exhibit():
    c = _content()
    c["sites"] = [{"name": "Fire Station 1", "address": "911 Franklin Street, Manitowoc, WI 54220"},
                  {"name": "City Hall", "address": "900 Quay St, Manitowoc, WI 54220"}]
    c["site_summary"] = {"stated": 2, "from_table": True}
    c["equipment_in_scope"] = [_fact("Fire Station 1 at 911 Franklin Street, Manitowoc, WI 54220 includes 8 physical phones to be replaced."),
                               _fact("Approximately 18 wall-mounted units are included in the physical installation scope.")]
    m._exhibit_not_equipment(c)
    assert _texts(c, "equipment_in_scope") == ["Approximately 18 wall-mounted units are included in the physical installation scope."]


def test_the_internal_team_is_the_customer():
    c = _content()
    c["overview"] = [_fact("Provider will handle desk phone installs, while the internal team handles rack hardware and gateways.")]
    c["out_of_scope"] = [_fact("Internal team will manage rack hardware.")]
    m._internal_team(c)
    assert _texts(c, "overview") == ["Provider will handle desk phone installs, while the Customer handles rack hardware and gateways."]
    assert _texts(c, "out_of_scope") == ["Customer will manage rack hardware."]


# --- 010419 B ------------------------------------------------------------------

def test_a_one_visit_job_says_the_visit():
    c = _content()
    c["schedule"] = [_fact("Estimated Duration: one (1) onsite workday of eight (8) hours per technician")]
    c["scope"] = [_fact("Document visit notes and next steps after each visit.")]
    m._one_visit_wording(c, [{"text": "We need two techs onsite for a day."}])
    assert _texts(c, "scope") == ["Document visit notes and next steps after the visit."]
    c["scope"] = [_fact("Document visit notes and next steps after each visit.")]
    m._one_visit_wording(c, [{"text": "Weekly onsite support visits."}])
    assert _texts(c, "scope") == ["Document visit notes and next steps after each visit."]


# --- 010115 A- -----------------------------------------------------------------

def test_an_arrival_time_is_a_schedule_value():
    c = _content()
    c["scope"] = [_std("Meet the onsite contact on arrival and review the hardware to be dismantled and packed."),
                  _fact("Arrive onsite at 8:00 am and meet Rachel Bennett.")]
    m._arrival_time_to_schedule(c)
    assert _texts(c, "scope") == ["Meet the onsite contact, Rachel Bennett, on arrival and review the hardware to be dismantled and packed."]
    assert _texts(c, "schedule") == ["Arrival Time: 8:00 AM local site time"]


def test_a_form_label_is_never_restored():
    assert m._FORM_LABEL_LINE.search("Hardware to be Installed (By Category) is N/A; no hardware installation is included in this work.")
    assert not m._FORM_LABEL_LINE.search("Scope of Work (SOW) is attached.")
    c = _content()
    line = _fact("Hardware to be Installed (By Category) is N/A; no hardware installation is included in this work.", ("f",))
    c["out_of_scope"] = []
    c["_merged"] = [{"target": line["text"], "list": "out_of_scope", "line": line}]
    assert m._merged_back(c, {"out_of_scope": [line]}, "form_chrome") == []
    assert not c["out_of_scope"]


def test_the_guard_does_not_restore_a_line_written_elsewhere():
    c = _content()
    line = _fact("Use common hand tools as needed to dismantle the hardware.", ("h",))
    c["provider_tools"] = [line]
    c["assumptions"] = [_std("The hardware can be safely dismantled using common hand tools.")]
    before, lists_before = m._snap(c), {k: list(c.get(k) or []) for k in m._LIST_SECTION}
    c["provider_tools"] = []
    out = m._guard_drops(c, "tidy", before, lists_before)
    assert not c["provider_tools"]
    assert out and out[0]["guard"] == "said_elsewhere"
    assert "h" in c["assumptions"][0]["cites"]


# --- 010085 A- -----------------------------------------------------------------

def test_a_reworded_neighbour_library_step_stays_out():
    c = _content()
    c["job_family"] = "network_switch_refresh"
    c["scope"] = [{"text": "Pull, remove, organize, or route cables where safely accessible.", "cites": [], "kind": "method"},
                  _fact("Install approximately twenty (20) network switches.")]
    m._library_other_family(c, [{"text": "physical installation and patch cabling only"}])
    assert _texts(c, "scope") == ["Install approximately twenty (20) network switches."]


# --- 010376 B ------------------------------------------------------------------

def test_a_bare_verb_sentence_goes():
    c = _content()
    c["overview"] = [_fact("Provider will remove."), _fact("Provider will remove the old UPS.")]
    m._sentence_integrity(c)
    assert _texts(c, "overview") == ["Provider will remove the old UPS."]


def test_the_title_spells_a_place_as_the_sources_do():
    c = {"project_name": "Smart Hands - Extton, PA - Guidewire UPS"}
    m._title_spelling(c, [{"text": "707 Eagleview Blvd. Exton, PA"}, {"text": "Exton site"}, {"text": "Guidewire UPS"}])
    assert c["project_name"] == "Smart Hands - Exton, PA - Guidewire UPS"


def test_a_means_duty_is_not_the_task_that_uses_it():
    c = _content()
    c["scope"] = [_fact("Move the removed equipment by cart and elevator from the third floor to the main door.", ("e",))]
    c["customer_responsibilities"] = [_fact("Provide cart access and the elevator path from the third floor to the main door.", ("e",))]
    m._duty_party(c, [{"id": "e", "text": "We can use the cart and the Elevator", "supplier": None}], {"atoms": []})
    assert len(c["customer_responsibilities"]) == 1
    m._scope_contradictions(c, [{"id": "e", "text": "We can use the cart and the Elevator", "when": ""}])
    assert len(c["scope"]) == 1
