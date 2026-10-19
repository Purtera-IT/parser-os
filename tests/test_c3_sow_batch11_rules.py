"""SOW writer rules, batch 11 (pass Y grades: 010326 D, 010279 B, 010082 B, 010209 B,
010373 A-, plus the stringified-cite drops 010115 and the 000123 count)."""
from __future__ import annotations

import json

from app.core import c3_sow as m


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


def _fact(t, cites=("a",), **kw):
    return {"text": t, "cites": list(cites), "kind": "fact", **kw}


def _std(t):
    return {"text": t, "cites": [], "kind": "method", "standard": True}


# --- 010115 pass Y: cites the writer sent as a stringified list -----------------

def test_a_stringified_cite_list_is_a_list_of_ids():
    x = m._item({"text": "Use common hand tools as needed.", "kind": "fact",
                 "cites": ["['atm_8e8d8cc3134002fb', 'atm_3c0f824b78b9c2c3']"]})
    assert x["cites"] == ["atm_8e8d8cc3134002fb", "atm_3c0f824b78b9c2c3"]
    assert m._item({"text": "t", "cites": "atm_8e8d8cc3134002fb"})["cites"] == ["atm_8e8d8cc3134002fb"]
    assert m._item({"text": "t", "cites": ["atm_8e8d8cc3134002fb"]})["cites"] == ["atm_8e8d8cc3134002fb"]
    assert m._item({"text": "t", "cites": None})["cites"] == []


def test_a_mistyped_cite_names_the_one_fact_it_nearly_spells():
    known = {"atm_3c0f824b78b9c2c3", "atm_8e8d8cc3134002fb", "atm_88c6acfe1909b383"}
    assert m._near_cite("atm_3c0f824d0a8fe246", known) == "atm_3c0f824b78b9c2c3"   # long prefix
    assert m._near_cite("atm_0a8c6cfe1909b383", known) == "atm_88c6acfe1909b383"   # long suffix
    assert m._near_cite("atm_ffff0000aaaa1111", known) is None
    # Two ids sharing the prefix: no guess.
    assert m._near_cite("atm_3c0f824d0a8fe246", known | {"atm_3c0f8240000000aa"}) is None


def test_the_check_keeps_a_line_whose_cites_were_stringified(monkeypatch):
    monkeypatch.setattr(m, "_complete", lambda prompt, *a, **k: json.dumps({"supported": list(range(50))}))
    c = _content()
    c["scope"] = [m._item({"text": "Use common hand tools as needed for dismantling and boxing.", "kind": "fact",
                           "cites": ["['atm_8e8d8cc3134002fb']"]}),
                  m._item({"text": "Complete dismantling and boxing of all everyday POS hardware.", "kind": "fact",
                           "cites": ["atm_3c0f824d0a8fe246"]})]
    lines = {"atm_8e8d8cc3134002fb": "Use common hand tools as needed for dismantling and boxing",
             "atm_3c0f824b78b9c2c3": "Complete dismantling and boxing of all everyday POS hardware"}
    m.check(c, lines)
    assert [x["cites"] for x in c["scope"]] == [["atm_8e8d8cc3134002fb"], ["atm_3c0f824b78b9c2c3"]]
    assert not c.get("_unknown_cites")


# --- 000123 pass Y: a line in dropped never renders ------------------------------

def test_a_dropped_line_never_renders():
    c = _content()
    c["overview"] = [_fact("Provider will install POS systems at one hundred sixty-one (161) BevMo locations.", ()),
                     _fact("Provider will install POS systems at 161 Customer locations.")]
    c["dropped_unsupported"] = [{"list": "overview", "text": "Provider will install POS systems at one hundred sixty-one (161) BevMo locations.",
                                 "cites": [], "reason": "method line with a number or name no fact gives"}]
    m._dropped_not_rendered(c)
    assert _texts(c, "overview") == ["Provider will install POS systems at 161 Customer locations."]


# --- 010326 pass Y D: "they" is the customer when PurTera writes it --------------

def _env_326():
    roster = {"people": [{"name": "AJ Evans", "email": "aj@purtera-it.com"}, {"name": "Chase Smith", "email": "chase@purtera-it.com"},
                         {"name": "Neal Zolt", "email": "nealzol@cdwg.com"}]}
    atoms = [
        {"id": "split", "text": "They will handle the gateways and the R&S, we would be handling phones on the desks, ATAs, wall mounting phones",
         "doc_origin": {"sender_email": "aj@purtera-it.com", "sent_at": "2026-09-17T20:18:11Z"}, "locator": {}},
        {"id": "rs", "text": "They will handle rack & stack gear", "doc_origin": {"sender_email": "aj@purtera-it.com", "sent_at": "2026-09-17T20:18:11Z"},
         "locator": {}},
        {"id": "ask", "text": "I found out that they will need help with racking/stacking/wall mounting the hardware.",
         "doc_origin": {"sender_email": "nealzol@cdwg.com", "sent_at": "2026-09-15T10:00:00Z"}, "locator": {}},
        {"id": "call", "text": "And then you guys are gonna be handling any of the, the gear that, that goes in the racks themselves.",
         "doc_origin": {}, "locator": {"speaker": "Chase Smith"}},
        {"id": "bom", "text": "Cisco VG400 Analog Voice Gateway | 2", "doc_origin": {}, "locator": {}},
    ]
    brief = {"sections": {"scope_of_work": [{"id": "rs", "text": "They will handle rack & stack gear", "supplier": ["us", 0.728]}]}}
    return {"atoms": atoms, "deal_roster": roster, "crm": {"account_name": "CDW Government"}, "c3_brief": brief}


def test_they_will_handle_is_a_customer_duty_and_an_exclusion_whatever_the_supplier_head_says():
    c = _content()
    c["scope"] = [{"text": "Racked hardware", "cites": [], "kind": "heading"},
                  _fact("Rack the designated hardware and install rack ears on the brackets for rack-mounted equipment.", ("ask", "call")),
                  {"text": "Phones and adapters", "cites": [], "kind": "heading"},
                  _fact("Install eighteen (18) Cisco ATA191 adapters, two (2) Cisco VG400 Analog Voice Gateway units, and three (3) "
                        "Cisco VG410 Analog Voice Gateway units at Customer-designated locations.", ("bom",)),
                  _fact("Wall mount fewer than ten (10) phones at Customer-designated locations.", ("split",))]
    c["customer_materials"] = [_fact("The Cisco VG400 and VG410 analog voice gateways.", ("bom",)), _fact("The Cisco ATA191 adapters.", ("bom",))]
    c["deliverables"] = [{"item": "Work Completion Report", "description": [
        "Completed racking and fewer than ten (10) wall-mounted phone installs as scoped."]}]
    m._they_handle(c, _env_326(), [{"id": "rs", "section": "scope_of_work"}])
    assert _texts(c, "scope") == ["Phones and adapters", "Install eighteen (18) Cisco ATA191 adapters at Customer-designated locations.",
                                  "Wall mount fewer than ten (10) phones at Customer-designated locations."]
    assert _texts(c, "customer_materials") == ["The Cisco ATA191 adapters."]
    assert c["deliverables"][0]["description"] == ["Completed fewer than ten (10) wall-mounted phone installs as scoped."]
    assert _texts(c, "customer_responsibilities") == ["Handle the gateways and the rack and stack."]
    assert _texts(c, "out_of_scope") == ["The gateways and the rack and stack, which the Customer handles."]
    assert set(c["customer_responsibilities"][0]["cites"]) >= {"split", "rs", "call"}
    rd = [r for r in c["rule_decisions"] if r["rule"] == "they_handle"]
    assert rd and rd[0]["value"] == "customer"


def test_the_customers_own_they_is_not_the_customer():
    env = _env_326()
    for a in env["atoms"]:
        a["doc_origin"] = {"sender_email": "jvaneyck@mpu.org"}
        a["locator"] = {}
    c = _content()
    c["scope"] = [_fact("Rack the designated hardware.", ("rs",))]
    m._they_handle(c, env)
    assert _texts(c, "scope") == ["Rack the designated hardware."]
    assert not c["customer_responsibilities"]


def test_a_later_purtera_line_taking_the_work_back_keeps_it_ours():
    env = _env_326()
    env["atoms"].append({"id": "back", "text": "Update: we will handle the rack and stack after all.",
                         "doc_origin": {"sender_email": "chase@purtera-it.com", "sent_at": "2026-09-20T10:00:00Z"}, "locator": {}})
    env["atoms"] = [a for a in env["atoms"] if a["id"] not in ("split", "call")]
    c = _content()
    c["scope"] = [_fact("Rack and stack the designated hardware.", ("rs",))]
    m._they_handle(c, env, [{"id": "rs", "section": "scope_of_work"}])
    assert _texts(c, "scope") == ["Rack and stack the designated hardware."]


def test_install_counts_come_from_the_site_table_not_the_bom():
    c = _content()
    c["site_summary"] = {"stated": 30, "unit": "phones", "unit_total": 317}
    line = ("Install three hundred fifty-one (351) Cisco phones: two hundred fifty-nine (259) Cisco Desk Phone 9851 units, "
            "seventy-eight (78) Cisco Desk Phone 9841 units, two (2) Cisco Desk Phone 9861 units, and twelve (12) Cisco IP Conference "
            "Phone 8832 units, at Customer-designated locations.")
    c["scope"] = [_fact(line)]
    c["overview"] = [_fact("Provider will install and wall mount three hundred seventeen (317) phones.", ()),
                     _fact("Provider will i" + line[1:].replace("Install", "install", 1), ())]
    m._site_unit_counts(c)
    assert _texts(c, "scope") == ["Install three hundred seventeen (317) Cisco phones (Cisco Desk Phone 9851, Cisco Desk Phone 9841, "
                                  "Cisco Desk Phone 9861 and Cisco IP Conference Phone 8832 models) at Customer-designated locations."]
    assert _texts(c, "overview") == ["Provider will install and wall mount three hundred seventeen (317) phones."]


def test_a_summary_site_count_no_cited_line_states_reads_the_site_table():
    c = _content()
    c["site_summary"] = {"stated": 30}
    c["overview"] = [_fact("Provider will install phones across fifty-three (53) Manitowoc, WI sites.", ()),
                     _fact("Provider will mount sixteen (16) monitor arms.", ()),
                     _fact("Provider will install phones with ninety-nine (99) brackets.", ())]
    c["scope"] = [_fact("Install sixteen (16) monitor arms.", ("x",))]
    m._overview_numbers(c, [{"id": "x", "text": "MPU Main Office | Phone Count: 53"}])
    assert _texts(c, "overview") == ["Provider will install phones across thirty (30) Manitowoc, WI sites.",
                                     "Provider will mount sixteen (16) monitor arms.", "Provider will install phones."]


def test_every_row_of_a_port_inventory_sheet_is_a_record_not_a_task():
    rows = [{"id": f"r{i}", "artifact_id": "inv", "text": f"PORT: {i} | DESCRIPTION: {d} | USAGE: Idle | SERVICE: In Service",
             "structured": {"sheet": "MO", "cells": {"PORT": str(i), "DESCRIPTION": d, "USAGE": "Idle", "SERVICE": "In Service"}}}
            for i, d in ((1, "Water HLP Meter"), (2, "CityFax Faxfinder4"), (3, "Boiler 9 Elevator Phone"))]
    rows.append({"id": "r9", "artifact_id": "inv", "text": "Description: Fax Line", "structured": {"_parent_atom_id": "r2"}})
    rows.append({"id": "p", "artifact_id": "inv", "text": "MPU Main Office | 53", "structured": {"sheet": "Sites", "cells": {"Site": "MPU", "Phones": "53"}}})
    env = {"atoms": rows}
    ids, names = m._inventory_rows(env)
    assert ids == {"r1", "r2", "r3", "r9"} and "cityfax faxfinder4" in names
    assert set(m._noise(env, [])[0]) >= ids
    c = _content()
    c["_inventory"] = (ids, names)
    c["scope"] = [_fact("Install a dial-up connection for the Water HLP Meter.", ("r1",)),
                  _fact("Install the CityFax Faxfinder4 fax line at City Hall.", ()),
                  _fact("Install the Boiler 9 Elevator Phone.", ("r3", "r9")),
                  _fact("Install the phones at MPU Main Office.", ("p",))]
    fs = [{"id": a["id"], "text": a["text"]} for a in rows]
    m._cell_not_task(c, fs)
    assert _texts(c, "scope") == ["Install the phones at MPU Main Office."]


# --- 010171 / 010012 / 010116 / 010376 pass Y ------------------------------------

def test_a_paraphrase_its_cite_covers_is_stated():
    cite = "We are currently waiting for our network drops to be installed next week."
    assert m._content_covered("Ensure network drops are installed before technician arrival.", cite)
    assert m._content_covered("Assume network drops are installed and functional before technician arrival.", cite)
    assert not m._content_covered("Network drops are not installed.", cite)
    assert not m._content_covered("Install the Rally Bar.", "We need a Logitech Rally Bar")


def test_a_device_of_a_several_room_job_is_each_rooms_in_the_summary():
    c = _content()
    c["overview"] = [_fact("Provider will install three (3) completely new conference setups.", ())]
    c["scope"] = [_fact("Mount the Rally Bar using a bracket that connects to the VESA mount behind the TV."),
                  _fact("Mount the NUC underneath the table."), _std("Mount each video bar at its designated location.")]
    m._summary_names_installs(c)
    assert _texts(c, "overview")[-1] == "Provider will also install each Rally Bar and the NUC."


def test_one_counted_thing_is_in_the_one_room_its_source_names():
    atoms = [{"id": "r", "artifact_id": "e", "text": "And one small conference room with a table with openings at the top.",
              "locator": {"line_start": 1}},
             {"id": "p", "artifact_id": "e", "text": "We will have available power (2x outlets) 1 and data port in the floor.",
              "locator": {"line_start": 2}}]
    c = _content()
    c["customer_responsibilities"] = [_fact("Ensure one (1) floor data port is available at the installation locations.", ("p",))]
    m._one_place_count(c, atoms)
    assert _texts(c, "customer_responsibilities") == ["Ensure one (1) floor data port is available in the small conference room."]


def test_a_task_sharing_a_cite_with_an_exclusion_is_not_that_exclusion():
    c = _content()
    c["writer_draft"] = {"assumptions": [_fact("Existing displays are not replaced or removed.", ("o",))]}
    c["scope"] = [_fact("Mount each Poly X72 bar at its designated location and connect it to its display.", ("o",))]
    m._exclusions_never_tasks(c, [{"id": "o", "text": "2x Poly X72 bars with TC10's"}])
    assert _texts(c, "scope") == ["Mount each Poly X72 bar at its designated location and connect it to its display."]


def test_every_drop_is_recorded_even_beside_a_line_sharing_its_cites():
    c = _content()
    c["scope"] = [_fact("Install two (2) Poly X72 bars.", ("o",)), _fact("Mount each Poly X72 bar on its wall mount kit.", ("o",))]
    m._traced(c, "demo", lambda c: c.__setitem__("scope", c["scope"][:1]), c)
    rd = [r for r in c["rule_decisions"] if r["rule"] == "demo"]
    assert [r["text"] for r in rd] == ["Mount each Poly X72 bar on its wall mount kit."] and rd[0]["value"] == "drop"


def test_equipment_in_scope_is_never_empty_beside_a_counted_model_install():
    c = _content()
    c["scope"] = [_fact("Install two (2) Poly X72 bars with TC10 controllers in the Conference Room 2102 and Shared Office 2105."),
                  _fact("Install the cabling neatly.")]
    m._equipment_from_installs(c)
    assert _texts(c, "equipment_in_scope") == ["Two (2) Poly X72 bars with TC10 controllers."]


def test_one_atom_never_yields_a_part_and_its_sum():
    c = _content()
    c["equipment_in_scope"] = [_fact("2 x QM65C Samsung 65IN Display", ("f",)), _fact("2 x TS525TU Chief Thinstall Wall Mount", ("f",)),
                               _fact("1 x QM75C Samsung 75IN Display", ("g",)), _fact("3 x TS525TU Chief Thinstall Wall Mount", ("g", "f"))]
    m._parts_and_sum(c)
    assert _texts(c, "equipment_in_scope") == ["2 x QM65C Samsung 65IN Display", "1 x QM75C Samsung 75IN Display",
                                               "3 x TS525TU Chief Thinstall Wall Mount"]


# --- 010147 / 010120 pass Y regressions ------------------------------------------

def test_a_counted_inventory_line_is_grounded_by_its_counts_and_thing():
    lines = {"q": "Quantity 25-30 CU34G2X", "s": "#of individual devices: 1U-4U Server Quantity 3"}
    c = _content()
    c["equipment_in_scope"] = [_fact("Approximately twenty-five (25) to thirty (30) CU34G2X curved monitors are in scope.", ("q",)),
                               _fact("Three (3) individual 1U-4U servers/devices are in scope.", ("s",)),
                               _fact("Four (4) backer boards.", ("s",))]
    m._supply_grounded(c, lines)
    assert _texts(c, "equipment_in_scope") == ["Approximately twenty-five (25) to thirty (30) CU34G2X curved monitors are in scope.",
                                               "Three (3) individual 1U-4U servers/devices are in scope."]


def test_moved_gear_sharing_only_counts_is_not_the_same_row():
    c = _content()
    c["customer_materials"] = [_fact("Twenty-five (25) to thirty (30) curved monitors.", ("a",)),
                               _fact("Twenty-five (25) to thirty (30) YeaLink phones.", ("b",))]
    m._moved_gear(c, "relocation_move")
    assert _texts(c, "equipment_in_scope") == ["Twenty-five (25) to thirty (30) curved monitors.",
                                               "Twenty-five (25) to thirty (30) YeaLink phones."]


def test_the_customers_own_team_member_is_never_re_partied():
    c = _content()
    c["customer_responsibilities"] = [_fact("Have a Lafayette team member available to meet the carrier and assist with packing.", ("t",))]
    fs = [{"id": "t", "text": "A member of our Lafayette team will be available to meet the carrier, assist with packing.",
           "supplier": ["us", 0.95]}]
    m._duty_party(c, fs, {"atoms": []})
    assert _texts(c, "customer_responsibilities") == ["Have a Lafayette team member available to meet the carrier and assist with packing."]
    assert not c["scope"]


def test_a_form_question_label_is_a_question():
    assert m._form_question("If new cables: Will Customer have them pre-hung in racks and labeled")
    assert m._form_question("Will cable/tracing and port mapping be in scope")
    assert not m._form_question("Will installs be performed after hours? [Betty] After hours, no weekends.")
    assert not m._form_question("Will Work be preformed during office hours: Enter Date | During office hours M-F: 2026-08-25")
    assert not m._form_question("Do not completely close the Time clock front.")


def test_a_sites_zip_comes_from_the_row_beside_its_street():
    c = _content()
    c["sites"] = [{"name": "Denver", "street": "1875 Lawrence Street, Suite 750", "city": "Denver", "state": "CO", "postal_code": ""}]
    m._site_zips(c, [{"text": "Origin Address: Origin City, State & Zip | 1875 Lawrence Street, Suite 750: Denver, CO 80204"}])
    assert c["sites"][0]["postal_code"] == "80204"


def test_an_empty_deliverable_row_goes():
    c = _content()
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Done."]}, {"item": "Chain of Custody Documentation", "description": []}]
    m._empty_deliverables(c)
    assert [d["item"] for d in c["deliverables"]] == ["Work Completion Report"]


def test_what_the_reseller_anticipates_including_is_customer_provided():
    c = _content()
    c["equipment_in_scope"] = [_fact("Provide and install a Cisco Meraki MS120 PoE switch to mount where the ceiling cable runs terminate.", ("r",))]
    m._reseller_includes(c, [{"id": "r", "text": "Due to limited ports, SHI anticipates including a Cisco Meraki MS120 PoE switch."}])
    assert not c["equipment_in_scope"]
    assert _texts(c, "scope") == ["Install the Customer-provided Cisco Meraki MS120 PoE switch to mount where the ceiling cable runs terminate."]
    assert _texts(c, "customer_materials") == ["The Cisco Meraki MS120 PoE switch."]


def test_a_workmanship_requirement_is_the_providers():
    c = _content()
    c["customer_responsibilities"] = [_fact("No cables should be visible after the mount is complete.")]
    c["scope"] = [_fact("Mount the device."), {"text": "Closeout", "cites": [], "kind": "heading"}]
    m._workmanship_not_duty(c)
    assert not c["customer_responsibilities"]
    assert _texts(c, "scope") == ["Mount the device.", "Conceal the cables so none are visible after the mount is complete.", "Closeout"]


# --- 010279 pass Y B -------------------------------------------------------------

def test_a_provider_task_whose_atom_an_exclusion_cites_goes():
    c = _content()
    c["out_of_scope"] = [_fact("Managed IT services or repair and replacement of defective Customer equipment.", ("f", "s"))]
    c["scope"] = [_fact("Service and resolve any hardware failures.", ("f",)),
                  _fact("Report unresolved issues to the Systems Support Desk.", ("s",)), _fact("Install the terminals.", ("t",))]
    m._exclusion_owns_atom(c, [{"id": "f", "text": "Service and resolve any hardware failures"},
                               {"id": "s", "text": "Report any issues to the Systems Support Desk"}])
    assert _texts(c, "scope") == ["Report unresolved issues to the Systems Support Desk.", "Install the terminals."]


def _rfp_atoms():
    loc = lambda n: {"page": 0, "line_start": n}
    return [{"id": "acc", "artifact_id": "rfp", "text": "The vendor will be required to accept & verify delivery of the following:", "locator": loc(71)},
            {"id": "bom", "artifact_id": "rfp", "locator": loc(72), "text":
             "Description | Part # | Quantity Needed | Supplied By | Toshiba 777 Terminals | TGCS 777 | 71 | HT | Cash Drawer Cables | "
             "TGCS Cash Drawer Cable | 242 | HT | Triplite 10 Ft DisplayPort Cable | P580-010 | 242 | HT | Toshiba 2X20 TCx Displays "
             "(Fuel Centers) | 3AA0079730 | 8 | HT"},
            {"id": "ins", "artifact_id": "rfp", "text": "The vendor will be required to transport and install the following equipment to each", "locator": loc(74)},
            {"id": "lst", "artifact_id": "rfp", "locator": loc(76),
             "text": "Description | Toshiba 777 Terminals | Cash Drawer Cables | Triplite 10 Ft DisplayPort Cable"}]


def test_an_accept_delivery_only_row_is_never_an_install_task():
    c = _content()
    c["scope"] = [_fact("Install 71 Toshiba 777 terminals at the self-checkout robots.", ("bom",)),
                  {"text": "Fuel centers", "cites": [], "kind": "heading"},
                  _fact("Install eight (8) Toshiba 2X20 TCx displays at the fuel centers.", ("bom",))]
    c["deliverables"] = [{"item": "Work Completion Report", "description": [
        "Document completed installation work for eight (8) Toshiba 2X20 TCx displays.", "Completion confirmed."]}]
    m._install_list_only(c, _rfp_atoms())
    assert _texts(c, "scope") == ["Install 71 Toshiba 777 terminals at the self-checkout robots."]
    assert c["deliverables"][0]["description"] == ["Completion confirmed."]


def test_a_table_rows_cable_type_is_never_its_neighbours():
    c = _content()
    c["scope"] = [_fact("Install 242 cash drawer cables.", ("bom",)), _fact("Run the HDMI-less display cables.", ("h",))]
    atoms = _rfp_atoms() + [{"id": "h", "text": "The existing connections will need to be reconnected (most likely HDMI cables)."}]
    m._cited_connector(c, atoms)
    assert _texts(c, "scope")[0] == "Install 242 cash drawer cables."


def test_a_sales_note_is_never_customer_text():
    c = _content()
    c["assumptions"] = [_fact("Travel is included in the store rate, or provide it as a separate line item if CDW prefers.", ("t",))]
    c["overview"] = [_fact("This is basic install work running from November through February.", ("b",))]
    c["schedule"] = [_fact("Work Window: a mix of business hours and after hours.", ("w",))]
    m._sales_notes_out(c, [{"id": "t", "text": "Travel included in the store rate, or clearly separated if CDW prefers"},
                           {"id": "b", "text": "This appears to be basic installs from Nov – Feb."},
                           {"id": "w", "text": "I think it looks like a mix of business hours and after hours."}])
    assert not c["assumptions"] and not c["overview"] and c["schedule"]


def test_a_dropped_line_strikes_only_the_outcome_it_names():
    c = _content()
    c["job_family"] = "pos_install_refresh"
    c["rule_decisions"] = [{"value": "drop", "text": "Review completed work and obtain Customer sign-off that installation and testing are complete."}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Completed Customer installation and testing checklists."]}]
    m._deliverables_hygiene(c)
    assert c["deliverables"][0]["description"] == ["Completed Customer installation and testing checklists."]


# --- 010082 pass Y B -------------------------------------------------------------

def test_a_customer_duty_is_admitted_only_when_its_source_puts_it_on_the_customer():
    atoms = [{"id": "v", "text": "Mike Stephens reminded that the vendor team must respect client drilling policies and site-specific safety requirements."},
             {"id": "o", "text": "Technician Arrival Protocol | Contact OSC prior to arrival"},
             {"id": "g", "text": "Technician Arrival Protocol | Customer greets the technician with an onsite manager on arrival."},
             {"id": "q", "text": "Aj, you'll get it over to the district manager, whoever needs to sign off on that."},
             {"id": "k", "text": "Mike Stephens agreed that if onsite contact cannot clarify installation conditions, a quick visit would be justified."},
             {"id": "p", "text": "Formal parking/loading dock process and permissions for drop-off", "structured": {"c3_label": {"type": "_keep", "p": 0.26}}},
             {"id": "l", "text": "Lift needed", "structured": {"c3_label": {"type": "requirement", "p": 0.9}}},
             {"id": "w", "text": "Provider (Chase/Delivery team): Reach out to Henry Hankins this week to gather details and schedule site visit if needed."}]
    c = _content()
    c["scope"] = [_fact("Follow client drilling rules and ladder-use protocols.", ("x",)),
                  _fact("Contact Henry Hankins to gather details and schedule a site visit if needed.", ("w",))]
    c["customer_responsibilities"] = [
        _fact("Customer will respect client drilling policies and site-specific safety requirements during installation.", ("v",)),
        _fact("Customer will coordinate any required contact with OSC prior to the technician's arrival.", ("o",)),
        _fact("Customer greets the technician with an onsite manager on arrival.", ("g",)),
        _fact("Customer will obtain any required signoff from the district manager.", ("q",)),
        _fact("If the onsite contact cannot clarify installation conditions, a quick site visit may be performed.", ("k",)),
        _fact("Customer will handle the formal parking/loading dock process and secure any permissions needed for drop-off.", ("p",)),
        _fact("Provide a lift for mounting as required.", ("l",))]
    m._duty_admission(c, atoms)
    assert _texts(c, "customer_responsibilities") == ["Customer greets the technician with an onsite manager on arrival.",
                                                      "Provide a lift for mounting as required."]
    assert _texts(c, "scope") == ["Follow client drilling rules and ladder-use protocols.",
                                  "Coordinate any required contact with OSC prior to the technician's arrival."]
    assert _texts(c, "assumptions") == ["If the onsite contact cannot clarify installation conditions, a quick site visit may be performed."]


def test_one_statement_per_topic():
    c = _content()
    c["assumptions"] = [_fact("A-frame ladder access is sufficient for the drop-tile ceiling work.", ("a",)),
                        _fact("The cable run remains within approximately 100 feet.", ("b",)),
                        _fact("The ceiling work can be performed using an A-frame ladder.", ("c",)),
                        _fact("All cable runs are no more than 100 feet.", ("d",)),
                        _fact("The replacement TVs are compatible with the existing content source and video wall layout.", ("e",)),
                        _fact("The existing 2x2 video wall structure and mounting points can safely support the two replacement TVs.", ("f",))]
    m._same_topic_once(c)
    assert _texts(c, "assumptions") == ["A-frame ladder access is sufficient for the drop-tile ceiling work.",
                                        "The cable run remains within approximately 100 feet.",
                                        "The replacement TVs are compatible with the existing content source and video wall layout.",
                                        "The existing 2x2 video wall structure and mounting points can safely support the two replacement TVs."]
    assert c["assumptions"][0]["cites"] == ["a", "c"]


# --- 010209 pass Y B -------------------------------------------------------------

def test_a_line_restating_its_sources_limit_is_not_broader():
    c = _content()
    c["scope"] = [_fact("Install rack mount hardware at sites even when no existing rack is present.", ("r",)),
                  _fact("All stores have a rack.", ("r",))]
    m._no_broader_than_source(c, [{"id": "r", "text": "Not all stores have a rack but still need rack mount hardware installed"}])
    assert _texts(c, "scope") == ["Install rack mount hardware at sites even when no existing rack is present."]


def test_a_line_naming_its_cited_models_is_grounded():
    lines = {"m": "The switches being removed are a combination of MS120-24 and MS120-48."}
    assert m._models_cited({"text": "Existing switches to be decommissioned include Cisco Meraki MS120-24 and MS120-48 models.", "cites": ["m"]}, lines)
    assert not m._models_cited({"text": "Install MS390-48 switches.", "cites": ["m"]}, lines)


def test_label_and_trace_come_before_removal():
    c = _content()
    c["scope"] = [{"text": "Switch replacement", "cites": [], "kind": "heading"},
                  _fact("Remove existing switches and install the new switches."),
                  _std("Verify available rack space for the new switch before removing the existing switch."),
                  {"text": "Cabling", "cites": [], "kind": "heading"},
                  _fact("Label all cables before removing the old switches."),
                  _fact("Trace each cable and create the patch panel to switch port mapping onsite."),
                  _fact("Reconnect patch cables to the new switches.")]
    m._label_before_removal(c)
    assert _texts(c, "scope") == ["Switch replacement", "Verify available rack space for the new switch before removing the existing switch.",
                                  "Label all cables before removing the old switches.",
                                  "Trace each cable and create the patch panel to switch port mapping onsite.",
                                  "Remove existing switches and install the new switches.", "Cabling", "Reconnect patch cables to the new switches."]


def test_a_no_charge_term_is_the_providers_error_owed_to_the_customer():
    c = _content()
    c["assumptions"] = [_fact("Any extra time spent identifying cables because cables were not labeled is at no charge to RL.", ("n",))]
    m._no_charge_bounds(c, [{"id": "n", "text": "the tech did not label EVERY cable and spent hours trying to identify 1 cable "
                                               "(this extra time must be spent at no charge to RL)"}])
    assert _texts(c, "assumptions") == ["Any extra time spent identifying cables because Provider did not label them is at no charge to the Customer."]


def test_a_boundary_around_the_devices_work_is_no_contradiction():
    c = _content()
    c["scope"] = [_fact("Install the new switches.")]
    c["out_of_scope"] = [_fact("Cleaning up or remediating other closet connections outside the switch area."),
                         _fact("Switch installation.", ())]
    m._tidy(c)
    assert "Cleaning up or remediating other closet connections outside the switch area." in _texts(c, "out_of_scope")


# --- 010373 pass Y A- ------------------------------------------------------------

def test_an_approval_duty_needs_a_source_that_asks_for_one():
    c = _content()
    c["customer_responsibilities"] = [
        {"text": "Approve any packing-material removal or disposal before Provider leaves the site.", "cites": ["h"], "kind": "method"},
        _fact("Approve the final mounting heights.", ("a",))]
    m._approval_needs_source(c, [{"id": "h", "text": "I just want to confirm the engineer's ability to haul away the packaging."},
                                 {"id": "a", "text": "The customer must approve the mounting heights before drilling."}])
    assert _texts(c, "customer_responsibilities") == ["Approve the final mounting heights."]


def test_the_check_reads_a_method_line_that_cites_atoms(monkeypatch):
    seen = []
    def fake(prompt, *a, **k):
        seen.append(prompt)
        return json.dumps({"supported": []})
    monkeypatch.setattr(m, "_complete", fake)
    c = _content()
    c["customer_responsibilities"] = [{"text": "Approve any packing-material removal or disposal.", "cites": ["h"], "kind": "method"}]
    m.check(c, {"h": "I just want to confirm the engineer's ability to haul away the packaging."})
    assert not c["customer_responsibilities"] and "Approve any packing-material removal" in seen[0]


def test_a_per_unit_count_is_not_a_summary_count():
    c = _content()
    c["overview"] = [_fact("Provider will install 16 Humanscale monitor arms and mount 48 new 27-inch monitors.", ())]
    c["scope"] = [_fact("Mount forty-eight (48) new 27-inch monitors on the installed arms, three (3) monitors per arm.")]
    m._overview_covers_scope(c, [])
    assert len(c["overview"]) == 1
