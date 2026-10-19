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
