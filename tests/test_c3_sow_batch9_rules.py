"""SOW writer rules, batch 9 (pass W grades: 010147 C+, 010116 B-; batch-8 ledger)."""
from __future__ import annotations

import copy as _copy
import json as _json
import os as _os

import pytest as _pytest

from app.core import c3_sow as m


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


def _std(family, section, needle):
    return next(t for t in m.purtera_standards(family)[section] if needle in t)


# --- 010147 C+: a site needs a street, a ZIP or a city and state ---------------

def test_a_sentence_is_never_a_site():
    assert not m._site_line("Origin site does not have an accessible 70' tractor-trailer loading dock.", 2)
    assert m._site_line("Denver, CO | 1875 Lawrence Street, Suite 750", 2)
    assert m._site_line("Lafayette, IN | 1000 Sagamore Parkway S", 2)
    assert m._site_line("Waste Water Treatment Facility (1015 S Lakeview Dr, Manitowoc, WI 54220) — 8 phones.", 30)
    # A short place name inside the one roster site still names it (010115).
    assert m._site_line("Specific Location(s) | Sodexo Dining CAFE", 1)
    assert not m._site_line("Some locations may require mounting on concrete or other hard wall surfaces.", 1)


def test_an_na_answer_is_never_a_negative_claim():
    c = _content()
    c["assumptions"] = [
        {"text": "Origin site does not have an accessible 70' tractor-trailer loading dock.", "cites": ["q1"], "kind": "fact"},
        {"text": "The destination does not have a freight elevator.", "cites": ["q2"], "kind": "fact"},
        {"text": "No console work is anticipated.", "cites": ["q3"], "kind": "fact"}]
    atoms = [{"id": "q1", "text": "Origin Address: Is loading dock tractor trailer (70' long) accessible? | 1875 Lawrence Street, Suite 750: N/A"},
             {"id": "q2", "text": "Is there a freight elevator? | No"},
             {"id": "q3", "text": "Will any console work need to be completed? N/a"}]
    m._na_not_negative(c, atoms)
    assert _texts(c, "assumptions") == ["The destination does not have a freight elevator.", "No console work is anticipated."]


# --- 010147 C+: the inventory is the source of truth for counts ---------------

def test_a_prose_count_of_inventory_gear_goes():
    c = _content()
    c["scope"] = [{"text": "Pack and move five (5) network switches.", "cites": ["inv"], "kind": "fact"},
                  {"text": "Pack and move the printer, smaller peripherals, and two (2) Cisco 1U switches.", "cites": ["mail"], "kind": "fact"},
                  {"text": "Pack and move two (2) switches.", "cites": ["mail"], "kind": "fact"},
                  {"text": "Pack and move twenty-five (25) to thirty (30) Yealink phones.", "cites": ["mail"], "kind": "fact"}]
    atoms = [{"id": "inv", "atom_type": "quantity", "text": "Quantity 5 Network Switch"},
             {"id": "ph", "atom_type": "vendor_line_item", "text": "Line item YeaLink Phones — qty 25-30"}]
    m._inventory_counts(c, atoms)
    assert _texts(c, "scope") == ["Pack and move five (5) network switches.",
                                  "Pack and move the printer and smaller peripherals.",
                                  "Pack and move twenty-five (25) to thirty (30) Yealink phones."]


# --- 010147 C+: a form's category label is not a verb of the work -------------

def test_label_verbs_leave_the_task():
    c = _content()
    c["scope"] = [{"text": "Pack, deinstall, and move three (3) individual 1U-4U servers/devices for deinstall/reinstall (not remaining in racks).",
                   "cites": ["h", "q"], "kind": "fact"},
                  {"text": "Deinstall and pack the two (2) racks.", "cites": ["r"], "kind": "fact"}]
    atoms = [{"id": "h", "text": "General Inventory Quantity: Individual Servers/Devices for Deinstall/Reinstall (not remaining in racks)"},
             {"id": "q", "text": "#of individual devices: 1U-4U Server Quantity 3"},
             {"id": "r", "text": "Provider will deinstall and pack the two racks."}]
    m._label_verbs(c, atoms)
    assert _texts(c, "scope") == ["Pack and move three (3) individual 1U-4U servers/devices.",
                                  "Deinstall and pack the two (2) racks."]


# --- 010147 C+: one sentence, one party ---------------------------------------

def test_a_split_sentence_stays_with_its_subject():
    s = ("A member of our Lafayette team will be available to meet the carrier, assist with packing, and address "
         "anything that may have been missed during the pre-audit.")
    c = _content()
    c["scope"] = [{"text": "Closeout", "cites": [], "kind": "heading"},
                  {"text": "Address anything missed during the pre-audit.", "cites": ["a"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Provide a member of the Lafayette team to meet the carrier and assist with packing.",
                                       "cites": ["a"], "kind": "fact"}]
    m._traced(c, "one_sentence_one_party", m._one_sentence_one_party, c, [{"id": "a", "text": s}])
    assert _texts(c, "scope") == ["Closeout"]
    assert _texts(c, "customer_responsibilities")[-1] == "Address anything missed during the pre-audit."
    assert any(d["rule"] == "one_sentence_one_party" and d["head"] == "sow_party" for d in c["rule_decisions"])


def test_a_we_sentence_is_left_alone():
    c = _content()
    c["scope"] = [{"text": "Mount the displays.", "cites": ["a"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Provide the mounts.", "cites": ["a"], "kind": "fact"}]
    m._one_sentence_one_party(c, [{"id": "a", "text": "We will mount the displays on the mounts you provide."}])
    assert _texts(c, "scope") == ["Mount the displays."]


# --- 010147 C+: moved gear, cables included -----------------------------------

def test_moved_equipment_reads_the_whole_verb_run_and_the_form_counts():
    c = _content()
    c["job_family"] = "relocation_move"
    c["equipment_in_scope"] = [{"text": "25–30 curved monitors.", "cites": ["mon"], "kind": "fact"}]
    c["scope"] = [{"text": "Pack and move twenty-five (25) to thirty (30) CU34G2X curved monitors.", "cites": ["mon"], "kind": "fact"},
                  {"text": "Pack and move the printer and smaller peripherals.", "cites": ["mail"], "kind": "fact"}]
    atoms = [{"id": "cab", "text": "Estimated # of Cables: 20", "structured": {"c3_label": {"act": "keep"}}},
             {"id": "zero", "text": "# of Complete SAN Frame/Cabinets: 0"}]
    m._moved_equipment(c, atoms)
    eq = _texts(c, "equipment_in_scope")
    assert not any(t.startswith("And move") for t in eq)
    assert "Printer and smaller peripherals." in eq
    assert "Twenty (20) cables." in eq
    assert not any("SAN" in t for t in eq)


def test_the_desk_duty_fires_on_office_technology_equipment():
    t = _std("relocation_move", "customer_responsibilities", "Remove desktop and office equipment from desks.")
    ok, _ = m._standard_applies(t, ["We need assistance moving office technology equipment from our Denver office."])
    assert ok


# --- 010116 B-: room bar, one supply list, rooms in the summary, tool lines ----

def test_room_bar_triggers_the_enrollment_and_remote_engineering_duties():
    for needle in ("Complete device enrollment", "Provide remote AV or network engineering support"):
        t = _std("av_conference_room", "customer_responsibilities", needle)
        assert m._standard_applies(t, ["1: CS-BAR-T-C-K9 Cisco Room Bar in Carbon Black w/Navigator -36mo"])[0]


def test_one_bom_is_one_supply_list():
    atoms = [{"id": f"r{i}", "artifact_id": "xlsx", "text": t} for i, t in enumerate(
        ["1: QM75C Samsung 75IN 4K UHD LED Display", "1: MMX4X2-HT200 Lightware 4x2 Matrix Switcher",
         "1: CS-BAR-T-C-K9 Cisco Room Bar in Carbon Black w/Navigator"])]
    c = _content()
    c["equipment_in_scope"] = [{"text": "QM75C Samsung 75IN 4K UHD LED Display.", "cites": ["r0"], "kind": "fact"},
                               {"text": "CS-BAR-T-C-K9 Cisco Room Bar in Carbon Black w/Navigator.", "cites": ["r2"], "kind": "fact"}]
    c["customer_materials"] = [{"text": "MMX4X2-HT200 Lightware 4x2 Matrix Switcher.", "cites": ["r1"], "kind": "fact"}]
    m._bom_lists(c, atoms)
    assert _texts(c, "equipment_in_scope") == []
    assert len(_texts(c, "customer_materials")) == 3


def test_the_summary_puts_each_item_in_its_own_room():
    atoms = [{"id": "b1", "text": "1: QM75C Samsung 75IN 4K UHD LED Display", "locator": {"section_path": ["Sheet1", "Boardroom"]}},
             {"id": "b2", "text": "1: CS-BAR-T-C-K9 Cisco Room Bar", "locator": {"section_path": ["Sheet1", "Boardroom"]}},
             {"id": "c1", "text": "2: 2 | QM65C: TS525TU | Samsung 65IN Display: Chief Mount",
              "locator": {"section_path": ["Sheet1", "Common Area Displays"]}},
             {"id": "b3", "text": "1: MMX4X2-HT200 Lightware Matrix Switcher", "locator": {"section_path": ["Sheet1", "Boardroom"]}},
             {"id": "p", "text": "QTY 3", "locator": {"section_path": ["Sheet1", "EXT. PRICE"]}}]
    c = _content()
    c["overview"] = [{"text": "Provider will install one (1) QM75C Samsung 75IN Display, two (2) QM65C Samsung 65IN Displays, "
                              "and one (1) CS-BAR-T-C-K9 Cisco Room Bar in the Boardroom.", "cites": ["b1", "c1", "b2"], "kind": "fact"},
                     {"text": "Provider will also install one (1) MMX4X2-HT200 Lightware Matrix Switcher.", "cites": ["b3"], "kind": "fact"},
                     {"text": "Provider will install three (3) cameras.", "cites": ["p"], "kind": "fact"}]
    m._summary_rooms(c, {"atoms": atoms})
    ov = _texts(c, "overview")
    assert ov[0] == ("Provider will install one (1) QM75C Samsung 75IN Display and one (1) CS-BAR-T-C-K9 Cisco Room Bar in the "
                     "Boardroom and two (2) QM65C Samsung 65IN Displays in the Common Area.")
    assert ov[1].endswith("Matrix Switcher in the Boardroom.")
    assert ov[2] == "Provider will install three (3) cameras."  # a price column is no room


def test_an_eponym_hand_tool_is_not_an_unsupported_name(monkeypatch):
    monkeypatch.setattr(m, "_complete", lambda *a, **k: _json.dumps({"supported": list(range(50))}))
    c = _content()
    c["provider_tools"] = [{"text": "Allen key set for mounts and brackets.", "cites": [], "kind": "method"},
                           {"text": "Velcro cable ties.", "cites": [], "kind": "method"},
                           {"text": "Fluke DSX-8000 certifier.", "cites": [], "kind": "method"}]
    m.check(c, {"f": "Mount the Samsung display."})
    assert _texts(c, "provider_tools") == ["Allen key set for mounts and brackets.", "Velcro cable ties."]


def test_the_av_toolkit_holds_the_allen_key_set():
    t = _std("av_conference_room", "tools", "Allen key set")
    assert m._standard_applies(t, ["1: TS525TU Chief Thinkstall Wall Mount"])[0]


# --- batch-8 ledger: 010083 / 010279 / 010376 ----------------------------------

def test_the_reseller_rule_is_traced():
    c = _content()
    c["scope"] = [{"text": "Provide professional services for central receiving, then scheduled shipping of the needed switch.",
                   "cites": ["a"], "kind": "fact"},
                  {"text": "Install the new switches.", "cites": ["b"], "kind": "fact"}]
    fs = [{"id": "z", "text": "CDW will ship all switches directly to each site."}]
    m._traced(c, "reseller_supplies", m._reseller_supplies, c, fs)
    assert _texts(c, "scope") == ["Install the new switches."]
    assert [d["rule"] for d in c["rule_decisions"] if d["value"] == "drop"] == ["reseller_supplies"]


def test_the_ledger_knows_a_reworded_line_and_an_unknown_cite():
    c = _content()
    c["customer_materials"] = [{"text": "One APC Smart-UPS SRT 5000VA UPS.", "cites": [], "kind": "fact"}]
    c["writer_draft"] = {"customer_materials": [{"text": "Supply one APC Smart-UPS SRT 5000VA UPS.", "cites": ["garbled1"]}],
                         "scope": [{"text": "Provide post installation support.", "cites": ["garbled2"]},
                                   {"text": "Install the router.", "cites": ["real"]}]}
    c["_unknown_cites"] = ["Provide post installation support."]
    m._untraced_ledger(c)
    got = {d["text"]: d["rule"] for d in c.get("rule_decisions") or []}
    assert "Supply one APC Smart-UPS SRT 5000VA UPS." not in got
    assert got["Provide post installation support."] == "uncited"
    assert got["Install the router."] == "untraced"
    assert not m._shown_reworded("Supply two APC Smart-UPS units.", {"One APC Smart-UPS units."})


# --- replays (skipped where the saved envelope is absent) ----------------------

from tests.test_c3_sow_batch4_rules import _SAVED, _replay  # noqa: E402


def _have(deal):
    p = f"{_SAVED}/final_{deal}_env.json"
    return _os.path.exists(p)


def test_010147_replay_has_two_sites_and_a_route_summary(monkeypatch):
    if not _have("010147"):
        _pytest.skip("saved envelope not on this machine")
    c = _replay("010147", monkeypatch)
    alltxt = " ".join(str(x.get("text")) for k in m.LISTS for x in c.get(k) or [] if isinstance(x, dict))
    assert "tractor-trailer loading dock" not in alltxt
    assert not any("loading dock" in str(s.get("name")) for s in c["sites"])
    assert "deinstall/reinstall" not in alltxt.lower()
    assert "two (2) Cisco 1U switches" not in alltxt
    assert not any("pre-audit" in t for t in _texts(c, "scope"))
    ov = " ".join(_texts(c, "overview"))
    if "Denver, CO" in ov and "Lafayette, IN" in ov:
        assert "Denver, CO to Lafayette, IN" in ov  # a route, never widened into a site list


def test_010116_replay_puts_the_common_area_displays_in_the_common_area(monkeypatch):
    if not _have("010116"):
        _pytest.skip("saved envelope not on this machine")
    c = _replay("010116", monkeypatch)
    ov = " ".join(_texts(c, "overview"))
    if "QM65C" in ov:
        assert "Common Area" in ov
        assert not _copy.copy(ov).split("QM65C", 1)[1].split(".")[0].rstrip().endswith("in the Boardroom")
