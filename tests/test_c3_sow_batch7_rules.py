"""SOW writer rules, batch 7 (pass V grades: 010373 B, 010116 B-, 010083 B, 010147 B, 010264 C-)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


# --- 010373 B: a capability sentence is never a supply line ------------------

def test_a_capability_reply_is_not_a_provider_material():
    c = _content()
    c["provider_materials"] = [{"text": "Sixteen (16) monitor arms for the installation.", "cites": ["f1", "f2"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Provide all monitor arms, monitors, mounting hardware and cables.", "cites": [], "standard": True}]
    fs = [_fact("f1", "Each desk/arm will have 3 new 27\" monitors mounted to them for a total of 48 monitors."),
          _fact("f2", "We can handle the monitor arms and monitor installation.")]
    m._capability_not_supply(c, fs)
    assert c["provider_materials"] == []
    assert _texts(c, "customer_materials") == ["Sixteen (16) monitor arms for the installation."]


def test_a_capability_reply_with_no_customer_side_goes_to_equipment():
    c = _content()
    c["provider_materials"] = [{"text": "Four (4) access points.", "cites": ["f1"], "kind": "fact"}]
    m._capability_not_supply(c, [_fact("f1", "We can install the four access points next week.")])
    assert c["provider_materials"] == [] and _texts(c, "equipment_in_scope") == ["Four (4) access points."]


def test_a_line_saying_we_supply_stays_ours():
    c = _content()
    c["provider_materials"] = [{"text": "Cat6 cable for the four drops.", "cites": ["f1", "f2"], "kind": "fact"}]
    fs = [_fact("f1", "We can handle the cabling."), _fact("f2", "Our tech will provide the Cat6 cable.")]
    m._capability_not_supply(c, fs)
    assert _texts(c, "provider_materials") == ["Cat6 cable for the four drops."]


def test_one_supplier_reads_a_spelled_count_and_a_procurement_exclusion():
    c = _content()
    c["provider_materials"] = [{"text": "Sixteen (16) monitor arms for the installation.", "cites": ["f1"], "kind": "fact"}]
    c["customer_responsibilities"] = [{"text": "Provide all monitor arms, monitors, mounting hardware and cables.", "cites": [], "standard": True}]
    c["out_of_scope"] = [{"text": "Procurement of monitors, arms or accessories.", "cites": [], "standard": True}]
    m._one_supplier(c, [_fact("f1", "16 monitor arms on new desks")])
    assert c["provider_materials"] == [] and _texts(c, "customer_materials") == ["Sixteen (16) monitor arms for the installation."]


def test_a_counted_step_in_a_standard_slot_keeps_the_steps_later_work(monkeypatch):
    monkeypatch.setattr(m, "purtera_standards", lambda fam: {"method_steps": ["Mount the monitors on each arm, then align and secure them."]})
    c = _content()
    c["scope"] = [{"text": "Mount forty-eight (48) new 27-inch monitors on the installed arms, three (3) per desk/arm.", "cites": ["f1"], "kind": "fact"}]
    m._apply_standards(c, "smart_hands_general", ["48 monitors mounted, 3 per arm"])
    assert "Mount forty-eight (48) new 27-inch monitors on the installed arms, three (3) per desk/arm, then align and secure them." in _texts(c, "scope")


# --- 010279 B+: who a split RFP sentence names, reuse, the report ------------

def _atom(i, text, line, sp, art="a1"):
    return {"id": i, "text": text, "artifact_id": art, "section_path": sp, "locator": {"line_start": line}}


def test_a_split_rfp_sentence_keeps_its_subject():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide a resource at the corporate office for assistance and escalation as needed.", "cites": ["f1"]}]
    f = _fact("f1", "This work will require no more than 2 tech POS Installer team. Harris Teeter will provide a resource at the "
                    "HT Corporate Office for assistance and escalation as needed.")
    f["supplier"] = ("us", 0.95)
    m._duty_party(c, [f], {"atoms": []})
    assert _texts(c, "customer_responsibilities") == ["Provide a resource at the corporate office for assistance and escalation as needed."]
    assert c["scope"] == []


def test_a_supplier_head_duty_with_no_named_party_is_ours():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide proof of delivery for each site.", "cites": ["f1"]}]
    f = _fact("f1", "Proof of delivery for each site.")
    f["supplier"] = ("us", 0.95)
    m._duty_party(c, [f], {"atoms": []})
    assert _texts(c, "scope") == ["Provide proof of delivery for each site."]


def test_a_duty_under_a_technician_lead_line_is_the_technicians():
    sp = ["2.3 Post-Installation"]
    atoms = [_atom("h", "2.3 Post-Installation", 59, sp), _atom("l", "Technician to provide post installation support", 60, sp),
             _atom("f1", "Verifies all tasks complete and store ready for business", 61, sp)]
    c = _content()
    c["scope"] = [{"text": "Closeout", "kind": "heading", "cites": []}, {"text": "Confirm completion.", "cites": [], "standard": True}]
    c["customer_responsibilities"] = [{"text": "Verify all tasks complete and store ready for business.", "cites": ["f1"]}]
    m._duty_party(c, [_fact("f1", "Verifies all tasks complete and store ready for business")], {"atoms": atoms})
    assert c["customer_responsibilities"] == []
    assert _texts(c, "scope")[-1] == "Verify all tasks complete and store ready for business."


def test_a_connector_pin_count_is_not_a_secret_and_the_report_never_shows_one():
    assert "[withheld]" not in m._withhold("Toshiba 12V Powered USB to 6 Pin | 42M5630 | 8")
    assert "[withheld]" in m._withhold("Door PIN: 4471")
    c = {"deliverables": [{"item": "Work Completion Report", "description": ["8 cables installed.", "Gate code PIN: 4471 used [withheld]"]}]}
    m._scrub(c)
    assert c["deliverables"][0]["description"] == ["8 cables installed."]


def test_a_removal_never_takes_what_the_job_reuses():
    c = _content()
    c["scope"] = [{"text": "Disconnect and remove the existing POS terminals and peripherals as specified.", "cites": ["f1"]}]
    c["assumptions"] = [{"text": "Existing power, network, and peripheral connections are operational and can be reused.", "cites": []}]
    m._reused_not_removed(c, [])
    assert _texts(c, "scope") == ["Disconnect and remove the existing POS terminals as specified."]


def test_the_report_states_outcomes_not_a_bill_of_materials():
    c = _content()
    c["scope"] = [{"text": "Install 242 Toshiba Garage Units and 242 cash drawer cables.", "cites": ["f1"]}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["242 Toshiba Garage Units", "8 Toshiba 2X20 TCx Displays",
                                                                             "Completion photos of each installed location."]}]
    m._report_counts_are_outcomes(c)
    assert c["deliverables"][0]["description"] == ["242 Toshiba Garage Units installed.", "Completion photos of each installed location."]


def test_a_long_dropped_line_does_not_strike_an_outcome_it_only_mentions():
    c = _content()
    c["job_family"] = "pos_install_refresh"
    c["rule_decisions"] = [{"value": "drop", "text": "Provider will de-install the existing terminals, apply asset tags, record terminal MAC "
                                                     "addresses by store, palletize old hardware for return/disposal, and complete "
                                                     "installation and testing checklists."}]
    c["deliverables"] = [{"item": "Work Completion Report", "description": ["Completed Customer installation and testing checklists."]}]
    m._deliverables_hygiene(c)
    assert c["deliverables"][0]["description"] == ["Completed Customer installation and testing checklists."]
