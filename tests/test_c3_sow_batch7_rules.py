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
