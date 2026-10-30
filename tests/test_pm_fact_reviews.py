"""A PM's saved fixes override the line (owner 2026-10-29)."""
from types import SimpleNamespace as NS

from app.core.pm_fact_reviews import apply_reviews


def _a(i, t, text, fn="sheet.xlsx"):
    return NS(id=i, atom_type=t, raw_text=text, value={}, entity_keys=["site:x"], source_refs=[NS(filename=fn, locator={})])


def test_not_a_site_and_fix_address_and_looks_right():
    atoms = [_a("a1", "physical_site", "Location | Houston, TX", "mail.eml"), _a("a2", "physical_site", "Store: 64 | City: Greensboro"),
             _a("a3", "physical_site", "Store: 65 | City: Raleigh")]
    types = {"a1": ["physical_site", 0.99]}
    reviews = [
        {"status": "fixed", "fixedValue": "not a site", "reason": "A city or region mention", "note": "region line",
         "factText": "Location | Houston, TX", "sourceFilename": "mail.eml", "reviewer": "pm@x"},
        {"status": "fixed", "fixedValue": "123 Main St, Greensboro, NC", "reason": "Address wrong", "atomId": "a2"},
        {"status": "looks_right", "factText": "store: 65 | city: raleigh"},
    ]
    assert sorted(apply_reviews(atoms, reviews, types)) == ["a1", "a2", "a3"]
    assert atoms[0].atom_type in ("deal_metadata",) or getattr(atoms[0].atom_type, "value", None) == "deal_metadata"
    assert atoms[0].entity_keys == [] and types["a1"][0] == "deal_metadata"
    assert atoms[0].value["pm_fix"]["why"] == "A city or region mention · region line"
    assert atoms[1].value["street_address"] == "123 Main St, Greensboro, NC" and atoms[1].value["c3_label"]["act"] == "pm"
    assert atoms[2].value["c3_label"]["pm_confirmed"] is True


def test_no_reviews_changes_nothing():
    atoms = [_a("a1", "physical_site", "x")]
    assert apply_reviews(atoms, []) == []
