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


def test_site_edit_renames_and_carries_tier_upstream():
    from app.core import rule_ledger

    atoms = [_a("s1", "physical_site", "Plant 7 | 7 Mill Rd"), _a("s2", "physical_site", "Plant 7 dock"), _a("m1", "deal_metadata", "Plant 7")]
    atoms[0].value = {"facility_name": "Plant 7", "site_id": "P7"}
    atoms[1].entity_keys = ["site:plant_7"]
    edits = [{"status": "site_edit", "atomIds": ["s1", "m1"], "siteKey": "site:plant_7", "name": "Waco Plant", "tier": "T1",
              "constraints": ["badge", 3], "reviewer": "pm@x"}]
    rule_ledger.take("pm_fix.")
    assert apply_reviews(atoms, edits) == ["s1"]  # ids win; the metadata line is never edited into a site
    v = atoms[0].value
    assert v["facility_name"] == "Waco Plant" and v["name"] == "Waco Plant" and v["site_id"] == "P7"
    assert v["pm_name_was"] == "Plant 7" and v["pm_site"] == {"tier": "T1", "constraints": ["badge"]}
    assert v["c3_label"]["pm_confirmed"] is True and v["c3_label"]["act"] == "pm"
    assert atoms[0].atom_type == "physical_site" and atoms[0].entity_keys == ["site:x"]
    assert "pm_fix.site_edit" in rule_ledger.take("pm_fix.")


def test_site_edit_follows_its_site_key_when_ids_moved():
    atoms = [_a("new1", "physical_site", "Plant 7 dock")]
    atoms[0].entity_keys = ["site:plant_7"]
    assert apply_reviews(atoms, [{"status": "site_edit", "atomIds": ["old1"], "siteKey": "site:plant_7", "tier": "T2"}]) == ["new1"]
    assert atoms[0].value["pm_site"] == {"tier": "T2"} and "facility_name" not in atoms[0].value


def test_load_site_edits_reads_the_blob():
    import json as _json
    from app.core import pm_fact_reviews as m

    class _B:
        def __init__(self, d):
            self.d = d

        def readall(self):
            return _json.dumps(self.d).encode()

    class _CC:
        def download_blob(self, path):
            if path.endswith("site_edits.json"):
                return _B({"edits": [{"atomIds": ["a"], "name": "X"}, {"nothing": 1}]})
            raise FileNotFoundError(path)

    assert m.load_site_edits("d", _CC()) == [{"atomIds": ["a"], "name": "X", "status": "site_edit"}]
