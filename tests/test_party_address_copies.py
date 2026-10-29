"""010358 (2026-10-29): bare copies of the customer's office address read as sites; a sheet summary read as a site."""
from types import SimpleNamespace as NS

from app.core import c3_type_base as c3
from app.core.party_address_veto import follow_party_copies


def _a(i, t, text, loc=None):
    return NS(id=i, atom_type=t, raw_text=text, value={}, entity_keys=["site:x"] if t == "physical_site" else [],
              source_refs=[NS(locator=loc or {})], artifact_id="art")


def test_bare_copies_follow_the_contact_line():
    atoms = [
        _a("s1", "stakeholder", "Scott Morgan | 1361 N Glenville Dr | scott@traversa.net | 214-707-0631"),
        _a("s2", "stakeholder", "Richardson TX 75081"),
        _a("p1", "physical_site", "1361 N Glenville Dr"),
        _a("p2", "physical_site", "Richardson TX 75081"),
        _a("p3", "physical_site", "4290 Thurman Road, Conley, GA 30288"),
    ]
    types = {"p1": ["physical_site", 0.997], "p2": ["physical_site", 0.9], "p3": ["physical_site", 0.99]}
    assert sorted(follow_party_copies(atoms, types)) == ["p1", "p2"]
    assert atoms[2].atom_type in ("stakeholder",) or getattr(atoms[2].atom_type, "value", None) == "stakeholder"
    assert atoms[2].entity_keys == [] and types["p1"][0] == "stakeholder"
    assert atoms[4].atom_type == "physical_site"


def test_a_roster_row_never_moves():
    atoms = [
        _a("s1", "stakeholder", "Jane Doe | 100 Main St | jane@acme.com"),
        _a("r1", "physical_site", "100 Main St", loc={"sheet": "Sites", "row": 4}),
    ]
    assert follow_party_copies(atoms, {}) == []


def test_our_sheet_summary_never_becomes_a_site():
    summary = NS(atom_type="derived_finding", value={}, entity_keys=[])
    assert c3.overwrite_bar(summary, "physical_site", 0.5) == float("inf")
    assert c3.overwrite_bar(summary, "_keep", 0.5) == 0.5


def test_quote_sites_keep_zip_specific_keys():
    roster = {k: {"keys": [k], "atom_ids": [k]} for k in
              ("site:houston_77030", "site:houston_77054_row_32", "site:houston_77054_row_33",
               "site:basking_ridge_nj", "site:1_main_st_basking_ridge_nj_07920")}
    c3._fold_contained(roster)
    assert "site:houston_77030" in roster and "site:houston_77054_row_32" in roster
    assert "site:basking_ridge_nj" not in roster and len(roster) == 4


def test_quote_hours_from_the_site_rows():
    by_id = {f"a{i}": NS(value={"extras": {"Total Hrs": str(h)}}) for i, h in enumerate((7.5, 12.0, 3.77))}
    q = {"lines": [{"techs": 2, "hours_per_tech": 8, "crew_hours": 16, "rate_line": "L2", "by": {"hours": "heads:read:hours_stated"}}]}
    sites = [{"site": f"s{i}", "atom_ids": [f"a{i}"]} for i in range(3)]
    out = c3._with_site_row_hours(q, sites, by_id)["lines"][0]
    # per site: the prefill applies the line at each site, so the kit's total is the sheet's 23.3 h
    assert out["crew_hours"] == 7.77 and out["hours_per_tech"] == 3.88
    assert out["by"]["hours"] == "sheet:site_rows" and len(out["site_hours"]) == 3
    assert c3._with_site_row_hours(q, sites + [{"site": f"x{i}", "atom_ids": []} for i in range(5)], by_id) is q
