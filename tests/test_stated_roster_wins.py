"""roster.stated_count_wins: the deal states N sites and one site list has N; sites only a larger list names
(010447's 244-store "annette source" tab beside its "2027_145_Sites" tab) leave the quote."""
from __future__ import annotations

from app.core import rule_ledger
from app.core import stated_roster as R


def _row(i, sheet, art="wb1"):
    return {"id": i, "artifact_id": art, "raw_text": f"Site No.: {i}",
            "locator": {"sheet": sheet, "extraction": "xlsx_site_roster_v1", "roster_rank": 1 if sheet == "145" else 2}}


def _deal(n_stated=10, n_list=10, n_super=25, overlap=8):
    atoms = [{"id": "say", "raw_text": f"The project covers {n_stated} sites across the region."}]
    sites = []
    for k in range(n_list):
        atoms.append(_row(f"L{k}", "145"))
        sites.append({"atom_ids": [f"L{k}"] + ([f"S{k}"] if k < overlap else [])})
    for k in range(n_super):
        atoms.append(_row(f"S{k}", "superset"))
        if k >= overlap:
            sites.append({"atom_ids": [f"S{k}"]})
    return sites, atoms


def test_superset_only_sites_leave():
    sites, atoms = _deal()
    kept = R.apply(sites, atoms)
    assert len(kept) == 10
    assert rule_ledger.fires().get(R.RULE_ID, {}).get("no_head", 0) >= 1


def test_no_stated_count_no_change():
    sites, atoms = _deal()
    atoms[0]["raw_text"] = "Thanks for the list."
    assert len(R.apply(sites, atoms)) == len(sites)


def test_no_list_of_the_stated_size_no_change():
    sites, atoms = _deal(n_stated=40)
    assert len(R.apply(sites, atoms)) == len(sites)


def test_lines_outside_any_list_stay():
    sites, atoms = _deal()
    atoms.append({"id": "mail", "raw_text": "Also the warehouse at 1 Main St", "locator": {}})
    sites.append({"atom_ids": ["mail"]})
    assert any(s["atom_ids"] == ["mail"] for s in R.apply(sites, atoms))
