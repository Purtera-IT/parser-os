"""Roster rows that are one building merge; distinct buildings never do."""
from app.core.site_identity import merge_roster


def _row(site, fac="", street="", city="", zip_=""):
    return {"site": site, "facility_name": fac, "street_address": street, "city": city, "zip": zip_,
            "readiness": 0.5, "aliases": []}


def test_same_store_three_keys_one_row():
    rows = [_row("site:106", "SMF_Auburn_1435", "2745 Bell Road"),
            _row("site:bevmosto_106", "SMF_Auburn_1435", "2745 Bell Road", "Auburn", "95603"),
            _row("site:1435_0", "Auburn", "2745 Bell Rd")]
    out, folded = merge_roster(rows)
    assert folded == 2 and len(out) == 1
    assert out[0]["zip"] == "95603"  # the most complete row is kept
    assert {m["site"] for m in out[0]["merged_from"]} == {"site:106", "site:1435_0"}


def test_neighbors_and_other_cities_stay_apart():
    rows = [_row("site:a", "facility 9999957", "37 Cherry Hill Drive", "Danvers", "01923"),
            _row("site:b", "facility 1000519", "35 Cherry Hill Drive", "Danvers", "01923"),
            _row("site:c", "", "100 Main St", "Springfield", "62701"),
            _row("site:d", "", "100 Main St", "Portland", "97201")]
    out, folded = merge_roster(rows)
    assert folded == 0 and len(out) == 4
