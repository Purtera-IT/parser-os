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


def test_mention_only_sites_are_set_aside_not_lost():
    from app.core.site_identity import filter_roster

    rows = [_row("site:1401", "Southern Farm Bureau", "1401 Livingston Ln", "Jackson", "39213"),
            _row("site:mariners", "Seattle Mariners Stadium"),
            _row("site:easterling", "Easterling, 600")]
    atoms = {"site:1401": ["a1"], "site:mariners": ["a2"], "site:easterling": ["a3", "a4"]}
    types = {"a1": ["physical_site", 0.86], "a2": ["small_talk", 0.44],
             "a3": ["small_talk", 0.71], "a4": ["physical_site", 0.65]}
    kept, aside = filter_roster(rows, atoms, types)
    assert [r["site"] for r in kept] == ["site:1401"]
    assert {r["site"] for r in aside} == {"site:mariners", "site:easterling"}
    assert all(r["set_aside"] == "mention_only" for r in aside)


def test_street_in_display_name_and_town_rows_fold_into_the_one_site():
    rows = [{"site": "site:6125_tyvola_centre_dr", "display_name": "6125 Tyvola Centre Drive, Charlotte, NC 28217",
             "street_address": "6125 Tyvola Centre Drive", "city": "Charlotte", "zip": "28217"},
            {"site": "site:6125_tyvola_centre_drive", "display_name": "6125 Tyvola Centre Drive"},
            {"site": "site:charlotte_nc_28217", "display_name": "Charlotte NC 28217"}]
    out, folded = merge_roster(rows)
    assert len(out) == 1 and folded == 2


def test_town_row_stays_apart_when_two_sites_share_the_town():
    rows = [_row("site:a", "", "37 Cherry Hill Drive", "Danvers", "01923"),
            _row("site:b", "", "35 Cherry Hill Drive", "Danvers", "01923"),
            {"site": "site:danvers_ma_01923", "display_name": "Danvers MA 01923"}]
    out, folded = merge_roster(rows)
    assert folded == 0


def test_a_signature_address_with_a_street_is_set_aside():
    from app.core.site_identity import filter_roster

    rows = [_row("site:323_lakeside", "Brooksource", "323 Lakeside Ave", "Cleveland", "44113"),
            _row("site:625_w_adams", "CDW", "625 W. Adams", "Chicago", "60661"),
            _row("site:2390_satellite", "SG", "2390 Satellite Blvd", "Buford", "30518")]
    atoms = {"site:323_lakeside": ["a1"], "site:625_w_adams": ["a2"], "site:2390_satellite": ["a3"]}
    types = {"a1": ["physical_site", 0.77], "a2": ["stakeholder", 0.97], "a3": ["physical_site", 0.8]}
    flags = {"a3": ["vendor_address_suspected"]}
    kept, aside = filter_roster(rows, atoms, types, flags)
    assert [r["site"] for r in kept] == ["site:323_lakeside"]
    assert {r["site"] for r in aside} == {"site:625_w_adams", "site:2390_satellite"}
