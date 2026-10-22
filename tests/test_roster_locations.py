"""Roster sites take their city/state from the sibling sheet keyed by the same id (010447)."""
from types import SimpleNamespace

from app.core.roster_locations import _clean_city, join_sheet_locations, share_roster_locations


def _atom(sid, sheet="2027_145_Sites", **value):
    ref = SimpleNamespace(extraction_method="xlsx_site_roster_v1", locator={"sheet": sheet})
    v = {"site_id": sid, "cells": {"Site No.": sid.lower().replace("dc", "dc "), "Visits": "1"}, **value}
    return SimpleNamespace(value=v, source_refs=[ref])


SOURCE = [
    ["Site No.", "Site Location", "State", "Location", "Type", "Site Type"],
    ["dc 5841", "Hialeah - Miami DFC", "FL", "USA", "2027", "DFC"],
    ["dc 5123", "Carlisle Crown Bolt", "PA", "USA", "2027", "CB"],
    ["dc 7019", "Mississauga BDC", None, "Canada", "2027", "BDC"],
]


def test_joins_city_state_country_by_the_same_id_column():
    atoms = [_atom("DC5841"), _atom("DC5123"), _atom("DC7019"), _atom("DC9999")]
    n = join_sheet_locations(atoms, [{"name": "2027_145_Sites", "rows": []}, {"name": "annette source", "rows": SOURCE}])
    assert n == 3
    a, b, c, d = (x.value for x in atoms)
    assert (a["city"], a["state"], a["country"], a["site_location"]) == ("Hialeah", "FL", "US", "Hialeah - Miami DFC")
    assert (b["city"], b["state"]) == ("Carlisle", "PA")
    assert (c["city"], c["country"]) == ("Mississauga", "CA") and "state" not in c
    assert "city" not in d
    assert a["location_from"] == {"sheet": "annette source", "row": "2", "by": "Site No."}


def test_never_overwrites_a_site_that_already_says_where_it_is():
    atoms = [_atom("DC5841", city="Miami", state="FL")]
    assert join_sheet_locations(atoms, [{"name": "annette source", "rows": SOURCE}]) == 0
    assert atoms[0].value["city"] == "Miami"


def test_a_sheet_without_the_id_column_or_a_city_column_is_not_used():
    atoms = [_atom("DC5841")]
    no_id = [["Store", "City", "State"], ["dc 5841", "Miami", "FL"]]
    no_city = [["Site No.", "Director"], ["dc 5841", "Someone"]]
    assert join_sheet_locations(atoms, [{"name": "a", "rows": no_id}, {"name": "b", "rows": no_city}]) == 0


def test_the_same_roster_in_two_files_shares_the_city():
    found, bare = _atom("dc5841", city="Hialeah", state="FL", country="US"), _atom("DC5841")
    assert share_roster_locations([bare, found]) == 1
    assert (bare.value["city"], bare.value["state"]) == ("Hialeah", "FL")


def test_city_from_a_location_name():
    assert _clean_city("Keasbey BDC", "BDC") == "Keasbey"
    assert _clean_city("Dallas FDC BDC", "FDC/BDC") == "Dallas"
    assert _clean_city("Luckey, OH", "", "OH") == "Luckey"
    assert _clean_city("Bolton", "") == "Bolton"
