"""tools/geo_tier.py: OMB class -> location_tier, city/ZIP resolution, splits
and remote_miles, on a tiny made-up table (no package data needed)."""
from __future__ import annotations

import pytest

from tools.geo_tier import GeoTable, Zip, miles, principal_city, tier_for


def z(zip_, city, st, cbsa=None, name=None, kind=None, pop=1000, lat=0.0, lon=0.0,
      county="C", fips="00001", standard=True):
    return Zip(zip=zip_, city=city, state=st, county=county, county_fips=fips, lat=lat,
               lon=lon, cbsa_code=cbsa, cbsa_name=name, cbsa_type=kind, population=pop,
               standard=standard)


@pytest.fixture
def table():
    return GeoTable([
        # a 1.2M metro whose principal city sits at (35, -90)
        z("10001", "Bigtown", "AA", "100", "Bigtown-Other, AA-BB", "Metro", 1_000_000, 35.0, -90.0),
        z("10002", "Other", "BB", "100", "Bigtown-Other, AA-BB", "Metro", 200_000, 35.5, -90.5),
        # a 300k metro, a micro area and a town outside every CBSA
        z("20001", "Midcity", "AA", "200", "Midcity, AA", "Metro", 300_000, 36.0, -89.0),
        z("30001", "Smallton", "AA", "300", "Smallton, AA", "Micro", 30_000, 34.0, -88.7),
        z("30002", "Smallton", "AA", "300", "Smallton, AA", "Micro", 10_000, 34.1, -88.7),
        z("30003", "Smallton", "AA", pop=0, standard=False),  # PO box: ignored by city
        z("40001", "Farmville", "AA", None, None, None, 2_000, 33.0, -88.0),
        # a town split between a metro and a micro area
        z("50001", "Splitburg", "AA", "200", "Midcity, AA", "Metro", 6_000, 36.1, -89.1),
        z("50002", "Splitburg", "AA", "300", "Smallton, AA", "Micro", 4_000, 36.2, -89.2),
    ])


def test_tier_table():
    assert tier_for("Metro", 1_000_000) == "major_metro"
    assert tier_for("Metro", 999_999) == "mid_metro"
    assert tier_for("Micro", 5_000_000) == "small_town"
    assert tier_for(None, None) == "rural"


def test_principal_city():
    assert principal_city("Memphis, TN-MS-AR") == ("Memphis", "TN")
    assert principal_city("Minneapolis-St. Paul-Bloomington, MN-WI") == ("Minneapolis", "MN")


def test_city_and_zip_resolve_to_tiers(table):
    assert table.lookup("Bigtown, AA").location_tier == "major_metro"
    assert table.lookup("Midcity, aa").location_tier == "mid_metro"
    r = table.lookup("Smallton, AA")
    assert (r.location_tier, r.zips, r.resolved_by) == ("small_town", ["30001", "30002"], "city")
    assert table.lookup("1 Main St, Farmville, AA 40001").location_tier == "rural"
    assert table.lookup("Nowhere, AA").error


def test_remote_miles(table):
    inside = table.lookup("Other, BB")
    assert inside.remote_miles == 0 and "Bigtown, AA metro" in inside.remote_miles_text
    r = table.lookup("40001")
    assert r.remote_miles == round(miles(33.0, -88.0, 35.0, -90.0))
    assert r.remote_miles_text.endswith("miles from Bigtown, AA")


def test_split_town_reports_both(table):
    r = table.lookup("Splitburg, AA")
    assert r.location_tier == "mid_metro"
    assert [(s["location_tier"], s["share"]) for s in r.split] == [("mid_metro", 0.6), ("small_town", 0.4)]


def test_near_threshold_flag():
    t = GeoTable([z("1", "Edge", "AA", "9", "Edge, AA", "Metro", 1_030_000)])
    assert t.lookup("Edge, AA").near_threshold
