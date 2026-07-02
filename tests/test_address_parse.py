"""Universal address parsing — MBrany-class fixes."""

from __future__ import annotations

from app.core.address_parse import (
    enrich_location_fields,
    parse_city_state_field,
    parse_us_address_line,
)


def test_mbrany_park_blvd_not_city() -> None:
    parsed = parse_us_address_line("Park BLvd. Highland Park, MI 48203")
    assert parsed.city == "Highland Park"
    assert parsed.state == "MI"
    assert parsed.zip == "48203"
    assert parsed.street_address is not None
    assert "Park" in parsed.street_address


def test_full_street_with_city_state_zip() -> None:
    parsed = parse_us_address_line("12575 Oakland Park Blvd, Highland Park, MI 48203")
    assert parsed.city == "Highland Park"
    assert parsed.state == "MI"
    assert parsed.zip == "48203"
    assert "12575" in (parsed.street_address or "")


def test_santa_fe_bare_city_line() -> None:
    parsed = parse_us_address_line("Santa Fe, NM 87506")
    assert parsed.city == "Santa Fe"
    assert parsed.state == "NM"
    assert parsed.zip == "87506"


def test_hubspot_note_address_without_state_zip_space() -> None:
    parsed = parse_us_address_line(
        "GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359"
    )
    assert parsed.street_address == "100 S COMMONS STE 145"
    assert parsed.city == "PITTSBURGH"
    assert parsed.state == "PA"
    assert parsed.zip == "15212"
    assert parsed.aliases == ("GECKO ROBOTICS",)


def test_city_state_field_split() -> None:
    city, state = parse_city_state_field("Highland Park, MI")
    assert city == "Highland Park"
    assert state == "MI"


def test_enrich_combined_address_column() -> None:
    loc = enrich_location_fields(
        street_address="12575 Oakland Park Blvd, Highland Park, MI 48203",
        facility_name="Mbrany Highland Park Office",
    )
    assert loc["city"] == "Highland Park"
    assert loc["state"] == "MI"
    assert loc["zip"] == "48203"


def test_enrich_separate_city_state_columns() -> None:
    loc = enrich_location_fields(
        street_address="12575 Oakland Park Blvd",
        city="Highland Park",
        state="MI",
        zip_code="48203",
    )
    assert loc["city"] == "Highland Park"
    assert loc["state"] == "MI"


def test_enrich_repairs_misparsed_city_from_facility_name() -> None:
    loc = enrich_location_fields(
        facility_name="Park BLvd. Highland Park, MI 48203",
        city="Park BLvd. Highland Park",
        state="MI",
        zip_code="48203",
    )
    assert loc["city"] == "Highland Park"
    assert loc["state"] == "MI"
    assert loc["zip"] == "48203"
