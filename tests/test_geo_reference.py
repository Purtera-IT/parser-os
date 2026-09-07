"""The gazetteer that replaced the city-name word lists."""

from __future__ import annotations

import pytest

from app.core.address_parse import _city_looks_valid, city_only_cell
from app.core.geo_reference import (
    available,
    city_state_for_zip,
    is_known_place,
    resolve,
    states_for_city,
)

pytestmark = pytest.mark.skipif(not available(), reason="reference data not present")


def test_zip_determines_city_and_state() -> None:
    assert city_state_for_zip("97132") == ("Newberg", "OR")
    assert city_state_for_zip("99362-1234") == ("Walla Walla", "WA")
    assert city_state_for_zip("00000") is None
    assert city_state_for_zip(None) is None


def test_abbreviated_place_names_resolve_to_their_full_form() -> None:
    """The case the old word list existed to paper over."""
    assert is_known_place("St. Louis", "MO") is True
    assert is_known_place("Saint Louis", "MO") is True
    assert is_known_place("Mt. Pleasant", "SC") is True
    assert is_known_place("Ft. Worth", "TX") is True


def test_street_fragments_are_not_places() -> None:
    assert is_known_place("Broad St") is False
    assert is_known_place("Suite 400") is False


def test_wrong_state_is_rejected() -> None:
    assert is_known_place("Newberg", "OR") is True
    assert is_known_place("Newberg", "HI") is False


def test_resolve_fills_only_blanks() -> None:
    # A ZIP names both halves.
    assert resolve(postal_code="97132") == ("Newberg", "OR")
    # The document always wins over the reference data.
    assert resolve(city="Dundee", postal_code="97132") == ("Dundee", "OR")
    # A city in exactly one state names that state.
    assert resolve(city="Newberg") == ("Newberg", "OR")
    # A city in many states does not.
    assert len(states_for_city("Springfield")) > 1
    assert resolve(city="Springfield") == ("Springfield", None)


def test_city_validity_no_longer_depends_on_a_token_list() -> None:
    """"St." leads a city name here and a street suffix in the next line."""
    assert _city_looks_valid("St. Louis") is True
    assert _city_looks_valid("Broad St") is False
    # The trim that repairs "Park BLvd. Highland Park" still reaches the city.
    assert _city_looks_valid("BLvd. Highland Park") is False
    assert _city_looks_valid("Highland Park") is True


def test_city_only_cell_needs_a_real_place() -> None:
    assert city_only_cell("Newberg") == "Newberg"
    assert city_only_cell("Walla Walla") == "Walla Walla"
    assert city_only_cell("Springfield, Springfield") is None
    assert city_only_cell("1001 Providence Dr") is None
    assert city_only_cell("TEN") is None
    assert city_only_cell("OR") is None
