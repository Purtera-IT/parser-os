"""Type a cell by its value, not by its label.

A label whitelist is one vendor wide: the next customer writes "Site Address" or
"Location" or leaves the cell unlabelled, and the reader finds nothing — which
looks exactly like a document that contains nothing.

Measured on the ten 010215 SOWs: shape alone finds 10/10 addresses, one
candidate each, with no label vocabulary at all.
"""

from __future__ import annotations

from app.parsers.value_shapes import (
    classify_value,
    street_addresses,
    looks_like_site_block,
)


def test_real_addresses_from_the_corpus():
    for a in ("601 Gurley St", "1123 Sandy Bluff Rd", "600 E Northside Ave",
              "747 Millers Rd", "105 Charles St", "1205 S Main St"):
        assert classify_value(a) == "address", a


def test_apostrophes_are_part_of_street_names():
    # "305 O'Neal St" was missed by a class allowing only word chars and
    # hyphens, and the row fell through to a looser pattern that matched
    # "1 UKG DX Clock" — a hardware quantity read as an address.
    assert classify_value("305 O’Neal St") == "address"
    assert classify_value("305 O'Neal St") == "address"


def test_a_trailing_route_number_is_still_one_address():
    assert classify_value("6641 South Hwy 41") == "address"


def test_a_quantity_is_not_an_address():
    # The exact false positive that shape v1 produced.
    assert classify_value("1 UKG DX Clock") is None
    assert classify_value("2 Clocks") is None
    assert classify_value("94575001") is None


def test_the_other_shapes():
    assert classify_value("29571") == "postal"
    assert classify_value("SC") == "state"
    assert classify_value("bernie.donnelly@sodexo.com") == "email"
    assert classify_value("404-918-0783") == "phone"
    assert classify_value("") is None
    assert classify_value(None) is None


def test_merged_repeats_collapse_but_order_survives():
    cells = ["Address Line 1", "601 Gurley St", "601 Gurley St", "Address Line 2"]
    assert street_addresses(cells) == ["601 Gurley St"]


def test_co_location_separates_a_place_from_a_line_item():
    # An address with a zip or a state beside it is a place. A number-led string
    # standing alone is not.
    assert looks_like_site_block(["Address Line 1", "601 Gurley St", "City", "Marion", "SC"])
    assert looks_like_site_block(["601 Gurley St", "29571"])
    assert not looks_like_site_block(["1 UKG DX Clock", "qty", "1"])
    assert not looks_like_site_block(["601 Gurley St"])


def test_non_iterable_input_is_safe():
    assert street_addresses(None) == []
    assert looks_like_site_block(None) is False
