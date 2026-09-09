"""Two rows with the same street and postcode are the same building.

Deal 02557291 published 2205 Gregg St THREE times: once from the note's roster
(``site:2205_gregg_st``), once from the entity backfill
(``site:loc_2205_gregg_st``, whose city read "Gregg St" -- a fragment of its own
street), and once as ``site:site_1``. Three extractors, three keys, one place.

The learned fusion pass asked about all six pairs and returned
``fallback:None`` for every one -- nothing taught, LLM tier budgeted off -- so
the duplicate reached the published roster. It should never have been a
question: that is what an address IS.
"""

from __future__ import annotations

from app.core.entity_resolution import _address_identity, address_identity_groups


def _row(key, street, city, state, zipc):
    return {
        "site": key, "street_address": street,
        "city": city, "state": state, "zip": zipc,
    }


ROWS = {
    "site:2205_gregg_st": _row("site:2205_gregg_st", "2205 Gregg St", "Columbia", "SC", "29201"),
    "site:loc_2205_gregg_st": _row("site:loc_2205_gregg_st", "2205 Gregg St", "Gregg St", "SC", "29201"),
    "site:site_1": _row("site:site_1", "2205 Gregg St", "Columbia", "SC", "29201"),
    "site:3501_merrill_pl": _row("site:3501_merrill_pl", "3501 Merrill Pl", "Mt Pleasant", "SC", "29466"),
}


def test_one_building_is_one_group_however_many_keys_name_it():
    groups = address_identity_groups(set(ROWS), ROWS)
    assert [sorted(g) for g in groups] == [
        ["site:2205_gregg_st", "site:loc_2205_gregg_st", "site:site_1"]
    ]


def test_a_wrong_city_does_not_stop_the_merge():
    """The city is the field most often wrong, and carries nothing the postcode
    does not. One of these rows has its own street name as its city."""
    groups = address_identity_groups(set(ROWS), ROWS)
    merged = groups[0]
    assert "site:loc_2205_gregg_st" in merged


def test_a_different_address_is_a_different_site():
    groups = address_identity_groups(set(ROWS), ROWS)
    assert not any("site:3501_merrill_pl" in g for g in groups)


def test_the_same_street_in_a_different_town_is_not_the_same_place():
    """Street alone is not identity -- Main St exists everywhere."""
    rows = {
        "site:a": _row("site:a", "100 Main St", "Austin", "TX", "78701"),
        "site:b": _row("site:b", "100 Main St", "Dallas", "TX", "75201"),
    }
    assert address_identity_groups(set(rows), rows) == []


def test_no_postcode_means_no_judgment():
    """A site known only by name abstains completely rather than guessing."""
    rows = {
        "site:a": _row("site:a", "100 Main St", "Austin", "TX", ""),
        "site:b": _row("site:b", "100 Main St", "Austin", "TX", ""),
    }
    assert address_identity_groups(set(rows), rows) == []


def test_no_street_means_no_judgment():
    rows = {
        "site:a": _row("site:a", "", "Austin", "TX", "78701"),
        "site:b": _row("site:b", "", "Austin", "TX", "78701"),
    }
    assert address_identity_groups(set(rows), rows) == []


def test_canadian_postcodes_normalise_by_spacing():
    """L4T 3L8 and L4T3L8 are one postcode."""
    rows = {
        "site:a": _row("site:a", "7675 Torbram Rd", "Mississauga", "ON", "L4T 3L8"),
        "site:b": _row("site:b", "7675 Torbram Rd", "Mississauga", "ON", "l4t3l8"),
    }
    groups = address_identity_groups(set(rows), rows)
    assert [sorted(g) for g in groups] == [["site:a", "site:b"]]


def test_identity_is_street_and_postcode_only():
    a = _address_identity(_row("site:a", "2205 Gregg St", "Columbia", "SC", "29201"))
    b = _address_identity(_row("site:b", "2205 Gregg St", "Gregg St", "SC", "29201"))
    assert a == b != ""


def test_a_single_key_is_never_a_group():
    rows = {"site:a": _row("site:a", "100 Main St", "Austin", "TX", "78701")}
    assert address_identity_groups(set(rows), rows) == []


def test_rows_it_knows_nothing_about_are_skipped():
    assert address_identity_groups({"site:a", "site:b"}, None) == []
