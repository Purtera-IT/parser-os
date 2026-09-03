"""A town is not a site.

``city|STATE|zip`` buckets let a city-only atom attach to the full-address atom
for the same place. But every facility in a town shares that bucket, so on its
own it unioned ten Marion County schools into two "sites" -- and the eight lost
schools were reported as nothing at all.

These guard the two vetoes, and the case the buckets exist for.
"""

from __future__ import annotations

from app.core.semantic_dedup import _merge_grouped_by_location_buckets


class _Atom:
    def __init__(self, **value):
        self.value = value


def _site(name, address, city="Marion", state="SC", zipc="29571"):
    return _Atom(name=name, address=address, city=city, state=state, zip=zipc)


def test_one_town_is_not_one_site():
    grouped = {
        "loc_601_gurley_st_johnakin": [_site("Johnakin Middle School", "601 Gurley St")],
        "loc_1205_s_main_st_marion_high": [_site("Marion High School", "1205 S Main St")],
        "loc_6641_south_hwy_41_creek_bridge": [_site("Creek Bridge STEM Academy", "6641 South Hwy 41")],
    }
    assert len(_merge_grouped_by_location_buckets(grouped)) == 3


def test_a_city_only_variant_still_attaches_to_its_full_address_site():
    """The case the buckets exist for -- it must keep working."""
    grouped = {
        "loc_601_gurley_st_johnakin": [_site("Johnakin Middle School", "601 Gurley St")],
        "marion_sc_29571": [_Atom(city="Marion", state="SC", zip="29571")],
    }
    assert len(_merge_grouped_by_location_buckets(grouped)) == 1


def test_a_shared_street_does_not_delete_a_differently_named_school():
    """Two SOWs copy-pasted the preceding school's address. Both schools are real."""
    grouped = {
        "loc_600_e_northside_ave_academy": [_site("Academy of Early Learning", "600 E Northside Ave")],
        "loc_600_e_northside_ave_easterling": [_site("Easterling Primary School", "600 E Northside Ave")],
    }
    assert len(_merge_grouped_by_location_buckets(grouped)) == 2, (
        "a customer copy-paste must not silently erase a school"
    )


def test_a_longer_spelling_of_the_same_name_still_merges():
    """The veto compares token sets with subset as agreement, not a vocabulary."""
    grouped = {
        "loc_601_gurley_st_johnakin_a": [_site("Johnakin Middle", "601 Gurley St")],
        "loc_601_gurley_st_johnakin_b": [_site("Johnakin Middle School", "601 Gurley St")],
    }
    assert len(_merge_grouped_by_location_buckets(grouped)) == 1
