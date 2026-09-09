"""Two rows at one address are the plainest duplicate, and were never asked about.

`site_duplicate_candidates` looked for an UNLOCATED row -- a name that matched
nothing -- and paired it with a located one on a shared token. That is the right
shape for "Hillview Office" vs "3300 Hillview Ave", and it is blind to the
simplest case of all: two rows that both know where they are and say the same
place.

Deal 02557291 published 2205 Gregg St twice, as `site:2205_gregg_st` and
`site:site_1`. Both anchored, both addressed, so neither was ever "unlocated" --
the deal asked nothing while showing one building under two names.
"""

from __future__ import annotations

from app.core.site_duplicate_candidates import (
    _address_identity,
    site_duplicate_candidates,
)

GREGG = {
    "site": "site:2205_gregg_st", "display_name": "2205 Gregg St, Columbia, SC 29201",
    "street_address": "2205 Gregg St", "city": "Columbia", "state": "SC",
    "zip": "29201", "anchored": True,
}
SITE_1 = {
    "site": "site:site_1", "facility_name": "Site 1",
    "street_address": "2205 Gregg St", "city": "Columbia", "state": "SC",
    "zip": "29201", "anchored": True,
}
MERRILL = {
    "site": "site:3501_merrill_pl", "display_name": "3501 Merrill Pl, Mt Pleasant, SC 29466",
    "street_address": "3501 Merrill Pl", "city": "Mt Pleasant", "state": "SC",
    "zip": "29466", "anchored": True,
}


def test_the_deal_now_asks_about_its_duplicate():
    out = site_duplicate_candidates([GREGG, MERRILL, SITE_1])
    assert len(out) == 1
    pair = {out[0]["unlocated"], out[0]["located"]}
    assert pair == {"site:site_1", "site:2205_gregg_st"}


def test_it_says_why_in_terms_a_person_can_check():
    out = site_duplicate_candidates([GREGG, SITE_1])
    assert "same street and postcode" in out[0]["why"]


def test_a_different_address_is_never_asked_about():
    """Two anchored sites with different addresses are two sites. Asking about
    them trains people to click through."""
    assert site_duplicate_candidates([GREGG, MERRILL]) == []


def test_the_same_street_in_another_town_is_not_a_pair():
    other_town = dict(GREGG, site="site:elsewhere", city="Dallas", zip="75201")
    assert site_duplicate_candidates([GREGG, other_town]) == []


def test_a_row_with_no_postcode_is_not_paired_on_street_alone():
    no_zip = dict(SITE_1, zip="")
    assert site_duplicate_candidates([GREGG, no_zip]) == []


def test_the_pair_is_stable_however_the_rows_are_ordered():
    a = site_duplicate_candidates([GREGG, SITE_1])[0]
    b = site_duplicate_candidates([SITE_1, GREGG])[0]
    assert (a["unlocated"], a["located"]) == (b["unlocated"], b["located"])
    assert a["exemplar"] == b["exemplar"], "one pair, one exemplar, or the answer is unfindable"


def test_it_decides_nothing():
    """A candidate is a question. Nothing here merges or drops a site."""
    rows = [dict(GREGG), dict(SITE_1)]
    before = [dict(r) for r in rows]
    site_duplicate_candidates(rows)
    assert rows == before


def test_address_identity_matches_the_merge_rule():
    """This module decides what a PERSON is asked; entity_resolution decides
    what is merged WITHOUT asking. A pair the merge would collapse but this
    never surfaces is a duplicate nobody can act on, so the two rules must
    agree."""
    from app.core.entity_resolution import _address_identity as merge_identity

    for row in (GREGG, SITE_1, MERRILL, dict(SITE_1, zip=""), dict(SITE_1, street_address="")):
        assert _address_identity(row) == merge_identity(row)


def test_canadian_postcodes_pair_on_spacing():
    a = {"site": "site:a", "street_address": "7675 Torbram Rd", "zip": "L4T 3L8", "anchored": True}
    b = {"site": "site:b", "street_address": "7675 Torbram Rd", "zip": "l4t3l8", "anchored": True}
    assert len(site_duplicate_candidates([a, b])) == 1
