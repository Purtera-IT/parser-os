"""A named place and a street address, judged from the deal and not from each other.

Deal 010003 compared "checkout llc new york office" with "40 10th ave fl 4" and
judged them distinct_site. They are one place: the customer's New York office is
40 10th Ave, Fl 4. The two strings share no word, which says nothing -- a name
and an address are not two descriptions of the same kind.
"""

from __future__ import annotations

import os

from app.core.site_duplicate_candidates import (
    SAME_SITE,
    UNCERTAIN,
    named_place_address_verdict,
    site_duplicate_candidates,
)

NAMED = {"site": "site:checkout_llc_new_york_office", "facility_name": "Checkout LLC New York office"}
ADDRESS = {"site": "site:40_10th_ave_fl_4", "street_address": "40 10th Ave, Fl 4", "city": "New York", "state": "NY"}
# The same address known only by its slug, as the fusion pass describes it.
ADDRESS_SLUG = {"site": "site:40_10th_ave_fl_4", "facility_name": "40 10th ave fl 4"}
EVIDENCE = "Install at Checkout LLC, 40 10th Avenue, Floor 4, New York, NY 10014. Badge in at the lobby."


def test_the_name_printed_beside_the_street_is_one_site() -> None:
    verdict, why = named_place_address_verdict(NAMED, ADDRESS_SLUG, [NAMED, ADDRESS_SLUG], EVIDENCE)
    assert verdict == SAME_SITE
    assert "beside" in why


def test_the_only_address_in_the_named_city_is_one_site() -> None:
    verdict, _ = named_place_address_verdict(NAMED, ADDRESS, [NAMED, ADDRESS])
    assert verdict == SAME_SITE


def test_two_addresses_in_that_city_need_a_person() -> None:
    other = {"site": "site:1_main_st", "street_address": "1 Main St", "city": "New York", "state": "NY"}
    verdict, _ = named_place_address_verdict(NAMED, ADDRESS, [NAMED, ADDRESS, other])
    assert verdict == UNCERTAIN


def test_nothing_tying_them_is_uncertain_not_distinct() -> None:
    verdict, _ = named_place_address_verdict(NAMED, ADDRESS_SLUG, [NAMED, ADDRESS_SLUG])
    assert verdict == UNCERTAIN


def test_a_different_city_is_left_to_the_other_rules() -> None:
    dallas = dict(NAMED, city="Dallas")
    assert named_place_address_verdict(dallas, ADDRESS, [dallas, ADDRESS])[0] is None


def test_two_addresses_or_two_names_are_not_this_shape() -> None:
    other = {"site": "site:1_main_st", "street_address": "1 Main St", "city": "New York"}
    assert named_place_address_verdict(ADDRESS, other)[0] is None
    assert named_place_address_verdict(NAMED, {"site": "site:x", "facility_name": "Boston Office"})[0] is None


def test_the_shortlist_proposes_the_tied_pair_with_its_verdict() -> None:
    out = site_duplicate_candidates([NAMED, ADDRESS])
    assert len(out) == 1
    assert {out[0]["unlocated"], out[0]["located"]} == {NAMED["site"], ADDRESS["site"]}
    assert out[0]["verdict"] == SAME_SITE


def test_the_shortlist_does_not_ask_about_untied_pairs() -> None:
    assert site_duplicate_candidates([NAMED, ADDRESS_SLUG]) == []
    assert len(site_duplicate_candidates([NAMED, ADDRESS_SLUG], EVIDENCE)) == 1


def _fusion(rows: dict, evidence: str, answer: str | None, source: str = "llm"):
    from app.core.entity_resolution import semantic_site_fusion_groups
    import app.core.decide as decide_mod

    class _D:
        verdict = answer

    _D.source = source
    real = decide_mod.decide
    decide_mod.decide = lambda **kw: _D()
    os.environ["SOWSMITH_NEURAL_SITE_FUSION"] = "1"
    try:
        return semantic_site_fusion_groups(set(rows), rows, evidence=evidence)
    finally:
        decide_mod.decide = real
        os.environ.pop("SOWSMITH_NEURAL_SITE_FUSION", None)


def test_the_fusion_pass_merges_what_the_deal_ties_whatever_a_model_says() -> None:
    rows = {r["site"]: r for r in (NAMED, ADDRESS_SLUG)}
    groups = _fusion(rows, EVIDENCE, "distinct_site")
    assert groups == [{NAMED["site"], ADDRESS_SLUG["site"]}]


def test_the_fusion_pass_keeps_untied_pairs_apart() -> None:
    rows = {r["site"]: r for r in (NAMED, ADDRESS_SLUG)}
    assert _fusion(rows, "", "distinct_site") == []
