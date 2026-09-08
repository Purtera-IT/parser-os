"""Pairs that look like one place — proposed for a person, decided by nobody.

Merging sites is destructive: it deletes a location and moves the project tier,
which is scored on site count. A distinctive-token rule agreed with a human
about half the time on a 140-envelope sample, which is useless for an automatic
merge and perfectly good for a shortlist.
"""

from __future__ import annotations

from app.core.site_duplicate_candidates import (
    pair_exemplar,
    site_duplicate_candidates,
)

PALO_ALTO = {
    "site": "site:palo_alto_ca_94304",
    "facility_name": "Palo Alto Office",
    "address": "3300 Hillview Ave",
    "city": "Palo Alto",
    "state": "CA",
    "anchored": True,
}
HILLVIEW = {
    "site": "site:symphonyai_hillview_office",
    "facility_name": "Symphony Ai Hillview Office",
    "anchored": False,
}


def test_the_symphony_pair_is_proposed() -> None:
    out = site_duplicate_candidates([PALO_ALTO, HILLVIEW])
    assert len(out) == 1
    assert out[0]["unlocated"] == "site:symphonyai_hillview_office"
    assert out[0]["located"] == "site:palo_alto_ca_94304"
    assert out[0]["shared_token"] == "hillview"


def test_the_exemplar_carries_the_evidence_the_judgement_turns_on() -> None:
    """The slug alone ("palo alto ca 94304") hides the address, which is the
    only reason anyone would think these are the same place."""
    ex = pair_exemplar(PALO_ALTO, HILLVIEW)
    assert "3300 Hillview Ave" in ex
    assert "Palo Alto, CA" in ex
    assert "Symphony Ai Hillview Office" in ex


def test_the_exemplar_does_not_depend_on_argument_order() -> None:
    """An unordered pair that embeds two ways is two lessons about one
    judgement, and neither is retrievable from the other."""
    assert pair_exemplar(PALO_ALTO, HILLVIEW) == pair_exemplar(HILLVIEW, PALO_ALTO)


def test_two_located_sites_are_never_proposed() -> None:
    """Two anchored sites with different addresses are two sites. Asking about
    them trains people to click through."""
    other = {
        "site": "site:spokane_wa", "facility_name": "Spokane Office",
        "address": "101 W 8th Ave", "city": "Spokane", "state": "WA", "anchored": True,
    }
    assert site_duplicate_candidates([PALO_ALTO, other]) == []


def test_an_ambiguous_name_is_not_evidence() -> None:
    """A token owned by two sites could belong to either, so it is dropped
    rather than guessed at."""
    a = dict(PALO_ALTO, site="site:a", address="1 Hillview Ave")
    b = dict(PALO_ALTO, site="site:b", address="2 Hillview Ave")
    assert site_duplicate_candidates([a, b, HILLVIEW]) == []


def test_it_proposes_and_never_decides() -> None:
    """Nothing in the output merges anything — it is a question with its
    reasoning attached."""
    out = site_duplicate_candidates([PALO_ALTO, HILLVIEW])[0]
    assert set(out) == {"unlocated", "located", "exemplar", "shared_token", "why"}
    assert "no address of its own" in out["why"]


def test_a_deal_with_nothing_to_ask_asks_nothing() -> None:
    assert site_duplicate_candidates([]) == []
    assert site_duplicate_candidates([PALO_ALTO]) == []
    assert site_duplicate_candidates([HILLVIEW]) == []
