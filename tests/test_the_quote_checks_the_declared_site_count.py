"""A quote that prices more places than the documents declare says so.

010447's documents say "145 locations nationwide", "145 sites in scope" and "145 Location Palo Alto
deployment", and its roster carries 243 keys. `declared_scope._declared_site_count` already reads
that number -- it returns 145 on this deal and nothing on 010171, which declares no count -- and
nothing downstream of the roster ever asked it. The roster is not overruled: which number is right
is a judgment. It is reported, with the line that says it.
"""
from app.core.c3_type_base import _with_declared_sites


def test_a_disagreement_is_reported_with_the_line_that_says_it():
    q = _with_declared_sites({"sites": {"count": 243}, "flags": []},
                             (145, "The Home Depot has 145 locations nationwide"), 243)
    rec = q["sites"]["declared"]
    assert rec["declared"] == 145 and rec["roster"] == 243 and rec["disagrees"] is True
    assert "145 locations nationwide" in rec["said_on"]
    assert q["flags"] == ["the documents declare 145 places and the roster has 243: "
                          "check the site list before pricing per site"]


def test_agreement_is_recorded_without_a_flag():
    q = _with_declared_sites({"sites": {"count": 145}, "flags": []}, (145, "145 sites in scope"), 145)
    assert q["sites"]["declared"]["declared"] == 145
    assert "disagrees" not in q["sites"]["declared"]
    assert q["flags"] == []


def test_a_deal_that_declares_nothing_is_left_alone():
    q = _with_declared_sites({"sites": {"count": 1}, "flags": []}, None, 1)
    assert q == {"sites": {"count": 1}, "flags": []}


def test_an_empty_roster_is_not_flagged_here():
    """site_roster_empty_why already answers an empty roster; this is about the count."""
    q = _with_declared_sites({"sites": {"count": 0}, "flags": []}, (145, "145 sites"), 0)
    assert "disagrees" not in q["sites"]["declared"]
    assert q["flags"] == []


def test_it_rides_its_own_key_when_sites_is_not_a_dict():
    q = _with_declared_sites({"sites": None, "flags": []}, (145, "145 sites"), 243)
    assert q["sites_declared"]["declared"] == 145


def test_a_degraded_quote_is_returned_as_is():
    assert _with_declared_sites(None, (145, "x"), 243) is None
