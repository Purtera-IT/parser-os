"""The roster's receipts ride on the quote the envelope keeps, not on the request.

They were first written onto `body["deal"]`, which the C3 service reads and never echoes. The
envelope stores the service's REPLY, so the receipt was invisible by construction: a compile of
010171 ran with the diagnostic in place and the envelope came back with no receipt at all.
"""
from app.core.c3_type_base import _with_roster_why

HELD = [{"site": "site:x", "atom_id": "a1", "head": "jdg:site_role", "answer": "vendor_or_billing_address",
         "p": 0.95, "why": "kept: vetoing it would leave the deal with no site"}]
EMPTY = [{"atom_id": "a2", "why": "type 'scope_item' and the role head did not name it a job site",
          "head": "jdg:site_role", "answer": None, "p": None, "line_was_read": True,
          "heads_on_line": ["col:admission"], "roster": "empty"}]


def test_the_held_back_record_reaches_the_reply():
    q = _with_roster_why({"site_count": 1}, HELD, [])
    assert q["site_role_held_back"] == HELD
    assert "site_roster_empty_why" not in q


def test_the_empty_roster_receipt_reaches_the_reply():
    q = _with_roster_why({"site_count": 0}, [], EMPTY)
    assert q["site_roster_empty_why"] == EMPTY
    assert "site_role_held_back" not in q


def test_both_can_ride_together():
    q = _with_roster_why({"site_count": 0}, HELD, EMPTY)
    assert q["site_role_held_back"] == HELD and q["site_roster_empty_why"] == EMPTY


def test_a_quote_with_a_roster_is_untouched():
    q = _with_roster_why({"site_count": 3, "sites": {"count": 3}}, [], [])
    assert q == {"site_count": 3, "sites": {"count": 3}}


def test_a_degraded_quote_that_is_not_a_dict_is_returned_as_is():
    assert _with_roster_why(None, HELD, EMPTY) is None
