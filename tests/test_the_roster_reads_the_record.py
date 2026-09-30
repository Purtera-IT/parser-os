# -*- coding: utf-8 -*-
"""The deal's own people belong in the deal's roster.

`build_deal_roster` got each person's address by parsing it back OUT of the
atom's words, with `read_signature(atom.raw_text)`, and skipped anyone it could
not find an email for. A stakeholder atom carries the resolved record in
`value`:

    raw_text  'Gregory Rivers | CEO'
    value     {'name': 'Gregory Rivers', 'email': 'gregory.rivers@norvetmsp.com'}

`read_signature('Gregory Rivers | CEO')` returns a name and a role and no
email, because the email is not in the text. So the `continue` skipped the
person.

Live 000113 (Columbus AFB premise wiring): three stakeholder atoms, each with a
name, a role and an address in `value`, and `deal_roster.people == []`. The
deal's own people were absent from the roster while `stakeholder_load` -- which
reads the same atoms a different way -- listed all three. Two sections
describing the same people, one blank.

This is the same shape as the rest of the audit: a consumer re-deriving from
text what the producer already resolved, and losing whatever the text does not
happen to say.
"""
from __future__ import annotations

from app.core.deal_roster import build_deal_roster, read_signature


class _Atom:
    def __init__(self, text, value=None, atom_type="stakeholder"):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.value = dict(value or {})
        self.entity_keys = []
        self.review_flags = []
        self.source_refs = []
        self.receipts = []


def _people():
    return [
        _Atom("Gregory Rivers | CEO",
              {"name": "Gregory Rivers", "email": "gregory.rivers@norvetmsp.com"}),
        _Atom("Chase Smith | Director of Operations",
              {"name": "Chase Smith", "email": "chase@purtera-it.com"}),
        _Atom("Patrick Kelly | Account Manager",
              {"name": "Patrick Kelly", "email": "patrick@purtera-it.com"}),
    ]


def test_the_email_in_the_value_is_enough() -> None:
    """No document carries a sender on this deal -- HubSpot notes and a PDF --
    so the atoms are the only source, and the text has no address in it."""
    roster = build_deal_roster(atoms=_people(), documents=[])
    emails = {p["email"] for p in roster["people"]}
    assert emails == {
        "gregory.rivers@norvetmsp.com",
        "chase@purtera-it.com",
        "patrick@purtera-it.com",
    }


def test_the_name_and_role_travel_too() -> None:
    roster = build_deal_roster(atoms=_people(), documents=[])
    greg = next(p for p in roster["people"] if p["email"].startswith("gregory"))
    assert greg["name"] == "Gregory Rivers"
    assert greg["role"] == "CEO", "the role is in the words, the address in the value"


def test_the_text_still_wins_when_it_has_more() -> None:
    """Parsing the words is not removed -- it is the fallback, and where the
    signature carries a fuller record it is still used."""
    atom = _Atom("Rhonda Sharp | Professional Services Manager | rhonda.sharp@cdw.com",
                 {"name": "Rhonda Sharp"})
    roster = build_deal_roster(atoms=[atom], documents=[])
    assert roster["people"][0]["email"] == "rhonda.sharp@cdw.com"
    assert roster["people"][0]["role"] == "Professional Services Manager"


def test_a_person_with_no_address_anywhere_is_still_skipped() -> None:
    """The roster is keyed on email; a record with none cannot be a row, and
    inventing one would be worse than omitting it."""
    atom = _Atom("Somebody Unknown | Title", {"name": "Somebody Unknown"})
    assert build_deal_roster(atoms=[atom], documents=[])["people"] == []


def test_read_signature_is_unchanged() -> None:
    """The parser itself was not the bug and must keep behaving as it did."""
    assert read_signature("Gregory Rivers | CEO") == {
        "name": "Gregory Rivers", "role": "CEO"}
