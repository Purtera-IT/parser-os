"""Who is on this deal: one small table, not a card per signature.

The stakeholder atoms on 010288 are referenced by zero packets and the
workload matrix they nominally feed is all zeros -- five "this person exists"
cards that reach nothing. But they cannot simply be dropped: before this, an
atom's own speaker record said ``{"email": "t@purtera-it.com", "name": "T"}``,
because the name was derived from the address. The signature was the only
place that knew it was Trent Torrence.
"""
from __future__ import annotations

import pytest

from app.core.deal_roster import build_deal_roster, read_signature


class Atom:
    def __init__(self, text, atom_type="stakeholder"):
        self.atom_type = atom_type
        self.raw_text = text


SIGS = [
    "Trent Torrence | Executive Vice President of Sales | t@purtera-it.com | 404.771.3490",
    "Chase Smith | Director of Operations | chase@purtera-it.com | 770.500.5062",
    "Octavian Mitroi | octavian@purtera-it.com",
    "AJ Evans | Account Executive | aj@purtera-it.com | 407.562.7871",
    "Alec Burns | Senior Client Executive, Commercial Majors | alecbur@cdw.com",
]
DOCS = (
    [{"sender_email": "aj@purtera-it.com", "filename": f"m{i}.eml"} for i in range(5)]
    + [{"sender_email": "chase@purtera-it.com", "filename": "m5.eml"},
       {"sender_email": "chase@purtera-it.com", "filename": "m6.eml"},
       {"sender_email": "t@purtera-it.com", "filename": "m7.eml"},
       {"sender_email": "alecandrich@cdw.com", "filename": "m8.eml"},
       {"sender_email": "alecandrich@cdw.com", "filename": "m9.eml"}]
)


def roster():
    return build_deal_roster(atoms=[Atom(s) for s in SIGS], documents=DOCS,
                             our_domains={"purtera-it"})


def by_email(r, email):
    return next(p for p in r["people"] if p["email"] == email)


def test_a_header_never_carries_a_name_and_a_signature_does():
    """The whole reason the table has to exist before the footers become
    chrome: the address is all a header knows."""
    trent = by_email(roster(), "t@purtera-it.com")
    assert trent["name"] == "Trent Torrence"
    assert trent["role"] == "Executive Vice President of Sales"
    assert trent["phone"] == "404.771.3490"


def test_a_title_with_a_comma_in_it_is_one_title():
    """"Senior Client Executive, Commercial Majors" was being split into two
    people by entity extraction -- the deal ended up with nine stakeholders
    for five humans, four of them job titles."""
    assert by_email(roster(), "alecbur@cdw.com")["role"] == (
        "Senior Client Executive, Commercial Majors")


def test_the_side_comes_from_the_domain_not_the_signature():
    r = roster()
    assert by_email(r, "aj@purtera-it.com")["side"] == "ours"
    assert by_email(r, "alecbur@cdw.com")["side"] == "theirs"
    assert r["by_side"] == {"ours": 4, "theirs": 2}


def test_who_actually_wrote_something_is_counted():
    """Headers are the only evidence a person wrote rather than was quoted."""
    r = roster()
    assert by_email(r, "aj@purtera-it.com")["messages_sent"] == 5
    assert by_email(r, "chase@purtera-it.com")["messages_sent"] == 2


def test_somebody_named_in_a_footer_who_never_sent_is_flagged():
    """On the deal, worth seeing. Never worth emailing without asking."""
    assert set(roster()["named_but_silent"]) == {
        "octavian@purtera-it.com", "alecbur@cdw.com"}


def test_a_line_that_is_not_a_signature_yields_nothing():
    assert read_signature("Anything in Orange") == {}
    assert read_signature("") == {}


def test_a_signature_with_no_address_is_not_a_person():
    """Without an address there is nothing to key on, and a row keyed on a
    name would merge two people who share one."""
    r = build_deal_roster(atoms=[Atom("Trent Torrence | EVP of Sales")], documents=[])
    assert r["people"] == []


@pytest.mark.parametrize("docs", [[], [{"sender_email": "not-an-address"}], [{}]])
def test_rubbish_headers_do_not_make_rows(docs):
    assert build_deal_roster(atoms=[], documents=docs)["people"] == []


def test_nothing_at_all_is_not_an_error():
    r = build_deal_roster(atoms=[], documents=[])
    assert r["people"] == [] and r["by_side"] == {}
