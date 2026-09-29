# -*- coding: utf-8 -*-
"""Three ways a signature block filed the wrong person.

Live 010237, from the phase-3 fold audit. The deal's stakeholder list carried

    WIFI Example | Thank you | chase@purtera-it.com | 770.500.5062

beside the real record for Chase Smith, Director of Operations, on the same
address and the same direct line. Three separate faults stacked up:

* `_is_titleish` only rejected a sign-off that ENDED IN A COMMA. "Thank you"
  on its own line is two words with a leading capital and passed every other
  test, so it became a job title.
* a cluster that had collected nothing -- no title, no address, no number --
  kept reading past the next name-shaped line, so a document heading three
  lines above a signature took that signature's contact details.
* `_clean` stripped only the ENDS of a line, so the non-breaking space Outlook
  writes between a first and last name rode into the stored name. Dedup then
  correctly folded "Tim\u00a0Penney" with "Tim Penney" and kept the
  non-breaking one, and every later match on the plain spelling missed a
  person the deal had already identified.
"""
from __future__ import annotations

from app.parsers.signature_block import _clean, _is_titleish, people_from_signature_lines


def test_a_signoff_is_not_a_title() -> None:
    for line in ("Thank you", "Thanks", "Best regards", "Kind Regards",
                 "Cheers", "Sincerely", "Talk soon", "Sent from my iPhone"):
        assert not _is_titleish(line), f"{line!r} was accepted as a job title"


def test_a_real_title_still_is_one() -> None:
    for line in ("Director of Operations", "Global Account Executive",
                 "Sr. Account Manager", "Field Solutions Director"):
        assert _is_titleish(line), f"{line!r} stopped being a job title"


def test_a_heading_does_not_take_the_signature_below_it() -> None:
    people = people_from_signature_lines([
        "WIFI Example",
        "Thank you",
        "Chase Smith",
        "Director of Operations",
        "chase@purtera-it.com",
        "770.500.5062",
    ])
    assert len(people) == 1, [p.get("name") for p in people]
    assert people[0]["name"] == "Chase Smith"
    assert people[0]["role"] == "Director of Operations"
    assert people[0]["email"] == "chase@purtera-it.com"


def test_a_name_keeps_no_non_breaking_space() -> None:
    people = people_from_signature_lines([
        "Tim\u00a0Penney",
        "Global\u00a0Account\u00a0Executive",
        "tim_penney@shi.com",
        "+17324218011",
    ])
    assert len(people) == 1
    assert people[0]["name"] == "Tim Penney"
    assert people[0]["role"] == "Global Account Executive"
    assert "\u00a0" not in _clean("Tim\u00a0Penney")


def test_two_signatures_in_one_mail_stay_two_people() -> None:
    people = people_from_signature_lines([
        "Carl Painter",
        "Sr. Account Manager",
        "carlpai@cdw.com",
        "847-968-9740",
        "",
        "Rhonda Sharp",
        "Professional Services Manager",
        "rhonda.sharp@cdw.com",
        "847-363-7372",
    ])
    assert [p["name"] for p in people] == ["Carl Painter", "Rhonda Sharp"]
