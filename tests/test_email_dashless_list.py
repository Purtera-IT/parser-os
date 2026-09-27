"""A list typed without dashes is still a list.

Live 010180. One email set out the floorplan like this --

    Offices and Rooms
    The floorplan includes:
    Executive offices
    Private offices
    Board room
    Huddle rooms
    Phone room
    IT room/closet
    Reception area
    Print/copy areas

-- and eight room facts reached no artifact at all. The chain:

1. ``The floorplan includes:`` is a framing lead-in, so branch 2b held it in
   ``pending_lead_in`` waiting for an ``Include:`` header. None ever came --
   most framing lead-ins are the whole label -- so the line was dropped and
   ``list_label`` was never armed.
2. The rooms beneath it were therefore not ``list_item``. Each became a bare
   one- or two-word ``scope_item``.
3. ``drop_email_non_scope`` in the substance gate deletes a short scope_item
   with no verb, no digit and no entity -- correctly, because "Huddle rooms"
   on its own is not a statement. Its one exemption is ``list_item``:
   "one word under a label is an item, not chatter."

So the gate was right and the parser had removed the only evidence that would
have saved them. The same hole had a second mouth: ``Included:`` DID set a
section, but ``section_for_line`` served it to bullets only, so a dashless
list under an explicit header lost it just the same.

What made this expensive to find is that it was invisible. The rooms survived
the PREVIOUS compile of this deal, because ``typed_atom_classification`` -- a
3B model -- happened to promote them to ``site_room_mix`` that run, and the
gate only examines ``scope_item``/``stakeholder``. Next run it left them
``scope_item`` and all eight died. Whether a fact reaches the SOW must not be
a coin flip; a statement survives on its own structure.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.atom_substance_gate import apply_substance_gate
from app.parsers.registry import choose_parser

#: Verbatim from 010180-hs-email-117380900193.eml.
ROOMS = [
    "Executive offices",
    "Private offices",
    "Board room",
    "Huddle rooms",
    "Phone room",
    "IT room/closet",
    "Reception area",
    "Print/copy areas",
]


def _parse(body: str):
    eml = (
        "From: patrick@purtera-it.com\n"
        "To: chase@purtera-it.com\n"
        "Subject: RE: FlexTrade New Office Cabling\n"
        "Date: Thu, 24 Sep 2026 14:44:34 +0000\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n" + body
    )
    path = Path(tempfile.mkdtemp()) / "m.eml"
    path.write_text(eml, encoding="utf-8")
    parser, _, _ = choose_parser(path)
    return list(parser.parse(path))


def _by_text(atoms):
    return {" ".join(a.raw_text.split()): a for a in atoms}


def _value(atom) -> dict:
    return atom.value if isinstance(atom.value, dict) else {}


FLOORPLAN = "The floorplan includes:\n" + "\n".join(ROOMS) + "\n"


def test_the_rooms_are_items_of_the_list_that_announced_them():
    found = _by_text(_parse(FLOORPLAN))
    for room in ROOMS:
        assert room in found, f"{room} produced no atom"
        assert _value(found[room]).get("list_item") is True, room


def test_they_carry_the_lead_in_that_makes_them_statements():
    """"Board room" is not a fact. "The floorplan includes: Board room" is --
    and the intro is what ``typed_atom_classifier`` reads to call it a room."""
    found = _by_text(_parse(FLOORPLAN))
    value = _value(found["Board room"])
    assert value.get("list_label") == "The floorplan includes:"


def test_the_substance_gate_keeps_every_one():
    """The measurement that matters: before this fix, one of nine survived."""
    atoms = _parse(FLOORPLAN)
    kept, dropped = apply_substance_gate(atoms)
    lost = [a.raw_text for a in dropped if " ".join(a.raw_text.split()) in ROOMS]
    assert lost == [], f"gate still deletes {lost}"
    assert len([a for a in kept if " ".join(a.raw_text.split()) in ROOMS]) == len(ROOMS)


def test_the_lead_in_is_not_itself_an_atom():
    """It is the label its items are read with, not a fact of its own."""
    assert "The floorplan includes:" not in _by_text(_parse(FLOORPLAN))


# --------------------------------------------------------------------------
# the second mouth: an explicit header, a list with no dashes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("header,section", [("Included:", "include"),
                                            ("Excluded:", "exclude")])
def test_a_dashless_list_under_an_explicit_header_inherits_it(header, section):
    """Every item, not only the first -- the first non-bullet line used to end
    the section for the lines after it."""
    found = _by_text(_parse(header + "\n" + "\n".join(ROOMS[:4]) + "\n"))
    for room in ROOMS[:4]:
        assert _value(found[room]).get("list_section") == section, room


def test_prose_still_closes_the_list():
    """The guard on the above: a real sentence ends the list, and what follows
    it is not swept in as an item."""
    body = ("Included:\n"
            "Executive offices\n"
            "That is everything we agreed on the call yesterday afternoon.\n"
            "Board room\n")
    found = _by_text(_parse(body))
    assert _value(found["Executive offices"]).get("list_section") == "include"
    assert _value(found["Board room"]).get("list_section") is None


def test_a_labelled_list_still_works_without_a_framing_verb():
    """"Provided by us:" and "Deliverables:" went through branch 3b and were
    never broken. They must keep working -- the fix arms `list_label` earlier,
    and the two mechanisms must not fight."""
    for label in ("Provided by us:", "Deliverables:", "Rooms:"):
        found = _by_text(_parse(label + "\nExecutive offices\nBoard room\n"))
        assert _value(found["Board room"]).get("list_item") is True, label
