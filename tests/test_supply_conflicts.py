"""Two documents, one part, two different companies buying it.

010288 arrived with a reseller's supply list and the vendor drawing it was
quoting from. The email says outright that the drawing is wrong -- "the
'Installer Supplied Components' are not accurate, as we provide several of
those pieces" -- and never says which several. Finding them took reading
eighteen labels off a picture and laying them against a ten-line list by hand.
"""
from __future__ import annotations

import pytest

from app.core.supply_conflicts import find_supply_conflicts, side_of


class Ref:
    def __init__(self, loc):
        self.filename = "f"
        self.locator = loc
        self.artifact_type = None


class Atom:
    def __init__(self, artifact_id, text, heading, sender=None, sheet=None):
        self.id = "atm_" + text[:10]
        self.project_id = "p"
        self.artifact_id = artifact_id
        self.raw_text = text
        self.value = {}
        loc = {"section_path": [heading]}
        if sender:
            loc["sender"] = sender
        if sheet:
            loc["sheet"] = sheet
        self.source_refs = [Ref(loc)]


CDW = "alecandrich@cdw.com"
OURS = "Provided by us"
CLUB = "Provided by Club/installer"
INSTALLER = "Installer supplied Components"
VENDOR = "Huzzard supplied Components"


def deal():
    """010288, as both documents state it."""
    email = [Atom("art_email", t, OURS, sender=CDW) for t in (
        "PC with Access Control Software",
        "USB Cable connecting PC to RS232 to USB converter",
        "RS232 to USB Converter",
        "Relay",
        "Local and Remote Extender",
    )]
    email += [Atom("art_email", t, CLUB, sender=CDW) for t in (
        "Mag Lock Cable",
        "Power Supply for mag lock/locking mechanism",
        "Mag/Electric Lock",
    )]
    drawing = [Atom("art_draw", t, INSTALLER, sheet="BPW061725 Rev1") for t in (
        "PC with Access Control Software", "USB Cable", "RS232 to USB converter",
        "Relay", "Mag Lock Cable", "Power Supply", "Mag / Electric Lock",
    )]
    drawing += [Atom("art_draw", "Remote Extender", VENDOR, sheet="Huzzard BPW061725")]
    return email + drawing


def test_it_finds_the_four_the_email_would_not_name():
    items = {c.value["item"] for c in find_supply_conflicts(deal())}
    assert items == {
        "PC with Access Control Software",
        "USB Cable",
        "RS232 to USB converter",
        "Relay",
    }


def test_it_stays_silent_where_the_two_documents_agree():
    """"Provided by Club/installer" and "Installer supplied Components" are the
    same answer in different words. A rule that fired on wording alone would
    flag all eight shared parts and be worth nothing."""
    said = " ".join(c.raw_text for c in find_supply_conflicts(deal()))
    for agreed in ("Mag Lock Cable", "Power Supply", "Mag / Electric Lock"):
        assert agreed not in said


def test_a_reseller_shipping_its_vendors_kit_is_not_a_disagreement():
    """The email says CDW provides the extender; the drawing says Huzzard
    supplies it. Both documents are claiming it for their OWN side, which is a
    supply chain, not a conflict."""
    said = " ".join(c.raw_text for c in find_supply_conflicts(deal()))
    assert "Extender" not in said


def test_one_question_per_part_not_per_pair_of_lines():
    """A quote line can name two things: "USB Cable connecting PC to RS232 to
    USB converter" matches the drawing's "USB Cable" AND its "RS232 to USB
    converter". A PM is asked about each part once."""
    got = find_supply_conflicts(deal())
    assert len(got) == len({c.value["item"] for c in got})


def test_the_question_names_both_documents_words():
    got = [c for c in find_supply_conflicts(deal()) if c.value["item"] == "Relay"][0]
    assert OURS in got.raw_text and INSTALLER in got.raw_text
    assert got.atom_type.value == "open_question"
    assert got.review_status.value == "needs_review"
    # Both originals are named, so the finding can be traced without prose.
    assert len(got.value["atom_ids"]) == 2


@pytest.mark.parametrize("heading,authors,want", [
    ("Provided by us", ["cdw"], "self"),
    ("Provided by Us", [], "self"),
    ("Huzzard supplied Components", ["huzzard"], "self"),
    ("Huzzard supplied Components", ["cdw"], "other"),
    ("Installer supplied Components", ["huzzard"], "other"),
    ("Provided by Club/installer", ["cdw"], "other"),
])
def test_which_side_a_heading_names(heading, authors, want):
    assert side_of(heading, authors) == want


def test_a_document_does_not_conflict_with_itself():
    """A supply list that contradicts itself is a drafting problem for whoever
    wrote it, not a question about who buys the part."""
    atoms = [Atom("art_email", "Relay", OURS, sender=CDW),
             Atom("art_email", "Relay", CLUB, sender=CDW)]
    assert find_supply_conflicts(atoms) == []


def test_headings_that_are_not_about_supply_are_ignored():
    """Otherwise every bulleted section in every email becomes a claim about
    who is buying something."""
    atoms = [Atom("art_email", "Relay", "Next steps", sender=CDW),
             Atom("art_draw", "Relay", "Questions", sheet="X")]
    assert find_supply_conflicts(atoms) == []


def test_nothing_to_compare_is_not_an_error():
    assert find_supply_conflicts([]) == []
    assert find_supply_conflicts([Atom("a", "Relay", OURS, sender=CDW)]) == []
