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
    got = find_supply_conflicts(deal())
    # One question, because the deal has one disagreement -- between the
    # reseller's own side and the installer's -- and it covers four parts.
    assert len(got) == 1
    assert {x["item"] for x in got[0].value["items"]} == {
        "PC with Access Control Software",
        "USB Cable",
        "RS232 to USB converter",
        "Relay",
    }


def test_one_card_not_four_near_identical_ones():
    """Four questions of the form "Who supplies X? One document puts it under
    ... and another under ..." are 0.95 similar to each other: the boilerplate
    is most of the sentence. Near-duplicate collapse then eats them, and which
    ones survive is luck -- on the real deal three of four did. Four cards were
    the wrong shape anyway: the email says it in one breath, "we provide
    several of those pieces", and a PM answers it in one."""
    got = find_supply_conflicts(deal())
    assert len(got) == 1
    said = got[0].raw_text
    assert "4 parts are claimed by both sides" in said
    assert "Provided by us" in said and "Installer supplied Components" in said


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


def test_a_part_is_listed_once_even_when_two_lines_name_it():
    """A quote line can name two things: "USB Cable connecting PC to RS232 to
    USB converter" matches the drawing's "USB Cable" AND its "RS232 to USB
    converter". Each part appears once in the list."""
    items = [x["item"] for x in find_supply_conflicts(deal())[0].value["items"]]
    assert len(items) == len(set(items))


def test_the_question_names_both_documents_words():
    got = find_supply_conflicts(deal())[0]
    assert OURS in got.raw_text and INSTALLER in got.raw_text
    assert got.atom_type.value == "open_question"
    assert got.review_status.value == "needs_review"
    # Every original is named, so the finding can be traced without prose.
    assert len(got.value["atom_ids"]) == 8
    relay = [x for x in got.value["items"] if x["item"] == "Relay"][0]
    assert len(relay["atom_ids"]) == 2


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


def test_a_drawing_read_onto_an_email_is_still_a_separate_document():
    """A linked drawing is READ ONTO the mail that carried it, so the vendor's
    parts list and the reseller's supply list share an artifact id while being
    two entirely separate statements of who buys what. Comparing by artifact
    found nothing at all on the real deal."""
    same = "art_email"
    email = Atom(same, "Relay", OURS, sender=CDW)
    drawing = Atom(same, "Relay", INSTALLER, sheet="BPW061725 Rev1")
    got = find_supply_conflicts([email, drawing])
    assert [x["item"] for x in got[0].value["items"]] == ["Relay"]


def test_a_sheet_speaks_for_its_own_vendor():
    """"Huzzard supplied Components" on a Huzzard drawing is that document
    saying "we do", the same as "Provided by us" on a reseller's mail. Without
    the vendor's name the comparison reads it as a third party and raises a
    disagreement where a reseller is shipping its vendor's kit."""
    email = Atom("art_email", "Remote Extender", OURS, sender=CDW)
    drawing = Atom("art_email", "Remote Extender", VENDOR, sheet="BPW061725 Rev1")
    drawing.value = {"vendor": "Huzzard"}
    assert find_supply_conflicts([email, drawing]) == []

    # And with no vendor recorded it is a third party, so the question stands.
    bare = Atom("art_email", "Remote Extender", VENDOR, sheet="BPW061725 Rev1")
    assert len(find_supply_conflicts([email, bare])) == 1


def test_a_vendors_part_never_collapses_the_reseller_s_own_line():
    """`cross_type_dedup_atoms` groups by text alone: "Relay" typed bom_line by
    the mail and deal_metadata by the sheet read onto it is one group spanning
    two types, and the loser is dropped. They are not one sentence emitted
    twice -- they are two companies each claiming to buy the part, which is the
    disagreement the deal turns on. On 010288 this deleted four lines of the
    list being quoted from."""
    from app.core.semantic_dedup import _cross_type_text_key

    class R:
        def __init__(self, sheet=None):
            self.locator = {"sheet": sheet} if sheet else {}

    class A:
        def __init__(self, text, sheet=None):
            self.raw_text = text
            self.source_refs = [R(sheet)]

    typed = A("Relay connecting the controller")
    drawn = A("Relay connecting the controller", sheet="BPW061725 Rev1")
    assert _cross_type_text_key(typed) != _cross_type_text_key(drawn)
    # Two lines off the SAME sheet still share a key -- that is what the stage
    # is for.
    other = A("Relay connecting the controller", sheet="BPW061725 Rev1")
    assert _cross_type_text_key(drawn) == _cross_type_text_key(other)
