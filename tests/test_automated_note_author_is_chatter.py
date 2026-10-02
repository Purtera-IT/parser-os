"""A HubSpot note written by an automated author is one rejected chatter atom.

Live 010353: "OxBlue customer portal intake received.", posted by the HubSpot
system user, became eight deal_metadata atoms restating the intake. The
labelling checklist says automated senders are always rejects and never
stakeholders.
"""

from __future__ import annotations

import pytest

from app.parsers.hubspot_note_parser import HubspotNoteParser, _is_automated_note_author

_BODY = """
OxBlue customer portal intake received.

Request: OXR-1042
Site: Camden, TN
Type: removal
Priority: standard

Summary: Requesting a quote for the removal of one camera system at the Camden, TN site.
"""


def _note(tmp_path, author_lines: str):
    p = tmp_path / "deal-hs-note-98765432101.txt"
    p.write_text(
        "HubSpot Note: Note\nHubSpot Note ID: 98765432101\n"
        "Date: 2026-08-14T15:02:11Z\n" + author_lines + "\n" + _BODY
    )
    return HubspotNoteParser().parse_artifact("p", "a1", p)


@pytest.mark.parametrize("author_lines", [
    "Author: HubSpot\n",
    # What the export writes when no owner resolved -- every API-created
    # note, the intake notification among them. Decided by the body's shape.
    "Author: HubSpot user\n",
    "Author: HubSpot System User\n",
    "Author: Integration User\n",
    "Author: Portal Bot\nAuthor-Email: noreply@hubspot.com\n",
])
def test_automated_note_is_one_rejected_chatter_atom(tmp_path, author_lines):
    atoms = _note(tmp_path, author_lines)
    # The note's header metadata is non_deal and may already be chatter on
    # its own (a robot's address in it); the BODY is what becomes one atom.
    chatter = [a for a in atoms if "chatter" in a.review_flags
               and a.value.get("kind") != "hubspot_note_meta"]
    assert len(chatter) == 1
    c = chatter[0]
    assert c.review_status.value == "rejected"
    assert c.value["automated_sender"]
    assert "automated_sender" in c.review_flags
    assert "OxBlue customer portal intake received." in c.raw_text
    assert not any(a.atom_type.value == "stakeholder" for a in atoms)
    texts = {a.raw_text for a in atoms}
    # Restatements of the intake are gone ...
    for gone in ("Site: Camden, TN", "Type: removal", "Priority: standard"):
        assert gone not in texts
    # ... and the identifier stated only here stays as its own fact.
    assert "Request: OXR-1042" in texts


def test_an_unresolved_author_with_a_person_s_prose_is_untouched(tmp_path):
    p = tmp_path / "deal-hs-note-1.txt"
    p.write_text(
        "HubSpot Note: Walkthrough\nHubSpot Note ID: 1\nAuthor: HubSpot user\n\n"
        "Walked the Camden site with Jamal today.\n"
        "Site: Camden, TN\n"
        "The mount is on the ground and the solar unit is cracked, so plan a lift.\n"
    )
    atoms = HubspotNoteParser().parse_artifact("p", "a1", p)
    assert not any("automated_sender" in a.review_flags for a in atoms)


def test_a_person_s_note_is_untouched(tmp_path):
    atoms = _note(tmp_path, "Author: Dana Whitfield\nAuthor-Email: dana@purtera-it.com\n")
    assert not any("automated_sender" in a.review_flags for a in atoms)
    assert "Site: Camden, TN" in {a.raw_text for a in atoms}


@pytest.mark.parametrize("name,email,expected", [
    ("HubSpot", "", True),
    ("System", "", True),
    ("", "no-reply@notifications.hubspot.com", True),
    ("Christopher Picchetti", "cpicchetti@oxblue.com", False),
    ("Bot Smith", "", False),
    ("HubSpot user", "", False),  # ambiguous alone; see _note_is_automated
])
def test_author_detection(name, email, expected):
    assert _is_automated_note_author(name, email) is expected
