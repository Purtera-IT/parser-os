"""A note whose first or last line is just its author's name.

Live 010353: Megan Blevins' Oct 1 note opened with "Megan Blevins" -- the
author line -- and it became an atom read as a stakeholder. Who wrote a note
is the note's metadata (the section's author), not a statement in it.
"""

from __future__ import annotations

import pytest

from app.parsers.hubspot_note_parser import HubspotNoteParser, parse_hubspot_note_text

_HEAD = (
    "HubSpot Note: Note\nHubSpot Note ID: 77\nDate: 2026-10-01T14:00:00Z\n"
    "Author: Megan Blevins\nAuthor-Email: megan.blevins@example.com\n\n"
)


def _parse(tmp_path, body: str, head: str = _HEAD):
    p = tmp_path / "deal-hs-note-77.txt"
    p.write_text(head + body)
    out = HubspotNoteParser().parse_artifact_full("p", "a1", p)
    return out


@pytest.mark.parametrize("body", [
    "Megan Blevins\nSpoke with the site lead today.\nThey need the camera removed by Friday.\n",
    "Spoke with the site lead today.\nThey need the camera removed by Friday.\n- Megan\n",
    "Spoke with the site lead today.\nThey need the camera removed by Friday.\nMegan Blevins\n",
])
def test_author_name_line_is_not_an_atom(tmp_path, body):
    atoms = _parse(tmp_path, body).atoms
    texts = [a.raw_text for a in atoms]
    assert not any("Megan" in t for t in texts if "author=" not in t)
    assert not any(a.atom_type.value == "stakeholder" for a in atoms)
    assert "Spoke with the site lead today." in texts
    assert "They need the camera removed by Friday." in texts


def test_author_line_becomes_section_metadata(tmp_path):
    out = _parse(tmp_path, "Megan Blevins\nSpoke with the site lead today.\nCall back Monday.\n")
    doc = next(d.content_json for d in out.derived_files if d.content_kind == "json")
    sec = doc["pages"][0]["sections"][0]
    assert sec["author"] == "Megan Blevins"
    assert sec["author_line"] == "Megan Blevins"
    assert "Megan Blevins" not in sec["blocks"][0]["text"]
    parsed = parse_hubspot_note_text(_HEAD + "Megan Blevins\nSpoke with the site lead today.\n")
    assert parsed["author_line"] == "Megan Blevins"
    assert parsed["body_lines"] == ["Spoke with the site lead today."]


def test_a_different_person_s_name_stays(tmp_path):
    """Only the note's own author is metadata; another person named on a line
    of their own is content."""
    atoms = _parse(tmp_path, "Jamal Carter\nSpoke with the site lead today.\nCall back Monday.\n").atoms
    assert "Jamal Carter" in {a.raw_text for a in atoms}


def test_a_note_that_is_only_the_name_keeps_it(tmp_path):
    parsed = parse_hubspot_note_text(_HEAD + "Megan Blevins\n")
    assert parsed["body_lines"] == ["Megan Blevins"]
    assert parsed["author_line"] == ""


def test_author_from_email_when_name_unresolved(tmp_path):
    head = (
        "HubSpot Note: Note\nHubSpot Note ID: 78\nAuthor: HubSpot user\n"
        "Author-Email: megan.blevins@example.com\n\n"
    )
    parsed = parse_hubspot_note_text(head + "Megan Blevins\nSpoke with the site lead today.\n")
    assert parsed["author_line"] == "Megan Blevins"


_HEAD_EMAIL_ONLY = (
    "HubSpot Note: Note\nHubSpot Note ID: 78\nDate: 2026-04-16T13:08:00Z\n"
    "Author-Email: {email}\n\n"
)


@pytest.mark.parametrize("email", ["ttorrence@example.com", "trent@example.com",
                                   "trenttorrence@example.com", "t.torrence@example.com"])
def test_full_name_line_matches_an_email_only_author(tmp_path, email):
    body = "Trent Torrence\nHey Guys -\nTwo small decomm jobs in Phoenix.\n"
    atoms = _parse(tmp_path, body, head=_HEAD_EMAIL_ONLY.format(email=email)).atoms
    texts = [a.raw_text for a in atoms]
    assert "Trent Torrence" not in texts
    assert "Two small decomm jobs in Phoenix." in texts


def test_first_name_author_strips_full_name_line(tmp_path):
    head = _HEAD.replace("Author: Megan Blevins", "Author: Megan (Purtera)")
    atoms = _parse(tmp_path, "Megan Blevins\nSpoke with the site lead today.\n", head=head).atoms
    assert "Megan Blevins" not in [a.raw_text for a in atoms]


def test_someone_elses_name_line_stays(tmp_path):
    body = "Adam Smith\nSpoke with the site lead today.\n"
    atoms = _parse(tmp_path, body, head=_HEAD_EMAIL_ONLY.format(email="ttorrence@example.com")).atoms
    assert any(a.raw_text.startswith("Adam Smith") for a in atoms)
