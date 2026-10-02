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
