"""A call-notes paragraph is several facts, and the heading is a pointer.

010180's notes email arrives as one line per section:

    Notes below:
    Layout & Technical Requirements: The team discussed the office layout,
    including 106 workstations, conference rooms, phone rooms, IT room, and
    pantry. CAD drawings and plans were shared for review. The setup will
    require Cat 6A cabling, two Ethernet connections per workstation, and AV
    work for conference rooms. 2:01

Left whole that is ONE atom, and it is three different facts of three
different types -- a room mix, a document exchange, and a cabling
specification. Typed as any one of them it is wrong about the other two, and a
head asked to learn from it learns that a room count and a Cat 6A spec are the
same kind of thing.

The heading was swallowed into the text too, so the thing that says WHICH
section a fact belongs to was not available as context. `Notes below:` is the
outer pointer, the heading is the inner one, and neither is an atom.

And the `2:01` on the end is an offset into a call recording. It times a moment
in a conversation, not anything about the job, and left on it became part of
the last sentence's atom.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.parsers.email_body import split_notes_entries
from app.parsers.registry import choose_parser

#: Verbatim from 010180-hs-email-114272444842.eml, non-breaking space and all.
NOTES = (
    "Notes below:\n"
    "Layout & Technical Requirements: The team discussed the office layout, "
    "including 106 workstations, conference rooms, phone rooms, IT room, and "
    "pantry. CAD drawings and plans were shared for review. The setup will "
    "require Cat 6A cabling, two Ethernet connections per workstation, and AV "
    "work for conference rooms. 2:01\n"
    "Security & Access Control: The team discussed implementing a new security "
    "system, likely using their own swipe card system similar to Great Neck, and "
    "considered whether to include Brevo or let the building handle physical "
    "security. 7:08\n"
)


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in split_notes_entries(text).splitlines() if ln.strip()]


def test_each_sentence_becomes_its_own_line():
    out = _lines(NOTES)
    assert out[0] == "Notes below:"
    assert out[1] == "Layout & Technical Requirements:"
    assert out[2].startswith("The team discussed the office layout")
    assert out[3] == "CAD drawings and plans were shared for review."
    assert out[4].startswith("The setup will require Cat 6A cabling")


def test_the_heading_leaves_the_text_even_for_a_single_sentence():
    """One fact under a heading deserves the pointer as much as three do."""
    out = _lines(NOTES)
    assert "Security & Access Control:" in out
    body = next(ln for ln in out if ln.startswith("The team discussed implementing"))
    assert not body.startswith("Security")


def test_the_recording_offset_goes():
    out = " | ".join(_lines(NOTES))
    assert "2:01" not in out
    assert "7:08" not in out
    assert "conference rooms." in out
    assert "physical security." in out


def test_a_label_with_a_value_is_left_alone():
    """The guard. Only a heading-shaped label over PROSE is split, so the
    label/value lines everywhere else keep their shape."""
    for line in ("Passcode: jz7o5CE9",
                 "Total workstation drops: 212.",
                 "Meeting ID: 228 859 003 479 315",
                 "Direct: (732) 982-0189"):
        assert split_notes_entries(line) == line


def _parse(body: str):
    eml = ("From: erick.villalobos@cdw.com\nTo: patrick@purtera-it.com\n"
           "Subject: RE: FlexTrade New Office Cabling\n"
           "Date: Wed, 5 Aug 2026 11:11:00 +0000\n"
           "Content-Type: text/plain; charset=utf-8\n\n" + body)
    path = Path(tempfile.mkdtemp()) / "m.eml"
    path.write_text(eml, encoding="utf-8")
    parser, _, _ = choose_parser(path)
    return [a for a in parser.parse(path)
            if isinstance(a.value, dict) and a.value.get("kind") == "email_body_line"]


def test_the_parser_makes_one_atom_per_fact():
    atoms = _parse(NOTES)
    texts = [" ".join(a.raw_text.split()) for a in atoms]
    assert any(t.startswith("The team discussed the office layout") for t in texts)
    assert "CAD drawings and plans were shared for review." in texts
    assert any(t.startswith("The setup will require Cat 6A cabling") for t in texts)
    # ...and NOT one atom carrying all three.
    assert not any("CAD drawings" in t and "Cat 6A cabling" in t for t in texts)


def test_every_fact_carries_the_heading_that_governs_it():
    """The pointer. Without it a head cannot tell which section a fact is
    from, and `Notes below:` / the heading were being read as atoms in their
    own right or not at all."""
    atoms = _parse(NOTES)
    for atom in atoms:
        value = atom.value
        if value.get("list_item"):
            assert value.get("list_label"), atom.raw_text[:60]
    governed = {v: True for v in
                (a.value.get("list_label") for a in atoms if a.value.get("list_item"))}
    assert any(str(k).startswith("Layout & Technical Requirements") for k in governed)
    assert any(str(k).startswith("Security & Access Control") for k in governed)
