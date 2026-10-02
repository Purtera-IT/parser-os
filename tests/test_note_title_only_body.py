"""A note whose whole content is its title still says something.

Live 000132: "Need Troy and Wilmington sites removed." was the note's title
and its body, and the note produced only its header atom -- the caption branch
in _mint_prose_one set a type and then minted nothing. "PO!!" went the same
way. A bare "Note" is the CRM's default title and genuinely empty.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.parsers.hubspot_note_parser import HubspotNoteParser


def _note(tmp_path: Path, title: str, body: str | None) -> Path:
    p = tmp_path / "000132-hs-note-1234.txt"
    text = (f"HubSpot Note: {title}\nHubSpot Note ID: 1234\nDate: 2026-06-02T10:00:00Z\n"
            f"Author: Trent Torrence\nAuthor-Email: t@purtera-it.com\n\n")
    if body is not None:
        text += f"{body}\n"
    p.write_text(text, encoding="utf-8")
    return p


def _content(atoms):
    return [a for a in atoms if (a.value or {}).get("kind") != "hubspot_note_meta"]


@pytest.mark.parametrize("line", ["Need Troy and Wilmington sites removed.", "PO!!"])
@pytest.mark.parametrize("shape", ["body_equals_title", "title_repeated_in_body", "no_body"])
def test_title_only_note_yields_one_content_atom(tmp_path: Path, line: str, shape: str):
    body = {"body_equals_title": line, "title_repeated_in_body": f"{line}\n\n{line}", "no_body": None}[shape]
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art_note", _note(tmp_path, line, body)))
    assert [a.raw_text for a in atoms] == [line]
    assert atoms[0].value.get("kind") == "hubspot_note_body"


def test_placeholder_title_is_empty(tmp_path: Path):
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art_note", _note(tmp_path, "Note", "Note")))
    assert atoms == []


def test_upload_caption_is_minted_as_metadata(tmp_path: Path):
    atoms = _content(HubspotNoteParser().parse_artifact(
        "p", "art_note", _note(tmp_path, "psow from current partner", "psow from current partner")))
    assert [(a.atom_type.value, a.raw_text) for a in atoms] == [("deal_metadata", "psow from current partner")]


def test_compile_keeps_the_one_line_note(tmp_path: Path):
    from app.core.compiler import compile_project

    line = "Need Troy and Wilmington sites removed."
    _note(tmp_path, line, line)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    assert any(a.raw_text == line for a in r.atoms)
