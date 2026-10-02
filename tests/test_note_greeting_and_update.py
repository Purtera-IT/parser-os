"""A greeting does not make a fact into chatter, and an update names what it revises.

Deal 010003, Patrick's note:

    Hello, we already have wall mounts, and I believe parking is not free
    Update on that they do have wall mounts and parking

HubSpot made line 1 the note's title too. It was retyped to conversation_meta
("greeting") and flagged small talk because it opens with "Hello,", and its
source pointed at no line, so the viewer found the identical title in the
``HubSpot Note:`` header first. Line 2 revises line 1 and nothing said so.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.deal_chatter import is_chatter
from app.core.hybrid_summary_transcript import classify_transcript_turn_role
from app.parsers.hubspot_note_parser import HubspotNoteParser, _is_upload_caption

LINE1 = "Hello, we already have wall mounts, and I believe parking is not free"
LINE2 = "Update on that they do have wall mounts and parking"


def _note(tmp_path: Path, title: str, body: str, name: str = "010003-hs-note-1.txt") -> Path:
    p = tmp_path / name
    p.write_text(
        f"HubSpot Note: {title}\nHubSpot Note ID: 1\nDate: 2026-01-10T18:58:14.007Z\n"
        f"Author: Patrick\nAuthor-Email: patrick@purtera-it.com\n\n{body}\n",
        encoding="utf-8",
    )
    return p


def _content(atoms):
    return [a for a in atoms if (a.value or {}).get("kind") != "hubspot_note_meta"]


def test_patrick_note_parses_both_lines_as_statements_from_the_body(tmp_path: Path) -> None:
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art", _note(tmp_path, LINE1, f"{LINE1}\n{LINE2}")))
    by_text = {a.raw_text: a for a in atoms}
    assert by_text[LINE1].atom_type.value == "scope_item"
    assert by_text[LINE2].atom_type.value == "scope_item"
    # Lines 1-5 are the header (line 1 repeats the title); the body is 7-8.
    assert by_text[LINE1].source_refs[0].locator["line_start"] == 7
    assert by_text[LINE2].source_refs[0].locator["line_start"] == 8


def test_update_line_supersedes_the_statement_before_it(tmp_path: Path) -> None:
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art", _note(tmp_path, LINE1, f"{LINE1}\n{LINE2}")))
    first = next(a for a in atoms if a.raw_text == LINE1)
    update = next(a for a in atoms if a.raw_text == LINE2)
    assert update.value["supersedes"] == first.id
    assert first.value["superseded_by"] == update.id
    assert "supersedes" not in first.value


@pytest.mark.parametrize("cue", ["Update: the lift is on site", "Correction: the lift is on site",
                                 "Actually, the lift is on site", "Scratch that, the lift is on site"])
def test_update_cues_link_to_the_nearest_preceding_statement(tmp_path: Path, cue: str) -> None:
    body = "The customer has no lift.\nParking is in the rear lot.\n" + cue
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art", _note(tmp_path, "Site notes", body)))
    parking = next(a for a in atoms if a.raw_text.startswith("Parking"))
    revised = next(a for a in atoms if a.raw_text == cue)
    assert revised.value["supersedes"] == parking.id
    assert parking.value["superseded_by"] == revised.id
    assert not any("superseded_by" in a.value for a in atoms if a.raw_text.startswith("The customer"))


def test_no_cue_no_link(tmp_path: Path) -> None:
    atoms = _content(HubspotNoteParser().parse_artifact(
        "p", "art", _note(tmp_path, "Site notes", "The customer has no lift.\nParking is in the rear lot.")))
    assert not any("supersedes" in a.value or "superseded_by" in a.value for a in atoms)


def test_greeting_never_makes_a_fact_line_a_caption() -> None:
    assert not _is_upload_caption("Hello, parking is not free")
    assert not _is_upload_caption("parking is not free")
    assert _is_upload_caption("psow from current partner")
    assert _is_upload_caption("SOW")


def test_short_greeted_fact_that_is_also_the_title_is_a_statement(tmp_path: Path) -> None:
    line = "Hello, parking is not free"
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art", _note(tmp_path, line, line)))
    assert [(a.atom_type.value, a.raw_text) for a in atoms] == [("scope_item", line)]
    assert atoms[0].source_refs[0].locator["line_start"] == 7


def test_greeting_with_content_is_not_chatter() -> None:
    assert classify_transcript_turn_role(LINE1) == "deal"
    assert not is_chatter(LINE1)
    assert not is_chatter("Good morning, the parking is not free and the lift is broken")


@pytest.mark.parametrize("line", ["Hello,", "Hi Bob!", "Good morning", "Hello"])
def test_bare_greeting_is_still_chatter(line: str) -> None:
    assert is_chatter(line)
    assert classify_transcript_turn_role(line) == "greeting"


def test_greeting_then_small_talk_stays_a_greeting() -> None:
    assert classify_transcript_turn_role("Hi Bob, how are you?") == "greeting"


def test_compiled_patrick_note_keeps_line_one_as_content(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    _note(tmp_path, LINE1, f"{LINE1}\n{LINE2}")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    first = next(a for a in r.atoms if a.raw_text == LINE1)
    update = next(a for a in r.atoms if a.raw_text == LINE2)
    assert first.atom_type.value == "scope_item"
    assert (first.value or {}).get("kind") != "conversation_meta"
    assert "chatter" not in (first.review_flags or [])
    assert first.source_refs[0].locator["line_start"] == 7
    assert update.value["supersedes"] == first.id
    assert first.value["superseded_by"] == update.id


@pytest.mark.parametrize("line", ["Update the firmware on all APs.", "Updated pricing attached."])
def test_update_as_a_verb_is_not_a_revision(tmp_path: Path, line: str) -> None:
    body = f"Parking is in the rear lot.\n{line}"
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art", _note(tmp_path, "Site notes", body)))
    assert not any("supersedes" in a.value or "superseded_by" in a.value for a in atoms)
