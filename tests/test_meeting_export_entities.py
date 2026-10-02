"""A HubSpot meeting recap is read as its author wrote it, not HTML-escaped.

Live 010087 (2026-10-02): the recap's header atom read "Summit 360 &amp;amp;
PurTera IT" -- HubSpot stores the title as HTML and the export escapes it
again. Notes and emails were already decoded; the meeting export was not.
"""
from app.parsers.transcript_parser import TranscriptParser


def _parse(tmp_path, title, line):
    p = tmp_path / "000001-hs-meeting-1-recap-Acme Corp.txt"
    p.write_text(
        f"HubSpot Meeting: {title}\nHubSpot Meeting ID: 1\nFacet: recap\n"
        f"Start: 2026-07-24T15:00:00Z\n\n{line}\n",
        encoding="utf-8",
    )
    return TranscriptParser().parse_artifact("p", "art_recap", p)


def test_a_double_escaped_title_and_body_are_decoded(tmp_path):
    atoms = _parse(tmp_path, "Acme &amp;amp; Widget Co", "Pat Lee: Racks &amp;amp; cabling next week.")
    header = next(a for a in atoms if a.value.get("kind") == "meeting_header")
    assert header.value["title"] == "Acme & Widget Co"
    assert header.raw_text.startswith("Meeting: Acme & Widget Co")
    body = next(a for a in atoms if "Racks" in a.raw_text)
    assert "Racks & cabling" in body.raw_text and "&amp;" not in body.raw_text
    assert body.source_refs[0].locator["line_start"] == 6


def test_a_single_escape_is_decoded_too(tmp_path):
    atoms = _parse(tmp_path, "Acme &amp; Widget Co", "Pat Lee: We need the R&amp;D lab first.")
    header = next(a for a in atoms if a.value.get("kind") == "meeting_header")
    assert header.value["title"] == "Acme & Widget Co"
    assert any("R&D lab" in a.raw_text for a in atoms)


def test_an_entity_the_author_typed_survives(tmp_path):
    # The author typed "&amp;" literally: HubSpot stores "&amp;amp;", the
    # export makes it "&amp;amp;amp;". Two rounds give back what was typed.
    atoms = _parse(tmp_path, "Escaping &amp;amp;amp; in titles", "Pat Lee: Fine.")
    header = next(a for a in atoms if a.value.get("kind") == "meeting_header")
    assert header.value["title"] == "Escaping &amp; in titles"


def test_a_url_query_is_not_read_as_an_entity(tmp_path):
    atoms = _parse(tmp_path, "Acme kickoff", "Pat Lee: See https://x.example/a?b=1&region=us for the list.")
    assert any("&region=us" in a.raw_text for a in atoms)
