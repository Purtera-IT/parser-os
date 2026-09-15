"""A meeting export's header block is one metadata record, never speech.

Live 010095 (2026-09-15): the CRM meeting export opens with ``HubSpot Meeting:
010095 Lantronix device installation and setup(with Orcle)`` and six more
``Label: value`` lines. Read as speech, "HubSpot Meeting" was a speaker and the
deal's own name became a task the Deal Kit never priced.
"""
from app.core.schemas import AtomType
from app.parsers.transcript_parser import TranscriptParser, peel_export_header

EXPORT = (
    "HubSpot Meeting: 010095 Lantronix device installation and setup(with Orcle)\n"
    "HubSpot Meeting ID: 112650820173\n"
    "Facet: recap\n"
    "Start: 2026-07-09T17:00:00Z\n"
    "End: 2026-07-09T17:30:00.001Z\n"
    "Transcript-Id: 01KX3F1T4B429G3ZC3F55A7XK9\n"
    "Recorded-At: 2026-07-09T17:18:23.876Z\n"
    "\n"
    "Customer: We will need the two Lantronix boxes racked and reachable from the Oracle host.\n"
)


def test_the_header_is_peeled_and_the_body_knows_where_it_started():
    fields, body, offset = peel_export_header(EXPORT)
    assert list(fields)[0] == "HubSpot Meeting"
    assert fields["Start"] == "2026-07-09T17:00:00Z"
    assert body.startswith("Customer:") and offset == 8


def test_the_title_is_metadata_not_a_task_or_an_utterance(tmp_path):
    p = tmp_path / "010095-hs-meeting-112650820173-recap-010095 Lantronix.txt"
    p.write_text(EXPORT, encoding="utf-8")
    atoms = TranscriptParser().parse_artifact("p", "art_recap", p)
    texts = [a.raw_text for a in atoms]
    assert not any("Lantronix device installation and setup" in t and a.atom_type != AtomType.deal_metadata
                   for a, t in zip(atoms, texts))
    assert not any(t.strip() in ("112650820173", "recap") for t in texts)
    header = [a for a in atoms if a.value.get("kind") == "meeting_header"]
    assert len(header) == 1 and header[0].atom_type == AtomType.deal_metadata
    assert header[0].value["title"] == "010095 Lantronix device installation and setup(with Orcle)"
    assert header[0].value["document_date"] == "2026-07-09T17:00:00Z"  # same shape as the JSON call header
    # The body after the header is still read as speech, at its true line.
    body = [a for a in atoms if "Lantronix boxes racked" in a.raw_text]
    assert body and body[0].source_refs[0].locator["line_start"] == 9


def test_speech_that_happens_to_start_with_speaker_lines_is_not_a_header():
    speech = ("Alex: Thanks for joining, let's walk the closet layout first.\n"
              "Jordan: Sounds good, the racks are on the east wall.\n"
              "Alex: And the fiber comes in from the north side?\n"
              "\n"
              "Jordan: Yes.\n")
    assert peel_export_header(speech) == (None, speech, 0)
    # Fewer than three fields, or a repeated label, is not an export header either.
    assert peel_export_header("Title: x\nId: 1\n\nbody\n") == (None, "Title: x\nId: 1\n\nbody\n", 0)
    dup = "A: one\nA: two\nB: three\n\nbody\n"
    assert peel_export_header(dup) == (None, dup, 0)
    # A header with nothing after it is a file with no speech; leave it alone.
    only = "HubSpot Meeting: x\nID: 1\nFacet: recap\n"
    assert peel_export_header(only) == (None, only, 0)
