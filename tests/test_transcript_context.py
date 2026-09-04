"""A call transcript is context with a date, not a review queue.

Live 010300 after the Carl Painter call was linked: the call sorted after
every document (nothing dated it), 181 raw turns were queued for review,
and the shape typer had "And they play Youngstown State, right?" as an
open_question. These pin the fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.atom_substance_gate import apply_substance_gate, demote_transcript_smalltalk
from app.core.confidence_recalibration import accept_verified_high_confidence
from app.core.orbitbrief_envelope import _document_header_date
from app.core.schemas import AtomType
from app.parsers.transcript_parser import TranscriptParser, _iso_date, _ulid_iso


def test_ulid_encodes_its_creation_time() -> None:
    assert _ulid_iso("01M1KWDX5FJCYZ5BAF5JC8W0QC") == "2026-09-03T14:58:07Z"
    assert _ulid_iso("not-a-ulid") is None
    assert _iso_date("2026-09-03T15:00:00.000Z") == "2026-09-03T15:00:00Z"
    from datetime import datetime, timezone
    ms = int(datetime(2026, 9, 3, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert _iso_date(ms) == "2026-09-03T15:00:00Z"


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "010301-fireflies-01M1KWDX5FJCYZ5BAF5JC8W0QC-Carl-Painter-CDW-transcript.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


_CALL = {
    "schema": "fireflies.transcript.utterances.v1",
    "id": "01M1KWDX5FJCYZ5BAF5JC8W0QC",
    "title": "Carl Painter CDW",
    "utterances": [
        {"speaker": "Speaker 1", "text": "Yeah.", "start": 0.3, "index": 0},
        {"speaker": "Speaker 2", "text": "And they play Youngstown State, right?", "start": 2.0, "index": 1},
        {"speaker": "Speaker 1", "text": "Their schedule this year is insane.", "start": 4.0, "index": 2},
        {"speaker": "Speaker 2", "text": "There's like 170 sites total across the country.", "start": 6.0, "index": 3},
        {"speaker": "Speaker 1", "text": "That has got to be done on Saturday, Sunday only after hours.", "start": 8.0, "index": 4},
        {"speaker": "Speaker 2", "text": "Send me NewBold's quote and we will install the 4 units.", "start": 10.0, "index": 5},
        {"speaker": "Speaker 1", "text": "Okay.", "start": 12.0, "index": 6},
    ],
}


def test_call_header_dates_the_transcript_from_its_ulid(tmp_path: Path) -> None:
    atoms = TranscriptParser().parse_artifact("p", "art_ff", _write(tmp_path, _CALL))
    hdr = [a for a in atoms if (a.value or {}).get("kind") == "transcript_header"]
    assert len(hdr) == 1
    assert hdr[0].value["document_date"] == "2026-09-03T14:58:07Z"
    assert "Carl Painter CDW" in hdr[0].raw_text
    assert _document_header_date(atoms, None) == "2026-09-03T14:58:07Z"


def test_stated_date_and_participants_win_over_the_ulid(tmp_path: Path) -> None:
    payload = dict(_CALL, date="2026-09-03T15:00:00.000Z", participants=["t@purtera-it.com"])
    atoms = TranscriptParser().parse_artifact("p", "art_ff", _write(tmp_path, payload))
    hdr = [a for a in atoms if (a.value or {}).get("kind") == "transcript_header"][0]
    assert hdr.value["document_date"] == "2026-09-03T15:00:00Z"
    assert hdr.value["participants"] == ["t@purtera-it.com"]


def test_small_talk_loses_its_type_and_filler_is_dropped(tmp_path: Path) -> None:
    atoms = TranscriptParser().parse_artifact("p", "art_ff", _write(tmp_path, _CALL))
    kept, dropped = apply_substance_gate(atoms)
    texts = {a.raw_text for a in kept}
    assert "Yeah." not in texts and "Okay." not in texts
    typed = {a.raw_text: a.atom_type for a in kept if a.atom_type != AtomType.raw_utterance and (a.value or {}).get("kind") != "transcript_header"}
    assert "And they play Youngstown State, right?" not in typed
    assert "Their schedule this year is insane." not in typed
    # Substance keeps its type: a figure, a scope verb, an entity.
    assert any("after hours" in t for t in typed), typed
    assert any("NewBold" in t for t in typed), typed
    assert any("170 sites" in a.raw_text for a in kept)
    demoted = [a for a in kept if "transcript_smalltalk_demoted" in (a.review_flags or [])]
    assert demoted and all(a.atom_type == AtomType.raw_utterance for a in demoted)


def test_raw_turns_are_context_not_review_work(tmp_path: Path) -> None:
    atoms = TranscriptParser().parse_artifact("p", "art_ff", _write(tmp_path, _CALL))
    kept, _ = apply_substance_gate(atoms)
    accept_verified_high_confidence(kept)
    raw = [a for a in kept if a.atom_type == AtomType.raw_utterance]
    assert raw
    assert all(a.review_status.value == "auto_accepted" for a in raw)
    assert any("context_only" in (a.review_flags or []) for a in raw)


def test_grounding_in_the_deal_documents_keeps_a_turns_type() -> None:
    """A turn about something the documents talk about keeps its claim even
    with no scope verb or figure; small talk that shares nothing with the
    documents loses it, however long it runs."""
    from tests.test_yealink_audit_fixes import _atom
    from app.core.schemas import SourceRef

    def turn(atype, text, i):
        a = _atom(atype, text)
        a.source_refs = [SourceRef(id=f"s{i}", artifact_id="ff", artifact_type="transcript", filename="call.json",
                                   extraction_method="t", parser_version="t", locator={"utterance_index": i})]
        return a

    doc = _atom(
        AtomType.scope_item,
        "All Dentistry for Children sites are worked after hours; the quote covers 4 units. "
        "Provider installs Yealink handsets, configures the network switch, labels cabling, "
        "verifies dial tone, photographs each rack, removes packaging, escalates outages, "
        "documents serial numbers, schedules technicians, confirms customer signoff, "
        "tracks materials, reports progress weekly and closes tickets.",
    )
    atoms = [
        doc,
        turn(AtomType.open_question, "Is the children dentistry done after hours?", 1),
        turn(AtomType.open_question, "When are you getting on the road to go see your daughter and the grandkids?", 2),
        turn(AtomType.action_item, "But I said, you know, Georgia got lucky this year with their travel because this is not.", 3),
    ]
    demote_transcript_smalltalk(atoms)
    assert atoms[1].atom_type == AtomType.open_question
    assert atoms[2].atom_type == AtomType.raw_utterance
    assert atoms[3].atom_type == AtomType.raw_utterance
