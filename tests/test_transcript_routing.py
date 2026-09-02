"""A large transcript must reach the transcript parser, not the JSON flattener.

TranscriptParser recognised a ``.json`` transcript only if ``json.loads``
succeeded on ``sample_text`` -- which is a truncated head of the file. A
Fireflies transcript runs ~77 KB, so the sample is cut mid-array, the parse
raises, and TranscriptParser scored 0.0.

JsonParser's *deliberate* 0.55 deferral then won by default and flattened the
call into key/value atoms. On one deal that was 1,315 atoms typed
``scope_item`` -- including ``utterances[25].speaker: Trent Torrence``, a
speaker's name presented as extracted scope. Corpus-wide it was 147,132 atoms,
35% of all evidence, made of conversational fragments.

Both parsers already agreed on the signature. Only this side demanded proof it
cannot have from a truncated sample.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.parsers.json_parser import JsonParser
from app.parsers.transcript_parser import TranscriptParser

TRANSCRIPT_NAME = Path("010215-fireflies-01M0TJC0-transcript.json")


def _fireflies_blob(n: int) -> str:
    """The shape ingest actually writes: schema, id, title, utterances[]."""
    return json.dumps(
        {
            "schema": "fireflies.transcript.utterances.v1",
            "id": "01M0TJC0NPQ2Q83QEAPC56KR16",
            "title": "CDW PurTera IT Sodexo Program plan",
            "utterances": [
                {"speaker": "Trent Torrence", "text": f"Utterance number {i}.", "start": i * 3.0, "index": i}
                for i in range(n)
            ],
        },
        indent=2,
    )


def _match(sample: str):
    return (
        TranscriptParser().match(TRANSCRIPT_NAME, sample, None),
        JsonParser().match(TRANSCRIPT_NAME, sample, None),
    )


def test_truncated_transcript_still_routes_to_the_transcript_parser():
    raw = _fireflies_blob(500)
    assert len(raw) > 20000, "fixture must be big enough to be truncated"
    t, j = _match(raw[:4000])
    assert t.confidence > j.confidence, (
        f"transcript {t.confidence} must beat json {j.confidence} on a truncated sample"
    )
    assert "json_transcript_shape_truncated_sample" in t.reasons


def test_a_whole_transcript_still_scores_higher_than_a_truncated_one():
    # A shape read off a cut-off head is the weaker claim and should say so.
    raw = _fireflies_blob(3)
    whole, _ = _match(raw)
    truncated, _ = _match(_fireflies_blob(500)[:4000])
    assert whole.confidence > truncated.confidence
    assert "json_transcript_candidate" in whole.reasons


def test_the_json_parser_still_defers_rather_than_withdrawing():
    # 0.55 is a deferral, not a withdrawal: if the transcript parser cannot take
    # the file, somebody still has to read it.
    _, j = _match(_fireflies_blob(500)[:4000])
    assert j.confidence == 0.55
    assert "defer_to_transcript" in j.reasons


def test_ordinary_json_is_not_stolen_by_the_transcript_parser():
    # "segments" is an ordinary business word -- network segments, customer
    # segments, cable segments. An intake payload must stay with JsonParser.
    payload = json.dumps(
        {"case_id": "C-1", "segments": [{"name": "core", "vlan": 10}, {"name": "edge", "vlan": 20}]}
    )
    j = JsonParser().match(Path("case_manifest.json"), payload, None)
    tp = TranscriptParser().match(Path("case_manifest.json"), payload, None)
    assert j.confidence >= tp.confidence, "an intake manifest must not route to the transcript parser"


def test_a_speaker_name_is_never_an_atom_of_its_own():
    # The JSON flattener emitted one atom per scalar leaf, so `.speaker` became
    # `utterances[25].speaker: Trent Torrence`, typed scope_item.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / TRANSCRIPT_NAME.name
        p.write_text(_fireflies_blob(40), encoding="utf-8")
        out = TranscriptParser().parse_artifact_full(project_id="p", artifact_id="a", path=p)
    texts = [(a.raw_text or "") for a in out.atoms]
    assert not any("speaker" in t and "utterances[" in t for t in texts)
    assert all("Trent Torrence" != t.strip() for t in texts)


def test_unclassified_speech_is_kept_but_ranked_below_everything():
    # The point is not to delete conversation -- it is to stop it carrying the
    # authority of extracted scope. Every word stays, at 0.40 and flagged.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / TRANSCRIPT_NAME.name
        p.write_text(_fireflies_blob(40), encoding="utf-8")
        out = TranscriptParser().parse_artifact_full(project_id="p", artifact_id="a", path=p)
    unclassified = [a for a in out.atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "raw_utterance"]
    assert unclassified, "ordinary speech must still produce an atom"
    assert all(a.confidence <= 0.45 for a in unclassified)
    assert all("unclassified_utterance" in (a.review_flags or []) for a in unclassified)
