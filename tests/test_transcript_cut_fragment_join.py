"""Fireflies cuts one spoken sentence across consecutive cues of the same
speaker and punctuates each piece ("Can you send the." / "Floor plans for
both buildings."). The parser made one atom per cue, so the brief carried a
dangling half sentence and a second atom without its subject. Mirrors the
shape of a real deal's transcript (schema fireflies.transcript.utterances.v1:
index, speaker, start, text; no end time). Synthetic text only.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.atom_type_sanity import strip_document_chrome
from app.parsers.transcript_parser import TranscriptParser


CUES = [
    # complete sentences from one speaker: never joined
    ("Dana Field", 100.0, "But two."),
    ("Dana Field", 102.1, "Is that workable?"),
    ("Dana Field", 103.4, "Is that too fast?"),
    # cut on an article, then a complete sentence that must stay alone
    ("Omar Lane", 106.0, "I have to check the."),
    ("Omar Lane", 106.9, "I mean the rooms, since it should be workable."),
    ("Omar Lane", 109.9, "I only have to check the rooms and draft a schedule."),
    # cut on a preposition into a question
    ("Dana Field", 120.4, "Do you bill per."),
    ("Dana Field", 122.2, "Is it per room or per building?"),
    # different speaker never absorbs the cut
    ("Omar Lane", 127.0, "I was thinking about one per building."),
    # cut on a possessive; the next cue repeats it
    ("Omar Lane", 200.0, "It was only a quick pass where I was checking the routes to see how I could plan my."),
    ("Omar Lane", 206.5, "My visits around there."),
    # restarts that repeat a function word stay separate
    ("Omar Lane", 209.0, "So that's."),
    ("Omar Lane", 209.6, "That's sort of what."),
    # ends on " a." -- the cut word must survive the compile
    ("Omar Lane", 210.4, "What I thought if we add a few more and then merge week one and week two, I mean, that's not a."),
    ("Omar Lane", 216.6, "That's not an issue."),
    ("Omar Lane", 218.4, "Got it."),
    # cut on an auxiliary, then an echoed content word
    ("Omar Lane", 300.0, "And we'll do."),
    ("Omar Lane", 301.8, "We'll do about five."),
    ("Omar Lane", 303.0, "Five to six a week."),
    ("Rae Cole", 305.1, "Sounds fine."),
]


def _write(tmp_path: Path) -> Path:
    payload = {
        "schema": "fireflies.transcript.utterances.v1",
        "utterances": [
            {"index": i, "speaker": s, "start": st, "text": t} for i, (s, st, t) in enumerate(CUES)
        ],
    }
    path = tmp_path / "000000-fireflies-01ABCDEFGHJKMNPQRSTVWXYZ01-Demo-transcript.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _turns(tmp_path: Path):
    atoms = TranscriptParser().parse_artifact("p", "art_demo", _write(tmp_path))
    by_index: dict[int, object] = {}
    for a in atoms:
        loc = a.source_refs[0].locator
        if loc.get("utterance_index") is not None:
            by_index.setdefault(loc["utterance_index"], a)
    return by_index


def _cue_group(by_index, i):
    loc = by_index[i].source_refs[0].locator
    return loc.get("joined_utterance_indexes") or [loc["utterance_index"]]


def test_cut_sentences_are_rejoined_with_a_spanning_locator(tmp_path):
    turns = _turns(tmp_path)
    assert sorted(turns) == [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 19]

    assert turns[3].raw_text == "I have to check the. I mean the rooms, since it should be workable."
    loc = turns[3].source_refs[0].locator
    assert (loc["utterance_index"], loc["line_start"], loc["line_end"]) == (3, 4, 5)
    assert loc["joined_utterance_indexes"] == [3, 4]
    assert loc["timestamp_start"] == "106.0"

    assert _cue_group(turns, 6) == [6, 7]
    assert turns[6].raw_text.endswith("per room or per building?")
    assert _cue_group(turns, 9) == [9, 10]
    assert _cue_group(turns, 13) == [13, 14]
    assert _cue_group(turns, 16) == [16, 17, 18]
    loc = turns[16].source_refs[0].locator
    assert (loc["line_start"], loc["line_end"]) == (17, 19)


def test_complete_sentences_restarts_and_other_speakers_stay_apart(tmp_path):
    turns = _turns(tmp_path)
    for i in (0, 1, 2, 5, 8, 11, 12, 15, 19):
        assert _cue_group(turns, i) == [i], (i, turns[i].raw_text)
    assert "joined_utterance_indexes" not in turns[5].source_refs[0].locator


def test_auxiliary_answer_is_not_a_cut():
    from app.parsers.transcript_parser import join_cut_fragments

    segs = [
        {"utterance_index": 0, "line_start": 1, "line_end": 1, "speaker": "A", "timestamp_start": "1.0", "text": "Yes, we can."},
        {"utterance_index": 1, "line_start": 2, "line_end": 2, "speaker": "A", "timestamp_start": "2.0", "text": "Monday works for us."},
    ]
    assert len(join_cut_fragments(segs)) == 2


def test_trailing_article_of_a_spoken_turn_survives_chrome_stripping(tmp_path):
    from app.parsers.transcript_parser import join_cut_fragments
    import app.parsers.transcript_parser as tp

    original = tp.join_cut_fragments
    tp.join_cut_fragments = lambda segs: segs  # the cue on its own, as when nothing follows it
    try:
        atoms = TranscriptParser().parse_artifact("p", "art_demo", _write(tmp_path))
    finally:
        tp.join_cut_fragments = original
    turn = next(a for a in atoms if a.source_refs[0].locator.get("utterance_index") == 13)
    strip_document_chrome([turn])
    assert turn.raw_text.endswith("that's not a.")


def test_column_bleed_enumerator_is_still_stripped_from_page_text():
    from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

    text = "Access Point (Indoor Only) c."
    atom = EvidenceAtom(
        id="atm_x", project_id="p", artifact_id="art_pdf", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text, value={"text": text}, entity_keys=[],
        source_refs=[SourceRef(id="src_x", artifact_id="art_pdf", artifact_type=ArtifactType.pdf,
                               filename="x.pdf", locator={"page": 1}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )
    strip_document_chrome([atom])
    assert atom.raw_text == "Access Point (Indoor Only)"
