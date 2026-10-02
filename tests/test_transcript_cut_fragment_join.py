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


# ---------------------------------------------------------------------------
# Wider join: read the completeness of both sides, not only a dangling last
# word, and talk over a one-word interjection. Mirrors the misses seen on a
# real deal (synthetic text).
# ---------------------------------------------------------------------------

def _seg(i, speaker, start, text):
    return {"utterance_index": i, "line_start": i + 1, "line_end": i + 1, "speaker": speaker,
            "timestamp_start": str(start), "text": text}


def _groups(cues):
    from app.parsers.transcript_parser import join_cut_fragments

    segs = [_seg(i, s, st, t) for i, (s, st, t) in enumerate(cues)]
    return [s.get("joined_utterance_indexes") or [s["utterance_index"]] for s in join_cut_fragments(segs)]


def test_prepositional_continuation_after_a_complete_looking_sentence():
    assert _groups([
        ("A", 10.0, "We won't have totals for those until Tuesday."),
        ("A", 13.4, "On what's really in the rooms."),
        ("B", 16.0, "Understood."),
    ]) == [[0, 1], [2]]
    assert _groups([
        ("A", 10.0, "Andy, when do you think the floor survey will be done?"),
        ("A", 14.0, "With dates, just so Priya knows."),
    ]) == [[0, 1]]


def test_verbless_head_runs_into_a_fragment_and_a_short_tail_closes_the_question():
    assert _groups([
        ("B", 515.8, "12 in Denver, 9 in Ohio."),
        ("A", 522.4, "Like rooms."),
        ("A", 523.0, "Do you bill per."),
        ("A", 524.8, "Is it per floor or per building?"),
        ("A", 529.8, "How do you sort of."),
        ("C", 532.0, "I was thinking about one per building."),
    ]) == [[0], [1, 2, 3, 4], [5]]
    # a verbless head before a complete question stays alone
    assert _groups([("A", 280.0, "But two."), ("A", 282.0, "Is that workable?")]) == [[0], [1]]


def test_verbless_phrase_closes_the_sentence_before_it():
    assert _groups([
        ("A", 10.0, "We're going to need a lot of those for the new term."),
        ("A", 14.0, "A lot of floor leads."),
        ("B", 16.0, "Makes sense."),
    ]) == [[0, 1], [2]]
    # a verbless phrase that picks up nothing of the sentence before is its own
    for a, b in (("Thanks everyone.", "Talk soon."), ("We have three sites.", "Chicago, Denver and Austin."),
                 ("I sent it over.", "In the meantime, review it."), ("We finished.", "Of course.")):
        assert _groups([("A", 1.0, a), ("A", 3.0, b)]) == [[0], [1]], (a, b)
    # two complete sentences stay apart
    assert _groups([
        ("A", 10.0, "I'll send you the old names and then the new names."),
        ("A", 13.0, "The names changed a little bit."),
        ("A", 15.0, "Because they changed a bit."),
    ]) == [[0], [1], [2]]


def test_restart_is_not_a_continuation_and_the_cut_head_closes_the_sentence_before():
    assert _groups([
        ("A", 10.0, "Because we'll need to know exactly where the racks sit in the rooms."),
        ("A", 15.0, "How are we gonna."),
        ("A", 16.2, "How are we thinking?"),
    ]) == [[0, 1], [2]]
    # an elliptical answer ending on an auxiliary is not a cut tail
    assert _groups([
        ("A", 12.0, "Yeah, if you have a form from another job, we can reuse it."),
        ("A", 15.0, "I believe we do."),
        ("B", 17.0, "Great."),
    ]) == [[0], [1], [2]]


def test_long_pause_joins_only_when_both_sides_are_fragments_and_never_past_the_ceiling():
    # 12.4 s between the cut and a verbless continuation
    assert _groups([("A", 18.6, "I'll do the same with the."), ("A", 31.0, "The other rooms as well.")]) == [[0, 1]]
    # past the ceiling nothing joins
    assert _groups([("A", 18.6, "I'll do the same with the."), ("A", 40.0, "The other rooms as well.")]) == [[0], [1]]
    # one-sided evidence allows a short pause only
    assert _groups([("A", 10.0, "I have to check the."), ("A", 25.0, "Is it ready?")]) == [[0], [1]]


def test_interjection_from_another_speaker_is_skipped_and_kept_in_order(tmp_path):
    cues = [
        ("Omar Lane", 10.0, "Next Monday is when we asked for all counts to be entered and locked."),
        ("Dana Field", 16.5, "Okay."),
        ("Omar Lane", 17.2, "At the latest."),
        ("Dana Field", 19.0, "That works."),
    ]
    payload = {"schema": "fireflies.transcript.utterances.v1",
               "utterances": [{"index": i, "speaker": s, "start": st, "text": t} for i, (s, st, t) in enumerate(cues)]}
    path = tmp_path / "000000-fireflies-01ABCDEFGHJKMNPQRSTVWXYZ02-Demo-transcript.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    atoms = TranscriptParser().parse_artifact("p", "art_demo", path)
    turns = [a for a in atoms if a.source_refs[0].locator.get("utterance_index") is not None]
    order = []
    for a in turns:
        ui = a.source_refs[0].locator["utterance_index"]
        if ui not in order:
            order.append(ui)
    assert order == [0, 1, 3]
    joined = next(a for a in turns if a.source_refs[0].locator["utterance_index"] == 0)
    loc = joined.source_refs[0].locator
    assert loc["joined_utterance_indexes"] == [0, 2]
    assert (loc["line_start"], loc["line_end"]) == (1, 3)
    assert joined.raw_text.endswith("entered and locked. At the latest.")
    okay = next(a for a in turns if a.source_refs[0].locator["utterance_index"] == 1)
    assert okay.source_refs[0].locator["speaker"] == "Dana Field"
    assert "joined_utterance_indexes" not in okay.source_refs[0].locator


def test_interjection_is_not_skipped_into_a_new_sentence():
    assert _groups([
        ("A", 10.0, "Next Monday is when we asked for counts."),
        ("B", 13.0, "Okay."),
        ("A", 14.0, "I'll send the list today."),
    ]) == [[0], [1], [2]]
    # a real answer from the other speaker is never skipped
    assert _groups([
        ("A", 10.0, "Is that the full list?"),
        ("B", 12.0, "Probably next week."),
        ("A", 14.0, "At the latest."),
    ]) == [[0], [1], [2]]
    # after a question the other speaker's word is an answer, not an interjection
    assert _groups([
        ("A", 10.0, "Is there parking at the depot?"),
        ("B", 12.0, "Yeah."),
        ("A", 13.0, "At the north gate."),
    ]) == [[0], [1], [2]]
