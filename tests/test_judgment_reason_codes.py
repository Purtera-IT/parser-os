"""A judgment's `reason` is a closed code per head (app/core/judgment_reasons.json).

Every saved reason on one deal was a sentence, and `_judgment_rows` turned each
into its own `<relation>_reason` class: one example per class, nothing to learn.
Only a listed code makes a reason row now; a sentence moves to the front of the
note's WHY, so no labeler's words are lost, and the report counts it.
Synthetic text only.
"""
from __future__ import annotations

import json

from app.core.judgment_reasons import codes_for, is_reason_code, load_reasons
from app.core.label_heads import load_heads
from app.learning.human_labels import REASON_NOT_A_CODE, IngestReport, _note_with_reason, rows_for_deal

WHY = "The question asks for a date the customer has already given in the kickoff email, so it is answered."
SENTENCE = "The kickoff email already gives the start date."


def _gap(**kw):
    return {"head": "gap", "target_key": "g1", "verdict": "invalid", "labeler": "a@example.com",
            "text": "What is the requested start date for the work?", **kw}


def _ingest(*judgments):
    report = IngestReport()
    rows = rows_for_deal({"deal_id": "d-reasons", "labels": [], "judgments": list(judgments)}, report=report)
    return rows, report


def test_every_listed_head_is_a_judgment_head_with_described_codes():
    judged = {j for h in load_heads()["heads"] for j in h.get("judgments", [])}
    for head, codes in load_reasons()["heads"].items():
        assert head in judged, f"{head} is not a judgment head in label_heads.json"
        values = [c["value"] for c in codes]
        assert len(values) == len(set(values)), head
        for c in codes:
            assert c["value"].replace("_", "").isalnum() and c["value"].islower(), c
            assert c["desc"].strip() and "\n" not in c["desc"], c
    # The parser's own provenance on a card is not a judgment reason.
    assert not any(is_reason_code(h, "deterministic_fallback") for h in load_reasons()["heads"])


def test_a_listed_code_makes_a_reason_row_and_rides_in_the_prompt():
    rows, report = _ingest(_gap(reason="answered", note=WHY))
    reason_rows = [r for r in rows if r["relation"] == "gap_valid_reason"]
    assert [r["label"] for r in reason_rows] == ["answered"]
    (rat,) = [r for r in rows if r["relation"] == "rationale:gap"]
    assert rat["raw_text"].endswith("VERDICT: invalid (answered)")
    assert rat["label"] == WHY
    assert REASON_NOT_A_CODE not in report.skipped


def test_a_sentence_is_no_class_and_opens_the_why_instead():
    rows, report = _ingest(_gap(reason=SENTENCE, note=WHY))
    assert not [r for r in rows if r["relation"].endswith("_reason")]
    assert report.skipped[REASON_NOT_A_CODE] == 1
    (rat,) = [r for r in rows if r["relation"] == "rationale:gap"]
    assert rat["label"] == f"{SENTENCE}\n{WHY}"
    assert "(" not in rat["raw_text"], "a sentence never rides in the prompt"
    # The verdict itself still trains.
    assert ("gap_valid", "invalid") in {(r["relation"], r["label"]) for r in rows}


def test_a_code_from_another_head_is_not_a_code_here():
    assert is_reason_code("gap", "answered") and not is_reason_code("site", "answered")
    rows, report = _ingest({"head": "site", "target_key": "s1", "verdict": "distinct_site",
                            "labeler": "a@example.com", "text": "Springfield, IL / Shelbyville, IL",
                            "reason": "answered", "note": ""})
    assert not [r for r in rows if r["relation"].endswith("_reason")]
    assert report.skipped[REASON_NOT_A_CODE] == 1


def test_a_sentence_alone_still_reaches_the_rationale():
    long_sentence = "Two different towns in two different states, so these are two separate sites."
    rows, _ = _ingest({"head": "site", "target_key": "s1", "verdict": "distinct_site",
                       "labeler": "a@example.com", "text": "Springfield, IL / Shelbyville, KY",
                       "reason": long_sentence})
    (rat,) = [r for r in rows if r["relation"] == "rationale:site"]
    assert rat["label"] == long_sentence


def test_note_with_reason_keeps_the_marker_first_and_never_repeats():
    assert _note_with_reason("", SENTENCE) == SENTENCE
    assert _note_with_reason(WHY, SENTENCE) == f"{SENTENCE}\n{WHY}"
    # Already in the note (a bulk accept once wrote the reason there): unchanged.
    both = f"{SENTENCE}  {WHY}"
    assert _note_with_reason(both, SENTENCE) == both
    marked = "[EXCLUDE_FROM_TRAINING: old manual kit]\nA rate row."
    assert _note_with_reason(marked, SENTENCE) == f"[EXCLUDE_FROM_TRAINING: old manual kit]\n{SENTENCE}\nA rate row."
    bare = "EXCLUDE_FROM_TRAINING: old manual kit.\nA rate row."
    assert _note_with_reason(bare, SENTENCE) == f"EXCLUDE_FROM_TRAINING: old manual kit.\n{SENTENCE}\nA rate row."
    assert _note_with_reason("[EXCLUDE_FROM_TRAINING: kit]", SENTENCE) == f"[EXCLUDE_FROM_TRAINING: kit]\n{SENTENCE}"


def test_a_proposal_never_trains_whatever_its_reason():
    rows, report = _ingest(_gap(reason="answered", note=WHY, labeler="drafter (assistant)"))
    assert rows == []
    assert REASON_NOT_A_CODE not in report.skipped


def test_the_file_is_plain_json_with_heads():
    data = json.loads(json.dumps(load_reasons()))
    assert set(data) >= {"version", "doc", "heads"}
    assert codes_for("not_a_head") == frozenset()
