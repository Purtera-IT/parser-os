"""Labelling alone cannot train a parser rule. This is why, and what fixes it.

A SemanticRule decides, while a document is being cut up, whether a line
becomes an atom and what it belongs to. The labels we collect are about atoms
that already exist. Two questions, two moments — and they meet at one point:
accepting or rejecting an atom also says whether the decision that made it was
right.

The measurement that forced this module. Joining the labels ALONE to the
`list_item_under_label` rule, across both labelled deals:

    010180: 54 atoms the rule created, 54 labelled by us
    010288: 11 atoms the rule created, 11 labelled by us
    -> 65 rows, 65 of them positive, 0 negative

That is structural. An atom exists only where the rule FIRED, so a line it
wrongly skipped leaves nothing on screen to label and no negative to learn
from. A threshold fitted on 65 positives and no negatives collapses to zero and
fires on everything — the rule would admit every line in the corpus. It is the
same thing the corpus loss audit reports every run: "40 of 65 atoms carry no
`rejected`, so they teach a point and not a boundary."

So the log supplies the candidates INCLUDING the ones that got away, and the
labels supply the truth. Neither half works alone.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.learning.rule_feedback import (join, read_decisions, truth_from_labels,
                                        write_labelled)

RULE = "list_item_under_label"


def _log(rows) -> Path:
    path = Path(tempfile.mkdtemp()) / "rules.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _d(text, fired, **kw):
    return {"rule": RULE, "text": text, "best_pos": kw.get("pos", 0.7),
            "best_neg": kw.get("neg", 0.5), "threshold": 0.62, "decision": fired}


#: All four shapes, taken from 010180.
DECISIONS = [
    _d("Board room", True),                                    # right to fire
    _d("as: cial Leasi", True),                                # fired, made junk
    _d("Erick offered a Cisco-funded wireless site survey.", False),  # should have
    _d("Thanks, let me know if you need anything else.", False),      # right to skip
]
LABELS = [
    {"text": "Board room", "rejected": False},
    {"text": "as: cial Leasi", "rejected": True},
    {"text": "Erick offered a Cisco-funded wireless site survey.", "rejected": False},
]


def _joined():
    return list(join(read_decisions(_log(DECISIONS)), truth_from_labels(LABELS)))


def test_a_rejected_atom_is_the_negative_labelling_cannot_give():
    """The whole point. `as: cial Leasi` is wreckage a reviewer rejected, and
    it is the only kind of row that teaches the rule where NOT to fire."""
    rows = _joined()
    neg = [r for r in rows if r["label"] == 0]
    assert [r["text"] for r in neg] == ["as: cial Leasi"]


def test_a_kept_atom_is_a_positive():
    rows = _joined()
    assert any(r["text"] == "Board room" and r["label"] == 1 for r in rows)


def test_a_line_the_rule_skipped_that_survived_anyway_is_a_positive():
    """It did not fire, and the words are on an atom a reviewer kept — so
    something else rescued the line and the rule should have fired. This row
    exists ONLY because the log records the no-fires."""
    rows = _joined()
    row = next(r for r in rows if r["text"].startswith("Erick offered"))
    assert row["decision"] is False
    assert row["label"] == 1


def test_a_silent_skip_is_left_unlabelled():
    """It did not fire and no atom carries the words. We cannot tell a correct
    skip from a silent loss, and guessing would manufacture exactly the
    negatives this whole exercise is short of."""
    rows = _joined()
    assert not any(r["text"].startswith("Thanks, let me know") for r in rows)


def test_a_reject_beats_an_accept_on_the_same_words():
    """Two labels on one string, one of them a reject, means somebody looked
    and said no."""
    truth = truth_from_labels([{"text": "Huddle rooms", "rejected": False},
                               {"text": "Huddle rooms", "rejected": True}])
    assert truth["Huddle rooms"] is False


def test_the_written_rows_are_what_the_trainer_reads():
    """`_train_semantic_rules.py` takes rows with `rule`, `text` and a
    ground-truth `label` of 0 or 1, one JSON object per line."""
    out = Path(tempfile.mkdtemp()) / "labelled.jsonl"
    n = write_labelled(_joined(), out)
    assert n == 3
    for line in out.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert rec["rule"] == RULE
        assert rec["text"]
        assert rec["label"] in (0, 1)


def test_no_log_means_no_rows_rather_than_a_crash():
    assert read_decisions(Path(tempfile.mkdtemp()) / "absent.jsonl") == []
