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


# ---------------------------------------------------------------------------
# The Rules stage teaches
# ---------------------------------------------------------------------------

class TestAJudgmentIsAWholeTrainingRow:
    """The Rules stage was the one labelling surface that reached nothing.

    Measured 2026-10-01 against the live service: a verdict made there was
    stored in Postgres, refused by /feedback/correction with
    `422 unknown head 'rule'`, and skipped by the nightly retrain. Three ways to
    the trainer and none of them connected.

    It needs no decide() relation -- nothing decides a rule at compile time, the
    thresholds are fitted offline into models/semantic_rule_thresholds.json and
    loaded at rule construction. What it needed was this: the judgment turned
    into the row `_train_semantic_rules._feedback_examples` already reads.
    """

    def _judgment(self, verdict: str, *, rule: str = "list_item_under_label",
                  text: str = "  (2) terminate  both   ends ", fired: bool = True):
        return {
            "head": "rule",
            "verdict": verdict,
            "labeler": "griffin@purtera-it.com",
            "target": {
                "rule": rule, "text": text, "fired": fired,
                "bestPos": 0.61, "bestNeg": 0.54, "threshold": 0.58, "margin": 0.03,
            },
        }

    def test_the_card_already_carries_every_feature_the_trainer_wants(self):
        from app.learning.rule_feedback import rows_from_judgments

        rows = rows_from_judgments([self._judgment("should_fire")])
        assert len(rows) == 1
        r = rows[0]
        # The exact shape `_feedback_examples` filters on: rule, text, label.
        assert r["rule"] == "list_item_under_label"
        assert r["text"] == "(2) terminate both ends", "text is normalised both sides of the join"
        assert r["label"] == 1
        # ...plus the decision's own features, so nothing has to be joined to
        # the log for a judged row.
        assert r["best_pos"] == 0.61 and r["best_neg"] == 0.54
        assert r["threshold"] == 0.58
        assert r["decision"] is True
        assert r["truth_source"] == "judgment"

    def test_should_not_fire_is_the_negative_the_corpus_has_none_of(self):
        from app.learning.rule_feedback import rows_from_judgments

        rows = rows_from_judgments([self._judgment("should_not_fire", fired=False)])
        assert rows[0]["label"] == 0
        # 65 rows from two labelled deals, 65 positive, 0 negative. A threshold
        # fitted on that collapses to zero and the rule admits every line in
        # the corpus. This is the only route to a negative on a line that made
        # no atom.
        assert rows[0]["decision"] is False

    def test_a_verdict_it_does_not_know_is_dropped_not_guessed(self):
        from app.learning.rule_feedback import rows_from_judgments

        assert rows_from_judgments([self._judgment("maybe")]) == []
        assert rows_from_judgments([{"head": "gap", "verdict": "valid"}]) == []
        assert rows_from_judgments([]) == []

    def test_a_target_stored_as_json_text_is_read(self):
        # Postgres hands back jsonb as a dict through the driver, but an export
        # or a backup round-trips it as a string.
        import json as _json
        from app.learning.rule_feedback import rows_from_judgments

        j = self._judgment("should_fire")
        j["target"] = _json.dumps(j["target"])
        assert rows_from_judgments([j])[0]["label"] == 1

    def test_a_rule_card_set_aside_does_not_train_the_threshold(self):
        from app.learning.human_labels import IngestReport
        from app.learning.rule_feedback import judged_index, rows_from_judgments

        for note in ("EXCLUDE_FROM_TRAINING: old manual Deal Kit row.",
                     "[EXCLUDE_FROM_TRAINING: old manual Deal Kit] the rule was right here.",
                     "  exclude_from_training: lower case, leading space"):
            j = {**self._judgment("should_not_fire", fired=False), "note": note}
            report = IngestReport()
            assert rows_from_judgments([j, self._judgment("should_fire")], report) == \
                rows_from_judgments([self._judgment("should_fire")]), note
            assert report.skipped["rule judgment excluded from training"] == 1
            assert judged_index([j]) == {}
        # Other marks human_labels reads as excluded, and a note that only
        # mentions the marker later, behave as they do there.
        assert rows_from_judgments([{**self._judgment("should_fire"), "weight_tier": "exclude"}]) == []
        kept = {**self._judgment("should_fire"), "note": "Kept; not EXCLUDE_FROM_TRAINING."}
        assert len(rows_from_judgments([kept])) == 1

    def test_a_person_overrules_the_inference(self):
        """`join` labels a decision by what became of the atom. A judgment says
        it directly, and where both exist the direct statement wins."""
        from app.learning.rule_feedback import join, judged_index

        text = "(2) terminate both ends"
        decisions = [{"rule": "list_item_under_label", "text": text, "decision": True,
                      "best_pos": 0.61, "best_neg": 0.54, "threshold": 0.58}]
        # The atom was accepted, so inference alone says the rule was right (1).
        inferred = list(join(decisions, {text: True}))
        assert inferred[0]["label"] == 1 and inferred[0]["truth_source"] == "inferred"

        # A person says it should not have fired. That wins.
        judged = judged_index([self._judgment("should_not_fire")])
        overruled = list(join(decisions, {text: True}, judged))
        assert overruled[0]["label"] == 0
        assert overruled[0]["truth_source"] == "judgment"

    def test_it_labels_a_silent_skip_that_inference_leaves_unlabelled(self):
        """The case `join`'s own docstring calls unlabelled: the rule did not
        fire and left nothing behind, so there is no atom to look at."""
        from app.learning.rule_feedback import join, judged_index

        text = "page 3 of 14 confidential"
        decisions = [{"rule": "list_item_under_label", "text": text, "decision": False,
                      "best_pos": 0.55, "best_neg": 0.57, "threshold": 0.58}]
        assert list(join(decisions, {})) == [], "inference cannot label this, by design"

        judged = judged_index([self._judgment("should_not_fire", text=text, fired=False)])
        rows = list(join(decisions, {}, judged))
        assert len(rows) == 1 and rows[0]["label"] == 0
