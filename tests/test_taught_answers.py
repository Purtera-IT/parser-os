"""A question somebody has already answered must stop being asked.

`resolve_open_questions` closes a question when a fact elsewhere in the corpus
shares an answer-bearing entity key with it. That catches "how many drops?"
answered by an atom carrying `count:212`. It is blind to an answer phrased in
words the question does not use, and those are most answers.

Live 010180, one thread, three messages:

    "Can you try listening to the recording below, and see if we can get
     budgetary numbers together?"          -> answered=True  (next line)
    "Any chance you have another way of sharing the recording?"
                                           -> needs_review, still
    "I think the best bet is reviewing the notes I sent over, that has
     everything."

The second is still put to a PM on every compile, two months after the
customer withdrew it. "The notes I sent over, that has everything" shares not
one entity key with "the recording".

A labeler had already said so on the first pass -- one of 15 `answers` edges on
this deal. Labels forward to the store, judgments forward, links did not, so
those edges reached Postgres and the training blob and no compile at all.

Guess-free, and deliberately stricter than the rest: only a STORE hit closes a
question. A model that wrongly closes one produces a fact nobody ever chases,
which is the failure this entire deal has been an exercise in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.core import taught_answers as TA


@dataclass
class _Atom:
    raw_text: str
    atom_type: str = "open_question"
    value: dict = field(default_factory=dict)
    review_flags: list = field(default_factory=list)
    review_status: Any = None


RECORDING = "Any chance you have another way of sharing the recording?"


class _D:
    def __init__(self, verdict, source="store"):
        self.verdict = verdict
        self.source = source
        self.confidence = 0.9


def _patch(monkeypatch, decision, *, store=True):
    import app.core.decide as D

    monkeypatch.setattr(D, "get_store", lambda: object() if store else None, raising=False)
    monkeypatch.setattr(D, "decide", lambda *a, **k: decision, raising=False)


def test_a_taught_answer_closes_the_question(monkeypatch):
    _patch(monkeypatch, _D("answered"))
    atom = _Atom(RECORDING)
    assert TA.resolve_taught_answers([atom], project_id="d1") == 1
    assert atom.value["answered"] is True
    assert atom.value["answered_by"] == "teacher"
    assert TA.TAUGHT_FLAG in atom.review_flags
    # Same flag key-overlap writes, so no consumer has to know which route ran.
    assert "answered_in_corpus" in atom.review_flags


def test_a_model_guess_does_not(monkeypatch):
    """Stricter than the other taught seams on purpose. Wrongly closing a
    question produces a fact nobody ever chases."""
    _patch(monkeypatch, _D("answered", source="model"))
    atom = _Atom(RECORDING)
    assert TA.resolve_taught_answers([atom], project_id="d1") == 0
    assert atom.value == {}


@pytest.mark.parametrize("verdict", ["open", None, ""])
def test_anything_but_answered_leaves_it_open(monkeypatch, verdict):
    _patch(monkeypatch, _D(verdict))
    atom = _Atom(RECORDING)
    assert TA.resolve_taught_answers([atom], project_id="d1") == 0
    assert atom.review_flags == []


def test_only_questions_are_considered(monkeypatch):
    _patch(monkeypatch, _D("answered"))
    atom = _Atom("Two drops per workstation.", atom_type="quantity")
    assert TA.resolve_taught_answers([atom], project_id="d1") == 0


def test_a_question_key_overlap_already_closed_is_left_alone(monkeypatch):
    """No second opinion on a settled question, and no second store call."""
    calls = []
    import app.core.decide as D

    monkeypatch.setattr(D, "get_store", lambda: object(), raising=False)
    monkeypatch.setattr(D, "decide", lambda *a, **k: calls.append(1) or _D("answered"),
                        raising=False)
    atom = _Atom(RECORDING, value={"answered": True})
    assert TA.resolve_taught_answers([atom], project_id="d1") == 0
    assert calls == []


def test_no_store_is_a_no_op(monkeypatch):
    _patch(monkeypatch, _D("answered"), store=False)
    atom = _Atom(RECORDING)
    assert TA.resolve_taught_answers([atom], project_id="d1") == 0


def test_the_vocabulary_is_the_one_the_labeller_forwards():
    """platform-labeling posts head `question_answered` with these two
    candidates when an `answers` edge is saved. A mismatch here means the edge
    is stored under a relation nothing reads -- the `_reject` mistake again."""
    assert TA.RELATION == "question_answered"
    assert TA.CANDIDATES == ("answered", "open")
