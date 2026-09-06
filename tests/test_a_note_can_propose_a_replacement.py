"""A PM who says what to ask INSTEAD must get it asked.

Live, 2026-09-05, the PM's own note on deal 010300:

    "So above this you ask who the site POC — the site POC is the same person
     who will sign, so these are basically redundant questions, so just wrap
     them in one: who's site POC who will do X Y and Z"

Two things said. Only the first was learned: the acceptance ask was suppressed
and nothing took its place, so the next brief loses a question and gains
nothing. The PM watches their own replacement never appear.

Grammar separates them — a proposal is interrogative, a reason is not — and the
card decides where the subject ends.
"""

from __future__ import annotations

from app.core.pm_note_router import route_note, split_clause

NO_LLM = lambda *a, **k: None  # noqa: E731 - the pattern path is what is under test

ACCEPTANCE = (
    "Who signs site acceptance at 2970 brandywine rd ste 200, "
    "and what is the pass/fail checklist?"
)
CONTACT = (
    "Who is the day-of onsite contact for 2970 brandywine rd ste 200, "
    "and how do we reach them?"
)
NOTE = (
    "Stop asking who signs site acceptance — the site POC is the same person who will sign, "
    "so these are redundant: who's the site POC who will do access, escort and sign-off?"
)


def _by_verdict(lessons):
    return {l.new_value: l for l in lessons}


def test_one_note_both_rejects_and_proposes() -> None:
    lessons = route_note(NOTE, questions=[ACCEPTANCE, CONTACT], synthesize=NO_LLM).lessons
    by = _by_verdict(lessons)
    assert set(by) == {"invalid", "valid"}, "rejecting without replacing loses a question"
    assert by["invalid"].exemplar == ACCEPTANCE
    assert "site POC" in by["valid"].exemplar


def test_the_proposal_is_marked_as_needing_authoring() -> None:
    """It grounds to no existing card, so somebody has to create it."""
    by = _by_verdict(route_note(NOTE, questions=[ACCEPTANCE, CONTACT], synthesize=NO_LLM).lessons)
    assert by["valid"].source == "proposal"


def test_a_proposal_that_already_exists_reinforces_it_instead() -> None:
    """Authoring a duplicate of a live ask is how a shortlist doubles. When the
    replacement IS a card already on screen, that card is the survivor."""
    note = (
        "Stop asking who signs site acceptance — redundant: "
        "who is the day-of onsite contact for 2970 brandywine rd ste 200, and how do we reach them?"
    )
    by = _by_verdict(route_note(note, questions=[ACCEPTANCE, CONTACT], synthesize=NO_LLM).lessons)
    assert by["valid"].exemplar == CONTACT, "learn on the real card, not the paraphrase"
    assert by["valid"].source == "proposal_existing"


def test_a_plain_reason_is_still_only_a_reason() -> None:
    """The common case must not grow a phantom question."""
    note = "Stop asking who signs site acceptance — Carl Painter signs every one of them"
    lessons = route_note(note, questions=[ACCEPTANCE], synthesize=NO_LLM).lessons
    assert [l.new_value for l in lessons] == ["invalid"]
    assert lessons[0].rationale == "Carl Painter signs every one of them"


def test_the_reason_and_the_proposal_are_separated() -> None:
    subject, reason, proposed = split_clause(NOTE, [ACCEPTANCE, CONTACT])
    assert subject.startswith("Stop asking who signs")
    assert reason.startswith("the site POC is the same person")
    assert proposed.lower().startswith("who's the site poc")
    assert "redundant" not in proposed, "the reason must not leak into the ask"


def test_the_proposal_carries_the_same_reason() -> None:
    """Both halves came from one sentence; the why explains both."""
    by = _by_verdict(route_note(NOTE, questions=[ACCEPTANCE, CONTACT], synthesize=NO_LLM).lessons)
    assert by["valid"].rationale == by["invalid"].rationale != ""


def test_a_fragment_is_not_a_question() -> None:
    _s, _r, proposed = split_clause(
        "Stop asking who signs site acceptance — who?", [ACCEPTANCE]
    )
    assert proposed == "" or len(proposed) < 12
