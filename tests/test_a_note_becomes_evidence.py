"""A grounded note states deal truth, and deal truth is an atom.

The user's argument, and it is the right one: the questions the brief asks
exist to get information about the deal. So information that arrives by note is
the same evidence as information that arrives by answer.

The only thing that ever separated them was provenance — an answer has a card
it is an answer TO, and a free-text note had nothing. Grounding supplies it: a
note is now matched to the exact question it was written about, so a grounded
note carries a question and a fact, exactly as an answer does.

    "Stop asking who stages the hardware because when Patrick Kelly owns the
     deal we stage it ourselves from the depot."

  grounded question: "What hardware is customer-furnished vs PurTera-furnished
                      — and who stages it to site?"
  the fact:          "when Patrick Kelly owns the deal we stage it ourselves
                      from the depot"

That is an answer in everything but the button pressed.
"""

from __future__ import annotations

from app.core.pm_answer_atoms import pm_answer_to_atom, pm_answers_to_atoms
from app.core.schemas import AuthorityClass, ReviewStatus

QUESTION = "What hardware is customer-furnished vs PurTera-furnished — and who stages it to site?"
FACT = "when Patrick Kelly owns the deal we stage it ourselves from the depot"


def _note_row(**over):
    row = {
        "action": "note_fact",
        "deal_id": "d1",
        "rule_id": "pmcover.hardware_furnish",
        "question_text": QUESTION,
        "edited_text": FACT,
        "created_at": "2026-09-05T19:30:00Z",
        "actor": "developer@purtera-it.com",
    }
    row.update(over)
    return row


def test_a_grounded_note_becomes_an_atom() -> None:
    atoms = pm_answers_to_atoms([_note_row()], project_id="p1")
    assert len(atoms) == 1
    a = atoms[0]
    assert FACT in a.claim
    assert "who stages it to site" in a.claim


def test_it_carries_the_same_authority_as_an_answer() -> None:
    """A PM stating deal truth is a PM stating deal truth. If a note landed at
    lower authority it would lose to the stale document line it was written to
    correct, which is the whole reason for saying it."""
    (a,) = pm_answers_to_atoms([_note_row()], project_id="p1")
    assert a.authority_class is AuthorityClass.pm_confirmed
    assert a.review_status is ReviewStatus.approved
    assert a.confidence == 0.95


def test_the_atom_says_it_came_from_a_note() -> None:
    """Same weight, honest provenance: a reader must be able to walk it back to
    the surface the PM actually used."""
    (a,) = pm_answers_to_atoms([_note_row()], project_id="p1")
    assert a.value["kind"] == "pm_note"
    assert a.source_refs[0].locator["kind"] == "pm_note"
    assert a.source_refs[0].locator["rule_id"] == "pmcover.hardware_furnish"


def test_an_answer_still_says_it_was_an_answer() -> None:
    (a,) = pm_answers_to_atoms(
        [_note_row(action="answered", edited_text="PurTera stages them.")], project_id="p1"
    )
    assert a.value["kind"] == "pm_answer"


def test_a_note_with_no_fact_in_it_is_not_evidence() -> None:
    """A note that only says "stop asking" carries no deal truth. The gap head
    still learns from it; evidence it is not."""
    assert pm_answers_to_atoms([_note_row(edited_text="")], project_id="p1") == []
    assert pm_answers_to_atoms([_note_row(edited_text="n/a")], project_id="p1") == []


def test_a_judgment_is_not_evidence() -> None:
    """dismiss / wrong_for_project state a view about the QUESTION, not a fact
    about the deal."""
    for action in ("dismiss", "wrong_for_project", "edit", "add"):
        assert pm_answers_to_atoms([_note_row(action=action)], project_id="p1") == []


def test_a_note_and_an_answer_on_one_question_do_not_stack() -> None:
    """Both are the PM's position on the same ask; the later one is current."""
    rows = [
        _note_row(action="answered", edited_text="Customer furnishes them.",
                  created_at="2026-09-05T10:00:00Z"),
        _note_row(created_at="2026-09-05T19:30:00Z"),
    ]
    atoms = pm_answers_to_atoms(rows, project_id="p1")
    assert len(atoms) == 1, "one question, one current position"
    assert FACT in atoms[0].claim


def test_the_kind_travels_through_the_single_atom_builder_too() -> None:
    a = pm_answer_to_atom(project_id="p1", question=QUESTION, answer=FACT, kind="pm_note")
    assert a is not None and a.value["kind"] == "pm_note"
