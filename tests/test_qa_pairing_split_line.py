"""A line the parser split into four atoms still has to pair.

010288 carries this as ONE source line:

    "Has the door been installed with the lock? Do we know the type of lock? -
    Defer to client - my understanding is client was working with installer for
    this, I am not sure if it is already installed/in place/etc. They are
    intending to use a maglock."

Two questions, two answers, four atoms, and every existing rule misses it:

  * the inline rule only fires while the answer is still inside the question's
    own text, and the split already took it away;
  * the next-line rule wants a strictly later line and breaks at the first
    question it meets, so the first question breaks on the second;
  * word overlap cannot help -- the answer to "what type of lock" is "a
    maglock", and `maglock` is not the token `lock`.

The sibling bullet on the same email, "How many doors - 1 external access point
[front door]", stayed in one atom and paired fine. Same shape in the document,
two different outcomes, decided by where the parser happened to cut.
"""
from __future__ import annotations

from app.core.qa_pairing import pair_within_one_line


class _Atom:
    def __init__(self, text: str, line: int = 116, kind: str = "email_body_line") -> None:
        self.id = "atm_" + text[:12]
        self.artifact_id = "art_email"
        self.raw_text = text
        self.atom_type = "open_question" if text.rstrip().endswith("?") else "scope_item"
        self.value = {"kind": kind, "line": line, "message_index": 2}
        self.locator = {"line_start": line, "message_index": 2}
        self.review_flags: list[str] = []


Q1 = "Has the door been installed with the lock?"
Q2 = "Do we know the type of lock?"
A1 = ("Defer to client - my understanding is client was working with installer "
      "for this, I am not sure if it is already installed/in place/etc.")
A2 = "They are intending to use a maglock."


def _line() -> list[_Atom]:
    return [_Atom(Q1), _Atom(Q2), _Atom(A1), _Atom(A2)]


def test_two_questions_and_two_answers_pair_in_order():
    atoms = _line()
    assert pair_within_one_line(atoms) == 2

    q1, q2, a1, a2 = atoms
    assert q1.value["answered"] is True
    assert q1.value["proposed_type"] == "answered_question"
    assert q1.value["answer"] == A1
    assert q2.value["answer"] == A2
    assert a1.value["answers_question"] == Q1
    assert a2.value["answers_question"] == Q2

    # ONE atom per pair, the way the unsplit form already is. Line 108 of this
    # same email is "How many doors - 1 external access point [front door]" --
    # a single atom, a single row. Leaving the answer standing as well would
    # put the same words on the card twice: once inside the question, once as a
    # loose line with nothing visibly tying it back.
    assert q1.raw_text == f"{Q1} - {A1}"
    assert q2.raw_text == f"{Q2} - {A2}"
    assert a1.value["absorbed_into"] == q1.id
    assert a2.value["absorbed_into"] == q2.id


def test_the_maglock_pair_is_the_one_word_overlap_could_not_find():
    """`lock` and `maglock` share no token, so the pairing must not depend on
    vocabulary. Order is what the line gives."""
    from app.core.qa_pairing import _answers_this_question

    q, a = _Atom(Q2), _Atom(A2)
    assert not _answers_this_question(q, a), "guard: overlap really does fail here"
    atoms = _line()
    assert pair_within_one_line(atoms) == 2
    assert atoms[1].value["answer"] == A2
    assert atoms[1].raw_text.endswith(A2)


def test_interleaved_is_left_alone():
    """"Q A Q A" is not the shape this rule claims. Guessing which half answers
    which is the judgement a head should learn, not a rule."""
    atoms = [_Atom(Q1), _Atom(A1), _Atom(Q2), _Atom(A2)]
    assert pair_within_one_line(atoms) == 0
    assert not atoms[0].value.get("answered")


def test_lopsided_is_left_alone():
    """Two questions and one statement says nothing about which it answers."""
    atoms = [_Atom(Q1), _Atom(Q2), _Atom(A2), _Atom("Thanks.")]
    # "Thanks." is one word, so it is not an answer and the counts no longer match
    assert pair_within_one_line(atoms) == 0


def test_a_single_question_on_its_line_is_left_to_the_inline_rule():
    atoms = [_Atom("How many doors?"), _Atom("1 external access point")]
    assert pair_within_one_line(atoms) == 0


def test_separate_lines_are_not_one_line():
    """The rule is about ONE source line. Atoms from different lines are the
    next-line rule's business, and pairing them here would reach across the
    document on nothing but list order."""
    atoms = [_Atom(Q1, line=116), _Atom(Q2, line=117),
             _Atom(A1, line=118), _Atom(A2, line=119)]
    assert pair_within_one_line(atoms) == 0
