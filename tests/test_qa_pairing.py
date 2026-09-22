"""Live 010289 asked three questions and got three answers, and the parser
filed six unrelated atoms. A question and its answer are one fact."""
from __future__ import annotations

from app.core.qa_pairing import pair_across_thread, pair_questions_with_answers
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _body(line: int, **extra):
    """An atom as the email parser emits it: body prose that knows its line."""
    return {"kind": "email_body_line", "message_index": 0, "line_start": line, **extra}


def _atom(text, atom_type=AtomType.scope_item, value=None, artifact="art_m", aid=None):
    return EvidenceAtom(
        id=aid or f"atm_{abs(hash((text, artifact))) % 10**8}", project_id="p", artifact_id=artifact,
        atom_type=atom_type, raw_text=text, normalized_text=text.lower(),
        value=value if value is not None else {"kind": "email_body_line"}, entity_keys=[],
        source_refs=[SourceRef(id="s1", artifact_id=artifact, artifact_type=ArtifactType.txt, filename="m.eml",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


def test_the_answer_on_the_same_line():
    q = _atom("How many doors - 1 external access point [front door]", AtomType.open_question, _body(10))
    assert pair_questions_with_answers([q]) == 1
    assert q.value["question"] == "How many doors"
    assert q.value["answer"] == "1 external access point [front door]"
    assert q.value["proposed_type"] == "answered_question"


def test_the_answer_on_the_next_line():
    q = _atom("Has the door been installed with the lock?", AtomType.open_question, _body(12))
    a = _atom("Defer to client - my understanding is client was working with installer for this.", value=_body(13))
    assert pair_questions_with_answers([q, a]) == 1
    assert q.value["answer"].startswith("Defer to client")
    assert q.value["answer_source"] == "next_line"
    # the answer keeps its own type and points back: it is scope in its own right
    assert a.atom_type == AtomType.scope_item
    assert a.value["answers_question"] == "Has the door been installed with the lock?"


def test_two_questions_in_a_row_do_not_answer_each_other():
    q1 = _atom("Do we know the type of lock?", AtomType.open_question, _body(12))
    q2 = _atom("Has the door been installed with the lock?", AtomType.open_question, _body(13))
    a = _atom("They are intending to use a maglock.", value=_body(14))
    assert pair_questions_with_answers([q1, q2, a]) == 1
    assert not q1.value.get("answered")
    assert q2.value["answer"] == "They are intending to use a maglock."


def test_the_reply_answers_it_but_only_the_direct_reply_from_the_other_person():
    def th(idx, sender):
        return {"kind": "email_body_line", "email_thread": {"thread_id": "T1", "thread_index": idx, "sender": sender}}

    q = _atom("Where is this site located?", AtomType.open_question, th(1, "aj@purtera-it.com"), artifact="art_1")
    mine = _atom("I will chase them today.", value=th(1, "aj@purtera-it.com"), artifact="art_1")
    reply = _atom("7832 Wisconsin Ave, Bethesda, MD 20814", AtomType.physical_site, th(2, "alec@cdw.com"), artifact="art_2")
    later = _atom("We can start next week.", value=th(3, "alec@cdw.com"), artifact="art_3")
    assert pair_across_thread([q, mine, reply, later]) == 1
    assert q.value["answer"].startswith("7832 Wisconsin Ave")
    assert q.value["answer_source"] == "cross_message"
    # a guess across messages is a proposal, not an assertion
    assert q.review_status == ReviewStatus.needs_review
    assert not later.value.get("answers_question")


def test_a_question_in_message_six_is_not_answered_by_message_two():
    def th(idx, sender):
        return {"kind": "email_body_line", "email_thread": {"thread_id": "T1", "thread_index": idx, "sender": sender}}

    q = _atom("Do we have the floorplan?", AtomType.open_question, th(2, "aj@purtera-it.com"), artifact="art_1")
    far = _atom("The riser room is locked after 6pm.", value=th(6, "alec@cdw.com"), artifact="art_2")
    assert pair_across_thread([q, far]) == 0
    assert not q.value.get("answered")


def test_without_line_numbers_it_does_not_guess():
    """A CRM note carries no line positions. "The next atom in the list" is
    not "the next line of the message": pairing on list order answered the
    wrong question, and paired a minted question with a sign-off."""
    q = _atom("Do we know the type of lock?", AtomType.open_question, {"kind": "hubspot_note_body"})
    a = _atom("They are intending to use a maglock.", value={"kind": "hubspot_note_body"})
    assert pair_questions_with_answers([q, a]) == 0


def test_a_question_the_compiler_minted_about_itself_is_never_paired():
    q = _atom("Referenced inline equipment image (cid:image002.png) could not be read",
              AtomType.open_question, {"kind": "unrecovered_region", "message_index": 0, "line_start": 3})
    a = _atom("Sorry, left that part off.", value=_body(4))
    assert pair_questions_with_answers([q, a]) == 0


def _thr(idx, sender):
    return {"kind": "email_body_line", "email_thread": {"thread_id": "T1", "thread_index": idx, "sender": sender},
            "message_index": 0, "line_start": idx}


def test_a_proposed_answer_has_to_fit_the_question():
    """First live run: "Where is this site located?" was answered with a
    recipient line, and "Has the door been installed?" with "Will be in
    touch." A reply's first sentence is usually not the answer."""
    from app.core.qa_pairing import _answers_this_question

    q_where = _atom("Where is this site located?", AtomType.open_question, _thr(1, "aj@purtera-it.com"))
    for text, atom_type, ok in [
        ("7832 Wisconsin Ave, Bethesda, MD 20814", AtomType.physical_site, True),
        ('"Albert Arzate" <albert@rd-systems.com>', AtomType.scope_item, False),
        ("Will be in touch.", AtomType.scope_item, False),
    ]:
        assert _answers_this_question(q_where, _atom(text, atom_type, _thr(2, "alec@cdw.com"))) is ok, text

    q_lock = _atom("Has the door been installed with the lock?", AtomType.open_question, _thr(1, "aj@purtera-it.com"))
    # no question word we can name: the answer has to talk about the question
    assert _answers_this_question(q_lock, _atom("The door is installed, the lock is not.", value=_thr(2, "a@cdw.com")))
    assert not _answers_this_question(q_lock, _atom("We can start next week.", value=_thr(2, "a@cdw.com")))


def test_a_reply_that_fits_nothing_leaves_the_question_open():
    q = _atom("Where is this site located?", AtomType.open_question, _thr(1, "aj@purtera-it.com"), artifact="art_1")
    reply = _atom('"Albert Arzate" <albert@rd-systems.com>', value=_thr(2, "alec@cdw.com"), artifact="art_2")
    assert pair_across_thread([q, reply]) == 0
    assert not q.value.get("answered")
