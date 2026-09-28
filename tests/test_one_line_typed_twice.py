"""A question and a statement can be ONE line, typed twice.

`cross_type_dedup_atoms` lets `open_question` and `decision` pass through
untouched, because a question is a distinct speech act: "MDF badge access?"
and a constraint saying "MDF badge access" are two things, and collapsing them
drops the only type that drives the missing_info packet.

That holds when they come from different places. Live 010180 showed what
happens when they do not.
"""
from app.core.semantic_dedup import cross_type_dedup_atoms

PATRICK = ("Would we need to have a walkthrough on the site? For 010180. This one "
           "we put budgetary numbers together for. Could do a site survey charge "
           "them a few hundred bucks and if they go with us just take it out of "
           "the total price of the cabling job?")


class _Ref:
    def __init__(self, filename, msg=0, line=1):
        self.filename = filename
        self.locator = {"message_index": msg, "line_start": line, "line_end": line,
                        "sender": "patrick@purtera-it.com"}


class _Atom:
    def __init__(self, atom_type, text, filename="mail.eml", msg=0, line=1):
        self.atom_type = atom_type
        self.raw_text = self.text = text
        self.confidence = 0.775
        self.artifact_id = "art:" + filename
        self.source_refs = [_Ref(filename, msg, line)]
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def test_one_line_typed_twice_collapses_to_the_question():
    """The real pair off 010180: one email, message 0, line 1, same sender,
    emitted as BOTH constraint and open_question. A PM reading the deal saw
    the same sentence twice."""
    out = cross_type_dedup_atoms([
        _Atom("constraint", PATRICK),
        _Atom("open_question", PATRICK),
    ])
    assert len(out) == 1
    assert out[0].atom_type == "open_question"   # the one that keeps the ask open


def test_the_question_keeps_what_the_statement_carried():
    q = _Atom("open_question", PATRICK)
    c = _Atom("constraint", PATRICK)
    c.entity_keys = ["site:7_penn_plaza"]
    out = cross_type_dedup_atoms([c, q])
    assert len(out) == 1
    assert "site:7_penn_plaza" in (out[0].entity_keys or [])


def test_a_question_elsewhere_still_passes_through():
    """The exception this narrows must survive: a question and a constraint
    from DIFFERENT lines are two things, and folding them would drop the only
    type that drives the missing_info packet."""
    out = cross_type_dedup_atoms([
        _Atom("constraint", "MDF badge access", line=4),
        _Atom("open_question", "MDF badge access?", line=9),
    ])
    assert len(out) == 2


def test_a_question_in_another_document_still_passes_through():
    out = cross_type_dedup_atoms([
        _Atom("constraint", PATRICK, filename="a.eml"),
        _Atom("open_question", PATRICK, filename="b.eml"),
    ])
    assert len(out) == 2


def test_two_questions_on_one_line_are_left_alone():
    """Nothing to choose between them here -- intra-type dedup is
    semantic_dedup's job, not this pass's."""
    out = cross_type_dedup_atoms([
        _Atom("open_question", PATRICK),
        _Atom("open_question", PATRICK),
    ])
    assert len(out) == 2


def test_an_atom_with_no_span_is_untouched():
    """Without a locator there is nothing to prove two atoms share a line, and
    guessing would collapse genuinely separate facts."""
    a, b = _Atom("constraint", PATRICK), _Atom("open_question", PATRICK)
    for x in (a, b):
        x.source_refs[0].locator = {}
    assert len(cross_type_dedup_atoms([a, b])) == 2
