"""Which words decided the label -- and which of a labeler's pointers is a span.

Every test here is a defect that was live. The module was written in one pass
and measured on 010288's 159 pointers, and the measurement found four things
wrong with it in a row. These pin all four.
"""
from __future__ import annotations

from app.learning.span_ranker import (
    MIN_OVERLAP, candidates_for, features_for, fit, pointer_kind, rank,
    training_pairs, FEATURE_NAMES,
)


def _atom(text: str, **kw):
    return {"text": text, "hint_refs": [], **kw}


def _ref(text: str, hint: str = "own_words", **kw):
    return {"text": text, "hint": hint, **kw}


# --------------------------------------------------------------------------
# Which head can learn a pointer at all
# --------------------------------------------------------------------------

def test_the_envelope_is_not_a_span():
    """`who_said_it` renders the sender, the affiliation and the direction. No
    amount of reading the page finds that text on it. 34 of 010288's 159
    pointers were this, and every one was being emitted for a span head to
    locate."""
    assert pointer_kind(_ref("alec@cdw.com (reseller, theirs) -> aj@ours",
                             "who_said_it")) == "field"


def test_provenance_is_not_a_span():
    assert pointer_kind(_ref("read off the drawing itself: OCR for the words",
                             "doc_type")) == "field"


def test_a_pointer_at_another_atom_is_retrieval_not_extraction():
    """Real evidence, real words -- on a surface the span head is not holding.
    Asking it to produce text that is not in front of it teaches it to
    invent."""
    assert pointer_kind(_ref("Mag/Electric Lock", "other_doc",
                             atomId="atom_9")) == "other_atom"


def test_words_in_this_atom_are_a_span():
    assert pointer_kind(_ref("but not the relay to the lock")) == "span"


def test_only_span_pointers_are_scored():
    """Otherwise a head that correctly declines to guess at the envelope is
    recorded as having missed."""
    atom = _atom("Mag Lock Cable",
                 hint_refs=[_ref("aj@purtera-it.com (ours)", "who_said_it")])
    assert not any(t for _, t in training_pairs(atom))


# --------------------------------------------------------------------------
# One pointer claims one candidate
# --------------------------------------------------------------------------

def test_a_neighbour_that_quotes_the_atom_does_not_claim_its_gold():
    """The first rule was containment, and the neighbour above the exclusion
    quotes it in parentheses -- so it contained the gold span and was marked
    gold itself. Four candidates, four positives, nothing to rank."""
    atom = _atom(
        "we provide the parts that connect the PC to the relay, "
        "but not the relay to the lock",
        neighbors_above=["The club/installer will need to source anything "
                         "beyond the Relay (we provide the parts that connect "
                         "the PC to the relay, but not the relay to the lock)"],
        hint_refs=[_ref("but not the relay to the lock")],
    )
    gold = [c for c, t in training_pairs(atom) if t]
    assert len(gold) == 1
    assert gold[0].source == "own_words"
    assert gold[0].text == "but not the relay to the lock"


def test_a_long_selection_is_still_the_same_span():
    """A threshold strict enough to stop the neighbour rejected a genuine
    140-character selection out of a 248-character line, for being 56% of it.
    An argmax needs no threshold."""
    body = ("Additionally I would note that the connectivity kit limits our "
            "power to 40-50ft (we're working on a better version with more "
            "distance) but we can extend the data as far as needed using the "
            "RS232 extenders provided (within realistic reason, anyway).")
    atom = _atom(body, hint_refs=[_ref(body[:140])])
    assert any(t for _, t in training_pairs(atom))


def test_the_whole_atom_is_a_candidate_however_long_it_runs():
    """MAX_SPAN caps generated clause fragments. Applied to the body it cost
    two atoms on 010288 every candidate they had."""
    body = "x" * 400 + " and then some words about the relay and the lock"
    assert any(c.text == body for c in candidates_for(_atom(body)))


def test_a_list_is_several_answers():
    """"PC with Access Control Software, USB Cable, RS232 to USB converter and
    Relay" was one candidate, and the labeler had pointed at the first item."""
    atom = _atom("Claimed by both: PC with Access Control Software, USB Cable, "
                 "RS232 to USB converter and Relay.")
    assert "PC with Access Control Software" in {c.text for c in candidates_for(atom)}


def test_prose_about_the_document_matches_nothing():
    """Below MIN_OVERLAP the labeler was writing, not selecting."""
    atom = _atom("Mag Lock Cable",
                 hint_refs=[_ref("this was decided by an entirely different "
                                 "consideration nobody wrote down")])
    assert not any(t for _, t in training_pairs(atom))


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def test_features_and_names_stay_in_step():
    atom = _atom("Mag Lock Cable", lead_in=["Provided by Club/installer:"])
    assert len(features_for(candidates_for(atom)[0], atom)) == len(FEATURE_NAMES)


def test_a_silent_line_is_decided_by_the_heading_above_it():
    """The whole reason context candidates exist. A pointwise model scored 85%
    and lost every one of these: "Mag Lock Cable" is a part name, and a global
    prior that the atom's own words win is right across the corpus and wrong
    here. Subtracting two candidates of the same atom cancels the prior."""
    bom = [
        _atom(part, lead_in=["Provided by Club/installer:"],
              section=["Provided by Club/installer"], weight_tier="ordinary",
              hint_refs=[_ref("Provided by Club/installer", "section")])
        for part in ("Mag Lock Cable", "Mag/Electric Lock",
                     "Power Supply for mag lock", "Door Contact")
    ]
    speech = [
        _atom("we provide the parts but not the relay to the lock",
              lead_in=["Provided by Club/installer:"], weight_tier="ordinary",
              hint_refs=[_ref("but not the relay to the lock")]),
        _atom("The club will need to source anything beyond the Relay",
              lead_in=["Provided by Club/installer:"], weight_tier="ordinary",
              hint_refs=[_ref("will need to source anything beyond the Relay")]),
    ]
    weights = fit(bom + speech)
    top = rank(bom[0], weights)[0][0]
    assert "Provided by Club/installer" in top.text

    # ...and a line that does speak keeps its own words.
    assert rank(speech[0], weights)[0][0].source == "own_words"


def test_an_atom_with_nothing_to_rank_is_not_trained_on():
    """A one-line question is its own answer: one candidate, and it is right.
    Counting it would inflate the score with atoms that cannot be got wrong."""
    atom = _atom("Where is this site located?",
                 hint_refs=[_ref("Where is this site located?")])
    pairs = training_pairs(atom)
    assert len(pairs) == 1 and all(t for _, t in pairs)
    try:
        fit([atom])
    except ValueError:
        return
    raise AssertionError("fit() should refuse a corpus with no wrong candidate")


def test_overlap_threshold_is_the_documented_one():
    assert 0.0 < MIN_OVERLAP < 1.0
