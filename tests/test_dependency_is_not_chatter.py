"""A wait or a dependency is never small talk (010003).

"But we are waiting for tv to arrive at their office (it is with the
shipping carrier now). I also need to keep my eye on the delivery status."
The first sentence says delivery gates the install; it must stay a normal,
non-chatter atom whatever chatty clause rides with it.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.core.deal_chatter import is_chatter, mark_chatter
from app.core.sentences import sentence_kind, split_by_kind

LINE = ("But we are waiting for tv to arrive at their office (it is with the shipping carrier now). "
        "I also need to keep my eye on the delivery status.")
WAIT = "But we are waiting for tv to arrive at their office (it is with the shipping carrier now)."


def _atom(text):
    return SimpleNamespace(atom_type="deal_metadata", raw_text=text, value={}, review_flags=[], entity_keys=[])


def test_the_010003_line_is_not_chatter():
    assert not is_chatter(LINE)
    assert not is_chatter(WAIT)
    assert sentence_kind(WAIT) == "work"
    atom = _atom(LINE)
    assert mark_chatter([atom]) == 0
    assert "chatter" not in atom.review_flags


def test_a_wait_never_reads_as_chatter_even_beside_a_handoff():
    for text in (
        "We are waiting on the carrier, looping in Sean.",
        "Install is on hold until the PO lands, looking forward to it.",
        "The TV is in transit, will be in touch.",
        "Still pending delivery -- sending it over once it arrives.",
    ):
        assert not is_chatter(text), text
        atom = _atom(text)
        mark_chatter([atom])
        assert "chatter" not in atom.review_flags, text
    # Plain relationship talk is still flagged.
    assert is_chatter("Thank you for bringing this our way")


def test_so_clause_splits_a_fact_from_process_talk():
    assert split_by_kind("The TV is with the shipping carrier now, so I am looping in Sean.") == [
        "The TV is with the shipping carrier now,", "so I am looping in Sean.",
    ]
    # Same kind on both sides: whole.
    assert split_by_kind("The racks arrived, so we need 40 drops pulled Monday.") == []
