"""One sentence, several statements: split at the clause boundary (010003).

"But we are waiting for tv to arrive at their office (it is with the shipping
carrier now), so I also need to keep my eye on the delivery status." stayed one
atom: the delivery fact and the writer's own status-watching remark, typed as
one thing.
"""

from __future__ import annotations

from app.core.sentences import split_compound_clauses, split_trigger_clause
from app.parsers.email_parser import _expand_lines_to_sentences

TV = (
    "But we are waiting for tv to arrive at their office (it is with the shipping "
    "carrier now), so I also need to keep my eye on the delivery status."
)


def _texts(lines: list[str]) -> list[str]:
    return [t for _, _, t in _expand_lines_to_sentences(lines, 1)]


def test_the_tv_sentence_is_two_statements() -> None:
    assert _texts([TV]) == [
        "But we are waiting for tv to arrive at their office (it is with the shipping carrier now)",
        "so I also need to keep my eye on the delivery status.",
    ]


def test_every_piece_is_source_text_verbatim() -> None:
    for piece in split_compound_clauses(TV):
        assert piece in TV


def test_semicolon_and_and_i_also_are_boundaries() -> None:
    s = (
        "We are waiting for the TV to arrive at their office; it is with the shipping "
        "carrier now and I also need to keep my eye on the delivery status."
    )
    assert split_compound_clauses(s) == [
        "We are waiting for the TV to arrive at their office",
        "it is with the shipping carrier now",
        "I also need to keep my eye on the delivery status.",
    ]


def test_multi_sentence_run_still_splits_per_sentence() -> None:
    out = _texts([
        "But we are waiting for tv to arrive at their office (it is with the shipping",
        "carrier now). I also need to keep my eye on the delivery status.",
    ])
    assert out == [
        "But we are waiting for tv to arrive at their office (it is with the shipping carrier now).",
        "I also need to keep my eye on the delivery status.",
    ]


def test_lists_and_addresses_stay_whole() -> None:
    for s in (
        "Please ship to 40 10th Ave; Fl 4; New York, NY 10014.",
        "Rack A; Rack B; the IDF closet; and the MDF.",
        "Cat6 cable, patch panels, and wall plates for all rooms.",
    ):
        assert split_compound_clauses(s) == [s]


def test_short_sides_and_parentheticals_stay_whole() -> None:
    assert split_compound_clauses("We need cables and I also want labels.") == [
        "We need cables and I also want labels."
    ]
    s = "The carrier says Friday (we checked; it is on the truck) for the delivery window."
    assert split_compound_clauses(s) == [s]


def test_trigger_clause_split_still_works_and_composes() -> None:
    s = "We are waiting for the TVs to arrive, once they are delivered we will schedule the install."
    assert split_trigger_clause(s) == [
        "We are waiting for the TVs to arrive",
        "Once they are delivered we will schedule the install.",
    ]
