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


# --- Live 010003 wrote the sentence with NO comma before "so" -------------
# The TV sentence above has ", so"; the real mail (text/plain, CRLF) reads
# "...<aside>) so I also need ...", so the comma-only boundary never fired and
# the atom stayed whole. The body below is synthetic but mirrors the real
# one's shape: a cheer, a blank line, one long paragraph whose last sentence
# follows "month…. ", then a signature and a quoted "From:" line.

NO_COMMA = (
    "But we are waiting for the switches to arrive at their office (they are with the "
    "freight carrier now) so I also need to keep an eye on the tracking status."
)
SYNTH_BODY = (
    "Great, thanks!!\r\n\r\n@Alex Doe<mailto:alex@example.com> and "
    "@Sam Roe<mailto:sam@example.com> \u2013 would you be handling the install? I am "
    "working with the customer, and they are hoping to wrap this up around the end of the "
    "month\u2026. " + NO_COMMA + "\r\n\r\n\r\n\r\nJordan Poe\r\n"
    "Account Manager | Example Services | Example Co\r\n"
    "1 Example Street | Springfield, ST 00000\r\n"
    "Direct: 555.010.0000 | Toll Free Number: 800.555.0100\r\n"
    "[cid:image001.png@01DC0000.00000000] [cid:image002.png@01DC0000.00000000]\r\n\r\n"
    "From: Pat Vendor <pat@vendor.example>\r\n"
)
SYNTH_HEADERS = (
    "From: Jordan Poe <jordan@example.com>\r\n"
    "To: Pat Vendor <pat@vendor.example>\r\n"
    "Subject: RE: switch install\r\n"
    "Date: Thu, 16 Jul 2026 16:31:12 +0000\r\n"
    "MIME-Version: 1.0\r\n"
    'Content-Type: text/plain; charset="utf-8"\r\n'
    "Content-Transfer-Encoding: 8bit\r\n\r\n"
)


def test_email_splits_a_paren_so_sentence_with_no_comma(tmp_path) -> None:
    from app.parsers.email_parser import EmailParser

    p = tmp_path / "m.eml"
    p.write_bytes((SYNTH_HEADERS + SYNTH_BODY).encode("utf-8"))
    texts = [a.raw_text for a in EmailParser().parse_artifact_full(
        project_id="p", artifact_id="a", path=p).atoms]
    assert NO_COMMA not in texts
    assert ("But we are waiting for the switches to arrive at their office "
            "(they are with the freight carrier now)") in texts
    assert "so I also need to keep an eye on the tracking status." in texts


def test_paren_so_and_bare_so_i_we_are_boundaries() -> None:
    assert split_compound_clauses(NO_COMMA) == [
        "But we are waiting for the switches to arrive at their office (they are with the freight carrier now)",
        "so I also need to keep an eye on the tracking status.",
    ]
    assert split_compound_clauses(
        "The switches are on backorder until August (per the distributor) so they will ship late."
    ) == [
        "The switches are on backorder until August (per the distributor)",
        "so they will ship late.",
    ]
    assert split_compound_clauses(
        "The site contact is out of office this week so we need to reschedule the walkthrough."
    ) == [
        "The site contact is out of office this week",
        "so we need to reschedule the walkthrough.",
    ]


def test_ordinary_so_stays_whole() -> None:
    for s in (
        "Please label every drop at both ends so that the techs can trace them later.",
        "We have pulled cable to the first three floors so far without any issues.",
        "Bring the ladders, the drills, the testers and so on to the site Monday.",
        "The carrier says it is not so we can expect it before Friday at the earliest.",
        "We need cables and so we ordered two more boxes from the distributor.",
        "It is a large building so they should plan for two full days onsite.",
        "Keep the wall plates handy so you can finish the second floor quickly.",
    ):
        assert split_compound_clauses(s) == [s], s
