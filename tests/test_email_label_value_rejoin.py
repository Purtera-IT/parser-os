"""A label and its value are one fact, however many blank lines sit between.

Deal 010288's wiring diagram was delivered by an email that wrote "Diagram:"
as one paragraph and the link as another, five empty paragraphs later. The
label was then a label with no value and the link a link with no claim, and
both were correctly discarded as not statements -- so the authoritative
document held no pointer to the drawing at all. It survived only because
somebody pasted the same link into a note six days later, on one line, where
it parsed.
"""
from __future__ import annotations

import pytest

from app.parsers.email_body import rejoin_label_and_value as rejoin

PNG = "https://huzzard.com/wp-content/uploads/2025/11/BPW061725-Rev-1.png"
#: Outlook's plain-text part writes a link as the display URL followed by its
#: real target in angle brackets. This is the exact shape 010288 arrived in.
OUTLOOK = f"{PNG}<https://urldefense.com/v3/__https:/huzzard.com/a.png__;!!HUqgN_M!pt$>"


def lines(text):
    return [ln.strip() for ln in rejoin(text).split("\n") if ln.strip()]


def test_a_label_and_its_link_become_one_line():
    assert lines(f"Diagram:\n\n\n\n\n{PNG}\n") == [f"Diagram: {PNG}"]


def test_the_shape_010288_actually_arrived_in():
    """Display URL plus bracketed target, four blank paragraphs up from its
    label. The display URL is what is kept."""
    assert lines(f"Diagram:\r\n\r\n\r\n\r\n{OUTLOOK}\r\n") == [f"Diagram: {PNG}"]


def test_a_label_over_something_that_is_not_a_link_is_left_alone():
    """"Provided by us:" heads a bullet list. Joining it to the first bullet
    would merge a section heading into a line item."""
    assert lines("Provided by us:\n-Anything in Orange\n-Relay\n") == [
        "Provided by us:", "-Anything in Orange", "-Relay"]


def test_prose_ending_in_a_colon_is_not_a_label():
    """The pattern is a SHORT line and nothing else. A sentence that happens to
    end in a colon is longer than a field name ever is."""
    sentence = ("Here are the details for the small job I was discussing, and the "
                "parts we will provide are as follows:")
    assert lines(f"{sentence}\n\n{PNG}\n") == [sentence, PNG]


def test_a_link_with_words_beside_it_is_already_a_statement():
    said = f"The drawing is at {PNG} if you want to look."
    assert lines(f"Diagram:\n\n{said}\n") == ["Diagram:", said]


def test_a_link_too_far_from_the_label_is_a_separate_thing():
    """Eight blank lines is already generous. Beyond it, a colon above and a
    link below are two unrelated parts of the message."""
    gap = "\n" * 12
    out = lines(f"Diagram:{gap}{PNG}\n")
    assert out == ["Diagram:", PNG]


def test_nothing_else_in_the_message_moves():
    body = ("Hey AJ,\n\nHere are the details.\n\nProvided by us:\n-Relay\n\n"
            f"Diagram:\n\n\n{PNG}\n\n-------\n\nThanks,\nAlec\n")
    got = lines(body)
    assert f"Diagram: {PNG}" in got
    assert got[0] == "Hey AJ,"
    assert got[-1] == "Alec"
    assert "-Relay" in got


@pytest.mark.parametrize("text", ["", "\n\n\n", "Diagram:", PNG])
def test_degenerate_input_survives(text):
    rejoin(text)
