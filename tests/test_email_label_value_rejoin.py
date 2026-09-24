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


# ── and the line has to survive the chrome filter ───────────────────
#
# Rejoining the label to its link is only half the job. `_is_link_only_line`
# strips the links off a line and calls it mail chrome when one short word is
# left -- which is exactly what "Diagram: <link>" looks like after stripping.
# It was written against signature-block brands glued to their own href, and
# it was eating the pointer to the drawing.

from app.parsers.email_parser import _is_link_only_line as chrome


@pytest.mark.parametrize("line", [
    f"Diagram: {PNG}",
    "Floorplan: https://example.com/floor.pdf",
    "Spec: https://example.com/spec.pdf",
])
def test_a_named_field_pointing_at_a_document_is_not_chrome(line):
    """The colon is a sender saying what they are handing over."""
    assert not chrome(line)


@pytest.mark.parametrize("line", [
    "PurTera-IT.com<https://purtera-it.com>",
    "Get Outlook for Mac<https://aka.ms/x>",
    "Report Suspicious<https://us-phishalarm",
    PNG,
])
def test_the_lines_that_rule_exists_to_kill_still_die(line):
    """No colon: the leftover word is a brand or a button, not a field name."""
    assert chrome(line)


def test_end_to_end_the_email_carries_its_drawing(tmp_path):
    """The whole path on a message built to 010288's exact shape: the label and
    the link as separate paragraphs, the link written the way Outlook's
    plain-text part writes one. The drawing has to reach an atom on the EMAIL,
    not only on a note somebody pasted days later.

    Built here rather than vendored: the real message carries a customer's
    address book and tenant ids inside its safelink, and none of that is needed
    to reproduce the shape.
    """
    from app.parsers.email_parser import EmailParser

    nl = "\r\n"
    wrapped = PNG + "<https://urldefense.com/v3/__https:/huzzard.com/a.png__;!!HUqgN_M!pt$>"
    body = nl.join([
        "Hey AJ,", "",
        "Here are the details for the small job.", "",
        "Provided by us:", "-Relay", "-Local and Remote Extender", "",
        "Diagram:", "", "", "",
        wrapped, "",
        "-------", "",
        "Thanks,", "Alec", "",
    ])
    head = nl.join([
        "From: alec@example-reseller.com",
        "To: aj@example-co.com",
        "Subject: Access Control",
        "Content-Type: text/plain; charset=utf-8",
        "", "",
    ])
    eml = tmp_path / "ask.eml"
    eml.write_text(head + body, encoding="utf-8")

    said = [str(getattr(a, "raw_text", "") or "") for a in EmailParser().parse(eml)]
    assert any(t.strip().lower().startswith("diagram:") and PNG in t for t in said), said
    # The bullet list above it is untouched.
    assert any("Relay" in t for t in said)
