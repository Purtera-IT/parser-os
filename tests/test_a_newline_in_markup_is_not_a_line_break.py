"""A newline INSIDE a text node is whitespace. HTML has always said so.

Unwrapping inline tags was only half the repair. A mail client also breaks its
source markup wherever it likes, and Outlook sent 010180 this:

    Are there any specific days the week of the 5
    <sup>th</sup>&nbsp;they want to do the walkthrough?

The `<sup>` unwraps and smooths back into one string — and that string still
carries the newline the markup had in front of the tag. The separator split on
it anyway, and the deal got two fragments:

    "Are there any specific days the week of the 5"    scope_item
    "th they want to do the walkthrough?"              open_question

Neither asks anything, from a question that is still open: nobody has said
which days the customer wants.
"""
from bs4 import BeautifulSoup

from app.parsers.email_body import _unwrap_inline_in_place

OUTLOOK = (
    '<div>Sounds good. Talked to my Ops team.</div>\n'
    '<div>\n Are there any specific days the week of the 5 \n'
    '<sup>th</sup>&nbsp;they want to do the walkthrough? '
    'Will there be someone that our engineer needs to meet onsite?\n</div>'
)


def _text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    _unwrap_inline_in_place(soup)
    return soup.get_text(separator="\n")


def test_the_question_survives_as_one_line():
    lines = [l.strip() for l in _text(OUTLOOK).splitlines() if l.strip()]
    asked = [l for l in lines if "specific days" in l]
    assert len(asked) == 1
    assert asked[0].startswith("Are there any specific days")
    assert "walkthrough?" in asked[0]
    assert not any(l.startswith("th ") for l in lines)


def test_the_ordinal_is_put_back_together():
    """The author wrote "5th" — the message's own text/plain says so — and a
    PM reading "the 5 th" sees a parse artifact and trusts the atom less."""
    assert "the 5th they want" in _text(OUTLOOK)


def test_only_a_suffix_after_a_digit_is_joined():
    """Anchored on a digit, so ordinary words starting st/nd/rd/th are safe."""
    assert "the standard" in _text("<div>the standard</div>")
    assert "5 things" in _text("<div>5 things</div>")
    assert "Suite 3rd" in _text("<div>Suite 3<sup>rd</sup></div>")


def test_block_tags_still_start_new_lines():
    """Collapsing source newlines must not weld separate paragraphs together —
    a block tag is the only thing that should break a line."""
    lines = [l.strip() for l in _text(
        "<div>First sentence.</div>\n<div>Second sentence.</div>") .splitlines()
        if l.strip()]
    assert lines == ["First sentence.", "Second sentence."]


def test_a_br_still_breaks():
    lines = [l.strip() for l in _text("<div>One<br>Two</div>").splitlines() if l.strip()]
    assert lines == ["One", "Two"]


def test_preformatted_text_keeps_its_newlines():
    """A newline in <pre> is a real line break, and collapsing it would ruin
    the one place in HTML where whitespace means something."""
    lines = [l for l in _text("<pre>line one\nline two</pre>").splitlines() if l.strip()]
    assert lines == ["line one", "line two"]
