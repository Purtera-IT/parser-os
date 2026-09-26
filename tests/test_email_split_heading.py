"""A heading and the sentence it introduces are one statement.

BeautifulSoup's `get_text(separator="\\n")` inserts a newline between an element
and the text after it, so `<b>Heading</b>: body` arrives as two lines and both
halves are worthless:

    Layout & Technical Requirements          a noun phrase asserting nothing,
                                             typed `scope_item` -- so "Security
                                             & Access Control" becomes work the
                                             job includes
    : The team discussed the office...       a sentence with no subject,
                                             starting on a bare colon

Measured across 461 dev envelopes: **41 orphaned bodies on 26 deals**, and the
matching count of heading atoms beside them.

The orphans do not merely read badly, they do not survive. On deal 010180 two
of four never reached the envelope at all:

  * the whole `Security & Access Control` discussion -- a customer-owned
    swipe-card system modelled on another of their sites, and an open question
    about whether the building handles physical security. An entire second
    low-voltage scope, invisible, while its heading sat in the deal looking
    like scope.
  * the sentence carrying "Cat 6A cabling, two Ethernet connections per
    workstation" -- the multiplier the deal's $110,108 is built on, stated on
    the customer call seven weeks before anything else restates it.

This is the mirror of `rejoin_label_and_value`, which handles `Label:` on one
line and its value on the next. Same defect, opposite side of the colon.
"""
from __future__ import annotations

from app.parsers.email_body import rejoin_split_heading as rejoin

#: Verbatim from 010180's 5 August call notes, as BeautifulSoup emits them.
NOTES = """Notes below:
Layout & Technical Requirements
: The team discussed the office layout, including 106 workstations, conference rooms, phone rooms, IT room, and pantry. CAD drawings and plans were shared for review. The setup will require Cat 6A cabling, two Ethernet connections per workstation, and AV work for conference rooms. 2:01
Sequence & Coordination
: Electrical connections will be provided by the landlord, with furniture vendors handling workstation hookups. 8:37
Security & Access Control
: The team discussed implementing a new security system, likely using their own swipe card system similar to Great Neck. 7:08
Wireless & Site Survey
: Erick offered a Cisco-funded wireless site survey to determine access point needs. 12:08
"""


def lines(text):
    return [ln for ln in rejoin(text).split("\n") if ln.strip()]


def test_the_010180_call_notes_come_back_as_four_statements():
    out = lines(NOTES)
    assert out[0] == "Notes below:"
    assert len(out) == 5
    for heading in ("Layout & Technical Requirements:", "Sequence & Coordination:",
                    "Security & Access Control:", "Wireless & Site Survey:"):
        assert any(ln.startswith(heading) for ln in out), heading


def test_no_line_is_left_starting_on_a_bare_colon():
    assert not [ln for ln in lines(NOTES) if ln.lstrip().startswith(":")]


def test_the_multiplier_survives_in_its_own_sentence():
    """"two Ethernet connections per workstation" is what 212 drops is built
    on, and it was inside the orphaned half."""
    joined = [ln for ln in lines(NOTES) if ln.startswith("Layout &")][0]
    assert "two Ethernet connections per workstation" in joined
    assert "106 workstations" in joined


def test_a_heading_with_no_body_is_left_alone():
    """A section heading over a bullet list is not this shape, and joining it
    to the first bullet would merge a heading into a line item."""
    body = "Provided by us:\n-Anything in Orange\n-Relay\n"
    assert lines(body) == ["Provided by us:", "-Anything in Orange", "-Relay"]


def test_prose_that_happens_to_be_short_is_not_a_heading():
    """The pattern requires the NEXT line to start with a colon. Without one
    there is nothing to rejoin and nothing is touched."""
    body = "Thanks for the update\nWe can do the week of October 5\n"
    assert lines(body) == ["Thanks for the update", "We can do the week of October 5"]


def test_a_blank_line_between_the_two_halves_is_crossed():
    body = "Wireless & Site Survey\n\n\n: Erick offered a survey.\n"
    assert lines(body) == ["Wireless & Site Survey: Erick offered a survey."]


def test_a_sentence_ending_in_punctuation_is_not_a_heading():
    """"We shipped it." followed by ": 3 boxes" is not a heading and a body."""
    body = "We shipped it.\n: 3 boxes\n"
    assert lines(body) == ["We shipped it.", ": 3 boxes"]


def test_text_with_no_headings_is_byte_identical():
    body = "Quote attached. 212 drops, two per workstation.\nLet me know.\n"
    assert rejoin(body) == body


# --------------------------------------------------------------------------
# The root cause: get_text splits at every text node, not every element
# --------------------------------------------------------------------------

def test_inline_markup_does_not_break_a_sentence():
    """`rejoin_split_heading` above patches one symptom. This is the cause.

    `get_text(separator="\n")` joins EVERY text node, so any tag a mail client
    wraps around part of a sentence splits it. Unwrapping the tag is not
    enough -- the text stays as separate strings -- so the fragments have to be
    smoothed back together too. On 010180's call notes this alone took the
    message from 102 lines to 70.
    """
    from bs4 import BeautifulSoup

    from app.parsers.email_body import _unwrap_inline_in_place

    html = ('<p>Suite 1316, 3<sup>rd</sup> Floor - Holmdel</p>'
            '<p><b>Layout &amp; Technical Requirements</b>'
            '<span>: The team discussed the office layout.</span></p>'
            '<p>To: <a href="mailto:p@x.com">Patrick Kelly</a> '
            '&lt;p@x.com&gt;</p>')
    soup = BeautifulSoup(html, "html.parser")
    _unwrap_inline_in_place(soup)
    out = [ln for ln in soup.get_text(separator="\n", strip=True).split("\n") if ln]
    assert out == [
        "Suite 1316, 3rd Floor - Holmdel",
        "Layout & Technical Requirements: The team discussed the office layout.",
        "To: Patrick Kelly <p@x.com>",
    ]


def test_table_cells_still_split_for_the_flattener():
    """Inline unwrapping must not dissolve the table structure -- the row
    flattener runs next and needs one string per cell."""
    from bs4 import BeautifulSoup

    from app.parsers.email_body import _unwrap_inline_in_place

    soup = BeautifulSoup("<table><tr><td><b>A</b></td><td>B</td></tr></table>",
                         "html.parser")
    _unwrap_inline_in_place(soup)
    assert soup.get_text(separator="\n", strip=True).split("\n") == ["A", "B"]
