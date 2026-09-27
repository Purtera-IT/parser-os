"""`When:` and `Where:` are a calendar invite's header, not the job.

Outlook writes a meeting invite's two extra fields directly under `Subject:`,
and 010180 carried them into the deal as scope:

    When: Monday, July 27, 2026 2:00 PM-3:00 PM (UTC-05:00) Eastern Time  -> constraint
    Where: Microsoft Teams Meeting                                        -> open_question

So a July calendar slot became a constraint on a cabling job, and Orbit would
have asked a PM where a meeting held two months ago took place.

`strip_meeting_invite` already knows both words. It reads them only once a
platform marker has OPENED a join block, and Outlook puts this pair ABOVE the
block, so no block was open when they arrived. They belong with `From:`,
`Sent:`, `To:` and `Subject:` in `_PSEUDO_HEADER_RE` instead -- they are header
fields, and the quoted-header branch already drops those.
"""
from __future__ import annotations

import pytest

from app.parsers.email_parser import _PSEUDO_HEADER_RE


@pytest.mark.parametrize("line", [
    "When: Monday, July 27, 2026 2:00 PM-3:00 PM (UTC-05:00) Eastern Time (US & Canada).",
    "Where: Microsoft Teams Meeting",
    "when: tuesday",
    "WHERE: Conference Room B",
])
def test_the_invite_fields_are_header_fields(line):
    assert _PSEUDO_HEADER_RE.match(line)


@pytest.mark.parametrize("line", [
    "Whenever you get a chance, send the floorplans.",
    "Where do you want the IDF?",
    "Wherever the landlord puts the riser is fine.",
])
def test_prose_that_merely_starts_with_the_word_is_not(line):
    """The colon is what makes it a field. A question is still a question."""
    assert not _PSEUDO_HEADER_RE.match(line)


# --------------------------------------------------------------------------
# A wider defect this test found, recorded rather than fixed here
# --------------------------------------------------------------------------
# On a QUOTED block -- which is what 010180 has, and what a forwarded invite
# always is -- the pseudo-header branch drops these lines, and the real file
# now yields no `When:`/`Where:` atom at all.
#
# On a NON-quoted body the whole leading run leaks: `From:`, `Sent:`, `To:` and
# `Subject:` each become a `scope_item`, not only the two invite fields.
# `_split_leading_pseudo_headers` peels exactly this run and returns the right
# answer when called directly, so it is not reached on that path. That is a
# bigger change than adding two words to a regex, on a core path, and it is
# written down instead of attempted half-awake.
