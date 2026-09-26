"""A Teams join block is not the job.

Teams, Zoom and Webex append a fixed block of join details, and left in it
parses. Measured across 461 dev envelopes: 87 atoms on 38 deals came out of one,
and the types are the problem rather than the count --

    42  scope_item   "Dial in by phone", "Reset dial-in PIN", "Meeting options"
    22  raw_utterance
    14  deal_metadata
     7  constraint   "Find a local number"
     2  site_access_restriction

So a cabling job at 7 Penn Plaza carried a Teams passcode as a restriction on
getting into the site, six deals were told to "Dial in by phone" as scope, and
"Need help?" became an `open_question` -- Orbit asking a PM the invite's own
rhetorical question.

It is also the one place a message carries live credentials. A bridge passcode
and a dial-in PIN belong in no artifact, no brief and no training corpus.

The shapes below are verbatim from deal 010180 (`010180-hs-email-115678001517`),
which is the deal this was found on and which must NOT be debugged against --
see TRAINING_DEALS.md. The fix was verified against 8aa9051a and 66f5f6fb,
which carry 12 and 11 of these atoms respectively.
"""
from __future__ import annotations

from app.parsers.email_body import strip_meeting_invite as strip

#: Exactly what Outlook wrote under deal 010180's message, line for line.
TEAMS = """Looking forward to speaking with you,
________________________________________________________________________________
Microsoft Teams meeting
Join: https://teams.microsoft.com/meet/228859003479315?p=3inkejx3nncDpjv8g2
Meeting ID:
228 859 003 479 315
Passcode:
jz7o5CE9
Need help?
|
System reference
Dial in by phone
+1 847-371-3000,,25104158#
United States, Libertyville
Find a local number
Phone conference ID:
251 041 58#
Join on a video conferencing device
Tenant key:
cdw@m.webex.com
Video ID:
119 856 328 6
More info
For organizers:
Meeting options
|
Reset dial-in PIN
________________________________________________________________________________
The contents of this email are intended only for the recipient(s) listed above.
"""


def lines(text):
    return [ln.strip() for ln in strip(text).split("\n") if ln.strip()]


def test_the_whole_teams_block_goes():
    out = lines(TEAMS)
    assert out == [
        "Looking forward to speaking with you,",
        "The contents of this email are intended only for the recipient(s) listed above.",
    ]


def test_the_credentials_are_gone():
    """The reason this matters more than the atom count."""
    out = strip(TEAMS)
    for secret in ("jz7o5CE9", "251 041 58#", "25104158#", "228 859 003 479 315"):
        assert secret not in out


def test_what_a_person_wrote_above_and_below_survives():
    body = ("is Rosa available next week to do a survey for this one for a day?\n"
            + TEAMS
            + "\nCost for the survey would be around $600 for us.\n")
    out = lines(body)
    assert out[0].startswith("is Rosa available")
    assert out[-1] == "Cost for the survey would be around $600 for us."


def test_a_zoom_block_goes_too():
    body = ("Call at 3 on Tuesday.\n"
            "Join Zoom Meeting\n"
            "https://zoom.us/j/9912345678?pwd=abcdEFGH\n"
            "Meeting ID:\n"
            "991 234 5678\n"
            "Passcode:\n"
            "8Kd2Qm\n"
            "One tap mobile\n"
            "+16465588656,,9912345678#\n"
            "Find a local number\n"
            "Let's cover the riser diagram.\n")
    assert lines(body) == ["Call at 3 on Tuesday.",
                          "Let's cover the riser diagram."]


def test_mentioning_a_teams_meeting_in_prose_is_not_a_block():
    """A block must say at least three invite-only things. One sentence about
    setting up a call is a statement a person made."""
    body = ("I'll send a Microsoft Teams meeting for Thursday so Rosa can walk "
            "us through the riser.\nDial in by phone if the VPN is down.\n")
    assert lines(body) == [
        "I'll send a Microsoft Teams meeting for Thursday so Rosa can walk "
        "us through the riser.",
        "Dial in by phone if the VPN is down.",
    ]


def test_a_real_door_code_is_not_a_meeting_passcode():
    """`Passcode:` counts only once a platform marker has opened a block. On its
    own it is a site fact, and the one kind of access restriction that is
    genuinely about the job."""
    body = ("Loading dock is on West 31st.\n"
            "Passcode:\n"
            "4417\n"
            "Ask for the super if it does not work.\n")
    assert "4417" in strip(body)


def test_a_message_with_no_invite_is_untouched():
    body = "Quote attached. 212 drops, two per workstation.\nLet me know.\n"
    assert strip(body) == body
