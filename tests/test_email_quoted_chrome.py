"""A quoted message's headers and signature are chrome, at every depth.

Signature detection was gated to the AUTHORED message only:

    if not block["quoted"] and _SIGNOFF_RE.match(cleaned):

so every quoted message's signature was atomised in full. A forward chain
carries one signature per message and deal 010215's runs sixteen deep — 64 of
its 75 chrome atoms sat at depth >= 1, emitting `t`, `Q`, `404.771.3490` and
`M: 404-918-0783` as scope_items. `t` and `Q` are the wrapped initials of Trent
Torrence and Quinton James.

HTML mail also splits a header label from its value across two lines, so
`_PSEUDO_HEADER_RE` (which needs "Label:" on the same line) never saw the value
and it became a scope item of its own.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.core.orbitbrief_envelope import _originating_sender
from app.parsers.email_parser import EmailParser

EML = """From: t@purtera-it.com
To: patrick@purtera-it.com
Subject: Fw: Time Clock Installs
Date: 2026-08-12T18:00:51.058Z

Load this please and reply to all that we are on it.

From:
Quinton James <quinton.james@cdw.com>
Sent:
Tuesday, August 12, 2026 10:20 AM
To:
mike.stephens <Mike.Stephens@sodexo.com>; Finn, Melody <Melody.Finn@sodexo.com>

Please review the request below.

Thanks,
Quinton James
Senior Account Executive
Q
404.771.3490

From:
Donnelly, Bernie <Bernie.Donnelly@sodexo.com>
Sent:
Tuesday, August 12, 2026 8:31 AM

We need to have 10 timeclocks installed for Marion County SD in SC.

Regards,
Bernie Donnelly
M: 404-918-0783
"""


def _atoms():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "010215-hs-email-1.eml"
        p.write_text(EML, encoding="utf-8")
        return EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms


def _texts(atoms):
    return [" ".join(str(a.raw_text or "").split()) for a in atoms]


def test_the_substance_of_every_message_survives():
    # The point of the whole feature: the customer's actual request.
    assert any("10 timeclocks installed for Marion County" in t for t in _texts(_atoms()))
    assert any("Please review the request below" in t for t in _texts(_atoms()))


def test_a_quoted_signature_is_not_atomised():
    texts = _texts(_atoms())
    for chrome in ["Q", "404.771.3490", "M: 404-918-0783", "Senior Account Executive", "Bernie Donnelly"]:
        assert chrome not in texts, f"{chrome!r} is signature chrome"


def test_a_header_value_on_its_own_line_is_not_a_scope_item():
    # HTML mail puts the label above the value, so the value line walked
    # straight through and became "Quinton James <quinton.james@cdw.com>".
    texts = _texts(_atoms())
    for chrome in [
        "Quinton James <quinton.james@cdw.com>",
        "Tuesday, August 12, 2026 10:20 AM",
        "mike.stephens <Mike.Stephens@sodexo.com>; Finn, Melody <Melody.Finn@sodexo.com>",
    ]:
        assert chrome not in texts, f"{chrome!r} is header chrome"


def test_attribution_still_resolves_to_the_customer():
    # The blocks must still be represented, or the originator is lost and the
    # customer's documents get credited to us again.
    assert _originating_sender(_atoms()) == "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"


def test_every_message_in_the_chain_is_still_represented():
    seen = set()
    for atom in _atoms():
        for ref in getattr(atom, "source_refs", None) or []:
            loc = getattr(ref, "locator", None)
            if isinstance(loc, dict) and loc.get("sender"):
                seen.add(loc.get("message_index"))
    assert seen == {0, 1, 2}, f"lost a message block: {seen}"


def test_signature_state_does_not_bleed_into_the_next_message():
    """`in_signature` latches per block.

    If it leaked, Quinton's sign-off would silence Bernie's message entirely --
    the one that carries the actual request.
    """
    assert any("10 timeclocks" in t for t in _texts(_atoms()))
