"""A wrapped display name must not produce a broken sender.

HTML mail wraps a long "From:" value across lines:

    From:
    Trent Torrence <
    t@purtera-it.com>

Rejoining those produced ``Trent Torrence <t@purtera-it.com`` -- an unbalanced
bracket, because the closing ">" had wrapped too, or sat inside a
``<mailto:...>`` decoration the text conversion left behind.

That is not cosmetic. ``_originating_sender`` takes the HIGHEST message index in
a forward chain -- the oldest message, the person a document actually came from
-- so it lands precisely on the deepest, most-wrapped blocks. On deal 010215 the
well-formed senders sat at index 1 and every index from 2 to 16 carried a
truncated one. The Aug 12 forward, whose chain starts with the customer, was
attributed to us: `originated_by: 'Trent Torrence <t@purtera-it.com'`.
"""
from __future__ import annotations

from app.parsers.email_parser import _normalize_sender


def test_an_unclosed_address_is_closed():
    assert _normalize_sender("Trent Torrence <t@purtera-it.com") == "Trent Torrence <t@purtera-it.com>"


def test_a_well_formed_sender_is_untouched():
    for value in [
        "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>",
        "Quinton James <quinton.james@cdw.com>",
        "t@purtera-it.com",
    ]:
        assert _normalize_sender(value) == value


def test_a_mailto_decoration_is_removed():
    # The text conversion renders the anchor and its href, leaving the address
    # twice: "t@purtera-it.com<mailto:T@purtera-it.com>".
    assert _normalize_sender("t@purtera-it.com<mailto:T@purtera-it.com>") == "t@purtera-it.com"


def test_a_bare_display_name_gains_no_invented_address():
    # A name with no address is incomplete, not broken. Manufacturing brackets
    # around nothing would assert structure the message never had.
    assert _normalize_sender("Quinton James") == "Quinton James"
    assert _normalize_sender("Donnelly, Bernie") == "Donnelly, Bernie"


def test_an_unclosed_bracket_with_no_address_is_left_alone():
    # "<" here is punctuation, not the start of an address.
    assert _normalize_sender("Sales <team") == "Sales <team"


def test_whitespace_and_wrapping_collapse():
    assert _normalize_sender("  Trent   Torrence\n<t@purtera-it.com>  ") == "Trent Torrence <t@purtera-it.com>"


def test_empty_stays_empty():
    # An unknown sender must not become a plausible-looking one.
    assert _normalize_sender("") == ""
    assert _normalize_sender(None) == ""


def test_the_originator_of_a_real_forward_is_the_customer():
    """End to end on the message that was actually mis-attributed."""
    from pathlib import Path
    import tempfile

    from app.core.orbitbrief_envelope import _originating_sender
    from app.parsers.email_parser import EmailParser

    eml = (
        "From: t@purtera-it.com\n"
        "To: patrick@purtera-it.com\n"
        "Subject: Fw: Time Clock Installs for Marion County School District\n"
        "Date: 2026-08-12T18:00:51.058Z\n"
        "\n"
        "Load this please and reply to all that we are on it.\n"
        "\n"
        "From:\n"
        "Quinton James <quinton.james@cdw.com>\n"
        "Sent:\n"
        "Tuesday, August 12, 2026 10:20 AM\n"
        "\n"
        "Passing along the site list.\n"
        "\n"
        "From:\n"
        "Donnelly, Bernie <\n"
        "Bernie.Donnelly@sodexo.com>\n"
        "Sent:\n"
        "Tuesday, August 12, 2026 8:31 AM\n"
        "\n"
        "Attached are the SOWs for each school.\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "010215-hs-email-114767132205.eml"
        p.write_text(eml, encoding="utf-8")
        out = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p)

    senders = {
        ref.locator.get("sender")
        for atom in out.atoms
        for ref in (getattr(atom, "source_refs", None) or [])
        if isinstance(getattr(ref, "locator", None), dict) and ref.locator.get("sender")
    }
    assert not [s for s in senders if "<" in s and not s.endswith(">")], f"malformed sender in {senders}"
    assert _originating_sender(out.atoms) == "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
