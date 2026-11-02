from types import SimpleNamespace

from app.parsers.email_parser import _header_first_in_quoted_block, _quoted_header_fields


def test_reads_to_cc_subject_until_blank_line():
    lines = [
        "From: A <a@x.com>",
        "Sent: Thursday, July 2, 2026 2:53 PM",
        "To: B <b@x.com<mailto:b@x.com>>; C <c@x.com>",
        "Cc: D <d@x.com>;",
        "E <e@x.com>",
        "Subject: Smarthands project",
        "",
        "To: not a header, body text",
    ]
    assert _quoted_header_fields(lines, 0) == [
        ("To", "B <b@x.com>; C <c@x.com>", 2),
        ("Cc", "D <d@x.com>; E <e@x.com>", 3),
        ("Subject", "Smarthands project", 5),
    ]


def test_header_with_no_fields_gives_nothing():
    assert _quoted_header_fields(["From: A", "Sent: Monday", "", "Body"], 0) == []


def test_contact_card_reads_at_the_signature():
    class Ref(SimpleNamespace):
        def model_copy(self, update):
            return Ref(locator=update["locator"])

    card = SimpleNamespace(value={"kind": "person"}, source_refs=[Ref(locator={"line_start": 9, "line_end": 30})])
    head = SimpleNamespace(value={"kind": "quoted_message_header"}, source_refs=[Ref(locator={"line_start": 9})])
    _header_first_in_quoted_block([head, card])
    assert card.source_refs[0].locator["line_start"] == 30
    assert head.source_refs[0].locator["line_start"] == 9
