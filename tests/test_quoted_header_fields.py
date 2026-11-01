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


def test_contact_card_reads_after_the_header():
    class Ref(SimpleNamespace):
        def model_copy(self, update):
            return Ref(locator=update["locator"])

    card = SimpleNamespace(value={"kind": "person", "quoted": True}, source_refs=[Ref(locator={"line_start": 9})])
    own = SimpleNamespace(value={"kind": "person", "quoted": False}, source_refs=[Ref(locator={"line_start": 1})])
    head = SimpleNamespace(value={"kind": "quoted_message_header", "quoted": True}, source_refs=[Ref(locator={})])
    _header_first_in_quoted_block([head, card, own])
    assert card.source_refs[0].locator["sentence_index"] == 4
    assert "sentence_index" not in own.source_refs[0].locator
    assert "sentence_index" not in head.source_refs[0].locator
