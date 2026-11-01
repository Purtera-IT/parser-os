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


def _atom(kind, line):
    class Ref(SimpleNamespace):
        def model_copy(self, update):
            return Ref(locator=update["locator"])

    return SimpleNamespace(artifact_id="a", value={"kind": kind},
                           source_refs=[Ref(locator={"message_index": 1, "line_start": line})])


def test_contact_on_the_header_line_reads_after_the_header():
    head, field, card, body = (_atom("quoted_message_header", 9), _atom("quoted_message_field", 9),
                               _atom("person", 9), _atom("person", 12))
    _header_first_in_quoted_block([head, field, card, body])
    assert card.source_refs[0].locator["sentence_index"] == 4
    assert "sentence_index" not in head.source_refs[0].locator
    assert "sentence_index" not in body.source_refs[0].locator
