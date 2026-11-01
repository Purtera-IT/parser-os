from app.parsers.email_parser import _quoted_header_fields


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
