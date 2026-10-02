"""One sentence on one line is one atom, curly-quoted phrases and all.

Live 000132 (multipart mail, quoted earlier message; the sentence is one line
in text/plain and one <span> in one <p> in the HTML) came out as two atoms:
"<request with two “quoted” phrases> would be helpful" | "so I can <purpose>."
The cut was #302's no-comma " so I ..." clause boundary, now removed. The text
below is synthetic with the same shape.
"""

from __future__ import annotations

from pathlib import Path

from app.core.sentences import split_trigger_clause
from app.parsers.email_parser import EmailParser, _expand_lines_to_sentences

SENT = (
    "Even a short “on track” or “on hold” reply would be great "
    "so I can keep our forecast current."
)


def test_sentence_is_one_piece() -> None:
    assert split_trigger_clause(SENT) == [SENT]
    assert [t for _, _, t in _expand_lines_to_sentences([SENT], 1)] == [SENT]
    assert [t for _, _, t in _expand_lines_to_sentences(["> " + SENT], 1)] == ["> " + SENT]


def test_comma_so_still_splits() -> None:
    s = "The racks arrived at the site this morning, so we need the crew there on Monday."
    assert split_trigger_clause(s) == [
        "The racks arrived at the site this morning",
        "so we need the crew there on Monday.",
    ]


def test_multipart_quoted_reply_keeps_the_sentence_whole(tmp_path: Path) -> None:
    plain = (
        "Hi Sam,\r\n\r\nBumping the note below.\r\n\r\nThanks,\r\nLee\r\n\r\n"
        "From: Lee Doe <lee@example.com>\r\nSent: Monday, July 20, 2026 2:00 PM\r\n"
        "To: Sam Roe <sam@example.org>\r\nSubject: Status\r\n\r\n"
        "Hi Sam,\r\n\r\n" + SENT + "\r\n\r\nThanks,\r\nLee\r\n"
    )
    html = (
        "<html><body><p>Hi Sam,</p><p>Bumping the note below.</p><p>Thanks,<br>Lee</p>"
        "<div><p><b>From:</b> Lee Doe<br><b>Sent:</b> Monday, July 20, 2026 2:00 PM<br>"
        "<b>To:</b> Sam Roe<br><b>Subject:</b> Status</p>"
        "<p><span>Hi Sam,</span></p><p><span>" + SENT + "</span></p></div></body></html>"
    )
    raw = (
        "From: Lee Doe <lee@example.com>\r\nTo: Sam Roe <sam@example.org>\r\nSubject: RE: Status\r\n"
        "Date: Mon, 03 Aug 2026 13:00:00 -0500\r\nMIME-Version: 1.0\r\n"
        'Content-Type: multipart/alternative; boundary="BB"\r\n\r\n--BB\r\n'
        "Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: 8bit\r\n\r\n" + plain
        + "\r\n--BB\r\nContent-Type: text/html; charset=utf-8\r\nContent-Transfer-Encoding: 8bit\r\n\r\n"
        + html + "\r\n--BB--\r\n"
    ).encode()
    path = tmp_path / "m.eml"
    path.write_bytes(raw)
    texts = [a.raw_text for a in EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=path).atoms]
    assert SENT in texts
    assert not any(t.startswith("so I can") for t in texts)
