"""An ellipsis that runs straight into the next word is not a sentence end.

Live 000132: a text/plain CRLF mail said "<greeting>.  <remark>....<more of
the same remark>." pysbd ended a sentence at the "...." even though a
lowercase word followed with no space, so one trailing-off thought became two
atoms (sentence_index 1 and 2 on the same line). "..." or "…" followed by
whitespace and a capital is still a real boundary.
"""

from __future__ import annotations

from pathlib import Path

from app.core.sentences import split_sentences, strip_list_marker
from app.parsers.email_parser import EmailParser, _expand_lines_to_sentences

LINE = "Good morning.  The hockey season starts today....only exhibition games but I'll watch anyway."


def _texts(lines: list[str]) -> list[str]:
    return [t for _, _, t in _expand_lines_to_sentences(lines, 1)]


def test_glued_ellipsis_does_not_split() -> None:
    assert [p.strip() for p in split_sentences(LINE)] == [
        "Good morning.",
        "The hockey season starts today....only exhibition games but I'll watch anyway.",
    ]
    assert [p.strip() for p in split_sentences("It was fine…mostly fine. Then it broke.")] == [
        "It was fine…mostly fine.",
        "Then it broke.",
    ]


def test_ellipsis_then_space_and_capital_is_still_a_boundary() -> None:
    assert [p.strip() for p in split_sentences("He waited a while... Then the crew arrived.")] == [
        "He waited a while...",
        "Then the crew arrived.",
    ]


def test_pieces_stay_verbatim() -> None:
    assert "".join(split_sentences(LINE)) == LINE


def test_email_line_keeps_the_remark_whole() -> None:
    out = _texts([LINE])
    assert "The hockey season starts today....only exhibition games but I'll watch anyway." in out
    assert not any(t == "The hockey season starts today...." for t in out)


def test_through_a_text_plain_crlf_email(tmp_path: Path) -> None:
    body = (
        "Hi Sam,\r\n\r\n\r\n\r\n" + LINE + "\r\n\r\n\r\n\r\n"
        "Just checking in… on the install date for the site.\r\n\r\nThanks,\r\nLee\r\n"
    )
    raw = (
        "From: Lee <lee@example.com>\r\nTo: Sam <sam@example.org>\r\nSubject: Check in\r\n"
        "Date: Mon, 10 Aug 2026 13:00:00 -0500\r\nMIME-Version: 1.0\r\n"
        "Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: 8bit\r\n\r\n" + body
    ).encode()
    path = tmp_path / "m.eml"
    path.write_bytes(raw)
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=path).atoms
    texts = [a.raw_text for a in atoms]
    assert "The hockey season starts today....only exhibition games but I'll watch anyway." in texts
    assert "The hockey season starts today...." not in texts
    assert not any(t.startswith("only exhibition games") for t in texts)


def test_list_marker_untouched() -> None:
    assert strip_list_marker("- item one")[1] == "item one"
