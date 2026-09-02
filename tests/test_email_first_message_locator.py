"""The first message of an .eml must carry who sent it and when.

Receipt verification anchors a quote through its locator. An empty locator
cannot be anchored, and on deal 010215 that cost 82% of email atoms their
verification against 0% for documents, meetings and notes -- 257 of them with
sender "unknown" and no date. Email is the largest evidence source in the
corpus, so most of what the system read was carrying claims it could not prove.
"""
from pathlib import Path

from app.parsers.email_parser import EmailParser


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "m.eml"
    p.write_text(
        "From: Octavian Mitroi <octavian@purtera-it.com>\n"
        "To: amy@sodexo.com\n"
        "Date: Thu, 27 Aug 2026 16:14:19 +0000\n"
        "Subject: RE: Sodexo - PSOW 202017\n"
        "Content-Type: text/plain\n\n" + body,
        encoding="utf-8",
    )
    return p


def _blocks(path: Path):
    p = EmailParser()
    doc = p.parse_artifact_full(path) if hasattr(p, "parse_artifact_full") else None
    return doc


class TestFirstMessageLocator:
    def test_first_block_locator_takes_the_envelope_sender_and_date(self, tmp_path):
        path = _write(tmp_path, "Please see the attached breakdown for all Sodexo sites.\n")
        parser = EmailParser()
        text = path.read_text(encoding="utf-8")
        # exercise the split directly with the envelope values the parser reads
        blocks = parser._split_blocks(
            "Please see the attached breakdown for all Sodexo sites.",
            envelope_sender="Octavian Mitroi <octavian@purtera-it.com>",
            envelope_sent_at="Thu, 27 Aug 2026 16:14:19 +0000",
        )
        assert blocks, "expected at least one message block"
        assert blocks[0]["locator_sender"] != "unknown"
        assert blocks[0]["locator_sent_at"] != ""

    def test_without_an_envelope_it_still_says_unknown_rather_than_guessing(self):
        parser = EmailParser()
        blocks = parser._split_blocks("Some body with no headers at all.")
        assert blocks[0]["locator_sender"] == "unknown"
        assert blocks[0]["locator_sent_at"] == ""

    def test_quoted_messages_keep_their_own_headers(self):
        # The envelope must never overwrite a quoted message's own From/Sent, or
        # every message in a thread is attributed to whoever sent the last reply.
        parser = EmailParser()
        text = (
            "Thanks, that works.\n"
            "\n"
            "From: Quinton James <quinton.james@cdw.com>\n"
            "Sent: Wednesday, August 12, 2026 2:23 PM\n"
            "Subject: RE: 010215\n"
            "Awesome, thanks!\n"
        )
        blocks = parser._split_blocks(
            text,
            envelope_sender="Patrick Kelly <patrick@purtera-it.com>",
            envelope_sent_at="Wed, 12 Aug 2026 18:31:00 +0000",
        )
        assert len(blocks) >= 2
        assert "patrick" in blocks[0]["locator_sender"].lower()
        assert "quinton" in blocks[1]["locator_sender"].lower()
        assert "August 12" in blocks[1]["locator_sent_at"]

    def test_body_headers_win_over_the_envelope_on_the_first_block(self):
        parser = EmailParser()
        blocks = parser._split_blocks(
            "From: Real Sender <real@x.com>\nSent: Mon, 1 Jun 2026 09:00:00 +0000\nBody text.\n",
            envelope_sender="Wrong <wrong@y.com>",
            envelope_sent_at="Tue, 2 Jun 2026 09:00:00 +0000",
        )
        assert "real@x.com" in blocks[0]["locator_sender"]
