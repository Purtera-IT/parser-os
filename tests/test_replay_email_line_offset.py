"""A line number only means something against the text the parser numbered.

EmailParser numbers the message BODY, having stripped the RFC822 headers.
_verify_line_range read the RAW file, so every email atom's line range was
offset by the header block, the snippet never matched, and the atom was marked
failed.

Measured on deal 010215: documents, meetings and notes verified at 0% failure
while email failed at 82% -- 422 of 514 atoms. Email is the corpus's largest
evidence source, so most of what the system read could not be receipt-verified.
"""
from pathlib import Path

from app.core.source_replay import _replay_lines


class _Ref:
    def __init__(self, locator):
        self.locator = locator


def _eml(tmp_path: Path) -> Path:
    p = tmp_path / "m.eml"
    p.write_text(
        "From: Octavian Mitroi <octavian@purtera-it.com>\n"
        "To: amy@sodexo.com\n"
        "Date: Thu, 27 Aug 2026 16:14:19 +0000\n"
        "Subject: RE: Sodexo - PSOW 202017\n"
        "MIME-Version: 1.0\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "FIRST BODY LINE\n"
        "second body line\n",
        encoding="utf-8",
    )
    return p


class TestEmailLineOffset:
    def test_line_one_of_an_eml_is_the_first_BODY_line(self, tmp_path):
        lines = _replay_lines(None, _Ref({"message_index": 0, "line_start": 1}), _eml(tmp_path))
        assert lines[0] == "FIRST BODY LINE"

    def test_the_raw_file_would_have_given_a_header(self, tmp_path):
        # The bug, stated as a fact about the input: reading the file whole puts
        # "From: ..." at line 1, so every atom's range is off by the header block.
        raw = _eml(tmp_path).read_text(encoding="utf-8").splitlines()
        assert raw[0].startswith("From:")

    def test_a_non_email_text_file_is_still_read_whole(self, tmp_path):
        # Keyed on message_index, not on the suffix: a text file that was not
        # parsed as email must keep its existing behaviour, or this breaks the
        # paths that already verify at 0% failure.
        p = tmp_path / "notes.txt"
        p.write_text("line one\nline two\n", encoding="utf-8")
        lines = _replay_lines(None, _Ref({"line_start": 1}), p)
        assert lines[0] == "line one"

    def test_a_missing_file_does_not_raise(self, tmp_path):
        assert _replay_lines(None, _Ref({"message_index": 0}), tmp_path / "nope.eml") == []

    def test_a_malformed_message_falls_back_rather_than_losing_the_receipt(self, tmp_path):
        p = tmp_path / "broken.eml"
        p.write_bytes(b"\xff\xfe not really a message at all\n")
        # Must not raise; a wrong-but-old answer beats no receipt.
        assert isinstance(_replay_lines(None, _Ref({"message_index": 0}), p), list)


class TestInlineImageOcrIsUnsupportedNotFailed:
    """OCR of an inline image is not in the body, so a line range cannot check it.

    Calling that "failed" asserts the receipt was checked and did not hold. It
    was not checked -- the method does not apply. On deal 010215 this was 91 of
    92 CID atoms, and it made email look like an 18% failure when every real
    body atom verified.
    """

    def test_a_cid_locator_yields_unsupported(self, tmp_path):
        from app.core.schemas import SourceRef
        from app.core.source_replay import _verify_line_range

        class _Atom:
            raw_text = "CDW) OFFICIAL PROVIDER"
            normalized_text = "cdw official provider"
            id = "atm_x"
            value: dict = {}

        from app.core.schemas import ArtifactType

        ref = SourceRef(
            id="src_1",
            artifact_id="art_1",
            artifact_type=ArtifactType.email,
            filename="m.eml",
            locator={"kind": "email_cid_inline", "content_id": "image007.png",
                     "message_index": 12, "line_start": 368, "line_end": 368},
            extraction_method="ocr",
            parser_version="test",
        )
        r = _verify_line_range(_Atom(), ref, _eml(tmp_path))
        assert r.replay_status == "unsupported"
        assert "not in the message body" in str(r.model_dump())
