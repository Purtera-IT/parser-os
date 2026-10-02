"""Header atoms whose text the parser composed are flagged, with the real
source lines attached.

"note_id=... | author=..." and "From: ... | To: ..." do not exist in the
files, so the review UI could not highlight them. Their text is kept (atom
ids / labels / downstream readers depend on it) but they carry
``synthetic_text`` and ``source_lines`` -- the verbatim lines to highlight.
"""
from __future__ import annotations

from pathlib import Path

from app.parsers.email_parser import EmailParser
from app.parsers.hubspot_note_parser import HubspotNoteParser

EML = (
    "Received: from mx.example.com\n"
    "From: Jane Customer <jane@acme.example>\n"
    "To: Trent Walsh <trent@purtera-it.com>\n"
    "Subject: Delphos and Lima site list\n"
    "Date: Tue, 01 Sep 2026 10:00:00 -0400\n"
    "Message-ID: <abc@acme.example>\n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/plain; charset=utf-8\n"
    "\n"
    "Hi Trent,\n\nPlease quote the deployment for the four sites we discussed.\n\n"
    "From: Bob Smith <bob@acme.example>\n"
    "Sent: Monday, August 31, 2026 9:00 AM\n"
    "To: Jane Customer <jane@acme.example>\n"
    "Subject: sites\n\n"
    "We need network switches installed at Delphos and Lima.\n"
)

NOTE = "\n".join([
    "HubSpot Note: Site walk recap",
    "HubSpot Note ID: 116539976562",
    "Date: 2026-09-02T10:33:00.000Z",
    "Author: AJ Evans",
    "Author-Email: aj@purtera-it.com",
    "",
    "Customer confirmed four live sites for the deployment.",
])


def _atoms(parser, path):
    out = parser.parse_artifact("p", "a", path)
    return out if isinstance(out, list) else out.atoms


def _check(atom, src_lines):
    assert atom.value["synthetic_text"] is True
    assert "synthetic_text" in atom.review_flags
    for sl in atom.value["source_lines"]:
        # each attached line is verbatim at its line number
        assert src_lines[sl["line"] - 1].strip().lstrip(">").strip() == sl["text"]
        assert sl["text"] in "\n".join(src_lines)


def test_eml_header_and_quoted_header(tmp_path: Path):
    p = tmp_path / "m.eml"
    p.write_text(EML)
    lines = EML.splitlines()
    atoms = _atoms(EmailParser(), p)
    (hdr,) = [a for a in atoms if (a.value or {}).get("kind") == "email_header"]
    assert hdr.raw_text not in EML  # composed
    _check(hdr, lines)
    texts = [sl["text"] for sl in hdr.value["source_lines"]]
    assert texts[0] == "From: Jane Customer <jane@acme.example>"
    assert "Subject: Delphos and Lima site list" in texts
    quoted = [a for a in atoms if (a.value or {}).get("kind") == "quoted_message_header"]
    for q in quoted:
        _check(q, lines)
        assert q.value["source_lines"][0]["text"].startswith("From: Bob Smith")


def test_note_meta(tmp_path: Path):
    p = tmp_path / "deal-hs-note-1.txt"
    p.write_text(NOTE)
    atoms = _atoms(HubspotNoteParser(), p)
    (meta,) = [a for a in atoms if (a.value or {}).get("kind") == "hubspot_note_meta"]
    assert "note_id=" in meta.raw_text and meta.raw_text not in NOTE
    _check(meta, NOTE.splitlines())
    texts = [sl["text"] for sl in meta.value["source_lines"]]
    assert "HubSpot Note ID: 116539976562" in texts and "Author: AJ Evans" in texts
    # body atoms are untouched
    assert not [a for a in atoms if "synthetic_text" in a.review_flags and a is not meta]
