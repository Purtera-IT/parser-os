"""A signed SOW that is really a Docusign PDF but named ".docx" is a PDF to
every reader, not only to the parser registry.

The registry already routed it by its %PDF magic bytes, but receipt replay,
the content census, the manifest and the pack router's preview dispatched on
the extension: replay opened the PDF with python-docx, every receipt came back
"failed", and the atom could not be located on the page.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from app.core.filetype import content_suffix
from app.core.manifest import build_artifact_fingerprint
from app.core.schemas import ArtifactType
from app.parsers.registry import choose_parser

_SENTENCE = "The contractor shall install 24 access points at the Phoenix office."


def _pdf_named_docx(tmp_path: Path) -> Path:
    path = tmp_path / "Signed_SOW.docx"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Docusign Envelope ID: 1234-ABCD")
    page.insert_text((72, 100), f"Statement of Work: {_SENTENCE}")
    doc.save(str(path))
    return path


def test_content_decides_the_suffix(tmp_path: Path) -> None:
    path = _pdf_named_docx(tmp_path)
    assert path.read_bytes()[:5] == b"%PDF-"
    assert content_suffix(path) == ".pdf"
    plain = tmp_path / "notes.txt"
    plain.write_text("From: someone\nhello", encoding="utf-8")
    assert content_suffix(plain) == ".txt"  # a text sniff never overrules a name


def test_registry_routes_it_to_the_pdf_parser(tmp_path: Path) -> None:
    parser, match, _ = choose_parser(_pdf_named_docx(tmp_path))
    assert match.artifact_type == ArtifactType.pdf
    assert type(parser).__name__ == "OrbitBriefPdfParser"


def test_manifest_records_a_pdf(tmp_path: Path) -> None:
    fp = build_artifact_fingerprint(_pdf_named_docx(tmp_path), "art", [])
    assert fp.artifact_type == ArtifactType.pdf


def test_receipts_replay_against_the_pdf(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    _pdf_named_docx(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    hits = [a for a in r.atoms if _SENTENCE in (a.raw_text or "")]
    assert hits, [a.raw_text for a in r.atoms]
    for a in hits:
        assert a.receipts
        assert all(rc.replay_status == "verified" for rc in a.receipts), [
            (rc.replay_status, getattr(rc, "replay_message", "")) for rc in a.receipts
        ]
    assert not [w for w in r.warnings if w.startswith("ERROR") and "failed receipt" in w]
