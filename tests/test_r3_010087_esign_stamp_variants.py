"""Every form of an e-signature page stamp is a doc_stamp reject (010087).

The DocuSign envelope stamp still came through as a content atom. The
stamp was only recognised at the start of a line with its id on the same
line; on real pages it also sits at the foot of the page beside a page
number, wraps its id onto the next line, or follows the page's own text on
one extracted line, and Adobe's reads "Adobe Acrobat Sign Transaction
Number".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.parsers.sow_sections import is_doc_stamp, split_doc_stamp

GUID = "3F2A9C1E-1B2C-4D5E-9F00-ABCDEF123456"


@pytest.mark.parametrize("text", [
    f"Docusign Envelope ID: {GUID}",
    f"Docusign Envelope ID:\n{GUID}",
    f"DocuSign Envelope ID: {GUID.replace('-', '')} Page 2 of 5",
    "Adobe Acrobat Sign Transaction Number: CBJCHBCAABAAx9yZqW",
])
def test_stamp_forms(text: str) -> None:
    assert is_doc_stamp(text)


def test_stamp_after_page_text_is_split_off() -> None:
    assert split_doc_stamp(f"Signatures Docusign Envelope ID: {GUID}") == (f"Docusign Envelope ID: {GUID}", "Signatures")
    assert split_doc_stamp("The DocuSign envelope id is sent separately.") is None


def test_pdf_stamps_are_rejects_not_content(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    doc = fitz.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((36, 30), "Docusign Envelope ID:", fontsize=8)
    p1.insert_text((36, 40), GUID, fontsize=8)
    p1.insert_text((36, 80), "Statement of Work", fontsize=12, fontname="hebo")
    p1.insert_text((36, 104), "PurTera will install 24 cameras across the warehouse for Acme Corp.", fontsize=10)
    p1.insert_text((36, 770), f"Docusign Envelope ID: {GUID}", fontsize=8)
    p1.insert_text((500, 770), "Page 1 of 2", fontsize=8)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((36, 80), "Signatures", fontsize=12, fontname="hebo")
    p2.insert_text((200, 80), f"Docusign Envelope ID: {GUID}", fontsize=8)
    p2.insert_text((36, 104), "Customer will provide power at each camera location.", fontsize=10)
    path = tmp_path / "Signed SOW.pdf"
    doc.save(str(path))
    atoms = list(getattr(OrbitBriefPdfParser().parse(path), "atoms", None) or OrbitBriefPdfParser().parse(path))
    stamped = [a for a in atoms if "envelope id" in a.raw_text.lower() or GUID in a.raw_text]
    assert stamped, [a.raw_text for a in atoms]
    for a in stamped:
        assert (a.value or {}).get("rejected_by") == "doc_stamp", (a.raw_text, a.atom_type, a.value)
        assert "chatter" in a.review_flags
    texts = [a.raw_text for a in atoms]
    assert "PurTera will install 24 cameras across the warehouse for Acme Corp." in texts
    assert "Customer will provide power at each camera location." in texts
