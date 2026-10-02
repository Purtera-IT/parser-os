"""A page footer can be several lines (CDW-style SOW, 010003).

The footer is a rule, then an 8pt line "Proprietary and Confidential | Page N |
<company>", then an 8pt document-number line under it. The first line was
recognised as page furniture, but the second line was read alone and became a
body atom of the page's last section. A small line directly under a footer line
is part of the footer band.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


BODY = [
    "Vendor will hang three screens in the client office and confirm each one powers on.",
    "Vendor will connect every screen to the guest wireless network before leaving site.",
    "Vendor will remove packaging from the work area at the end of the visit.",
]


def _sow(path: Path) -> None:
    doc = fitz.open()
    w, h = 612.3, 790.9
    for pno in range(3):
        p = doc.new_page(width=w, height=h)
        y = 80.0
        if pno == 0:
            p.insert_text((54, y), "PROJECT SCOPE", fontsize=13, fontname="tibo")
            y += 24
        for k in range(14):
            p.insert_text((54, y), f"{BODY[k % 3]} Item {pno}.{k}.", fontsize=10, fontname="tiro")
            y += 16 + (8 if k % 3 == 2 else 0)
        # Footer band at the real document's coordinates (tops 727.9 and 743.1).
        p.draw_line((53.8, 720.5), (562.5, 720.5), width=0.5)
        p.insert_text((57.3, 734.1), "Proprietary and Confidential", fontsize=8, fontname="tiro")
        p.insert_text((294.2, 734.1), f"Page {pno + 1}", fontsize=8, fontname="tiro")
        p.insert_text((478.0, 734.1), "Northwind Services LLC", fontsize=8, fontname="tiro")
        p.insert_text((57.6, 749.3), "SOW 554210", fontsize=8, fontname="tiro")
    doc.save(str(path))
    doc.close()


def test_footer_second_line_is_footer_not_body(tmp_path):
    pdf = tmp_path / "Signed SOW.pdf"
    _sow(pdf)
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    texts = [a.raw_text.strip() for a in atoms]
    assert "SOW 554210" not in texts, texts
    footers = [a for a in atoms if "Proprietary and Confidential" in a.raw_text]
    assert footers, texts
    for a in footers:
        # The document number rides in the footer atom, not after it.
        assert a.raw_text.rstrip().endswith("SOW 554210"), a.raw_text
    body = [a for a in atoms if "Vendor will" in a.raw_text]
    assert body
    assert not any("SOW 554210" in a.raw_text for a in body)


def test_single_line_footer_unchanged(tmp_path):
    pdf = tmp_path / "SOW.pdf"
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((54, 80), "PROJECT SCOPE", fontsize=13, fontname="tibo")
    p.insert_text((54, 104), BODY[0], fontsize=10, fontname="tiro")
    p.insert_text((57.3, 734.1), "Proprietary and Confidential | Page 1", fontsize=8, fontname="tiro")
    doc.save(str(pdf))
    doc.close()
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    assert [a.raw_text.strip() for a in atoms if "Confidential" in a.raw_text] == [
        "Proprietary and Confidential | Page 1"
    ]
