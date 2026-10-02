"""A PO's stacked label/value boxes read as one field each (010003 PO).

A purchase order sets each field as a box: a short bold label on a shaded
header bar ("Ship To"), with the value framed in a box directly under it. The
fill change read as a block boundary, so every label became an atom of its own
("Supplier", "Ship To", "Bill To", "Message", ...) apart from its value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

GREY = (0.85, 0.85, 0.85)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _box(page, x0, x1, y, label, values, h=None):
    page.draw_rect(fitz.Rect(x0, y, x1, y + 13), color=None, fill=GREY)
    page.insert_text((x0 + 3, y + 9.5), label, fontsize=8, fontname="hebo")
    h = h or (6 + 10 * max(1, len(values)))
    page.draw_rect(fitz.Rect(x0, y + 13, x1, y + 13 + h), color=(0, 0, 0), width=0.5)
    for k, v in enumerate(values):
        page.insert_text((x0 + 3, y + 23 + 10 * k), v, fontsize=8, fontname="helv")


def _po(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((21, 50), "Purchase Order", fontsize=16, fontname="hebo")
    _box(p, 21, 300, 110, "Supplier", ["Northwind Install Co", "40 River Road", "Albany, NY 12207"])
    _box(p, 312, 583, 110, "Ship To", ["Contoso Ltd", "12 Harbor Street Fl 3", "Boston, MA 02110"])
    _box(p, 312, 583, 170, "Bill To", ["Contoso Accounts Payable", "PO Box 4410", "Boston, MA 02111"])
    _box(p, 21, 195, 240, "Supplier Terms", ["Net 45"], h=26)
    _box(p, 21, 583, 600, "Message", ["Please reference the PO number on every invoice."])
    _box(p, 21, 583, 650, "Disclaimer", ["This order is governed by the buyer's standard terms."])
    doc.save(str(path))
    doc.close()


def test_box_label_joins_its_value(tmp_path):
    pdf = tmp_path / "PO-55012.pdf"
    _po(pdf)
    texts = [a.raw_text.strip() for a in getattr(OrbitBriefPdfParser().parse(pdf), "atoms", [])]
    for label in ("Supplier", "Ship To", "Bill To", "Supplier Terms", "Message", "Disclaimer"):
        assert label not in texts, texts
    assert "Ship To: Contoso Ltd 12 Harbor Street Fl 3 Boston, MA 02110" in texts, texts
    assert "Bill To: Contoso Accounts Payable PO Box 4410 Boston, MA 02111" in texts, texts
    assert "Supplier: Northwind Install Co 40 River Road Albany, NY 12207" in texts, texts
    assert "Supplier Terms: Net 45" in texts, texts
    assert "Message: Please reference the PO number on every invoice." in texts, texts
    assert "Disclaimer: This order is governed by the buyer's standard terms." in texts, texts


def test_shaded_banner_over_long_text_stays_a_heading(tmp_path):
    """A shaded section banner over a long framed body is a heading, not a field."""
    pdf = tmp_path / "SOW.pdf"
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    lines = [f"Vendor will mount and connect display number {k} in the client office." for k in range(10)]
    _box(p, 21, 583, 100, "Scope of Work", lines)
    doc.save(str(pdf))
    doc.close()
    texts = [a.raw_text.strip() for a in getattr(OrbitBriefPdfParser().parse(pdf), "atoms", [])]
    assert not any(t.startswith("Scope of Work:") for t in texts), texts
