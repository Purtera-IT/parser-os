"""A PO's label boxes read as one field each (010003 PO).

A purchase order sets each field under a label on a filled grey header cell
with no stroke, in a plain serif face ("Ship To", 8.3pt), over its value in
another face. The boxes are drawn with lines, and PyMuPDF's table finder
drops a one-column box, so they reach the layout reader as text, where the
fill change ended a block: every label became an atom apart from its value.
Two full-width unruled bands ("Disclaimer", "Message") label the lines under
them, and were read inside the left column, before the right column's boxes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

GREY = (0.5, 0.5, 0.5)
LIGHT = (0.75, 0.75, 0.75)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _box(p, x0, x1, top, label, values, size=8.3, vfont="helv", bottom=None):
    """A label on a filled header cell over a box ruled with lines."""
    bar = top + 13.0
    p.draw_rect(fitz.Rect(x0, top, x1, bar), color=None, fill=GREY)
    p.insert_text((x0 + 2.5, bar - 3.0), label, fontsize=size, fontname="tiro")
    bottom = bottom or bar + 6 + 11.5 * max(1, len(values))
    for x in (x0 + 0.4, x1 - 0.4):
        p.draw_line((x, top), (x, bottom), width=0.6)
    for y in (top + 0.4, bar + 0.4, bottom - 0.4):
        p.draw_line((x0, y), (x1, y), width=0.6)
    for k, v in enumerate(values):
        p.insert_text((x0 + 2.5, bar + 12 + 11.5 * k), v, fontsize=10, fontname=vfont)
    return bottom


def _band(p, top, label, lines, fill=GREY):
    p.draw_rect(fitz.Rect(21, top, 583, top + 11.7), color=None, fill=fill)
    p.insert_text((22, top + 9.5), label, fontsize=8.3, fontname="helv")
    for k, ln in enumerate(lines):
        p.insert_text((21.3, top + 22 + 11.5 * k), ln, fontsize=10, fontname="helv")


def _po(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    _box(p, 21, 209, 218, "Supplier", ["Northwind Install Co", "40 River Road", "Albany, NY 12207"], size=10)
    _box(p, 348, 583, 218, "Ship To", ["Pat Lee (5501)", "12 Harbor Street", "Suite 300", "Boston, MA 02110"])
    _box(p, 21, 209, 297, "Supplier Terms", ["Net 45"], size=10, vfont="tiro")
    # Two touching boxes with no values.
    _box(p, 21, 209, 332, "Supplier Method", [], size=10, bottom=362)
    _box(p, 21, 209, 365, "Comments", [], bottom=394)
    _box(p, 348, 583, 365, "Bill To", ["Accounts Payable Dept.", "PO Box 4410", "Boston, MA 02111"])
    _band(p, 591, "Disclaimer", ["Note: these orders are not transferable."])
    _band(p, 625, "Message", [
        "Send invoices to ap@example.com",
        "Shipping Address must appear on all invoices and packages.",
        "Purchase Order Number must appear on all correspondence, invoices and packages.",
    ], fill=LIGHT)
    doc.save(str(path))
    doc.close()


def _texts(pdf: Path) -> list[str]:
    return [a.raw_text.strip() for a in getattr(OrbitBriefPdfParser().parse(pdf), "atoms", [])]


def test_box_label_joins_its_value(tmp_path):
    pdf = tmp_path / "PO-55012.pdf"
    _po(pdf)
    texts = _texts(pdf)
    for label in ("Supplier", "Ship To", "Bill To", "Supplier Terms", "Message", "Disclaimer"):
        assert label not in texts, texts
    assert "Supplier: Northwind Install Co 40 River Road Albany, NY 12207" in texts, texts
    assert "Ship To: Pat Lee (5501) 12 Harbor Street Suite 300 Boston, MA 02110" in texts, texts
    assert "Bill To: Accounts Payable Dept. PO Box 4410 Boston, MA 02111" in texts, texts
    assert "Supplier Terms: Net 45" in texts, texts
    # Empty touching boxes stay their own labels, never one atom.
    assert "Supplier Method" in texts and "Comments" in texts, texts
    assert "Disclaimer: Note: these orders are not transferable." in texts, texts
    message = [t for t in texts if t.startswith("Message: ")]
    assert message == [
        "Message: Send invoices to ap@example.com",
        "Message: Shipping Address must appear on all invoices and packages.",
        "Message: Purchase Order Number must appear on all correspondence, invoices and packages.",
    ], texts
    # Both columns finish before the full-width bands.
    assert texts.index(next(t for t in texts if t.startswith("Bill To:"))) < texts.index(message[0])


def test_bold_banner_heading_is_not_a_field(tmp_path):
    """A bold heading on a filled banner over body text stays a heading."""
    pdf = tmp_path / "SOW.pdf"
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.draw_rect(fitz.Rect(21, 100, 583, 116), color=None, fill=GREY)
    p.insert_text((24, 112), "Scope of Work", fontsize=12, fontname="hebo")
    for k in range(4):
        p.insert_text((24, 132 + 12 * k), f"Vendor will mount and connect display number {k} in the office.",
                      fontsize=10, fontname="helv")
    doc.save(str(pdf))
    doc.close()
    assert not any(t.startswith("Scope of Work:") for t in _texts(pdf))
