"""Boxes standing side by side on a PDF page are never one atom (010003).

The CDW BOM's ship-to box ("SHIP TO:" / company / street / "NEW YORK, NY
10014-1066" / "Phone: ...") stands beside a "Shipping Method" box that
starts lower. The whitespace-column extractor paired their shared lines
into a table and the layout reader, cut off from the box's head, read the
two shared rows as a header over values: "NEW YORK, NY 10014-1066: Phone:
... | Shipping Method: DROP SHIP-GROUND: ...". The box's all-caps company
and street lines were taken for section headings and vanished. With three
framed boxes over a BOM, the BOM was read column by column under the boxes'
gutters ("CDW# 7506872 5502114").
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SHIP = ["SHIP TO:", "ACME CORPORATION", "Attention To: SARAH HALPERN", "40 10TH AVE FL 4",
        "NEW YORK, NY 10014-1066", "Phone: (212) 555-0199"]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _head(p) -> None:
    p.insert_text((36, 50), "QUOTE CONFIRMATION", fontsize=16, fontname="hebo")
    p.insert_text((36, 80), "Dear Sarah, thank you for considering CDW for your technology needs. "
                  "The details of your quote are below.", fontsize=9)


def _bom(p, y: float) -> None:
    cols = [36, 100, 170, 400, 445, 520]
    rows = [["CDW#", "Mfg#", "Description", "Qty", "Unit Price", "Ext Price"],
            ["7506872", "QM75C", "Samsung QM75C 75in display", "12", "$1,077.30", "$12,927.60"],
            ["5502114", "WMN6575SE", "Samsung Slim Fit Wall Mount", "12", "$89.99", "$1,079.88"]]
    for i, r in enumerate(rows):
        for x, c in zip(cols, r):
            p.insert_text((x, y + 14 * i), c, fontsize=8, fontname="hebo" if i == 0 else "helv")


def _ship_beside_method(p) -> None:
    _head(p)
    for i, line in enumerate(SHIP):
        p.insert_text((36, 120 + 12 * i), line, fontsize=9, fontname="hebo" if i == 0 else "helv")
    p.insert_text((250, 168), "Shipping Method: DROP SHIP-GROUND", fontsize=9)
    p.insert_text((250, 180), "Payment Terms: Net 30 Days", fontsize=9)
    _bom(p, 230)


def _three_framed_boxes(p) -> None:
    _head(p)
    boxes = (
        (40, SHIP),
        (220, ["BILL TO:", "ACME CORPORATION", "Accounts Payable", "PO BOX 1234", "NEW YORK, NY 10008-1234"]),
        (400, ["Shipping Method: DROP SHIP-GROUND", "Payment Terms: Net 30 Days"]),
    )
    for x, lines in boxes:
        for i, line in enumerate(lines):
            p.insert_text((x, 120 + 12 * i), line, fontsize=8, fontname="hebo" if line.endswith(":") else "helv")
    for x0, x1 in ((36, 210), (215, 390), (395, 580)):
        p.draw_rect(fitz.Rect(x0, 108, x1, 190))
    _bom(p, 230)


@pytest.mark.parametrize("build", [_ship_beside_method, _three_framed_boxes])
def test_boxes_side_by_side_are_separate_atoms(tmp_path: Path, build) -> None:
    path = tmp_path / "CDW Quote.pdf"
    doc = fitz.open()
    build(doc.new_page(width=612, height=792))
    doc.save(str(path))
    out = OrbitBriefPdfParser().parse_artifact("p", "a", path)
    texts = [" ".join(a.raw_text.split()) for a in (out if isinstance(out, list) else out.atoms)]
    for t in texts:
        assert not ("10014-1066" in t and "Shipping Method" in t), texts
        assert not ("Phone:" in t and "Payment Terms" in t), texts
    for value in ("ACME CORPORATION", "40 10TH AVE FL 4", "DROP SHIP-GROUND", "Net 30 Days"):
        assert any(value in t for t in texts), (value, texts)
    # the BOM below the boxes still reads row by row
    assert any("7506872" in t and "$12,927.60" in t for t in texts), texts
    assert not any("7506872" in t and "5502114" in t for t in texts), texts
