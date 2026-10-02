"""A PO header's shaded label column has no header row (010003 PO).

The PO's top-right header table sets each label ("Purchase Order Number",
Times 8.3) on a grey filled cell beside its value (Helvetica 10). Read with
row 0 as the header, every row came out "Purchase Order Number: Purchase
Order Date | PO-...: Jul 15, 2026". Label cells filled, value cells not, in
another face: it is a LABEL | VALUE grid, one field per row (#320's shape).
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

FIELDS = [("Order Number", "PO-00055012"), ("Order Date", "Aug 3, 2026"), ("Payment Terms", "Net 30"),
          ("Buyer", "Pat Lee (5501)"), ("Phone Number", "+1 (617) 5550100")]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _grid(path: Path, label_font: str, label_size: float) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((21, 90), "Northwind Purchasing", fontsize=10, fontname="helv")
    tops = [80.4, 104.4, 120.6, 138.0, 153.3, 171.5]
    for (lab, val), a, b in zip(FIELDS, tops, tops[1:]):
        p.draw_rect(fitz.Rect(347.9, a, 440.4, b), color=None, fill=(0.5, 0.5, 0.5))
        p.insert_text((350.4, b - 4), lab, fontsize=label_size, fontname=label_font)
        p.insert_text((442.9, b - 4), val, fontsize=10, fontname="helv")
    for y in tops:
        p.draw_line((347.9, y), (583.8, y), width=0.6)
    for x in (347.9, 440.4, 583.8):
        p.draw_line((x, tops[0]), (x, tops[-1]), width=0.6)
    doc.save(str(path))
    doc.close()


def _rows(pdf: Path) -> list[str]:
    atoms = getattr(OrbitBriefPdfParser().parse(pdf), "atoms", [])
    return [a.raw_text.strip() for a in atoms if (a.source_refs[0].locator or {}).get("block_kind") == "table"]


def test_shaded_label_column_reads_as_fields(tmp_path):
    pdf = tmp_path / "PO-55012.pdf"
    _grid(pdf, "tiro", 8.3)
    assert _rows(pdf) == [f"{lab}: {val}" for lab, val in FIELDS]


def test_shaded_column_in_the_value_face_keeps_its_header(tmp_path):
    """Same face and size on both sides: no label cue, the grid is unchanged."""
    pdf = tmp_path / "grid.pdf"
    _grid(pdf, "helv", 10)
    assert not any(r == "Order Number: PO-00055012" for r in _rows(pdf))
