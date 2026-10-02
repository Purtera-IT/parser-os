"""A row drawn as one cell across the grid still has columns.

A PO line grid rules its header cells but draws each item row with only
horizontal rules, so the grid finder reads the row as ONE cell spanning every
column. Keyed by position, all of that cell's text went to the first column:
on 010003's PO the row has no dates, and the atom read "Start Date: <spend
category> <amount> <description>". Each word now goes to the header column it
sits under by x, and the date columns stay empty.

Structure mirrors the real PO (header coordinates, a multi-line description,
no date values); the words are invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

HEADER = ["Start Date", "End Date", "Spend Category", "Ordered Amount", "Line Description"]
DESC = ["REQ0000012345-Example", "Services-2026.01-Onsite", "Support for HQ Lobby-", "Final Visit"]


def _po(path: Path, *, note: str | None = None) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((58, 80), "Purchase Order", fontsize=16, fontname="hebo")
    page.insert_text((58, 110), "Supplier: Example Services LLC   Currency: USD", fontsize=10)
    page.insert_text((58, 124), "Ship To: 1 Example Plaza, Springfield, IL 62701", fontsize=10)
    xs = [58, 150, 245, 370, 440, 582]
    top, hb, bot = 525, 539, 583
    for x in xs:
        page.draw_line((x, top), (x, hb), width=0.6)
    page.draw_line((xs[0], hb), (xs[0], bot), width=0.6)
    page.draw_line((xs[-1], hb), (xs[-1], bot), width=0.6)
    for y in (top, hb, bot):
        page.draw_line((xs[0], y), (xs[-1], y), width=0.6)
    for x, t in zip([62.92, 155, 250, 373, 445], HEADER):
        page.insert_text((x, 535.5), t, fontsize=8.5, fontname="hebo")
    y0 = 547.5
    if note:
        page.insert_text((62, y0), note, fontsize=8.5)
    else:
        page.insert_text((251.87, y0), "Facilities-Contract labor", fontsize=8.5)
        page.insert_text((385, y0), "$2,480.00", fontsize=8.5)
        for k, t in enumerate(DESC):
            page.insert_text((445, y0 + k * 10.3), t, fontsize=8.5)
    page.insert_text((58, 640), "Please quote the PO number on every invoice you send.", fontsize=10)
    doc.save(str(path))
    doc.close()


def test_each_word_goes_to_the_column_it_sits_under(tmp_path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "po.pdf"
    _po(pdf)
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    rows = [a for a in atoms if (a.value or {}).get("kind") == "table_row"]
    assert len(rows) == 1, [a.raw_text for a in atoms]
    v = rows[0].value
    assert v["columns"] == HEADER
    cells = {k: " ".join(str(c).split()) for k, c in v["cells"].items() if str(c).strip()}
    assert cells == {
        "Spend Category": "Facilities-Contract labor",
        "Ordered Amount": "$2,480.00",
        "Line Description": " ".join(DESC),
    }
    assert not rows[0].raw_text.startswith("Start Date")


def test_a_note_across_the_grid_is_left_whole(tmp_path):
    from app.parsers.pdf._shared import _table_rows_repaired

    pdf = tmp_path / "po_note.pdf"
    note = "All work under this order is billed against the master services agreement."
    _po(pdf, note=note)
    with fitz.open(str(pdf)) as doc:
        page = doc[0]
        tables = page.find_tables(strategy="lines").tables
        assert tables, "the grid was not found"
        rows = _table_rows_repaired(page, tables[0])
    assert [str(c or "").strip() for c in rows[0]] == HEADER
    assert " ".join(str(rows[1][0]).split()) == note
    assert all(c is None for c in rows[1][1:])


def _po_titled(path: Path) -> None:
    """The real PO's own drawing: a filled title band merged across the grid,
    filled header cells, and only the outer border round the item row."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((22, 80), "Purchase Order", fontsize=16, fontname="hebo")
    page.insert_text((22, 110), "Order lines are listed below.", fontsize=9)
    grey = (0.85, 0.85, 0.85)
    edges = [21.0, 133.4, 245.9, 358.3, 470.8, 583.2]

    def fill(r):
        page.draw_rect(fitz.Rect(*r), color=None, fill=grey, width=0)

    fill((21.0, 513.5, 583.2, 527.3))
    for x0, x1 in zip(edges, edges[1:]):
        fill((x0, 527.3, x1, 538.4))
    for a, b in [((22.1, 513.5), (22.1, 585.1)), ((582.1, 513.5), (582.1, 585.1)),
                 ((21.0, 514.6), (583.2, 514.6)), ((21.0, 584.0), (583.2, 584.0))]:
        page.draw_line(a, b, width=0.8)
    page.insert_text((265.5, 525), "Order Lines", fontsize=8)
    for x, t in zip([62.9, 175.4, 258.9, 389.1, 501.2], HEADER):
        page.insert_text((x, 536.7), t, fontsize=7)
    page.insert_text((251.9, 548), "Facilities-Contract labor", fontsize=7)
    page.insert_text((396.3, 548), "$2,480.00", fontsize=7)
    for k, t in enumerate(DESC):
        page.insert_text((476.0, 548 + k * 9.6), t, fontsize=7)
    doc.save(str(path))
    doc.close()


def test_a_title_band_above_the_header_does_not_hide_the_columns(tmp_path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "po_titled.pdf"
    _po_titled(pdf)
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    rows = [a for a in atoms if (a.value or {}).get("kind") == "table_row"
            and "Ordered Amount" in ((a.value or {}).get("columns") or [])]
    assert len(rows) == 1, [a.raw_text for a in atoms]
    cells = {k: " ".join(str(c).split()) for k, c in rows[0].value["cells"].items() if str(c).strip()}
    assert cells == {
        "Spend Category": "Facilities-Contract labor",
        "Ordered Amount": "$2,480.00",
        "Line Description": " ".join(DESC),
    }
