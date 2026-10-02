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
