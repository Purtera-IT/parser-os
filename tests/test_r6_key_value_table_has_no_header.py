"""A LABEL | VALUE grid has no header row.

A signed SOW's cover table sets each field's label in column 0 (bold, on a
grey band), its value in column 1, and two side cells that each span several
rows in column 2 (a sales contact, "Prepared By: <name>"). Read with row 0 as
the header, the first field named the columns: every row read "Job Name:
Client Name: | Lobby Display Refresh: Example Retail LLC" and no atom said
what the job was called (010003). Each field is now its own row atom, row 0
included, and each side cell is emitted once.

Geometry mirrors the real table (cell edges, filled label bands, side cells
spanning rows 1-3 and 4-5, the drawing order of the text); the words are
invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

LABELS = ["Job Name:", "Client Name:", "Vendor Name:", "Partner Firm:", "Date:"]
VALUES = ["Lobby Display Refresh", "Example Retail LLC", "Sample Services LLC",
          "Example Partner Inc", "March 03, 2026"]
SIDE_A = ["Account Manager:", "Jordan Lee", "+1 (555) 0100", "jordan.lee@example.com"]
SIDE_B = ["Prepared By:", "Alex Morgan"]

ROW_Y = [157.9, 169.8, 181.8, 204.3, 216.3, 240.1]
COL_X = [54.5, 157.4, 371.2, 558.0]


def _padding(page, y: float) -> None:
    page.insert_text((54, y), "SCOPE OF WORK", fontsize=11, fontname="tibo")
    for k in range(16):
        page.insert_text((54, y + 20 + k * 14),
                         f"Item {k + 1}: the vendor will mount and test one display, "
                         "route the cabling and confirm the picture works.",
                         fontsize=10, fontname="tiro")


def _kv_sow(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 100), "STATEMENT OF WORK", fontsize=14, fontname="tibo")
    page.insert_text((54, 140), "PROJECT DESCRIPTION", fontsize=11, fontname="tibo")
    x0, x1, x2, x3 = COL_X
    for top, bot in zip(ROW_Y, ROW_Y[1:]):
        page.draw_rect(fitz.Rect(x0, top + 0.1, x1 - 0.2, bot - 0.3),
                       color=None, fill=(0.85, 0.85, 0.85), width=0)
        page.draw_line((x1, top), (x1, bot), width=0.5)
    for x in (x0, x3):
        page.draw_line((x, ROW_Y[0]), (x, ROW_Y[-1]), width=0.5)
    page.draw_line((x2, ROW_Y[0]), (x2, ROW_Y[3]), width=0.5)
    page.draw_line((x2, ROW_Y[3]), (x2, ROW_Y[-1]), width=0.5)
    for i, y in enumerate(ROW_Y):
        # Rows 1-3 and 4-5 share one side cell: their rules stop at column 2.
        page.draw_line((x0, y), (x3 if i in (0, 3, 5) else x2, y), width=0.5)

    def label_row(i: int) -> None:
        base = ROW_Y[i] + 9.4
        page.insert_text((58.1, base), LABELS[i], fontsize=10, fontname="tibo")
        page.insert_text((161.3, base), VALUES[i], fontsize=10, fontname="tiro")

    for i in (0, 1, 2):
        label_row(i)
    for k, t in enumerate(SIDE_A):
        page.insert_text((375.1, 167.3 + k * 11.5), t, fontsize=10,
                         fontname="tibo" if k == 0 else "tiro")
    for i in (3, 4):
        label_row(i)
    for k, t in enumerate(SIDE_B):
        page.insert_text((375.1, 220.1 + k * 11.5), t, fontsize=10,
                         fontname="tibo" if k == 0 else "tiro")
    _padding(page, 280)
    doc.save(str(path))
    doc.close()


def _row_atoms(pdf: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    atoms = list(OrbitBriefPdfParser().parse(pdf).atoms)
    return atoms, [a for a in atoms if (a.value or {}).get("kind") == "table_row"]


def test_each_field_is_its_own_row_and_each_side_cell_is_emitted_once(tmp_path):
    pdf = tmp_path / "sow.pdf"
    _kv_sow(pdf)
    atoms, rows = _row_atoms(pdf)
    texts = [a.raw_text for a in rows]
    fields = [f"{lab} {val}" for lab, val in zip(LABELS, VALUES)]
    side_a, side_b = " ".join(SIDE_A), " ".join(SIDE_B)
    assert texts == [fields[0], side_a, fields[1], fields[2], fields[3], side_b, fields[4]], texts
    # No row is keyed by another row's label or value.
    for a in atoms:
        assert "Job Name: Client Name:" not in (a.raw_text or "")
        assert "Lobby Display Refresh:" not in (a.raw_text or "")
    # Each row's words sit together on the page, so the source pane finds them.
    from app.parsers.cell_fragments import in_source, pdf_page_text

    with fitz.open(str(pdf)) as doc:
        page_text = pdf_page_text(doc[0])[0]
    for t in texts:
        assert in_source(t, page_text), t


def test_key_value_rows_are_records_not_a_signature_block(tmp_path):
    from app.core.atom_type_sanity import merge_signature_rows

    pdf = tmp_path / "sow.pdf"
    _kv_sow(pdf)
    _atoms, rows = _row_atoms(pdf)
    before = [a.raw_text for a in rows]
    assert merge_signature_rows(rows) == 0
    assert [a.raw_text for a in rows] == before


def _header_table(path: Path, *, bold_first_column: bool) -> None:
    """An ordinary grid: a bold header row over plain data rows."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 80), "STATEMENT OF WORK", fontsize=14, fontname="hebo")
    xs = [54, 160, 300, 400, 558]
    ys = [110, 126, 142, 158]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), width=0.6)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=0.6)
    rows = [["VERSION", "QUOTED BY", "DATE", "REVISION"],
            ["v.1", "Jordan Lee", "03/01/2026", "First"],
            ["v.2", "Alex Morgan", "03/09/2026", "Second"]]
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            bold = r == 0 or (bold_first_column and c == 0)
            page.insert_text((xs[c] + 4, ys[r] + 11), t, fontsize=9,
                             fontname="hebo" if bold else "helv")
    _padding(page, 190)
    doc.save(str(path))
    doc.close()


@pytest.mark.parametrize("bold_first_column", [False, True])
def test_a_grid_with_a_header_row_keeps_it(tmp_path, bold_first_column):
    pdf = tmp_path / "rev.pdf"
    _header_table(pdf, bold_first_column=bold_first_column)
    _atoms, rows = _row_atoms(pdf)
    assert [a.raw_text for a in rows] == [
        "VERSION: v.1 | QUOTED BY: Jordan Lee | DATE: 03/01/2026 | REVISION: First",
        "VERSION: v.2 | QUOTED BY: Alex Morgan | DATE: 03/09/2026 | REVISION: Second",
    ]
    assert all(not (a.value or {}).get("key_value") for a in rows)


def _two_col(path: Path, header: tuple[str, str] | None) -> list[tuple[str, str]]:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 80), "STATEMENT OF WORK", fontsize=14, fontname="hebo")
    body = [("Site:", "Building A"), ("Contact:", "Jordan Lee"), ("Window:", "After hours")]
    grid = ([header] if header else []) + body
    ys = [110 + 16 * i for i in range(len(grid) + 1)]
    for x in (54, 200, 558):
        page.draw_line((x, ys[0]), (x, ys[-1]), width=0.6)
    for y in ys:
        page.draw_line((54, y), (558, y), width=0.6)
    for r, (a, b) in enumerate(grid):
        bold = header is not None and r == 0
        page.insert_text((58, ys[r] + 11), a, fontsize=9, fontname="hebo" if bold else "helv")
        page.insert_text((204, ys[r] + 11), b, fontsize=9, fontname="hebo" if bold else "helv")
    _padding(page, 190)
    doc.save(str(path))
    doc.close()
    return body


def test_a_two_column_label_grid_without_bold_or_fill_is_read_by_its_colons(tmp_path):
    pdf = tmp_path / "two.pdf"
    body = _two_col(pdf, None)
    _atoms, rows = _row_atoms(pdf)
    assert [a.raw_text for a in rows] == [f"{a} {b}" for a, b in body]


def test_a_two_column_grid_with_a_bold_header_keeps_it(tmp_path):
    pdf = tmp_path / "two_h.pdf"
    _two_col(pdf, ("Field", "Detail"))
    _atoms, rows = _row_atoms(pdf)
    assert [a.raw_text for a in rows][0].startswith("Field: Site:"), [a.raw_text for a in rows]
    assert all(not (a.value or {}).get("key_value") for a in rows)
