"""A table drawn from filled cell bands, with no ruling lines, is a table.

A Word table exported to PDF shades each cell instead of ruling it: a filled
band per cell, an inset band per cell for the text margin, hairline rule
rects only around the header. The 010087 fee table is that shape. Its header
is two lines ("STATED RATE" over "(USD)") with the first header cell merged
down across both lines, and the totals row is one cell merged across four
columns.

The grid finder took every band edge as a wall, so each column came back as
three (margin, text, margin). The merged totals cell then sat across the
walls of the rows above it, the "this grid cuts words, so it is a figure"
check counted that as cut words and threw the table out, and the page read
the header and the body row as prose: each header word a section heading,
each cell its own atom. Only the totals row survived, as a one-column table.

The structure here mirrors the real page (coordinates, fills, the merged
cells); the words are invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

BLUE = (0, 0.439, 0.753)
RULE = (0.749,) * 3
BAND = (0.949,) * 3
WHITE = (1, 1, 1)
COLS = [(58.8, 211.3), (211.8, 301.3), (301.8, 377.9), (378.4, 440.8), (441.5, 558.0)]
PAD = 5.4

HEADER = ["ITEM", "UNIT PRICE (USD)", "PRICING BASIS", "QTY. ORDERED", "LINE TOTAL (USD)"]
BODY = ["Technician (2 hrs per visit)", "$85.00", "Hourly", "40", "$3,400.00"]


def _fee_table(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((58.8, 330), "PROJECT PRICING", fontsize=12, fontname="hebo")
    page.insert_text((58.8, 352), "The services below are billed at the rates shown in the table.",
                     fontsize=10)
    page.insert_text((58.8, 366), "Quantities are estimates and are trued up on the final invoice.",
                     fontsize=10)

    def fill(r, c):
        page.draw_rect(fitz.Rect(*r), color=None, fill=c, width=0)

    fill((58.3, 380.8, 558.5, 381.3), RULE)
    # The first header cell is one band down both header lines; the others
    # carry an inset text band per line.
    fill((58.8, 381.2, 211.3, 406.7), BLUE)
    for x0, x1 in COLS[1:]:
        fill((x0, 381.2, x1, 406.7), BLUE)
        fill((x0 + PAD, 381.2, x1 - PAD, 394.0), BLUE)
        fill((x0 + PAD, 394.0, x1 - PAD, 406.7), BLUE)
    for x in (58.3, 211.3, 301.3, 377.9, 441.1, 558.0):
        fill((x, 381.2, x + 0.5, 406.7), RULE)
    for x0, x1 in COLS:
        fill((x0, 407.2, x1, 427.8), BAND)
        fill((x0 + PAD, 407.2, x1 - PAD, 427.8), BAND)
    # The totals row: one cell across the first four columns.
    fill((58.8, 428.3, 440.8, 456.0), BAND)
    fill((441.5, 428.3, 558.0, 456.0), BAND)

    def word(x, top, text, colour=(0, 0, 0), bold=False):
        page.insert_text((x, top + 8.6), text, fontsize=9.5,
                         fontname="hebo" if bold else "helv", color=colour)

    for x, t in [(222.0, "UNIT PRICE"), (317.9, "PRICING"), (394.0, "QTY."), (455.0, "LINE TOTAL")]:
        word(x, 382.8, t, WHITE, True)
    word(120.0, 389.3, "ITEM", WHITE, True)
    for x, t in [(241.3, "(USD)"), (322.0, "BASIS"), (390.0, "ORDERED"), (484.3, "(USD)")]:
        word(x, 395.6, t, WHITE, True)
    for x, t in zip([67.0, 239.9, 324.1, 403.3, 475.1], BODY):
        word(x, 408.8, t)
    word(320.0, 435.3, "TOTAL OF ALL LINES", bold=True)
    word(475.1, 435.3, "$3,400.00", bold=True)

    page.insert_text((58.8, 490), "INVOICING", fontsize=12, fontname="hebo")
    page.insert_text((58.8, 510), "Invoices are payable within thirty days of receipt.", fontsize=10)
    doc.save(str(path))
    doc.close()


def _atoms(pdf: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(pdf)
    return list(getattr(out, "atoms", out))


def _loc(atom) -> dict:
    return dict(atom.source_refs[0].locator) if atom.source_refs else {}


def test_the_grid_reads_as_five_named_columns(tmp_path):
    from app.parsers.orbitbrief_pdf import _extract_ruled_tables

    pdf = tmp_path / "pricing.pdf"
    _fee_table(pdf)
    blocks, bboxes = _extract_ruled_tables(pdf, 0)
    assert len(blocks) == 1, blocks
    assert blocks[0]["columns"] == HEADER
    assert blocks[0]["rows"][0] == dict(zip(HEADER, BODY))
    # the merged totals value sits under its own column, not the margin's
    assert blocks[0]["rows"][1] == {"ITEM": "TOTAL OF ALL LINES", "LINE TOTAL (USD)": "$3,400.00"}
    assert bboxes[0].y0 < 382 and bboxes[0].y1 > 455


def test_one_row_atom_per_body_row_and_no_header_sections(tmp_path):
    pdf = tmp_path / "pricing.pdf"
    _fee_table(pdf)
    atoms = _atoms(pdf)
    rows = [a for a in atoms if (a.value or {}).get("kind") == "table_row"]
    body = [a for a in rows if (a.value or {}).get("cells", {}).get("ITEM") == BODY[0]]
    assert len(body) == 1, [a.raw_text for a in rows]
    assert body[0].value["columns"] == HEADER
    assert body[0].value["cells"] == dict(zip(HEADER, BODY))

    header_words = {"ITEM", "UNIT PRICE", "PRICING", "BASIS", "QTY.", "ORDERED",
                    "LINE TOTAL", "(USD)"}
    for a in atoms:
        path = _loc(a).get("section_path") or []
        assert not header_words & set(path), (a.raw_text, path)
        if (a.value or {}).get("kind") != "table_row":
            # no cell of the table survives as prose of its own
            assert a.raw_text.strip() not in header_words | set(BODY), a.raw_text


def test_a_grid_through_words_is_still_not_a_table():
    """The figure check keeps its teeth: walls on a word's own line cut it."""
    from types import SimpleNamespace

    from app.parsers.pdf.page_kind import table_cuts_words

    cells = [(0, 0, 50, 20), (50, 0, 100, 20), (0, 20, 50, 40), (50, 20, 100, 40)]
    words = [(30, 2, 70, 12, "RECEPTION", 0, 0, 0), (35, 22, 65, 32, "OFFICE", 0, 1, 0)]
    page = SimpleNamespace(get_text=lambda kind: words)
    table = SimpleNamespace(cells=cells, bbox=(0, 0, 100, 40))
    assert table_cuts_words(page, table)

    # the same words inside cells merged across that wall are not cut
    merged = SimpleNamespace(cells=[(0, 0, 100, 20), (0, 20, 100, 40), (0, 40, 50, 60),
                                    (50, 40, 100, 60)], bbox=(0, 0, 100, 60))
    assert not table_cuts_words(page, merged)
