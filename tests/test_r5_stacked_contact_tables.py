"""Stacked ruled tables on a SOW's first page stay one record per row (010353, 010087).

Live 010353 (the SOW's PDF page 2) sets three framed
grids one above the other: the revision grid (SOW VERSION | QUOTED BY | DATE |
REVISION HISTORY), VENDOR SALES CONTACTS and CUSTOMER CONTACTS (FULL NAME |
JOB TITLE | EMAIL ADDRESS), the customer row wrapping onto a second line. The
parser read each grid right, but the compile then

* split every table row at its " | " into one atom per cell
  ("FULL NAME: Sam Carter |", "EMAIL ADDRESS: Lee"), and
* grouped those cells, and the rows, as a signature block -- "NAME:",
  "TITLE:" and "DATE:" look like signature labels -- read column by column
  across all three tables:
  "QUOTED BY: Alan Reyes | FULL NAME: Sam Carter, Lee Park-Owens /
  Chris Hall | JOB TITLE: Global Strategic Account Executive, ...",
  rewriting the first sales contact's own row as that record, so that contact had no row with their
  email. 010087's signed SOW showed the same ("QUOTED BY: Rob Lane |
  REVISION HISTORY: First | JOB T...").

The fixture uses the real page's coordinates (pdfplumber: column rules at
66.6/189.5/344.5/534.6 and 66.6/192.3/345.3/534.6, row tops 365.1..424.6 and
461.8..517.2, every rect drawn twice).
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")


def _text(pg, x, top, s, size=8, bold=False):
    pg.insert_text((x, top + size * 0.8), s, fontsize=size, fontname="hebo" if bold else "helv")


def _grid(pg, xs, ys):
    for _ in range(2):  # the real file strokes every cell twice
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                pg.draw_rect(fitz.Rect(xs[i], ys[j], xs[i + 1], ys[j + 1]), color=(0, 0, 0), width=0.5)


_HEAD = ["FULL NAME", "JOB TITLE", "EMAIL ADDRESS"]


def _sow(path: Path, *, revisions, sales, customer, customer_wrap=None) -> None:
    doc = fitz.open()
    pg = doc.new_page(width=612, height=792)
    _text(pg, 250, 40, "WWW.VENDOR.EXAMPLE")
    _text(pg, 72, 80, "INTRODUCTION", 12, True)
    xs = [66.6, 168.0, 290.0, 400.0, 534.6]
    ys = [110.0 + 21.0 * i for i in range(len(revisions) + 2)]
    _grid(pg, xs, ys)
    for row, top in zip([["SOW VERSION", "QUOTED BY", "DATE", "REVISION HISTORY"], *revisions], ys):
        for x, s in zip(xs, row):
            _text(pg, x + 5.5, top + 6.0, s, bold=row[0] == "SOW VERSION")
    _text(pg, 72, 195, "EXECUTIVE SUMMARY", 12, True)
    _text(pg, 72, 215, "This Project Services Statement of Work (\"SOW\") is made by and between "
                       "the Customer and Vendor LLC.", 9)
    # VENDOR SALES CONTACTS: the real geometry
    _text(pg, 72, 340, "VENDOR SALES CONTACTS", 12, True)
    ys = [365.1, 387.2] + [405.9 + 18.7 * i for i in range(len(sales))]
    _grid(pg, [66.6, 189.5, 344.5, 534.6], ys)
    for row, top in zip([_HEAD, *sales], [372.5, 389.0, 407.7, 426.4]):
        for x, s in zip([72.1, 194.9, 350.0], row):
            _text(pg, x, top, s, bold=row is _HEAD)
    # CUSTOMER CONTACTS: same header, one row wrapping onto a second line
    _text(pg, 72, 444.7, "CUSTOMER CONTACTS", 12, True)
    _grid(pg, [66.6, 192.3, 345.3, 534.6], [461.8, 484.5, 517.2])
    for x, s in zip([72.1, 197.8, 350.8], _HEAD):
        _text(pg, x, 469.5, s, bold=True)
    for i, x in enumerate([72.1, 197.8, 350.8]):
        _text(pg, x, 486.3, customer[i])
        if customer_wrap:
            _text(pg, x, 497.5, customer_wrap[i])
    _text(pg, 72, 540, "STATEMENT OF WORK (SOW)", 12, True)
    doc.save(str(path))


def _010353(tmp_path: Path) -> Path:
    pdf = tmp_path / "sow-a.pdf"
    _sow(pdf,
         revisions=[["1.0", "Alan Reyes", "September 17, 2026", "Site camera installation"]],
         sales=[["Sam Carter", "Director of Operations", "sam@vendor.example"],
                ["Dana Moore", "Global Strategic Account Executive", "dana@vendor.example"]],
         customer=["Lee Park-Owens / Chris", "Client Support Manager / Onsite",
                   "Lee Park-Owens: (555) 010-4200 / Chris"],
         customer_wrap=["Hall", "Contact", "Hall: Phone number not provided"])
    return pdf


def _010087(tmp_path: Path) -> Path:
    pdf = tmp_path / "sow-b.pdf"
    _sow(pdf,
         revisions=[["v.1", "Rob Lane", "07/09/2026", "First"],
                    ["v.2", "Sam Carter", "07/16/2026", "Second"]],
         sales=[["Kim Fox", "Executive VP Sales", "kim@vendor.example"],
                ["Rob Lane", "Solution Architect", "rob@vendor.example"]],
         customer=["Pat Ng", "Client Executive", "pat.ng@customer.example"])
    return pdf


SAM = "FULL NAME: Sam Carter | JOB TITLE: Director of Operations | EMAIL ADDRESS: sam@vendor.example"
DANA = ("FULL NAME: Dana Moore | JOB TITLE: Global Strategic Account Executive"
         " | EMAIL ADDRESS: dana@vendor.example")
LEE = ("FULL NAME: Lee Park-Owens / Chris Hall | JOB TITLE: Client Support Manager / Onsite Contact"
        " | EMAIL ADDRESS: Lee Park-Owens: (555) 010-4200 / Chris Hall: Phone number not provided")
REV = ("SOW VERSION: 1.0 | QUOTED BY: Alan Reyes | DATE: September 17, 2026"
       " | REVISION HISTORY: Site camera installation")


def test_each_stacked_grid_is_its_own_table_one_atom_per_row(tmp_path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse_artifact("p", "art", _010353(tmp_path))
    atoms = list(out if isinstance(out, list) else out.atoms)
    rows = {a.raw_text: a for a in atoms if (a.value or {}).get("kind") == "table_row"}
    assert set(rows) >= {REV, SAM, DANA, LEE}, list(rows)
    assert rows[SAM].value["cells"]["EMAIL ADDRESS"] == "sam@vendor.example"
    blocks = {rows[t].source_refs[0].locator.get("block_id") for t in (REV, SAM, LEE)}
    assert len(blocks) == 3, blocks
    for a in atoms:
        assert ", Lee" not in a.raw_text and not ("Carter" in a.raw_text and "Park-Owens" in a.raw_text), a.raw_text


@pytest.mark.parametrize("build", [_010353, _010087])
def test_compile_never_merges_the_tables_into_a_signature_block(tmp_path, build):
    from app.core.compiler import compile_project

    pdf = build(tmp_path)
    r = compile_project(pdf.parent, project_id="p", allow_errors=True, use_cache=False)
    texts = [a.raw_text for a in r.atoms]
    for a in r.atoms:
        assert (a.value or {}).get("kind") != "signature_block", a.raw_text
        t = a.raw_text
        # never the cells of two tables / two people in one atom
        assert not ("QUOTED BY" in t and "FULL NAME" in t), t
        assert not ("REVISION HISTORY" in t and "JOB TITLE" in t), t
        # a row is never cut into one atom per cell
        assert not t.rstrip().endswith("|"), t
        assert t not in ("EMAIL ADDRESS: Lee", "Park-Owens: (555) 010-4200 /"), t
    if build is _010353:
        assert SAM in texts and DANA in texts, texts
        assert REV in texts, texts
    else:
        assert ("FULL NAME: Kim Fox | JOB TITLE: Executive VP Sales"
                " | EMAIL ADDRESS: kim@vendor.example") in texts, texts
        assert ("SOW VERSION: v.1 | QUOTED BY: Rob Lane | DATE: 07/09/2026"
                " | REVISION HISTORY: First") in texts, texts


def test_a_record_row_is_not_a_signature_row_but_a_signature_table_still_is():
    from types import SimpleNamespace

    from app.core.atom_type_sanity import _is_table_record_row

    def _row(cells):
        return SimpleNamespace(raw_text=" | ".join(f"{k}: {v}" for k, v in cells.items()),
                               value={"kind": "table_row", "columns": list(cells), "cells": cells})

    assert _is_table_record_row(_row({"FULL NAME": "Sam Carter", "JOB TITLE": "Director",
                                      "EMAIL ADDRESS": "sam@vendor.example"}))
    assert _is_table_record_row(_row({"SOW VERSION": "1.0", "QUOTED BY": "Alan Reyes",
                                      "DATE": "September 17, 2026"}))
    # a signature table's cells carry their own label: still a signature row
    assert not _is_table_record_row(_row({"CDW Technologies LLC": "By: Mike Murphy",
                                          "NewBold LLC": "By: Shelly Lewis"}))
    # no header row (col_N) or not a table row: left to the signature test
    assert not _is_table_record_row(SimpleNamespace(raw_text="Name: A | Title: B",
                                                    value={"kind": "table_row", "columns": ["col_0", "col_1"]}))
    assert not _is_table_record_row(SimpleNamespace(raw_text="Name: A | Title: B", value={"kind": "paragraph"}))
