"""Atoms rebuilt from table cells name where each cell is.

Deal 010003, CDW BOM PDF: "SUBTOTAL | $4,309.20", "SHIPPING | $0.00", "QUOTE
#: PSNV676 | QUOTE DATE: ..." (headers in one row, values in the next) and
"Need Help? | My Account | Support | Call 800.800.4239" are not on the page
as one string, so the labeling view's source pane highlighted nothing for 21
atoms. Each such atom now carries ``locator["cell_fragments"]`` -- every
cell's text and where it is (PDF page + bbox; docx table/row/col; sheet
row/col) -- and the viewer marks them all. Grouping and text are untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsers.cell_fragments import locate_fragments, pieces_of


def _frags(atom):
    return (atom.source_refs[0].locator or {}).get("cell_fragments")


# ── the matcher ─────────────────────────────────────────────────────────────


def test_pieces_split_rows_and_header_value_pairs():
    assert pieces_of("SUBTOTAL | $4,309.20") == [["SUBTOTAL"], ["$4,309.20"]]
    assert pieces_of("QUOTE #: PSNV676 | QUOTE DATE: 1/14/2026") == [
        ["QUOTE #: PSNV676", "QUOTE #", "PSNV676"],
        ["QUOTE DATE: 1/14/2026", "QUOTE DATE", "1/14/2026"],
    ]


def test_verbatim_text_gets_no_fragments():
    src = "Line one\nSUBTOTAL | $4,309.20\n"
    assert locate_fragments("SUBTOTAL | $4,309.20", src) is None
    # whitespace never decides it
    assert locate_fragments("The   installer will\nmount", "The installer will mount it") is None


def test_header_row_over_value_row():
    src = "QUOTE # QUOTE DATE\nPSNV676 1/14/2026\n"
    got = locate_fragments("QUOTE #: PSNV676 | QUOTE DATE: 1/14/2026", src)
    assert [f["text"] for f in got] == ["QUOTE #", "PSNV676", "QUOTE DATE", "1/14/2026"]
    for f in got:
        assert src[f["char_start"]:f["char_end"]] == f["text"]


def test_a_repeated_value_is_taken_beside_its_label():
    src = "Item A $0.00\nItem B $0.00\nSUBTOTAL\n$9.00\nSHIPPING\n$0.00\n"
    got = locate_fragments("SHIPPING | $0.00", src)
    assert [f["text"] for f in got] == ["SHIPPING", "$0.00"]
    assert got[1]["char_start"] == src.rindex("$0.00")


def test_a_copy_an_earlier_atom_claimed_is_passed_over():
    src = "GRAND TOTAL\n$5.00\nSUBTOTAL $5.00\nGRAND TOTAL $5.00\n"
    taken: set = set()
    first = locate_fragments("GRAND TOTAL: $5.00", src, taken=taken)
    second = locate_fragments("GRAND TOTAL | $5.00", src, taken=taken)
    assert first[0]["char_start"] == 0
    assert second[0]["char_start"] == src.rindex("GRAND TOTAL")
    assert second[1]["char_start"] == src.rindex("$5.00")


def test_too_little_found_is_no_fragments():
    assert locate_fragments("Nothing | Here", "completely different words") is None


# ── PDF ─────────────────────────────────────────────────────────────────────

fitz = pytest.importorskip("fitz")


def _cdw_bom(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 60), "QUOTE CONFIRMATION", fontsize=18, fontname="hebo")
    page.insert_text((36, 90), "Thank you for considering CDW for your technology needs. "
                     "The details of your quote are below.", fontsize=10)
    xs = [36, 140, 240, 380, 480]
    y = 120
    for x, h in zip(xs, ["QUOTE #", "QUOTE DATE", "QUOTE REFERENCE", "CUSTOMER #", "GRAND TOTAL"]):
        page.insert_text((x, y), h, fontsize=9, fontname="hebo")
    y += 14
    for x, v in zip(xs, ["PSNV676", "1/14/2026", "SAMSUNG", "15018865", "$4,691.64"]):
        page.insert_text((x, y), v, fontsize=10)
    y = 180
    page.insert_text((36, y), "Need Help?", fontsize=9, fontname="hebo")
    page.insert_text((200, y), "My Account", fontsize=9)
    page.insert_text((300, y), "Support", fontsize=9)
    page.insert_text((400, y), "Call 800.800.4239", fontsize=9)
    y = 260
    for lab, val in [("SUBTOTAL", "$4,309.20"), ("SHIPPING", "$0.00"),
                     ("SALES TAX", "$382.44"), ("GRAND TOTAL", "$4,691.64")]:
        page.insert_text((380, y), lab, fontsize=9, fontname="hebo")
        page.insert_text((500, y), val, fontsize=9)
        y += 14
    doc.save(str(path))
    doc.close()


def _pdf_atoms(pdf: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(pdf)
    return list(getattr(out, "atoms", out))


def _page_text(pdf: Path) -> str:
    with fitz.open(str(pdf)) as d:
        return "".join(d[0].get_text().split()).lower()


def test_pdf_totals_rows_carry_their_cells(tmp_path):
    pdf = tmp_path / "CDW Quote.pdf"
    _cdw_bom(pdf)
    atoms = {a.raw_text: a for a in _pdf_atoms(pdf)}
    page = _page_text(pdf)
    for text, cells in [
        ("SUBTOTAL: $4,309.20", ["SUBTOTAL", "$4,309.20"]),
        ("SHIPPING: $0.00", ["SHIPPING", "$0.00"]),
        ("SALES TAX: $382.44", ["SALES TAX", "$382.44"]),
        ("Need Help? | My Account | Support | Call 800.800.4239",
         ["Need Help?", "My Account", "Support", "Call 800.800.4239"]),
    ]:
        assert text in atoms, list(atoms)
        # the bug: the joined text is not on the page
        assert "".join(text.split()).lower() not in page
        frags = _frags(atoms[text])
        assert frags, f"{text!r} has no cell fragments"
        assert [f["text"] for f in frags] == cells
        for f in frags:
            assert f["page"] == 0
            assert len(f["bbox"]) == 4
            assert "".join(f["text"].split()).lower() in page
    # the totals box's GRAND TOTAL, not the header row's
    gt = _frags(atoms["GRAND TOTAL: $4,691.64"])
    assert [f["text"] for f in gt] == ["GRAND TOTAL", "$4,691.64"]
    assert all(f["bbox"][1] > 250 for f in gt), gt
    # which copy of the words it is, for a viewer without geometry
    assert [f["nth"] for f in gt] == [1, 1]
    assert [f["nth"] for f in _frags(atoms["SUBTOTAL: $4,309.20"])] == [0, 0]
    # a cell's value sits on its label's row
    ship = _frags(atoms["SHIPPING: $0.00"])
    assert abs(ship[0]["bbox"][1] - ship[1]["bbox"][1]) < 3


def test_pdf_header_row_over_value_row(tmp_path):
    pdf = tmp_path / "CDW Quote.pdf"
    _cdw_bom(pdf)
    head = [a for a in _pdf_atoms(pdf) if a.raw_text.startswith("QUOTE #")]
    assert head, "header atom missing"
    frags = _frags(head[0])
    assert [f["text"] for f in frags][:4] == ["QUOTE #", "PSNV676", "QUOTE DATE", "1/14/2026"]
    labels, values = frags[0::2], frags[1::2]
    assert all(v["bbox"][1] > l["bbox"][1] for l, v in zip(labels, values))
    assert all(f["bbox"][1] < 140 for f in frags), frags


def test_pdf_verbatim_atoms_are_untouched_and_ids_do_not_move(tmp_path, monkeypatch):
    pdf = tmp_path / "CDW Quote.pdf"
    _cdw_bom(pdf)
    stamped = _pdf_atoms(pdf)
    assert not _frags(next(a for a in stamped if a.raw_text.startswith("Thank you for considering")))
    import app.parsers.cell_fragments as cf

    monkeypatch.setattr(cf, "stamp_pdf_cell_fragments", lambda atoms, path: atoms)
    plain = _pdf_atoms(pdf)
    assert [a.id for a in plain] == [a.id for a in stamped]
    assert [a.raw_text for a in plain] == [a.raw_text for a in stamped]


# ── docx / xlsx ─────────────────────────────────────────────────────────────


def test_docx_table_rows_carry_their_cells(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "sow.docx"
    d = docx.Document()
    d.add_paragraph("Statement of Work")
    t = d.add_table(rows=4, cols=3)
    for r, row in enumerate([["Site", "Address", "Qty"], ["Boston", "1 Main St", "4"],
                             ["Austin", "9 Elm Rd", "2"], ["Subtotal", "", "$4,309.20"]]):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    d.save(p)
    from app.parsers.docx_parser import DocxParser

    atoms = DocxParser().parse_artifact("p", "a", p)
    rows = [a for a in atoms if a.raw_text == "Subtotal | $4,309.20"]
    assert rows, [a.raw_text for a in atoms]
    for a in rows:
        frags = _frags(a)
        assert frags, a.source_refs[0].locator
        assert [(f["text"], f["table_index"], f["row"], f["col"]) for f in frags] == [
            ("Subtotal", 0, 3, 0), ("$4,309.20", 0, 3, 2)]
    boston = [a for a in atoms if a.raw_text == "Boston | 1 Main St | 4"]
    assert boston and all([f["text"] for f in _frags(a)] == ["Boston", "1 Main St", "4"] for a in boston)
    assert not _frags(next(a for a in atoms if a.raw_text == "Statement of Work"))


def test_xlsx_rows_carry_their_cells(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "po.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "PO"
    ws.append(["Purchase Order", None, None, None])
    ws.append(["Item", "Description", "Qty", "Unit Price"])
    ws.append(["AP-1", "Wireless access point", 4, 120.5])
    ws.append(["SW-2", "48-port switch", 1, 2000])
    wb.save(p)
    from app.parsers.cell_fragments import stamp_cell_fragments
    from app.parsers.parser_router import parse_artifact

    atoms = stamp_cell_fragments(parse_artifact("p", "a", p), p)
    row = [a for a in atoms if a.raw_text == "SW-2 | 48-port switch | 1 | 2000"]
    assert row, [a.raw_text for a in atoms]
    assert [(f["text"], f["sheet"], f["row"], f["col"]) for f in _frags(row[0])] == [
        ("SW-2", "PO", 4, "A"), ("48-port switch", "PO", 4, "B"), ("1", "PO", 4, "C"), ("2000", "PO", 4, "D")]
    # an atom the parser wrote from named columns points at those cells
    derived = [a for a in atoms if (a.source_refs[0].locator or {}).get("columns")
               and (a.source_refs[0].locator or {}).get("row") == 3]
    for a in derived:
        cols = {f["col"] for f in _frags(a)}
        assert cols == {v for v in a.source_refs[0].locator["columns"].values()}


def test_compiled_envelope_atom_keeps_the_fragments(tmp_path):
    from app.core.orbitbrief_envelope import _compact_atom

    pdf = tmp_path / "CDW Quote.pdf"
    _cdw_bom(pdf)
    atom = next(a for a in _pdf_atoms(pdf) if a.raw_text == "SUBTOTAL: $4,309.20")
    compact = _compact_atom(atom)
    assert [f["text"] for f in compact["locator"]["cell_fragments"]] == ["SUBTOTAL", "$4,309.20"]
