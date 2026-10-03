"""Table header cells are field names, not atoms (deal 010353 SOW tables).

The SOW's rate table put a group caption "Stated Rate" over the column names
"Business Hours | After Hours", and its contact table held a row with only
"Chase Smith" filled in. Each header cell, and the bare name, came out as an
atom of its own. A header row names the cells beneath it and is never an atom;
a lone value on its row is bound to its column's name ("Name: Chase Smith").
"""

from __future__ import annotations

import pytest

from app.parsers.table_headers import (
    bind_lone_cell,
    is_banner_row,
    is_header_only_row,
)


def _frags(atom):
    return (atom.source_refs[0].locator or {}).get("cell_fragments")


# ── the rule ────────────────────────────────────────────────────────────────


def test_a_label_row_over_values_is_a_header():
    rows = [["Role", "Business Hours", "After Hours"], ["Technician", "$125.00", "$187.50"]]
    assert is_header_only_row(rows, 0)


def test_a_row_of_names_is_data_not_a_header():
    # short value-free words, but the next row fills a column it does not name
    rows = [["Chase Smith", "Director", ""], ["Dana Whitfield", "Project Manager", "dana@x.com"]]
    assert not is_header_only_row(rows, 0)
    # and words over more words, with no value anywhere, decide nothing
    assert not is_header_only_row([["Boston", "Austin"], ["Denver", "Miami"]], 0)


def test_a_merged_title_is_not_a_header_but_a_lone_caption_over_one_is_a_banner():
    rows = [["Labor Rates"] * 3, ["Role", "Business Hours", "After Hours"], ["Tech", "$1", "$2"]]
    assert not is_header_only_row(rows, 0)
    assert is_banner_row(rows, 0)
    rows = [["", "Stated Rate", ""], ["Role", "Business Hours", "After Hours"], ["Tech", "$1", "$2"]]
    assert is_banner_row(rows, 0)
    assert not is_banner_row([["Boston"], ["Austin"]], 0)


def test_a_lone_value_takes_its_column_name():
    hdr = ["Name", "Title", "Email"]
    assert bind_lone_cell(["Chase Smith", "", ""], hdr) == "Name: Chase Smith"
    assert bind_lone_cell(["", "", "x@y.com"], hdr) == "Email: x@y.com"
    assert bind_lone_cell(["Chase Smith", "Director", ""], hdr) is None
    assert bind_lone_cell(["Name", "", ""], hdr) is None
    assert bind_lone_cell(["a", ""], ["col_1", "col_2"]) is None


# ── docx ────────────────────────────────────────────────────────────────────


def _sow(path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Statement of Work")
    # Rate table: a "Stated Rate" caption merged over the two rate columns,
    # the role column merged down through both header rows.
    t = d.add_table(rows=3, cols=3)
    t.cell(0, 1).merge(t.cell(0, 2)).text = "Stated Rate"
    t.cell(0, 0).merge(t.cell(1, 0)).text = "Role"
    t.cell(1, 1).text = "Business Hours"
    t.cell(1, 2).text = "After Hours"
    for c, v in enumerate(["Technician", "$125.00", "$187.50"]):
        t.cell(2, c).text = v
    # A caption row in its own cell over the column names.
    t = d.add_table(rows=3, cols=3)
    t.cell(0, 1).text = "Stated Rate"
    for c, v in enumerate(["Role", "Business Hours", "After Hours"]):
        t.cell(1, c).text = v
    for c, v in enumerate(["Engineer", "$150.00", "$225.00"]):
        t.cell(2, c).text = v
    # Contact table with a row holding only a name.
    t = d.add_table(rows=3, cols=3)
    for c, v in enumerate(["Name", "Title", "Email"]):
        t.cell(0, c).text = v
    t.cell(1, 0).text = "Chase Smith"
    for c, v in enumerate(["Dana Whitfield", "Project Manager", "dana@x.com"]):
        t.cell(2, c).text = v
    d.save(path)


def test_docx_header_cells_are_field_names_not_atoms(tmp_path):
    from app.parsers.docx_parser import DocxParser

    p = tmp_path / "010353 SOW.docx"
    _sow(p)
    atoms = DocxParser().parse_artifact("p", "a", p)
    texts = [a.raw_text for a in atoms]
    for header in ("Stated Rate", "Business Hours", "After Hours"):
        assert not any(header in t for t in texts), texts
    assert "Chase Smith" not in texts
    rows = [a for a in atoms if a.atom_type.value == "scope_item"
            and a.raw_text == "Technician | $125.00 | $187.50"]
    assert rows, texts
    assert rows[0].value["cells"] == {
        "Role": "Technician", "Business Hours": "$125.00", "After Hours": "$187.50"}
    eng = next(a for a in atoms if a.atom_type.value == "scope_item"
               and a.raw_text == "Engineer | $150.00 | $225.00")
    assert eng.value["cells"]["Business Hours"] == "$150.00"
    rtr = next(a for a in atoms if a.atom_type.value == "raw_table_row"
               and a.raw_text.startswith("Technician"))
    assert rtr.value["_columns"] == ["Role", "Business Hours", "After Hours"]

    chase = [a for a in atoms if a.raw_text == "Name: Chase Smith"]
    assert chase and all(a.atom_type.value in ("scope_item", "raw_table_row") for a in chase)
    frags = [f["text"] for f in (_frags(chase[0]) or [])]
    assert "Chase Smith" in frags
    # a full contact row is untouched
    assert "Dana Whitfield | Project Manager | dana@x.com" in texts


def test_docx_first_row_data_and_property_rows_are_kept(tmp_path):
    docx = pytest.importorskip("docx")
    from app.parsers.docx_parser import DocxParser

    p = tmp_path / "sow.docx"
    d = docx.Document()
    t = d.add_table(rows=3, cols=4)
    for c in range(4):
        t.cell(0, 0).merge(t.cell(0, c))
    t.cell(0, 0).text = "Site Information"
    for c, v in enumerate(["City", "Marion", "State", "SC"]):
        t.cell(1, c).text = v
    for c, v in enumerate(["Zip Code", "29571", "Phone", "843-555-0100"]):
        t.cell(2, c).text = v
    d.save(p)
    texts = [a.raw_text for a in DocxParser().parse_artifact("p", "a", p)]
    assert "City | Marion | State | SC" in texts
    assert "Zip Code | 29571 | Phone | 843-555-0100" in texts


# ── xlsx ────────────────────────────────────────────────────────────────────


def test_xlsx_rate_sheet_caption_and_contact_header_are_not_atoms(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from app.parsers.parser_router import parse_artifact

    p = tmp_path / "rates.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rates"
    ws.append(["Labor Rates"])
    ws.append([None, "Stated Rate", None])
    ws.append(["Role", "Business Hours", "After Hours"])
    ws.append(["Technician", 125, 187.5])
    ws.append(["Engineer", 150, 225])
    ws.append([])
    ws.append(["Name", "Title", "Email"])
    ws.append(["Chase Smith", None, None])
    ws.append(["Dana Whitfield", "Project Manager", "dana@x.com"])
    wb.save(p)
    atoms = parse_artifact("p", "a", p)
    texts = [a.raw_text for a in atoms]
    assert "Stated Rate" not in texts
    assert "Name | Title | Email" not in texts
    assert "Chase Smith" not in texts
    assert "Name: Chase Smith" in texts
    assert any("Business Hours: 125" in t for t in texts), texts


def test_xlsx_generic_table_binds_a_lone_name(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from app.parsers.parser_router import parse_artifact

    p = tmp_path / "contacts.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Project Team"
    ws.append(["Name", "Title", "Email", "Phone"])
    ws.append(["Chase Smith", None, None, None])
    ws.append(["Dana Whitfield", "Project Manager", "dana@x.com", "555-123-4567"])
    wb.save(p)
    texts = [a.raw_text for a in parse_artifact("p", "a", p)]
    assert "Chase Smith" not in texts
    assert "Name: Chase Smith" in texts
    # a full row reads in the same keyed shape as the lone cell above it
    assert "Name: Dana Whitfield | Title: Project Manager | Email: dana@x.com | Phone: 555-123-4567" in texts


# ── signature / e-sign blocks ───────────────────────────────────────────────


def test_esign_caption_is_the_name_of_its_fields_not_an_atom(tmp_path):
    """010003 / 010353: "The Buyer" over "Signature: | Name: | Date:" came
    out as an atom of its own (and bound onto the party header as
    "Purtera-IT LLC: The Buyer"). It is the caption of the fields beneath it."""
    docx = pytest.importorskip("docx")
    from app.parsers.docx_parser import DocxParser
    from app.parsers.table_headers import is_caption_row

    assert is_caption_row([["The Buyer", ""], ["Signature:", "____"]], 0)
    assert not is_caption_row([["Chase Smith", ""], ["Dana", "PM"]], 0)

    p = tmp_path / "sig.docx"
    d = docx.Document()
    t = d.add_table(rows=5, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Purtera-IT LLC", "Customer"
    t.cell(1, 0).text = "The Buyer"
    for r, (a, b) in enumerate([("Signature:", "______"), ("Name:", "Chase Smith"), ("Date:", "")], start=2):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    t = d.add_table(rows=3, cols=2)
    t.cell(0, 0).merge(t.cell(0, 1)).text = "The Seller"
    for r, (a, b) in enumerate([("Signature:", "______"), ("Name:", "Lee Park")], start=1):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    d.save(p)
    atoms = DocxParser().parse_artifact("p", "a", p)
    texts = [a.raw_text for a in atoms]
    assert not any("The Buyer" in t or "The Seller" in t for t in texts), texts
    assert not any(t.endswith(": Date:") for t in texts), texts
    chase = next(a for a in atoms if a.raw_text == "Name: | Chase Smith")
    assert chase.source_refs[0].locator["section_path"][-1] == "The Buyer"
    lee = next(a for a in atoms if a.raw_text == "Name: | Lee Park")
    assert lee.source_refs[0].locator["section_path"][-1] == "The Seller"
