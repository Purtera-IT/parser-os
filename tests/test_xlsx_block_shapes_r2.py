"""Spreadsheet shapes that were read under the wrong header.

* A two-column info box ("Customer | OxBlue" / "OPPTY # | 010246") read as a
  table whose header was its first pair: "Customer: OPPTY # | OxBlue: 010246".
* A Gantt "Financials" block whose first resources had uncomputed totals was
  walked into the header band: "Niagara Tech #1 Niagara Tech #2 PC: PPE Tech 1
  | ...", four rows in one atom, three atoms like it.
* A service block flush under a country matrix (no blank row) was bound to
  the matrix's columns.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

from app.core.schemas import ArtifactType
from app.parsers.sheet_classifier import SheetClassification, SheetRole
from app.parsers.xlsx_blocks import sheet_blocks
from app.parsers.xlsx_parser import XlsxParser, _commercial_header_band, _money_columns


def _parse(tmp_path: Path, wb, name="Book.xlsx"):
    p = tmp_path / name
    wb.save(p)
    return XlsxParser().parse(p)


# ── item: two-column label/value box ─────────────────────────────────────────


def test_info_box_reads_as_pairs_beside_a_table(tmp_path):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Scope"
    ws.append(["Customer", "OxBlue"])
    ws.append(["OPPTY #", "010246"])
    ws.append([])
    ws.append(["Task", "Hours", "Rate", "Total"])
    ws.append(["Install cameras", 40, 95, 3800])
    ws.append(["Configure NVR", 8, 120, 960])
    texts = [a.raw_text for a in _parse(tmp_path, wb)]
    assert "Customer: OxBlue" in texts, texts
    assert "OPPTY #: 010246" in texts, texts
    assert not [t for t in texts if "Customer: OPPTY" in t], texts
    assert any("Task: Install cameras | Hours: 40" in t for t in texts), texts


def test_info_box_alone_on_a_sheet_keeps_its_first_pair(tmp_path):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Scope"
    ws.append(["Customer", "OxBlue"])
    ws.append(["OPPTY #", "010246"])
    texts = [a.raw_text for a in _parse(tmp_path, wb)]
    assert "Customer: OxBlue" in texts, texts
    assert "OPPTY #: 010246" in texts, texts


def test_mixed_values_box_is_pairs_but_a_real_two_column_table_is_not():
    box = sheet_blocks([["Customer", "OxBlue"], ["Sites", 12], ["Rep", "PK"]])
    assert [b["kind"] for b in box] == ["keyval"]
    assert box[0]["pairs"][0] == ("Customer", "OxBlue")
    table = sheet_blocks([["Item", "Qty"], ["Cat6 cable", 40], ["Jacks", 20], ["Plates", 10]])
    assert [b["kind"] for b in table] == ["table"]
    assert table[0]["header"] == ["Item", "Qty"]
    names = sheet_blocks([["Site", "Address"], ["Bldg A", "1 Main St"], ["Bldg B", "2 Main St"]])
    assert [b["kind"] for b in names] == ["table"]


# ── item: Gantt Financials rows walked into the header band ─────────────────

_FIN = [
    ["Financials"],
    ["Resource", "Role", "Site", "Hours", "Cost Rate", "Sell Rate", "Total Cost", "Total Sell"],
    ["Niagara Tech #1", "Field Tech", "Onsite", 40, 55, None, None, None],
    ["Niagara Tech #2", "Field Tech", "Onsite", 40, 55, None, None, None],
    ["PC", "Coordinator", "Remote", 10, 60, None, None, None],
    ["PPE Tech 1", "Field Tech", "Onsite", 24, 50, 75, 1200, 1800],
    ["PPE Tech 2", "Field Tech", "Onsite", 24, 50, 75, 1200, 1800],
    ["PM", "Project Manager", "Remote", 12, 80, 120, 960, 1440],
]


def test_sparse_value_rows_are_not_a_header_band():
    headers, hrows, floor = _commercial_header_band(_FIN, _money_columns(_FIN))
    assert headers[:3] == ["Resource", "Role", "Site"], headers
    assert hrows == {1}
    assert floor == 2


def test_gantt_financials_one_atom_per_row(tmp_path):
    atoms = XlsxParser()._emit_commercial_sheet_rows(
        project_id="p", artifact_id="a", artifact_type=ArtifactType.xlsx, filename="Deal Kit.xlsx",
        sheet_name="Gantt Financials", rows=_FIN,
        classification=SheetClassification(role=SheetRole.FINANCIAL_SUMMARY, suppress=True,
                                           reason="t", confidence=1.0),
    )
    texts = [a.raw_text for a in atoms]
    names = ["Niagara Tech #1", "Niagara Tech #2", "PPE Tech 1", "PPE Tech 2", "PM"]
    for n in names:
        hits = [t for t in texts if f"Resource: {n} |" in t]
        assert len(hits) == 1, (n, texts)
    for t in texts:
        assert sum(1 for n in names if f": {n} |" in t) <= 1, t
        assert "Niagara Tech #1 Niagara Tech #2" not in t, t


# ── item: a new header row with no blank row before it ──────────────────────


def test_block_table_splits_at_a_new_header_row():
    rows = [["Country", "Request", "L1 Hourly", "L2 Hourly"]]
    rows += [[c, 50, 60, 70] for c in ("United States", "Canada", "Mexico")]
    rows += [["Service", "Sell", "Cost", None]]
    rows += [["L2 EUC 2 hour minimum", 96, 63.5, None], ["PC", 50, 30, None], ["PM", 120, 80, None]]
    tables = [b for b in sheet_blocks(rows) if b["kind"] == "table"]
    assert [t["header"][0] for t in tables] == ["Country", "Service"], tables
    assert [r[0] for r in tables[1]["rows"]] == ["L2 EUC 2 hour minimum", "PC", "PM"]
    assert all(r[0] not in ("PC", "PM", "Service") for r in tables[0]["rows"])
