from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from app.parsers.xlsx_parser import XlsxParser


def _write_case(path: Path, header_row: int, quantity_header: str, qty_a: str, qty_b: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "site_roster"
    for _ in range(header_row - 1):
        ws.append(["title row", "", "", "", "", ""])
    ws.append(["Site", "Floor", "Device", quantity_header, "Access Window", "Scope"])
    ws.append(["Main Campus", "1", "IP Camera", qty_a, "Weekdays", "Install"])
    ws.append(["West-Wing", "2", "IP Camera", qty_b, "Escort required", "Install"])
    ws.append(["Subtotal", "", "", "1250", "", ""])
    ws.append(["", "", "", "", "", ""])
    ws.append(["TOTAL", "", "", "1250", "", ""])
    wb.save(path)


def test_xlsx_adversarial_cases(tmp_path: Path) -> None:
    parser = XlsxParser()

    case_header_shift = tmp_path / "shifted.xlsx"
    _write_case(case_header_shift, 7, "Qty", "50 EA", "1,200")
    atoms = parser.parse_artifact("proj", "art", case_header_shift)
    assert atoms
    quantities = [atom.value.get("quantity") for atom in atoms if atom.atom_type.value == "quantity"]
    line_qty = [a.value.get("quantity") for a in atoms if a.atom_type.value == "quantity" and not a.value.get("aggregate")]
    agg_qty = [a.value.get("quantity") for a in atoms if a.atom_type.value == "quantity" and a.value.get("aggregate")]
    assert 50 in quantities
    assert 1200 in quantities
    assert 1250 in agg_qty
    assert 1250 not in line_qty
    assert all(atom.source_refs for atom in atoms)
    all_keys = {key for atom in atoms for key in atom.entity_keys}
    assert "site:west_wing" in all_keys

    case_count = tmp_path / "count.xlsx"
    _write_case(case_count, 3, "Count", "50", "41")
    atoms_count = parser.parse_artifact("proj", "art2", case_count)
    assert atoms_count

    case_hash = tmp_path / "hash.xlsx"
    _write_case(case_hash, 2, "#", "50", "41")
    atoms_hash = parser.parse_artifact("proj", "art3", case_hash)
    assert atoms_hash


def test_adversarial_wide_drop_schedule_totals(tmp_path: Path) -> None:
    path = tmp_path / "wide_totals.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "drops"
    ws.append(["Plate ID", "Location", "RJ45", "Cat6 UTP", "Cat6 STP"])
    ws.append(["X", "Room A", 1, 1, 0])
    ws.append(["TOTALS", "", 10, 8, 2])
    wb.save(path)
    atoms = XlsxParser().parse_artifact("p", "a", path)
    agg = [a for a in atoms if a.atom_type.value == "quantity" and a.value.get("aggregate")]
    assert {a.value.get("quantity") for a in agg} == {10, 8, 2}
    assert not any(a.atom_type.value == "entity" and "totals" in a.normalized_text for a in atoms)
