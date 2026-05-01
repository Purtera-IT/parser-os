from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from app.parsers.quote_parser import QuoteParser


def test_quote_adversarial_cases(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "vendor_quote.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["Part Number", "Description", "Quantity", "Unit Price", "Lead Time"])
    ws.append(["cam-ip-001", "IP Cam", "72", "$300.00", ""])
    ws.append(["TOTAL", "", "72", "", ""])
    wb.save(xlsx_path)

    atoms = QuoteParser().parse_artifact("proj", "art", xlsx_path)
    assert atoms
    assert any(atom.atom_type.value == "vendor_line_item" for atom in atoms)
    assert any(atom.atom_type.value == "quantity" and atom.value.get("quantity") == 72 for atom in atoms)
    all_keys = {key for atom in atoms for key in atom.entity_keys}
    assert "device:ip_camera" in all_keys

    txt_path = tmp_path / "vendor_quote.txt"
    txt_path.write_text(
        "Part Number|Description|Quantity|Unit Price|Lead Time\n"
        "CAM-IP-002|IP Cam|10|$280.00|1 week\n",
        encoding="utf-8",
    )
    txt_atoms = QuoteParser().parse_artifact("proj", "art2", txt_path)
    assert txt_atoms
