"""Per-sheet routing: text-bearing -> deterministic; scanned/icon -> vision."""
from __future__ import annotations
import pytest
fitz = pytest.importorskip("fitz")
from app.core.schematic_route import needs_vision, route_sheet, extract_text_rows


def _text_pdf(tmp_path):
    doc = fitz.open(); pg = doc.new_page(width=612, height=792)
    y = 40
    for i in range(40):  # 40 rows of schedule-like text
        pg.insert_text((40, y), f"CKT-{i:02d}  20A  1P  LIGHTING  ROOM {100+i}")
        y += 16
    p = tmp_path / "sched.pdf"; doc.save(str(p)); doc.close(); return str(p)


def _blank_pdf(tmp_path):
    doc = fitz.open(); doc.new_page(width=612, height=792)  # no text -> scanned-like
    p = tmp_path / "blank.pdf"; doc.save(str(p)); doc.close(); return str(p)


def test_text_sheet_routes_deterministic(tmp_path):
    pg = fitz.open(_text_pdf(tmp_path))[0]
    assert needs_vision(pg) is False
    r = route_sheet(pg)
    assert r["path"] == "deterministic"
    assert len(r["rows"]) >= 30   # all the schedule rows, exact, no VLM


def test_scanned_sheet_routes_vision(tmp_path):
    pg = fitz.open(_blank_pdf(tmp_path))[0]
    assert needs_vision(pg) is True
    assert route_sheet(pg)["path"] == "vision"


def test_extract_rows_returns_text_lines(tmp_path):
    pg = fitz.open(_text_pdf(tmp_path))[0]
    rows = extract_text_rows(pg)
    assert any("CKT-" in r for r in rows)
