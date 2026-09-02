"""A crash in block detection must not look like an empty sheet.

Both call sites return [] when sheet_blocks raises, which is right -- one bad
sheet must not cost a workbook its other sheets. What was wrong is that []
ALSO means "no blocks here", so the two were the same answer.

A TypeError in block detection silently produced a Deal Kit with ZERO block
atoms. The compiler saw a clean parse, no error was raised anywhere, and the
sheet simply read as empty. That is the exact failure this pipeline exists to
prevent: a silent zero and a real zero must never look alike.
"""
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

import app.parsers.xlsx_blocks as xlsx_blocks
from app.parsers.xlsx_parser import XlsxParser


def _book(tmp_path: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws["A1"], ws["B1"] = "Price", "QTY"
    ws["A2"], ws["B2"] = "Technician Dispatch", 75
    p = tmp_path / "kit.xlsx"
    wb.save(p)
    return p


class TestBlockFailureIsVisible:
    def test_a_crash_becomes_a_warning(self, monkeypatch, tmp_path):
        def boom(*_a, **_k):
            raise TypeError("too many values to unpack (expected 2)")

        monkeypatch.setattr(xlsx_blocks, "sheet_blocks", boom)
        res = XlsxParser().parse_artifact_full(project_id="p", artifact_id="a", path=_book(tmp_path))
        assert any("block detection failed" in w for w in (res.warnings or []))

    def test_the_warning_names_the_sheet_and_the_error(self, monkeypatch, tmp_path):
        def boom(*_a, **_k):
            raise TypeError("too many values to unpack (expected 2)")

        monkeypatch.setattr(xlsx_blocks, "sheet_blocks", boom)
        res = XlsxParser().parse_artifact_full(project_id="p", artifact_id="a", path=_book(tmp_path))
        w = " ".join(res.warnings or [])
        assert "Summary" in w and "TypeError" in w

    def test_a_healthy_workbook_raises_no_such_warning(self, tmp_path):
        res = XlsxParser().parse_artifact_full(project_id="p", artifact_id="a", path=_book(tmp_path))
        assert not any("block detection failed" in w for w in (res.warnings or []))

    def test_the_rest_of_the_workbook_still_parses(self, monkeypatch, tmp_path):
        # Reporting the failure must not turn one bad sheet into a dead file.
        def boom(*_a, **_k):
            raise TypeError("boom")

        monkeypatch.setattr(xlsx_blocks, "sheet_blocks", boom)
        res = XlsxParser().parse_artifact_full(project_id="p", artifact_id="a", path=_book(tmp_path))
        assert res.atoms is not None
