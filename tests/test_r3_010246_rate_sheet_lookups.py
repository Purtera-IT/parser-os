"""A rate sheet's dropdown lists are rejects; its base-rate row reads under
its headers.

Deal 010246's Deal Kit rate sheet (re-run on #268) keeps the option lists
its dropdowns read from on the sheet itself -- "COST RATES / SELL RATES",
"T&M / Fixed Fee / Per Site", ~50 SKUs, "L0..L4" -- side by side. Read
across they came back as ~50 content atoms ("COST RATES | T&M |
PS-TRAVEL-EXPENSES | L0"), and the compiler retyped the SKU-only rows to
bom_line with a device key. They are now chatter atoms, rejected_by
``lookup_list``, whether a data validation names them or only their shape
does. And the base-rate row above the matrix ("98 | 88 | 82 | 125") no
longer reads as bare numbers: it is read under the matrix's column headers.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

from app.parsers.xlsx_parser import XlsxParser

ROLES = ["Networking L1 Technician", "Networking L2 Technician", "EUC L1 Technician", "Project Manager"]
BASE = (98, 88, 82, 125)
COUNTRIES = ["United States", "Indonesia", "United Arab Emirates", "Hong Kong", "Germany", "Brazil"]
SKUS = ["PS-TRAVEL-EXPENSES"] + [f"PS-L{i % 4}-SVC-{i:02d}-ONSITE" for i in range(1, 30)]
LISTS = [["COST RATES", "SELL RATES"], ["T&M", "Fixed Fee", "Per Site"], SKUS, ["L0", "L1", "L2", "L3", "L4"]]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _build(path: Path, layout: str, validation: bool) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "SELL RATES"

    def put(r0: int, c0: int, rows) -> None:
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if v is not None:
                    ws.cell(row=r0 + i, column=c0 + j, value=v)

    n = max(len(lst) for lst in LISTS)
    grid = [[lst[i] if i < len(lst) else None for lst in LISTS] for i in range(n)]
    top = 1
    if layout == "top":
        put(1, 1, grid)
        top = n + 2
    put(top, 3, [list(BASE)])
    put(top + 1, 1, [["Country", "Request"] + ROLES])
    for ci, c in enumerate(COUNTRIES):
        put(top + 2 + ci, 1, [[c, 1.0 + ci / 10] + [round(b * (1 + ci / 10), 2) for b in BASE]])
    end = top + 2 + len(COUNTRIES)
    src = {"top": (1, "C"), "bottom": (end + 2, "C"), "right": (1, "L")}[layout]
    if layout == "bottom":
        put(end + 2, 1, grid)
    elif layout == "right":
        put(1, 10, grid)
    if validation:
        dv = DataValidation(type="list", formula1=f"${src[1]}${src[0]}:${src[1]}${src[0] + len(SKUS) - 1}")
        dv.add("A200")
        ws.add_data_validation(dv)
    wb.save(path)


def _atoms(path: Path):
    out = XlsxParser().parse_artifact("p", "a", path)
    return out if isinstance(out, list) else out.atoms


def _is_lookup_text(text: str) -> bool:
    return any(x in text for x in ("PS-TRAVEL-EXPENSES", "SVC-", "COST RATES |", "Fixed Fee", "Per Site"))


@pytest.mark.parametrize("layout,validation", [
    ("top", True), ("top", False), ("bottom", False), ("bottom", True), ("right", True), ("right", False),
])
def test_dropdown_lists_are_lookup_rejects(tmp_path: Path, layout: str, validation: bool) -> None:
    path = tmp_path / "Deal Kit.xlsx"
    _build(path, layout, validation)
    atoms = _atoms(path)
    lookups = [a for a in atoms if (a.value or {}).get("rejected_by") == "lookup_list"]
    assert len(lookups) >= len(SKUS) - 1, [a.raw_text for a in atoms]
    for a in lookups:
        assert a.atom_type.value == "deal_metadata"
        assert "chatter" in a.review_flags and a.entity_keys == []
    # No other atom carries a dropdown option, and no priced matrix row
    # has one glued onto it.
    for a in atoms:
        if (a.value or {}).get("rejected_by") == "lookup_list" or (a.value or {}).get("is_summary"):
            continue
        assert not _is_lookup_text(a.raw_text), (layout, validation, a.raw_text)
    rows = [a for a in atoms if a.atom_type.value == "service_line"]
    assert len([a for a in rows if "Country:" in a.raw_text]) == len(COUNTRIES)


@pytest.mark.parametrize("layout", ["top", "bottom", "right"])
def test_base_rate_row_reads_under_its_headers(tmp_path: Path, layout: str) -> None:
    path = tmp_path / "Deal Kit.xlsx"
    _build(path, layout, True)
    atoms = _atoms(path)
    base = [a for a in atoms if "98" in a.raw_text and "Country:" not in a.raw_text
            and not (a.value or {}).get("is_summary")]
    assert base, [a.raw_text for a in atoms]
    assert all("|" not in a.raw_text or "Networking L1 Technician: 98" in a.raw_text for a in base), \
        [a.raw_text for a in base]
    b = base[0]
    assert "Networking L1 Technician: 98" in b.raw_text and "Project Manager: 125" in b.raw_text
    assert not (b.value or {}).get("chatter")


def test_lookup_rows_stay_rejects_through_compile(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    deal = tmp_path / "deal"
    deal.mkdir()
    _build(deal / "Deal Kit.xlsx", "top", False)
    r = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)
    sku_rows = [a for a in r.atoms if "SVC-" in a.raw_text]
    assert sku_rows
    for a in sku_rows:
        # the SKU-only rows used to be retyped bom_line with a device key
        assert a.atom_type.value == "deal_metadata", (a.raw_text, a.atom_type)
        assert not any(k.startswith(("device:", "part_number:")) for k in a.entity_keys), a.entity_keys
        assert (a.value or {}).get("rejected_by") == "lookup_list"
