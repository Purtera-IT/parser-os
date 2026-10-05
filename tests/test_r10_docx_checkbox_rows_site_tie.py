"""A site list whose rows carry several checkbox cells each.

    | Location(s)     | Service(s)  (spans three columns)                 |
    | Springfield, IL | ☐ Alpha  | ☐ Gamma  | ☑ Epsilon |
    |                 | ☐ Beta   | ☐ Delta  | ☑ Zeta    |
    | Salem, OR       | (the same three cells, word for word)            |

The parser emitted one atom per checkbox cell ("☐ Alpha ☐ Beta", three a row,
every row alike), each tied to its site only by structure. Now each grid row
is ONE atom owned by its site: the site cell first, then only the ticked
options ("Springfield, IL: Epsilon, Zeta"). Unticked options are never atoms
and never in the text; they stay on the row atom as structure, with the
column header each came from. The row atom keeps the id it had as the site
cell's atom, and its locator names every cell of the row.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from app.core.ids import stable_id
from app.core.orbitbrief_envelope import _tie_checkbox_cells_to_site
from app.parsers import checkbox_cells
from app.parsers.docx_parser import DocxParser

SITES = ["Springfield, IL", "Salem, OR", "Dover, NH"]
COLS = [
    [(0, "Alpha"), (0, "Beta")],
    [(0, "Gamma"), (0, "Delta")],
    [(1, "Epsilon"), (1, "Zeta"), (0, "Eta")],
]
NONE_ROW = "Concord, NH"  # a row with no box ticked


def _r(t: str) -> str:
    return f"<w:r><w:t xml:space='preserve'>{t}</w:t></w:r>"


def _tc(paras: list[str], span: int | None = None) -> str:
    s = f"<w:gridSpan w:val='{span}'/>" if span else ""
    return f"<w:tc><w:tcPr><w:tcW w:w='2000' w:type='dxa'/>{s}</w:tcPr>" + "".join(
        f"<w:p>{p}</w:p>" for p in paras) + "</w:tc>"


def _row(site: str, cols) -> str:
    cells = "".join(_tc([_r(("☑" if c else "☐")) + _r(" " + l) for c, l in col]) for col in cols)
    return "<w:tr>" + _tc([_r(site)]) + cells + "</w:tr>"


def _doc(path: Path) -> Path:
    rows = ["<w:tr>" + _tc([_r("Location(s)")]) + _tc([_r("Service(s)")], 3) + "</w:tr>"]
    rows += [_row(site, COLS) for site in SITES]
    rows.append(_row(NONE_ROW, [[(0, l) for _c, l in col] for col in COLS]))
    tbl = (f"<w:tbl {nsdecls('w')}><w:tblPr><w:tblW w:w='8000' w:type='dxa'/></w:tblPr>"
           "<w:tblGrid>" + "<w:gridCol w:w='2000'/>" * 4 + "</w:tblGrid>" + "".join(rows) + "</w:tbl>")
    d = Document()
    d.add_heading("Designated Locations", 2)
    d.element.body.insert(len(d.element.body) - 1, parse_xml(tbl))
    d.add_paragraph("Services follow the table above.")
    # An ordinary table beside it: untouched.
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Item", "Owner"
    t.cell(1, 0).text, t.cell(1, 1).text = "Rack the switch", "Provider"
    d.save(path)
    return path


def _parse(tmp_path: Path):
    out = DocxParser().parse_artifact("p", "art_x", _doc(tmp_path / "sow.docx"))
    return out if isinstance(out, list) else out.atoms


def _loc(a) -> dict:
    return a.source_refs[0].locator


def _row_atom(atoms, row: int, table: int = 0):
    return next(a for a in atoms
                if _loc(a).get("extraction") == "docx_table_row_v1"
                and _loc(a).get("table_index") == table and _loc(a).get("row") == row)


TICKED = [l for col in COLS for c, l in col if c]
UNTICKED = [l for col in COLS for c, l in col if not c]


def test_each_grid_row_is_one_atom_site_first_then_ticked(tmp_path):
    atoms = _parse(tmp_path)
    # No per-cell checkbox atom, and no box glyph in any atom's text.
    assert not [a for a in atoms if "checkbox_cell" in (a.review_flags or [])]
    assert not [a for a in atoms if "☐" in a.raw_text or "☑" in a.raw_text]
    for row, site in enumerate(SITES, start=1):
        a = _row_atom(atoms, row)
        assert a.raw_text == f"{site}: {', '.join(TICKED)}"
        assert not [x for x in atoms if _loc(x).get("table_index") == 0 and _loc(x).get("row") == row
                    and x.atom_type.value != "raw_table_row" and x is not a]
        # The id the site cell's atom had: a label saved on it stays put.
        assert a.id == stable_id("atm", "art_x", "docx_row", 0, row, site)
    assert _row_atom(atoms, len(SITES) + 1).raw_text == f"{NONE_ROW}: {checkbox_cells.NONE_SELECTED}"


def test_row_atom_names_every_cell_and_keeps_unticked_as_structure(tmp_path):
    atoms = _parse(tmp_path)
    a = _row_atom(atoms, 2)
    loc, v = _loc(a), a.value
    assert loc["table_index"] == 0 and loc["row"] == 2
    assert loc["cells"] == [0, 1, 2, 3] and loc["row_key"] == "t0 r2"
    assert v["checkbox_row"] and v["site"] == v["subject"] == SITES[1]
    assert v["site_cell"] == 0 and v["site_column"] == "Location(s)"
    assert v["selected"] == TICKED and v["not_selected"] == UNTICKED
    assert v["options"] == [
        {"label": l, "checked": bool(c), "column": "Service(s)", "cell": i}
        for i, col in enumerate(COLS, start=1) for c, l in col
    ]
    # The header row and the grid's heading are pointers on the row.
    assert v["columns"][0] == "Location(s)" and v["option_columns"] == ["Service(s)"]
    assert v["heading"] == "Designated Locations"
    none = _row_atom(atoms, len(SITES) + 1).value
    assert none["selected"] == []
    assert none["not_selected"] == [l for col in COLS for _c, l in col]


def test_drop_unticked_fully_is_one_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(checkbox_cells, "DROP_UNTICKED_FULLY", True)
    atoms = _parse(tmp_path)
    v = _row_atom(atoms, 1).value
    assert v["selected"] == TICKED and "not_selected" not in v
    assert all(o["checked"] for o in v["options"])
    assert not [t for t in v["cells"].values() if "☐" in t or "☑" in t]
    assert _row_atom(atoms, 1).raw_text == f"{SITES[0]}: {', '.join(TICKED)}"


def test_ordinary_table_rows_are_untouched(tmp_path):
    atoms = _parse(tmp_path)
    a = _row_atom(atoms, 1, table=1)
    assert a.raw_text == "Rack the switch | Provider"
    assert "checkbox_row" not in a.value and "row_key" not in _loc(a)


def test_envelope_names_the_kept_site_atom_for_each_row():
    loc = lambda row, **kw: {"table_index": 1, "row": row, **kw}  # noqa: E731
    rows = [
        {"id": "row1", "artifact_id": "a", "atom_type": "scope_item", "locator": loc(1),
         "structured": {"kind": "table_row", "checkbox_row": True, "row_key": "t1 r1"}},
        {"id": "site1", "artifact_id": "a", "atom_type": "physical_site", "locator": loc(1), "structured": {}},
        {"id": "row2", "artifact_id": "a", "atom_type": "scope_item", "locator": loc(2),
         "structured": {"kind": "table_row", "checkbox_row": True, "row_key": "t1 r2"}},
        {"id": "row3", "artifact_id": "a", "atom_type": "scope_item", "locator": loc(3), "structured": {}},
        {"id": "cb3", "artifact_id": "b", "atom_type": "site_attribute", "locator": loc(1),
         "structured": {"kind": "checkbox_selection", "row_key": "t1 r1"}},
    ]
    out = {r["id"]: r["structured"] for r in _tie_checkbox_cells_to_site(rows)}
    assert out["row1"]["site_atom_id"] == "site1"  # the physical_site wins
    assert out["row2"]["site_atom_id"] == "row2"  # no site atom: the row is its own
    assert "site_atom_id" not in out["row3"]
    assert "site_atom_id" not in out["cb3"]  # never across documents


def test_sites_still_found_and_read_from_the_site_cell(tmp_path):
    import os

    from app.core.compiler import compile_project

    os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
    proj = tmp_path / "proj"
    proj.mkdir()
    _doc(proj / "sow.docx")
    r = compile_project(proj, project_id="p", allow_errors=True, use_cache=False)
    sites = [a for a in r.atoms if a.atom_type.value == "physical_site"]
    names = {a.raw_text for a in sites}
    for site in SITES:
        assert site in names, (site, names)
    for a in sites:
        if a.raw_text in SITES:
            assert a.source_refs[0].locator.get("cells") == [0]
