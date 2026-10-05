"""A site list whose rows carry several checkbox cells each.

    | Location(s)     | Service(s)  (spans three columns)                 |
    | Springfield, IL | ☐ Alpha  | ☐ Gamma  | ☑ Epsilon |
    |                 | ☐ Beta   | ☐ Delta  | ☑ Zeta    |
    | Salem, OR       | (the same three cells, word for word)            |

Every row's checkbox cells read the same, so the text alone cannot say which
site a cell belongs to. The parser emitted each row's checkbox cells BEFORE the
row's site cell, and nothing on a checkbox atom pointed at its row's site.

Now each row reads site cell first, then its boxes; each checkbox atom carries
its row key, the site cell's column and the row atom's id; and the envelope
names the kept atom for that row's site. Ids, text and tick state are
unchanged.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from app.core.ids import stable_id
from app.core.orbitbrief_envelope import _tie_checkbox_cells_to_site
from app.parsers.docx_parser import DocxParser

SITES = ["Springfield, IL", "Salem, OR", "Dover, NH"]
COLS = [
    [(0, "Alpha"), (0, "Beta")],
    [(0, "Gamma"), (0, "Delta")],
    [(1, "Epsilon"), (1, "Zeta"), (0, "Eta")],
]
CELL_TEXT = [" ".join(("☑" if c else "☐") + " " + l for c, l in col) for col in COLS]


def _r(t: str) -> str:
    return f"<w:r><w:t xml:space='preserve'>{t}</w:t></w:r>"


def _tc(paras: list[str], span: int | None = None) -> str:
    s = f"<w:gridSpan w:val='{span}'/>" if span else ""
    return f"<w:tc><w:tcPr><w:tcW w:w='2000' w:type='dxa'/>{s}</w:tcPr>" + "".join(
        f"<w:p>{p}</w:p>" for p in paras) + "</w:tc>"


def _doc(path: Path) -> Path:
    rows = ["<w:tr>" + _tc([_r("Location(s)")]) + _tc([_r("Service(s)")], 3) + "</w:tr>"]
    for site in SITES:
        cells = "".join(
            _tc([_r(("☑" if c else "☐")) + _r(" " + l) for c, l in col]) for col in COLS)
        rows.append("<w:tr>" + _tc([_r(site)]) + cells + "</w:tr>")
    tbl = (f"<w:tbl {nsdecls('w')}><w:tblPr><w:tblW w:w='8000' w:type='dxa'/></w:tblPr>"
           "<w:tblGrid>" + "<w:gridCol w:w='2000'/>" * 4 + "</w:tblGrid>" + "".join(rows) + "</w:tbl>")
    d = Document()
    d.add_heading("Designated Locations", 2)
    d.element.body.insert(len(d.element.body) - 1, parse_xml(tbl))
    d.add_paragraph("Services follow the table above.")
    d.save(path)
    return path


def _parse(tmp_path: Path):
    out = DocxParser().parse_artifact("p", "art_x", _doc(tmp_path / "sow.docx"))
    return out if isinstance(out, list) else out.atoms


def _loc(a) -> dict:
    return a.source_refs[0].locator


def _row_atom(atoms, row: int):
    return next(a for a in atoms
                if _loc(a).get("extraction") == "docx_table_row_v1" and _loc(a).get("row") == row)


def _cb(atoms):
    return [a for a in atoms if "checkbox_cell" in (a.review_flags or [])]


def test_site_cell_reads_before_its_checkbox_cells(tmp_path):
    atoms = _parse(tmp_path)
    for row, site in enumerate(SITES, start=1):
        site_atom = _row_atom(atoms, row)
        assert site_atom.raw_text == site
        boxes = [a for a in _cb(atoms) if _loc(a)["row"] == row]
        assert len(boxes) == 3
        assert all(_loc(site_atom)["line_start"] < _loc(b)["line_start"] for b in boxes)
        assert atoms.index(site_atom) < min(atoms.index(b) for b in boxes)


def test_each_checkbox_cell_is_tied_to_its_rows_site(tmp_path):
    atoms = _parse(tmp_path)
    cbs = _cb(atoms)
    assert len(cbs) == 3 * len(SITES)
    for a in cbs:
        loc, v = _loc(a), a.value
        row = loc["row"]
        assert v["site"] == SITES[row - 1]
        assert v["row_key"] == loc["row_key"] == f"t0 r{row}"
        assert v["site_cell"] == loc["site_cell"] == 0
        assert v["site_column"] == "Location(s)"
        assert v["row_atom_id"] == _row_atom(atoms, row).id
        # An exact cell locator: table, row, grid column.
        assert loc["table_index"] == 0 and loc["cell"] in (1, 2, 3)
    # The site's own atom names the cells its text came from.
    assert _loc(_row_atom(atoms, 1))["cells"] == [0]


def test_identical_cells_keep_their_ids_text_and_ticks(tmp_path):
    atoms = _parse(tmp_path)
    cbs = _cb(atoms)
    assert len({a.id for a in cbs}) == len(cbs)
    for a in cbs:
        col = _loc(a)["cell"]
        assert a.raw_text == CELL_TEXT[col - 1]
        assert a.value["selected"] == [l for c, l in COLS[col - 1] if c]
        assert a.value["not_selected"] == [l for c, l in COLS[col - 1] if not c]
        # The id is minted from the cell locator as before the tie was added.
        base = {"table_index": 0, "row": _loc(a)["row"], "cell": col,
                "section_path": list(_loc(a)["section_path"])}
        assert a.id == stable_id("atm", "art_x", "checkbox_cell",
                                 str(sorted(base.items(), key=str)), "Service(s)", a.raw_text)


def test_envelope_names_the_kept_site_atom_for_each_row():
    loc = lambda row, **kw: {"table_index": 1, "row": row, **kw}  # noqa: E731
    rows = [
        {"id": "row1", "artifact_id": "a", "atom_type": "scope_item", "locator": loc(1), "structured": {}},
        {"id": "site1", "artifact_id": "a", "atom_type": "physical_site", "locator": loc(1), "structured": {}},
        {"id": "row2", "artifact_id": "a", "atom_type": "scope_item", "locator": loc(2), "structured": {}},
        {"id": "cb1", "artifact_id": "a", "atom_type": "site_attribute", "locator": loc(1, cell=1),
         "structured": {"kind": "checkbox_selection", "row_key": "t1 r1", "row_atom_id": "row1"}},
        {"id": "cb2", "artifact_id": "a", "atom_type": "site_attribute", "locator": loc(2, cell=1),
         "structured": {"kind": "checkbox_selection", "row_key": "t1 r2", "row_atom_id": "row2"}},
        {"id": "cb3", "artifact_id": "b", "atom_type": "site_attribute", "locator": loc(1, cell=1),
         "structured": {"kind": "checkbox_selection", "row_key": "t1 r1", "row_atom_id": "zz"}},
    ]
    out = {r["id"]: r["structured"] for r in _tie_checkbox_cells_to_site(rows)}
    assert out["cb1"]["site_atom_id"] == "site1"  # the physical_site wins
    assert out["cb2"]["site_atom_id"] == "row2"
    assert "site_atom_id" not in out["cb3"]  # never across documents
    assert "site_atom_id" not in out["row1"]
