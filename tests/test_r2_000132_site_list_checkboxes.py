"""Deal 000132: a site list's checkbox column, and the site-count gap.

1. "Delphos, OH | ☐ Assessment ☐ Configuration…" -- a service-type checkbox
   cell was glued onto the site name. Checkbox cells are their own atoms
   (service-type facts) beside the site's name/address atom, in docx and
   xlsx site tables.
2. "Customer documents declare 6 locations; 3 identified" beside four live
   sites: the gap counted site: keys on physical_site atoms, the site list
   counts what build_site_readiness lists. They now agree.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
from docx import Document

from app.core.declared_scope import declared_scope_questions
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef,
)
from app.parsers.checkbox_cells import checkbox_options
from app.parsers.docx_parser import DocxParser
from app.parsers.xlsx_parser import XlsxParser

CB = {
    "Delphos, OH": "☐ Assessment ☐ Configuration ☒ Installation ☐ Decommission",
    "Lima, OH": "☒ Assessment ☐ Configuration ☒ Installation ☐ Decommission",
    "Troy, OH": "☐ Assessment ☒ Configuration ☐ Installation ☐ Decommission",
    "Wilmington, OH": "☒ Assessment ☒ Configuration ☒ Installation ☐ Decommission",
}
ADDR = {
    "Delphos, OH": "123 Main St, Delphos, OH 45833",
    "Lima, OH": "45 Elm St, Lima, OH 45801",
    "Troy, OH": "9 Oak Ave, Troy, OH 45373",
    "Wilmington, OH": "77 Pine Rd, Wilmington, OH 45177",
}


def _atoms(out):
    return out if isinstance(out, list) else out.atoms


def _assert_split(atoms):
    for name, cb in CB.items():
        glued = [a.raw_text for a in atoms if name in a.raw_text and "☐" in a.raw_text]
        assert not glued, glued
        cells = [a for a in atoms if a.raw_text == cb]
        assert cells, (name, [a.raw_text for a in atoms])
        v = cells[0].value
        assert v["kind"] == "checkbox_selection" and v["subject"] == name
        assert v["selected"] == [l for c, l in checkbox_options(cb) if c]
        assert "Installation" in v["selected"] or "Configuration" in v["selected"]


def test_checkbox_cell_reader():
    assert checkbox_options(CB["Delphos, OH"]) == [
        (False, "Assessment"), (False, "Configuration"), (True, "Installation"), (False, "Decommission")]
    assert checkbox_options("Delphos, OH") is None
    assert checkbox_options("■ Install cameras") is None  # a bullet, not a box


def test_docx_site_name_table(tmp_path: Path):
    doc = Document()
    doc.add_heading("Site List", 1)
    t = doc.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Site Location", "Service Type"
    for name, cb in CB.items():
        r = t.add_row().cells
        r[0].text, r[1].text = name, cb
    doc.save(tmp_path / "SOW v2.docx")
    atoms = _atoms(DocxParser().parse_artifact("p", "a", tmp_path / "SOW v2.docx"))
    _assert_split(atoms)
    rows = [a for a in atoms if a.atom_type == AtomType.scope_item and a.raw_text in CB]
    assert sorted(a.raw_text for a in rows) == sorted(CB)
    cbs = [a for a in atoms if a.raw_text in CB.values()]
    assert all(a.atom_type == AtomType.site_attribute for a in cbs)


def test_docx_roster_table(tmp_path: Path):
    doc = Document()
    t = doc.add_table(rows=1, cols=3)
    for i, h in enumerate(["Site Location", "Address", "Service Type"]):
        t.cell(0, i).text = h
    for name, cb in CB.items():
        r = t.add_row().cells
        r[0].text, r[1].text, r[2].text = name, ADDR[name], cb
    doc.save(tmp_path / "SOW v2.docx")
    atoms = _atoms(DocxParser().parse_artifact("p", "a", tmp_path / "SOW v2.docx"))
    _assert_split(atoms)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site
             and a.source_refs[0].extraction_method == "docx_site_roster_v1"]
    assert len(sites) == 4 and not [a for a in sites if "Service Type" in a.raw_text]
    for s in sites:
        name = s.value["facility_name"]
        (cb,) = [a for a in atoms if a.raw_text == CB[name]]  # one atom per cell
        assert set(cb.entity_keys) == set(s.entity_keys)


def test_xlsx_roster_sheet(tmp_path: Path):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Site List"
    ws.append(["Site Location", "Address", "Service Type"])
    for name, cb in CB.items():
        ws.append([name, ADDR[name], cb])
    wb.save(tmp_path / "Deal Kit v2.xlsx")
    atoms = _atoms(XlsxParser().parse(tmp_path / "Deal Kit v2.xlsx"))
    _assert_split(atoms)
    assert len([a for a in atoms if a.atom_type == AtomType.physical_site]) == 4


# ── the site-count gap agrees with the site list ────────────────────────────


def _atom(aid, atype, text, *, keys=(), value=None, authority=AuthorityClass.contractual_scope):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id="art", atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=value or {"text": text}, entity_keys=list(keys),
        source_refs=[SourceRef(id=stable_id("src", aid), artifact_id="art",
                               artifact_type=ArtifactType.docx, filename="SOW v2.docx",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=authority, confidence=0.9, review_status=ReviewStatus.auto_accepted,
        review_flags=[], parser_version="t",
    )


def test_gap_counts_every_listed_site():
    from app.core.orbitbrief_core import build_site_readiness

    atoms = [_atom("d", AtomType.scope_item, "The project covers 6 locations across Ohio.")]
    for i, name in enumerate(["Lima", "Troy", "Wilmington"]):
        sid = name.upper()
        atoms.append(_atom(f"s{i}", AtomType.physical_site, f"facility: {name}, OH",
                           keys=[f"site:{name.lower()}"],
                           value={"kind": "physical_site", "id": sid, "site_id": sid, "name": name}))
    # A listed site with its id but no site: key (a name-only roster row).
    atoms.append(_atom("s3", AtomType.physical_site, "facility: Delphos, OH",
                       value={"kind": "physical_site", "id": "DELPHOS-OH", "site_id": "DELPHOS-OH",
                              "name": "Delphos, OH"}))
    listed = build_site_readiness(atoms=atoms, edges=[])["site_count"]
    assert listed == 4
    (q,) = [a for a in declared_scope_questions(project_id="p", atoms=atoms)
            if a.value["declared_scope"]["kind"] == "site_count_gap"]
    assert q.value["declared_scope"]["found_count"] == listed
    assert "declare 6 locations; 4 identified" in q.raw_text


def test_no_gap_when_the_list_is_complete():
    atoms = [_atom("d", AtomType.scope_item, "The project covers 2 locations.")]
    for i, name in enumerate(["Lima", "Troy"]):
        atoms.append(_atom(f"s{i}", AtomType.physical_site, f"facility: {name}, OH",
                           value={"kind": "physical_site", "id": name.upper(), "name": name}))
    assert not [a for a in declared_scope_questions(project_id="p", atoms=atoms)
                if a.value["declared_scope"]["kind"] == "site_count_gap"]
