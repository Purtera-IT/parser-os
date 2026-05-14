"""Regression tests for PR2: structured CSV/XLSX row typing.

The corpus review found risk registers, asset inventories, support
matrices, site rosters, and EOL sheets all flattening into a generic
``scope_item`` row. After PR2 they should bucket into the new
structured AtomTypes (risk / asset_record / support_entitlement /
site_roster / lifecycle_status) and any in-cell exclusion / risk
language gets its own sub-atom anchored to the parent row.
"""
from __future__ import annotations

from pathlib import Path

from app.core.ids import stable_id
from app.core.schemas import AtomType, ArtifactType
from app.parsers.xlsx_parser import XlsxParser


def _emit_csv(parser: XlsxParser, tmp_path: Path, name: str, contents: str):
    p = tmp_path / name
    p.write_text(contents, encoding="utf-8")
    artifact_id = stable_id("art", str(p))
    return parser.parse_artifact(
        project_id="TEST",
        artifact_id=artifact_id,
        path=p,
    )


def test_risk_register_rows_become_risk_atoms(tmp_path: Path):
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "risk_register.csv",
        "Risk ID,Severity,Impact,Mitigation,Owner\n"
        "R-1,High,Outage,Add redundancy,Alex\n"
        "R-2,Medium,Slow,Tune SLA,Sam\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    risks = [a for a in atoms if a.atom_type == AtomType.risk]
    assert len(risks) >= 2
    assert any("R-1" in (a.raw_text or "") for a in risks)


def test_asset_inventory_rows_become_asset_record_atoms(tmp_path: Path):
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "asset_inventory.csv",
        "Asset ID,Serial,Model,IP Address,MAC Address\n"
        "AST-001,SN12345,WS-C3850-48,10.0.0.1,aa:bb:cc:dd:ee:01\n"
        "AST-002,SN67890,WS-C3850-48,10.0.0.2,aa:bb:cc:dd:ee:02\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    records = [a for a in atoms if a.atom_type == AtomType.asset_record]
    assert len(records) >= 2


def test_support_entitlement_rows_become_support_entitlement_atoms(tmp_path: Path):
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "support.csv",
        "Asset,Support Level,Contract ID,Renewal Date\n"
        "ws-1,24x7,CT-100,2027-01-15\n"
        "ws-2,8x5,CT-101,2026-12-31\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    sups = [a for a in atoms if a.atom_type == AtomType.support_entitlement]
    assert len(sups) >= 2


def test_site_roster_rows_become_site_roster_atoms(tmp_path: Path):
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "site_list.csv",
        "Site ID,Site Name,Address,Access Notes\n"
        "S-1,Banks High School,13050 NW Main St,After-hours escort req'd\n"
        "S-2,District Core,200 4th Ave,Badge required\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    rosters = [a for a in atoms if a.atom_type == AtomType.site_roster]
    assert len(rosters) >= 2


def test_lifecycle_rows_become_lifecycle_status_atoms(tmp_path: Path):
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "lifecycle.csv",
        "Asset ID,Serial,Model,Lifecycle,Status\n"
        "AST-1,SN1,WS-C3850-48,EOL,unsupported\n"
        "AST-2,SN2,WS-C9300,Current,supported\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    lifecycle = [a for a in atoms if a.atom_type == AtomType.lifecycle_status]
    assert len(lifecycle) >= 2


def test_cell_fact_emits_exclusion_subatom(tmp_path: Path):
    """A cell whose text says "not included" gets its own
    ``exclusion`` sub-atom anchored to the parent row."""
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "scope.csv",
        "Item,Notes\n"
        "Camera install,Fire alarm not included\n"
        "Cabling,Owner provides ceiling access\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    exclusions = [a for a in atoms if a.atom_type == AtomType.exclusion]
    assert any("not included" in (a.raw_text or "").lower() for a in exclusions)
    # The sub-atom should carry parent_row_atom_id.
    assert any(
        isinstance(a.value, dict) and a.value.get("parent_row_atom_id")
        for a in exclusions
    )


def test_cell_fact_emits_risk_subatom_for_eol(tmp_path: Path):
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "scope.csv",
        "Item,Notes\n"
        "Switch refresh,EOL hardware in MDF\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    risks = [a for a in atoms if a.atom_type == AtomType.risk]
    assert any("eol" in (a.raw_text or "").lower() for a in risks)


def test_legacy_generic_row_still_emits_scope_item(tmp_path: Path):
    """A row that doesn't match any structured profile keeps the
    legacy ``scope_item`` / ``contractual_scope`` / 0.84 behavior."""
    out = _emit_csv(
        XlsxParser(),
        tmp_path,
        "misc.csv",
        "Foo,Bar,Baz\n"
        "1,2,3\n"
        "4,5,6\n",
    )
    atoms = out if isinstance(out, list) else out.atoms
    scope_items = [a for a in atoms if a.atom_type == AtomType.scope_item]
    assert len(scope_items) >= 2
