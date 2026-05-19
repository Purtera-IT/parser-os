"""PR5 — legend-first wiring of OrbitbriefPdfParser."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def _build_drawing_set(path: Path) -> None:
    """Two-page PDF: page 0 is a global legend (T0.01), page 1 is a floor plan (E1.01)."""

    doc = fitz.open()
    # ── page 0: legend sheet T0.01 ──
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "4", fontsize=10)
    page.insert_text((72, 128), "DOM", fontsize=10)
    page.insert_text((180, 128), "FIXED DOME CAMERA", fontsize=10)
    page.insert_text((300, 128), "12", fontsize=10)
    page.insert_text((72, 146), "NVR", fontsize=10)
    page.insert_text((180, 146), "NETWORK VIDEO RECORDER", fontsize=10)
    page.insert_text((300, 146), "1", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    # ── page 1: floor plan E1.01 referencing T0.01 ──
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN - FIRST FLOOR", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def _build_rfp_only(path: Path) -> None:
    """Single-page text-only PDF with no legend / no sheet number."""

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Request for Proposal — Boring Project", fontsize=14)
    page.insert_text((72, 100), "1. Scope of work.", fontsize=12)
    page.insert_text((72, 120), "Provide widgets per spec.", fontsize=11)
    page.insert_text((72, 140), "2. Submission deadline.", fontsize=12)
    page.insert_text((72, 160), "Bids due by Friday.", fontsize=11)
    doc.save(str(path))
    doc.close()


def _parse(path: Path, pack_id: str | None = "security_camera"):
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack(pack_id) if pack_id else None
    artifact_id = stable_id("art", str(path))
    return parser.parse_artifact("proj_test", artifact_id, path, domain_pack=pack)


def test_schematic_pre_pass_emits_legend_and_target_set(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out = _parse(pdf, pack_id="security_camera")
    legend_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_legend]
    target_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    assert legend_atoms, "expected at least one schematic_legend atom"
    assert target_atoms, "expected at least one schematic_detection_target_set atom"
    # Legend should carry the PTZ / DOM / NVR entries.
    syms = {e["symbol"] for a in legend_atoms for e in a.value["entries"]}
    assert {"PTZ", "DOM", "NVR"}.issubset(syms)


def test_schematic_pre_pass_uses_resolved_legend_for_floor_plan(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out = _parse(pdf, pack_id="security_camera")
    target_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    pages = {a.value["page"] for a in target_atoms}
    # Floor plan page (index 1) must receive a target set via cross-sheet resolution.
    assert 1 in pages, f"floor plan page missing target set: pages={pages}"
    floor = next(a for a in target_atoms if a.value["page"] == 1)
    target_keys = {t["target_key"] for t in floor.value["targets"]}
    assert target_keys, "floor plan target set should not be empty"


def test_schematic_pre_pass_emits_warning_for_drawing_without_legend(tmp_path: Path) -> None:
    pdf = tmp_path / "lone_floor_plan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN - ORPHAN", fontsize=14)
    page.insert_text((500, 740), "E2.01", fontsize=10)
    # No legend anywhere in this PDF, but we declare detection_targets via the pack
    doc.save(str(pdf))
    doc.close()

    out = _parse(pdf, pack_id="security_camera")
    warning_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_warning]
    types = {a.value["warning_type"] for a in warning_atoms}
    assert "missing_legend" in types


def test_rfp_only_pdf_emits_no_schematic_atoms(tmp_path: Path) -> None:
    pdf = tmp_path / "rfp.pdf"
    _build_rfp_only(pdf)
    out = _parse(pdf, pack_id=None)
    schematic_kinds = {
        AtomType.schematic_legend,
        AtomType.schematic_detection_target_set,
        AtomType.schematic_symbol_detection,
        AtomType.schematic_warning,
    }
    schematic_atoms = [a for a in out.atoms if a.atom_type in schematic_kinds]
    assert not schematic_atoms, f"rfp-only PDF should not produce schematic atoms: {schematic_atoms}"


def test_schematic_pre_pass_writes_derived_jsons(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out = _parse(pdf, pack_id="security_camera")
    paths = [d.relative_path for d in out.derived_files]
    assert any(p.endswith("schematic_legends.json") for p in paths)
    assert any(p.endswith("schematic_targets.json") for p in paths)


def test_schematic_pre_pass_atoms_are_deterministic(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    a = _parse(pdf, pack_id="security_camera")
    b = _parse(pdf, pack_id="security_camera")
    a_ids = [atom.id for atom in a.atoms if atom.atom_type.value.startswith("schematic_")]
    b_ids = [atom.id for atom in b.atoms if atom.atom_type.value.startswith("schematic_")]
    assert a_ids == b_ids, "schematic atom IDs drifted across runs"


def test_target_set_carries_resolved_aliases_per_target(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out = _parse(pdf, pack_id="security_camera")
    target_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    # The legend has PTZ, DOM, NVR — at least one of those should map to a load_bearing pack target.
    assert target_atoms
    keys: set[str] = set()
    for atom in target_atoms:
        for t in atom.value["targets"]:
            keys.add(t["target_key"])
    # We expect ptz_camera and at least one of fixed_dome_camera / nvr matched.
    assert "ptz_camera" in keys or "fixed_dome_camera" in keys or "nvr" in keys, keys


def test_pre_pass_skips_when_no_legend_and_no_targets(tmp_path: Path) -> None:
    # default_pack has no detection_targets; an RFP-style PDF with no legend
    # should not emit any schematic atoms.
    pdf = tmp_path / "rfp.pdf"
    _build_rfp_only(pdf)
    out = _parse(pdf, pack_id=None)
    assert not [a for a in out.atoms if a.atom_type.value.startswith("schematic_")]
