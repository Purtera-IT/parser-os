"""PR6 — end-to-end schematic detections through OrbitBriefPdfParser."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.core.source_replay import replay_source_ref
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def _build_drawing_set(path: Path) -> None:
    doc = fitz.open()
    # Legend sheet T0.01
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "3", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Floor plan E1.01 — three PTZ markers + one unknown XYZ
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((200, 300), "PTZ", fontsize=10)
    page.insert_text((350, 300), "PTZ", fontsize=10)
    page.insert_text((500, 300), "PTZ", fontsize=10)
    page.insert_text((200, 500), "XYZ", fontsize=10)
    page.insert_text((350, 500), "XYZ", fontsize=10)
    page.insert_text((500, 500), "XYZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def _parse(path: Path):
    pack = load_domain_pack("security_camera")
    parser = OrbitBriefPdfParser()
    art_id = stable_id("art", str(path))
    out = parser.parse_artifact("proj_test", art_id, path, domain_pack=pack)
    return out, art_id


def test_three_ptz_tokens_produce_three_detections(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out, _ = _parse(pdf)
    dets = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection]
    ptz = [d for d in dets if d.value["target_key"] == "ptz_camera"]
    assert len(ptz) == 3, [d.value["bbox"] for d in ptz]
    for d in ptz:
        # Provenance contract: every detection carries page+bbox.
        loc = d.source_refs[0].locator
        assert loc["page"] == 1
        assert len(loc["bbox"]) == 4
        assert loc["bbox_units"] == "pdf_points"
        assert loc["crop_sha256"] and len(loc["crop_sha256"]) == 64


def test_unknown_xyz_token_produces_unknown_symbol_warning(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out, _ = _parse(pdf)
    warnings = [a for a in out.atoms if a.atom_type == AtomType.schematic_warning]
    unknowns = [w for w in warnings if w.value["warning_type"] == "unknown_symbol"]
    tokens = {w.value.get("detail", "") for w in unknowns}
    assert any("'XYZ'" in t for t in tokens), tokens


def test_detection_ids_are_stable_across_compiles(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out_a, _ = _parse(pdf)
    out_b, _ = _parse(pdf)
    a_ids = sorted(
        atom.id for atom in out_a.atoms if atom.atom_type == AtomType.schematic_symbol_detection
    )
    b_ids = sorted(
        atom.id for atom in out_b.atoms if atom.atom_type == AtomType.schematic_symbol_detection
    )
    assert a_ids == b_ids
    assert a_ids


def test_detection_source_replay_verifies(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out, art_id = _parse(pdf)
    dets = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection]
    assert dets
    artifact_paths = {art_id: pdf}
    for det in dets:
        src = det.source_refs[0]
        receipt = replay_source_ref(det, src, artifact_paths)
        assert receipt.replay_status == "verified", (
            f"detection {det.id} failed replay: {receipt.reason}"
        )


def test_detections_carry_legend_entry_id(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out, _ = _parse(pdf)
    dets = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection]
    for d in dets:
        assert d.value.get("legend_entry_id"), f"detection {d.id} missing legend_entry_id"


def test_detections_written_to_derived_json(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_drawing_set(pdf)
    out, _ = _parse(pdf)
    paths = [d.relative_path for d in out.derived_files]
    assert any(p.endswith("schematic_detections.json") for p in paths)
    detections_file = next(d for d in out.derived_files if d.relative_path.endswith("schematic_detections.json"))
    payload = detections_file.content_json
    assert payload and payload["detections"], "schematic_detections.json should not be empty"
