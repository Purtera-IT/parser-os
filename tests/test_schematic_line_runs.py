"""Tests for the line-run detector and ``schematic_line_run`` atom."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from app.parsers.schematic_models import SymbolDetection
from orbitbrief_page_os.segmentation.schematic.line_runs import (
    LineRun,
    detect_line_runs,
)


def _build_drawing_with_line_runs(path: Path) -> None:
    doc = fitz.open()
    # Legend page
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Floor plan with PTZ markers + a line run between them
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((150, 300), "PTZ", fontsize=10)
    page.insert_text((450, 300), "PTZ", fontsize=10)
    # Draw a long horizontal line from near PTZ#1 to near PTZ#2 — a
    # cable run between the two cameras.
    page.draw_line(fitz.Point(160, 310), fitz.Point(440, 310), color=(0, 0, 0), width=1)
    # A second, shorter line going off into nowhere — should still
    # be picked up because we pass detections=() (no snap filter).
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def test_line_run_detector_finds_polyline_between_detections() -> None:
    """A PyMuPDF page with one straight line between two detection
    centers should produce one LineRun whose endpoints snap to the
    two detections.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line(fitz.Point(100, 200), fitz.Point(400, 200), color=(0, 0, 0), width=1)
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    try:
        d1 = SymbolDetection.make(
            page_index=0,
            sheet_number="E1.01",
            target_key="ptz_camera",
            entity_key="device:ptz_camera",
            legend_entry_id=None,
            bbox_pdf=(95, 195, 105, 205),
            crop_sha256="aaa",
            modality="text_tag",
            confidence=0.9,
        )
        d2 = SymbolDetection.make(
            page_index=0,
            sheet_number="E1.01",
            target_key="ptz_camera",
            entity_key="device:ptz_camera",
            legend_entry_id=None,
            bbox_pdf=(395, 195, 405, 205),
            crop_sha256="bbb",
            modality="text_tag",
            confidence=0.9,
        )
        runs = detect_line_runs(
            page=doc.load_page(0),
            page_index=0,
            sheet_number="E1.01",
            detections=[d1, d2],
        )
    finally:
        doc.close()
    assert runs, "expected one line run"
    run = runs[0]
    assert {run.from_detection_id, run.to_detection_id} == {d1.detection_id, d2.detection_id}
    assert run.length_pt > 200  # ~300 pt long


def test_line_run_detector_filters_short_decoration_lines() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # 6pt line — way too short to be a run
    page.draw_line(fitz.Point(100, 200), fitz.Point(106, 200), color=(0, 0, 0), width=1)
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    try:
        runs = detect_line_runs(
            page=doc.load_page(0),
            page_index=0,
            sheet_number="E1.01",
            detections=(),
        )
    finally:
        doc.close()
    assert runs == []


def test_line_run_detector_filters_long_border_lines() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # ~750pt line — too long, almost certainly a border
    page.draw_line(fitz.Point(20, 200), fitz.Point(780, 200), color=(0, 0, 0), width=1)
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    try:
        runs = detect_line_runs(
            page=doc.load_page(0),
            page_index=0,
            sheet_number="E1.01",
            detections=(),
        )
    finally:
        doc.close()
    assert runs == []


def test_line_run_atom_emitted_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_drawing_with_line_runs(pdf)
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    line_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_line_run]
    assert line_atoms, "expected at least one schematic_line_run atom"
    atom = line_atoms[0]
    assert atom.value["from_detection_id"] is not None
    assert atom.value["to_detection_id"] is not None
    assert atom.value["from_detection_id"] != atom.value["to_detection_id"]
    # Replayable provenance
    loc = atom.source_refs[0].locator
    assert loc.get("bbox_units") == "pdf_points"
    assert loc.get("crop_sha256")


def test_line_run_atom_ids_stable_across_runs(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_drawing_with_line_runs(pdf)
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))

    def _run() -> list[str]:
        out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
        return sorted(
            a.id for a in out.atoms if a.atom_type == AtomType.schematic_line_run
        )

    assert _run() == _run()
