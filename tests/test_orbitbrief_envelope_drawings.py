"""Tests for the OrbitBrief envelope's schematic ``drawings`` section."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.manifest import build_artifact_fingerprint
from app.core.orbitbrief_envelope import (
    build_orbitbrief_envelope,
    envelope_to_markdown,
)
from app.core.schemas import CompileManifest, CompileResult
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def _build_rich_drawing_set(path: Path) -> None:
    doc = fitz.open()
    # Legend
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "3", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Floor plan with rooms, notes, schedule, detections
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((100, 280), "LOBBY 101", fontsize=10)
    page.insert_text((120, 320), "PTZ", fontsize=10)
    page.insert_text((400, 280), "CONFERENCE 301", fontsize=10)
    page.insert_text((420, 320), "PTZ", fontsize=10)
    page.insert_text((250, 320), "PTZ", fontsize=10)
    page.insert_text((72, 500), "KEYED NOTES", fontsize=11)
    page.insert_text((72, 520), "1. Verify mounting in field.", fontsize=9)
    page.insert_text((400, 700), "Project: Marriott Renovation", fontsize=9)
    page.insert_text((400, 715), "Sheet Title: First Floor", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def _project_with_pdf(tmp_path: Path) -> tuple[Path, Path]:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    artifacts = project_dir / "artifacts"
    artifacts.mkdir()
    pdf = artifacts / "drawings.pdf"
    _build_rich_drawing_set(pdf)
    return project_dir, pdf


def _compile_to_envelope(tmp_path: Path) -> dict:
    project_dir, pdf = _project_with_pdf(tmp_path)

    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    artifact_id = stable_id("art", str(pdf))
    parse_out = parser.parse_artifact("proj_test", artifact_id, pdf, domain_pack=pack)
    fingerprint = build_artifact_fingerprint(
        path=pdf,
        artifact_id=artifact_id,
        parsed_atoms=list(parse_out.atoms),
        filename=pdf.name,
        parser_name="orbitbrief_pdf",
        parser_version="test",
    )

    manifest = CompileManifest(
        compile_id="cmp_test",
        project_id="proj_test",
        artifact_fingerprints=[fingerprint],
        started_at="2024-01-01T00:00:00Z",
        deterministic_seed="seed",
        input_signature="sig",
    )
    compile_result = CompileResult(
        project_id="proj_test",
        compile_id="cmp_test",
        manifest=manifest,
        atoms=parse_out.atoms,
        edges=[],
        entities=[],
        packets=[],
        candidates=[],
        receipts=[],
    )
    return build_orbitbrief_envelope(
        project_dir=project_dir,
        compile_result=compile_result,
    )


def test_drawings_section_present_when_schematic_atoms_exist(tmp_path: Path) -> None:
    env = _compile_to_envelope(tmp_path)
    drawings = env.get("drawings")
    assert drawings is not None, "drawings section missing on schematic project"
    assert drawings["artifacts"], drawings
    arts = drawings["artifacts"]
    pages = arts[0]["pages"]
    assert pages
    floor = next((p for p in pages if p.get("sheet_number") == "E1.01"), None)
    assert floor is not None
    counts = floor.get("target_counts") or {}
    assert counts.get("ptz_camera") == 3


def test_drawings_section_indexes_collect_global_counts(tmp_path: Path) -> None:
    env = _compile_to_envelope(tmp_path)
    idx = env["drawings"]["indexes"]
    assert idx["detections_by_target_key"].get("ptz_camera") == 3
    assert "T0.01" in idx["drawings_by_sheet_number"] or "E1.01" in idx["drawings_by_sheet_number"]


def test_envelope_markdown_renders_drawings_block(tmp_path: Path) -> None:
    env = _compile_to_envelope(tmp_path)
    md = envelope_to_markdown(env)
    assert "## Drawings" in md
    assert "ptz_camera" in md
    assert "Sheet E1.01" in md
    assert "Rooms:" in md
    assert "Keyed notes:" in md


def test_drawings_section_absent_on_non_schematic_project(tmp_path: Path) -> None:
    """An RFP-only PDF with no domain pack produces no schematic atoms;
    the drawings section must be omitted (not just empty) so old
    consumers don't see a new key.
    """
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    artifacts = project_dir / "artifacts"
    artifacts.mkdir()
    pdf = artifacts / "rfp.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Request for Proposal", fontsize=14)
    page.insert_text((72, 100), "Scope: Provide widgets.", fontsize=11)
    doc.save(str(pdf))
    doc.close()

    parser = OrbitBriefPdfParser()
    artifact_id = stable_id("art", str(pdf))
    parse_out = parser.parse_artifact("proj_test", artifact_id, pdf, domain_pack=None)
    fingerprint = build_artifact_fingerprint(
        path=pdf,
        artifact_id=artifact_id,
        parsed_atoms=list(parse_out.atoms),
        filename=pdf.name,
        parser_name="orbitbrief_pdf",
        parser_version="test",
    )

    manifest = CompileManifest(
        compile_id="cmp_test",
        project_id="proj_test",
        artifact_fingerprints=[fingerprint],
        started_at="2024-01-01T00:00:00Z",
        deterministic_seed="seed",
        input_signature="sig",
    )
    compile_result = CompileResult(
        project_id="proj_test",
        compile_id="cmp_test",
        manifest=manifest,
        atoms=parse_out.atoms,
        edges=[],
        entities=[],
        packets=[],
        candidates=[],
        receipts=[],
    )
    env = build_orbitbrief_envelope(
        project_dir=project_dir,
        compile_result=compile_result,
    )
    assert "drawings" not in env
