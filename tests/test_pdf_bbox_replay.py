"""PR1 — PDF schematic bbox/crop_sha256 source replay."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.source_replay import replay_source_ref
from app.parsers.schematic_models import (
    BBOX_UNITS_PDF_POINTS,
    SCHEMATIC_REPLAY_DPI,
    crop_sha256_of_pixels,
)


def _build_drawing_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Draw a couple of deterministic shapes so a bbox crop has stable pixels.
    page.draw_rect(fitz.Rect(50, 50, 200, 150), color=(0, 0, 0), width=2)
    page.insert_text((60, 100), "LEGEND", fontsize=14, color=(0, 0, 0))
    page.draw_rect(fitz.Rect(300, 300, 360, 360), color=(0, 0, 0), width=1)
    page.insert_text((305, 330), "WN", fontsize=10, color=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _crop_hash(path: Path, page_index: int, bbox: tuple[float, float, float, float]) -> str:
    doc = fitz.open(str(path))
    try:
        page = doc.load_page(page_index)
        zoom = SCHEMATIC_REPLAY_DPI / 72.0
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            clip=fitz.Rect(*bbox),
            alpha=False,
            colorspace=fitz.csRGB,
        )
        return crop_sha256_of_pixels(pix.samples, pix.width, pix.height, pix.n)
    finally:
        doc.close()


def _build_atom_with_locator(artifact_id: str, filename: str, locator: dict) -> tuple[EvidenceAtom, SourceRef]:
    src = SourceRef(
        id="sr_test_1",
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator=locator,
        extraction_method="schematic_test",
        parser_version="test_v1",
    )
    atom = EvidenceAtom(
        id="atom_test_1",
        project_id="proj_test",
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_symbol_detection,
        raw_text="WN @ (300, 300)",
        normalized_text="wn 300 300",
        value={"target_key": "wireless.node"},
        entity_keys=["wireless:node"],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="test_v1",
    )
    return atom, src


def test_bbox_crop_replay_verifies_matching_hash(tmp_path: Path) -> None:
    pdf_path = tmp_path / "drawing.pdf"
    _build_drawing_pdf(pdf_path)

    bbox = (300.0, 300.0, 360.0, 360.0)
    expected_hash = _crop_hash(pdf_path, page_index=0, bbox=bbox)

    atom, src = _build_atom_with_locator(
        artifact_id="art_drawing",
        filename=pdf_path.name,
        locator={
            "page": 0,
            "bbox": list(bbox),
            "bbox_units": BBOX_UNITS_PDF_POINTS,
            "crop_sha256": expected_hash,
        },
    )
    receipt = replay_source_ref(atom, src, {"art_drawing": pdf_path})
    assert receipt.replay_status == "verified", receipt.reason


def test_bbox_crop_replay_fails_on_hash_mismatch(tmp_path: Path) -> None:
    pdf_path = tmp_path / "drawing.pdf"
    _build_drawing_pdf(pdf_path)
    bbox = (300.0, 300.0, 360.0, 360.0)
    atom, src = _build_atom_with_locator(
        artifact_id="art_drawing",
        filename=pdf_path.name,
        locator={
            "page": 0,
            "bbox": list(bbox),
            "bbox_units": BBOX_UNITS_PDF_POINTS,
            "crop_sha256": "0" * 64,
        },
    )
    receipt = replay_source_ref(atom, src, {"art_drawing": pdf_path})
    assert receipt.replay_status == "failed"
    assert "hash mismatch" in receipt.reason


def test_bbox_crop_replay_fails_on_out_of_range_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "drawing.pdf"
    _build_drawing_pdf(pdf_path)
    atom, src = _build_atom_with_locator(
        artifact_id="art_drawing",
        filename=pdf_path.name,
        locator={
            "page": 99,
            "bbox": [10.0, 10.0, 50.0, 50.0],
            "bbox_units": BBOX_UNITS_PDF_POINTS,
            "crop_sha256": "0" * 64,
        },
    )
    receipt = replay_source_ref(atom, src, {"art_drawing": pdf_path})
    assert receipt.replay_status == "failed"
    assert "out of range" in receipt.reason


def test_bbox_crop_replay_fails_on_degenerate_bbox(tmp_path: Path) -> None:
    pdf_path = tmp_path / "drawing.pdf"
    _build_drawing_pdf(pdf_path)
    atom, src = _build_atom_with_locator(
        artifact_id="art_drawing",
        filename=pdf_path.name,
        locator={
            "page": 0,
            "bbox": [50.0, 50.0, 50.0, 50.0],
            "bbox_units": BBOX_UNITS_PDF_POINTS,
            "crop_sha256": "0" * 64,
        },
    )
    receipt = replay_source_ref(atom, src, {"art_drawing": pdf_path})
    assert receipt.replay_status == "failed"
    assert "strictly positive" in receipt.reason


def test_existing_block_id_pdf_replay_path_unchanged(tmp_path: Path) -> None:
    # If a PDF locator has neither block_id nor bbox+crop, it remains
    # 'unsupported' rather than being silently routed elsewhere.
    pdf_path = tmp_path / "empty.pdf"
    _build_drawing_pdf(pdf_path)
    atom, src = _build_atom_with_locator(
        artifact_id="art_drawing",
        filename=pdf_path.name,
        locator={"page": 0},
    )
    receipt = replay_source_ref(atom, src, {"art_drawing": pdf_path})
    assert receipt.replay_status == "unsupported"
    assert "missing block_id" in receipt.reason
