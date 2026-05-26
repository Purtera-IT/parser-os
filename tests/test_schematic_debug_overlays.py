"""PR10 — schematic debug overlay rendering tests."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
PIL = pytest.importorskip("PIL")

from app.parsers.schematic_models import (
    ParsedLegend,
    ParsedLegendEntry,
    SymbolDetection,
)
from orbitbrief_page_os.segmentation.schematic.debug_overlay import render_overlay


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.draw_rect(fitz.Rect(50, 50, 562, 700), color=(0, 0, 0), width=1)
    doc.save(str(path))
    doc.close()


def _legend(page_index: int = 0) -> ParsedLegend:
    entry = ParsedLegendEntry.make(
        page_index=page_index,
        label_text="PTZ CAMERA",
        normalized_label="ptz camera",
        raw_symbol_text="PTZ",
        normalized_symbol_text="ptz",
        symbol_bbox_pdf=(60.0, 80.0, 90.0, 95.0),
        confidence=0.9,
    )
    return ParsedLegend.make(
        page_index=page_index,
        sheet_number="T0.01",
        title="SYMBOL LEGEND",
        scope="global",
        entries=(entry,),
        source_ref_locator={
            "page": page_index,
            "bbox": [55.0, 70.0, 200.0, 120.0],
            "bbox_units": "pdf_points",
        },
        confidence=0.9,
    )


def _detection(page_index: int = 0, x: float = 200.0, y: float = 300.0) -> SymbolDetection:
    return SymbolDetection.make(
        page_index=page_index,
        sheet_number="E1.01",
        target_key="ptz_camera",
        entity_key="device:ptz_camera",
        legend_entry_id="legend_entry_x",
        bbox_pdf=(x, y, x + 30, y + 14),
        crop_sha256="abc123",
        modality="text_tag",
        confidence=0.92,
    )


def test_render_overlay_writes_png(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _make_pdf(pdf)
    doc = fitz.open(str(pdf))
    try:
        page = doc.load_page(0)
        out = tmp_path / "overlay.png"
        result = render_overlay(
            page=page,
            legends_on_page=[_legend()],
            detections=[_detection(), _detection(x=300.0)],
            out_path=out,
        )
    finally:
        doc.close()
    assert result is not None
    assert out.is_file()
    assert out.stat().st_size > 0
    assert result.legend_count == 1
    assert result.detection_count == 2


def test_render_overlay_is_pixel_deterministic(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _make_pdf(pdf)
    doc_a = fitz.open(str(pdf))
    doc_b = fitz.open(str(pdf))
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    try:
        render_overlay(
            page=doc_a.load_page(0),
            legends_on_page=[_legend()],
            detections=[_detection()],
            out_path=a,
        )
        render_overlay(
            page=doc_b.load_page(0),
            legends_on_page=[_legend()],
            detections=[_detection()],
            out_path=b,
        )
    finally:
        doc_a.close()
        doc_b.close()
    assert a.read_bytes() == b.read_bytes(), "overlay PNG drifted across runs"


def test_render_overlay_handles_empty_legend_list(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _make_pdf(pdf)
    doc = fitz.open(str(pdf))
    out = tmp_path / "overlay.png"
    try:
        result = render_overlay(
            page=doc.load_page(0),
            legends_on_page=[],
            detections=[_detection()],
            out_path=out,
        )
    finally:
        doc.close()
    assert result is not None
    assert result.legend_count == 0
    assert result.detection_count == 1


def test_render_overlay_handles_empty_detections(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _make_pdf(pdf)
    doc = fitz.open(str(pdf))
    out = tmp_path / "overlay.png"
    try:
        result = render_overlay(
            page=doc.load_page(0),
            legends_on_page=[_legend()],
            detections=[],
            out_path=out,
        )
    finally:
        doc.close()
    assert result is not None
    assert result.legend_count == 1
    assert result.detection_count == 0


def test_render_overlay_creates_parent_dir(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _make_pdf(pdf)
    doc = fitz.open(str(pdf))
    nested = tmp_path / "deep" / "deeper" / "overlay.png"
    try:
        result = render_overlay(
            page=doc.load_page(0),
            legends_on_page=[_legend()],
            detections=[_detection()],
            out_path=nested,
        )
    finally:
        doc.close()
    assert result is not None
    assert nested.is_file()
