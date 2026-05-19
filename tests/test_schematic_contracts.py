"""PR1 — schematic data contracts and atom-type enum coverage."""
from __future__ import annotations

import pytest

from app.core.schemas import AtomType
from app.parsers.schematic_models import (
    BBOX_UNITS_PDF_POINTS,
    DetectionTarget,
    DetectionTargetSet,
    ParsedLegend,
    ParsedLegendEntry,
    SchematicWarning,
    SymbolDetection,
    crop_sha256_of_pixels,
)


def test_new_schematic_atom_types_exist() -> None:
    assert AtomType("schematic_legend") is AtomType.schematic_legend
    assert AtomType("schematic_detection_target_set") is AtomType.schematic_detection_target_set
    assert AtomType("schematic_symbol_detection") is AtomType.schematic_symbol_detection
    assert AtomType("schematic_warning") is AtomType.schematic_warning


def test_parsed_legend_entry_make_is_deterministic() -> None:
    a = ParsedLegendEntry.make(
        page_index=2,
        label_text="WIRELESS NODE",
        normalized_label="wireless node",
        raw_symbol_text="WN",
        normalized_symbol_text="wn",
        symbol_bbox_pdf=(10.0, 20.0, 30.0, 40.0),
        confidence=0.9,
    )
    b = ParsedLegendEntry.make(
        page_index=2,
        label_text="WIRELESS NODE",
        normalized_label="wireless node",
        raw_symbol_text="WN",
        normalized_symbol_text="wn",
        symbol_bbox_pdf=(10.0, 20.0, 30.0, 40.0),
        confidence=0.9,
    )
    assert a == b
    assert a.entry_id == b.entry_id
    assert a.entry_id.startswith("legend_entry_")


def test_parsed_legend_entry_requires_symbol_text_or_crop() -> None:
    with pytest.raises(ValueError):
        ParsedLegendEntry(
            entry_id="x",
            label_text="X",
            normalized_label="x",
            confidence=0.5,
        )


def test_parsed_legend_entry_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        ParsedLegendEntry.make(
            page_index=0,
            label_text="X",
            normalized_label="x",
            raw_symbol_text="X",
            normalized_symbol_text="x",
            confidence=1.5,
        )


def test_parsed_legend_make_assigns_stable_id() -> None:
    entry = ParsedLegendEntry.make(
        page_index=2,
        label_text="CARD READER",
        normalized_label="card reader",
        raw_symbol_text="CR",
        normalized_symbol_text="cr",
        confidence=0.8,
    )
    legend = ParsedLegend.make(
        page_index=2,
        sheet_number="T0.01",
        title="SYMBOLS & LEGENDS",
        scope="global",
        entries=(entry,),
        confidence=0.85,
    )
    legend_b = ParsedLegend.make(
        page_index=2,
        sheet_number="T0.01",
        title="SYMBOLS & LEGENDS",
        scope="global",
        entries=(entry,),
        confidence=0.85,
    )
    assert legend.legend_id == legend_b.legend_id
    assert legend.legend_id.startswith("legend_")


def test_detection_target_validates_modalities_present() -> None:
    with pytest.raises(ValueError):
        DetectionTarget(
            target_key="cam.ptz",
            entity_key="device:ptz_camera",
            completeness="load_bearing",
            expected_modalities=(),
        )


def test_detection_target_set_sorts_targets_deterministically() -> None:
    t1 = DetectionTarget(
        target_key="cam.ptz",
        entity_key="device:ptz_camera",
        completeness="load_bearing",
        expected_modalities=("text_tag",),
    )
    t2 = DetectionTarget(
        target_key="cam.dome",
        entity_key="device:dome_camera",
        completeness="load_bearing",
        expected_modalities=("glyph_template",),
    )
    a = DetectionTargetSet.make(
        page_index=4,
        sheet_number="E1.01",
        pack_id="security_camera",
        legend_id="legend_abc",
        targets=(t1, t2),
    )
    b = DetectionTargetSet.make(
        page_index=4,
        sheet_number="E1.01",
        pack_id="security_camera",
        legend_id="legend_abc",
        targets=(t2, t1),
    )
    assert a.targets == b.targets
    assert [t.target_key for t in a.targets] == ["cam.dome", "cam.ptz"]


def test_symbol_detection_rejects_zero_area_bbox() -> None:
    with pytest.raises(ValueError):
        SymbolDetection(
            detection_id="d",
            page_index=4,
            sheet_number="E1.01",
            target_key="cam.ptz",
            entity_key="device:ptz_camera",
            legend_entry_id=None,
            bbox_pdf=(10.0, 10.0, 10.0, 20.0),
            crop_sha256="deadbeef",
            modality="text_tag",
            confidence=0.9,
        )


def test_symbol_detection_locator_uses_pdf_points() -> None:
    det = SymbolDetection.make(
        page_index=4,
        sheet_number="E1.01",
        target_key="cam.ptz",
        entity_key="device:ptz_camera",
        legend_entry_id="legend_entry_abc",
        bbox_pdf=(10.0, 20.0, 30.0, 50.0),
        crop_sha256="abc123",
        modality="text_tag",
        confidence=0.95,
    )
    loc = det.locator_dict()
    assert loc["bbox_units"] == BBOX_UNITS_PDF_POINTS
    assert loc["bbox"] == [10.0, 20.0, 30.0, 50.0]
    assert loc["crop_sha256"] == "abc123"
    assert loc["target_key"] == "cam.ptz"


def test_schematic_warning_locator_minimal_shape() -> None:
    w = SchematicWarning.make(
        warning_type="missing_legend",
        page_index=4,
        sheet_number=None,
        detail="No legend resolvable for this drawing page",
    )
    loc = w.locator_dict()
    assert loc["warning_type"] == "missing_legend"
    assert loc["page"] == 4
    assert "bbox" not in loc


def test_crop_sha256_namespaces_dimensions() -> None:
    pixels = b"\x00" * 12  # 2x2 RGB
    a = crop_sha256_of_pixels(pixels, 2, 2, 3)
    b = crop_sha256_of_pixels(pixels, 4, 1, 3)
    c = crop_sha256_of_pixels(pixels, 2, 2, 3)
    assert a == c
    assert a != b
