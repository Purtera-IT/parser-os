"""PR6 — vector/text symbol detector unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.schematic_models import (
    DetectionTarget,
    DetectionTargetSet,
    ParsedLegend,
    ParsedLegendEntry,
)
from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    TextBlock,
    page_text_blocks,
)
from orbitbrief_page_os.segmentation.schematic.symbol_detector import (
    _deterministic_nms,
    detect_symbols,
)


def _wn_entry(page_index: int = 0) -> ParsedLegendEntry:
    return ParsedLegendEntry.make(
        page_index=page_index,
        label_text="WIRELESS NODE",
        normalized_label="wireless node",
        raw_symbol_text="WN",
        normalized_symbol_text="wn",
        symbol_bbox_pdf=(72.0, 100.0, 90.0, 112.0),
        confidence=0.9,
    )


def _wn_legend(page_index: int = 0) -> ParsedLegend:
    return ParsedLegend.make(
        page_index=page_index,
        sheet_number="T0.01",
        title="SYMBOL LEGEND",
        scope="global",
        entries=(_wn_entry(page_index),),
        confidence=0.9,
    )


def _wn_target_set(legend: ParsedLegend, page_index: int) -> DetectionTargetSet:
    target = DetectionTarget(
        target_key="wireless_node",
        entity_key="device:wireless_node",
        completeness="load_bearing",
        expected_modalities=("text_tag", "glyph_template"),
        legend_entry_id=legend.entries[0].entry_id,
        aliases=("wireless node", "wn"),
    )
    return DetectionTargetSet.make(
        page_index=page_index,
        sheet_number="E1.01",
        pack_id="test_pack",
        legend_id=legend.legend_id,
        targets=(target,),
    )


def _make_drawing(tmp_path: Path, wn_positions: list[tuple[float, float]]) -> Path:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    # Page 0: legend
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "WN", fontsize=10)
    page.insert_text((180, 100), "WIRELESS NODE", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Page 1: drawing body with N copies of WN
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    for (x, y) in wn_positions:
        page.insert_text((x, y), "WN", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_text_tag_detector_finds_three_wn_tokens(tmp_path: Path) -> None:
    pdf = _make_drawing(tmp_path, [(200.0, 300.0), (400.0, 350.0), (250.0, 500.0)])
    doc = fitz.open(str(pdf))
    try:
        legend_page = doc.load_page(0)
        # Build legend entry from the actual WN bbox on the legend page so
        # glyph template matching has a meaningful crop too.
        legend_blocks = page_text_blocks(legend_page)
        wn_block = next(b for b in legend_blocks if b.text.strip() == "WN")
        entry = ParsedLegendEntry.make(
            page_index=0,
            label_text="WIRELESS NODE",
            normalized_label="wireless node",
            raw_symbol_text="WN",
            normalized_symbol_text="wn",
            symbol_bbox_pdf=wn_block.bbox,
            confidence=0.9,
        )
        legend = ParsedLegend.make(
            page_index=0,
            sheet_number="T0.01",
            title="SYMBOL LEGEND",
            scope="global",
            entries=(entry,),
            confidence=0.9,
        )
        target = DetectionTarget(
            target_key="wireless_node",
            entity_key="device:wireless_node",
            completeness="load_bearing",
            expected_modalities=("text_tag",),  # text only to avoid glyph noise
            legend_entry_id=entry.entry_id,
            aliases=("wireless node", "wn"),
        )
        ts = DetectionTargetSet.make(
            page_index=1,
            sheet_number="E1.01",
            pack_id="test_pack",
            legend_id=legend.legend_id,
            targets=(target,),
        )
        page = doc.load_page(1)
        blocks = page_text_blocks(page)
        dets = detect_symbols(
            page=page,
            page_index=1,
            sheet_number="E1.01",
            blocks=blocks,
            target_set=ts,
            legend=legend,
            legend_page=legend_page,
            include_glyph=False,
        )
    finally:
        doc.close()
    assert len(dets) == 3, [(d.target_key, d.bbox_pdf) for d in dets]
    for d in dets:
        assert d.modality == "text_tag"
        assert d.target_key == "wireless_node"
        assert d.crop_sha256 and len(d.crop_sha256) == 64


def test_detector_is_deterministic_across_runs(tmp_path: Path) -> None:
    pdf = _make_drawing(tmp_path, [(200.0, 300.0), (400.0, 350.0)])

    def run() -> list[str]:
        doc = fitz.open(str(pdf))
        try:
            legend_page = doc.load_page(0)
            legend_blocks = page_text_blocks(legend_page)
            wn_block = next(b for b in legend_blocks if b.text.strip() == "WN")
            entry = ParsedLegendEntry.make(
                page_index=0,
                label_text="WIRELESS NODE",
                normalized_label="wireless node",
                raw_symbol_text="WN",
                normalized_symbol_text="wn",
                symbol_bbox_pdf=wn_block.bbox,
                confidence=0.9,
            )
            legend = ParsedLegend.make(
                page_index=0,
                sheet_number="T0.01",
                title="SYMBOL LEGEND",
                scope="global",
                entries=(entry,),
                confidence=0.9,
            )
            target = DetectionTarget(
                target_key="wireless_node",
                entity_key="device:wireless_node",
                completeness="load_bearing",
                expected_modalities=("text_tag",),
                legend_entry_id=entry.entry_id,
                aliases=("wn",),
            )
            ts = DetectionTargetSet.make(
                page_index=1,
                sheet_number="E1.01",
                pack_id="test_pack",
                legend_id=legend.legend_id,
                targets=(target,),
            )
            page = doc.load_page(1)
            blocks = page_text_blocks(page)
            dets = detect_symbols(
                page=page,
                page_index=1,
                sheet_number="E1.01",
                blocks=blocks,
                target_set=ts,
                legend=legend,
                legend_page=legend_page,
                include_glyph=False,
            )
        finally:
            doc.close()
        return [d.detection_id for d in dets]

    a = run()
    b = run()
    assert a == b
    assert len(a) == 2


def test_excluded_bboxes_suppress_legend_self_matches(tmp_path: Path) -> None:
    # The legend page itself contains "WN" — if we don't exclude it,
    # the detector would falsely match the legend row.
    pdf = _make_drawing(tmp_path, [])
    doc = fitz.open(str(pdf))
    try:
        legend_page = doc.load_page(0)
        legend_blocks = page_text_blocks(legend_page)
        wn_block = next(b for b in legend_blocks if b.text.strip() == "WN")
        legend = _wn_legend(page_index=0)
        target = DetectionTarget(
            target_key="wireless_node",
            entity_key="device:wireless_node",
            completeness="load_bearing",
            expected_modalities=("text_tag",),
            legend_entry_id=legend.entries[0].entry_id,
            aliases=("wn",),
        )
        ts = DetectionTargetSet.make(
            page_index=0,
            sheet_number="T0.01",
            pack_id="test_pack",
            legend_id=legend.legend_id,
            targets=(target,),
        )
        # Exclude WN's own bbox.
        dets = detect_symbols(
            page=legend_page,
            page_index=0,
            sheet_number="T0.01",
            blocks=legend_blocks,
            target_set=ts,
            legend=legend,
            legend_page=legend_page,
            excluded_bboxes=(wn_block.bbox,),
            include_glyph=False,
        )
    finally:
        doc.close()
    assert not dets, [(d.target_key, d.nearby_text) for d in dets]


def test_nms_collapses_overlapping_same_target() -> None:
    from app.parsers.schematic_models import SymbolDetection

    a = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="ptz",
        entity_key="device:ptz",
        legend_entry_id=None,
        bbox_pdf=(10, 10, 30, 30),
        crop_sha256="aaa",
        modality="text_tag",
        confidence=0.95,
    )
    b = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="ptz",
        entity_key="device:ptz",
        legend_entry_id=None,
        bbox_pdf=(12, 12, 28, 28),
        crop_sha256="bbb",
        modality="glyph_template",
        confidence=0.80,
    )
    out = _deterministic_nms([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.95


def test_nms_keeps_different_targets_at_same_location() -> None:
    from app.parsers.schematic_models import SymbolDetection

    a = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="ptz",
        entity_key="device:ptz",
        legend_entry_id=None,
        bbox_pdf=(10, 10, 30, 30),
        crop_sha256="aaa",
        modality="text_tag",
        confidence=0.95,
    )
    b = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="dome",
        entity_key="device:dome",
        legend_entry_id=None,
        bbox_pdf=(11, 11, 29, 29),
        crop_sha256="bbb",
        modality="text_tag",
        confidence=0.95,
    )
    out = _deterministic_nms([a, b])
    assert len(out) == 2
