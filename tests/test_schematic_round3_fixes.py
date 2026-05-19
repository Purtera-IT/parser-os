"""Round-3 fix coverage: items from the post-boss-review scorecard.

Each test pins one of the architectural gaps the audit listed as
"not fixed" so they can't quietly regress:

  - OCR words → TextBlocks
  - Resolver warnings carry bbox provenance
  - Title block / keyed notes / schedules excluded from detection
  - Multi-column construction legends (mounting height etc.)
  - Bounded _expand_block_downward
  - Drawing-index → legend resolution
  - Rotated title blocks
  - Glyph-metric sub-token bbox
  - Subtype → parent entity rollup
  - _verify_pdf_block handles block_id + bbox together
  - End-to-end realistic stress fixture

A separate ``test_schematic_marriott_stress.py`` couldn't share fixtures
without duplication, so we keep one suite.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from app.parsers.schematic_models import ParsedLegendEntry, ParsedLegend, DetectionTarget, DetectionTargetSet
from orbitbrief_page_os.segmentation.schematic.exclusion_zones import detect_exclusion_zones
from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    TextBlock,
    locate_legend_candidates,
    page_text_blocks,
)
from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend
from orbitbrief_page_os.segmentation.schematic.legend_resolver import (
    LegendResolver,
    extract_sheet_number_with_bbox,
)
from orbitbrief_page_os.segmentation.schematic.ocr import (
    OcrWord,
    words_to_textblocks,
)


def _b(text: str, x0: float, y0: float, w: float = 80, h: float = 12) -> TextBlock:
    return TextBlock(text=text, bbox=(x0, y0, x0 + w, y0 + h))


# ─── 1. OCR words → TextBlocks ───


def test_ocr_words_convert_to_pdf_point_textblocks() -> None:
    # Two words on the same line at 200 DPI = 2.78 px per pt.
    # A word at pixel (100, 100, 200, 130) lands at (36, 36, 72, 47) pt.
    words = [
        OcrWord(text="LEGEND", confidence=0.9, bbox=(100, 100, 200, 130)),
        OcrWord(text="KEY", confidence=0.9, bbox=(220, 102, 280, 128)),
        OcrWord(text="WN", confidence=0.95, bbox=(100, 200, 130, 226)),
    ]
    blocks = words_to_textblocks(words, page_dpi=200)
    assert len(blocks) == 2  # two y-bands
    # First block (y=100ish in pixel) groups LEGEND + KEY
    first = blocks[0]
    assert "LEGEND" in first.text and "KEY" in first.text
    # Second block (y=200) has WN alone
    second = blocks[1]
    assert second.text == "WN"
    # Confirm pt conversion: 200 dpi → 72/200 = 0.36 scale
    assert math.isclose(first.bbox[0], 100 * 72 / 200, rel_tol=0.01)


def test_ocr_words_empty_input_returns_empty() -> None:
    assert words_to_textblocks([], page_dpi=200) == []


# ─── 2. Resolver warnings carry bbox ───


def test_resolver_missing_legend_warning_attaches_sheet_token_bbox() -> None:
    res = LegendResolver()
    blocks = [
        _b("FLOOR PLAN", 72, 60, w=200),
        _b("E5.01", 500, 740),  # title-block sheet number
    ]
    res.ingest_page(page_index=4, blocks=blocks)
    resolved = res.resolve_for_page(4)
    warnings = [w for w in resolved.warnings if w.warning_type == "missing_legend"]
    assert warnings
    bbox = warnings[0].bbox_pdf
    assert bbox is not None
    # The bbox should be the sheet-number block we inserted at (500, 740).
    assert 490 < bbox[0] < 530


def test_resolver_unresolved_reference_attaches_reference_bbox() -> None:
    res = LegendResolver()
    blocks = [
        _b("E1.01", 500, 740),
        _b("see sheet Z9.99 for legend", 60, 100, w=300),
    ]
    res.ingest_page(page_index=4, blocks=blocks)
    resolved = res.resolve_for_page(4)
    unresolved = [w for w in resolved.warnings if w.warning_type == "unresolved_legend_reference"]
    assert unresolved
    bbox = unresolved[0].bbox_pdf
    assert bbox is not None
    # The reference block sits at x=60; should not be the title-block bbox.
    assert bbox[0] < 200


# ─── 3. Detection exclusion zones ───


def test_exclusion_zones_find_title_block_in_bottom_right() -> None:
    blocks = [
        _b("Project Name: Marriott Lobby", 60, 720),
        _b("Sheet Title: First Floor", 60, 740),
        _b("E1.01", 540, 760),
        _b("Drafter: ABC", 540, 740),
    ]
    page_bbox = (0.0, 0.0, 612.0, 792.0)
    zones = detect_exclusion_zones(blocks, page_bbox=page_bbox)
    title_zones = [z for z in zones if z.label == "title_block"]
    assert title_zones
    bbox = title_zones[0].bbox
    assert bbox[1] > 600  # bottom band


def test_exclusion_zones_find_keyed_notes_block() -> None:
    blocks = [
        _b("KEYED NOTES", 72, 200, w=120),
        _b("1. Provide P/N XYZ", 72, 220, w=300),
        _b("2. Coordinate w/ owner", 72, 236, w=300),
        _b("3. Verify height in field", 72, 252, w=300),
    ]
    zones = detect_exclusion_zones(blocks)
    notes = [z for z in zones if z.label == "keyed_notes"]
    assert notes
    # Region should include rows 1-3
    bbox = notes[0].bbox
    assert bbox[1] <= 200 and bbox[3] >= 250


def test_exclusion_zones_find_schedule_block() -> None:
    blocks = [
        _b("DOOR SCHEDULE", 72, 300, w=120),
        _b("DOOR 101 / READER / STRIKE", 72, 320, w=300),
        _b("DOOR 102 / READER / MAG LOCK", 72, 336, w=300),
    ]
    zones = detect_exclusion_zones(blocks)
    schedules = [z for z in zones if z.label == "schedule"]
    assert schedules


def test_ptz_room_in_title_block_does_not_count_as_detection(tmp_path: Path) -> None:
    """A token like "PTZ" inside the title block (a project name
    contains the word PTZ) must not get counted as a PTZ camera
    detection on the drawing body.
    """
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    # Legend page
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Floor plan: title-block at bottom mentions "PTZ" several times,
    # but the drawing body has zero PTZ markers.
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((500, 700), "Project: PTZ Imaging Suite", fontsize=10)
    page.insert_text((500, 720), "Sheet Title: PTZ Floor", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    detections = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    # The body has no real PTZ markers; the title block mentions PTZ
    # twice but those must be excluded.
    assert not detections, [d.value for d in detections]


# ─── 4. Multi-column legend parsing ───


def test_multi_column_legend_extracts_mounting_height_and_cable_count() -> None:
    blocks = [
        # Header row (must be recognized as a header)
        _b("SYMBOL", 50, 100, w=40),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("COUNT", 200, 100, w=40),
        _b("MOUNTING HEIGHT", 260, 100, w=80),
        _b("CABLE COUNT", 360, 100, w=60),
        _b("REMARKS", 440, 100, w=60),
        # Data row 1
        _b("CR", 50, 120, w=20),
        _b("CARD READER", 100, 120, w=80),
        _b("12", 200, 120, w=20),
        _b('48" AFF', 260, 120, w=60),
        _b("1 CAT6", 360, 120, w=60),
        _b("NIC AT REAR", 440, 120, w=80),
        # Data row 2
        _b("WN", 50, 140, w=20),
        _b("WIRELESS NODE", 100, 140, w=80),
        _b("4", 200, 140, w=20),
        _b('CEILING', 260, 140, w=60),
        _b("1 CAT6A", 360, 140, w=60),
        _b("BY OWNER", 440, 140, w=80),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is not None
    by_sym = {e.raw_symbol_text: e for e in legend.entries}
    cr = by_sym["CR"]
    attrs = dict(cr.attributes)
    assert attrs.get("mounting_height") == '48" AFF'
    assert "1 CAT6" in attrs.get("cable_count", "")
    assert "NIC AT REAR" in attrs.get("remarks", "")
    wn = by_sym["WN"]
    wn_attrs = dict(wn.attributes)
    assert wn_attrs.get("mounting_height") == "CEILING"
    assert wn_attrs.get("responsibility") == "BY OWNER" or "BY OWNER" in wn_attrs.get("remarks", "")


# ─── 5. Bounded expand_block_downward ───


def test_expand_block_downward_stops_at_font_size_jump() -> None:
    # Seed is a 12pt header; a large 24pt block sits 10pt below it.
    # Without the height-ratio bound, the legend bbox would swallow
    # the heading.
    blocks = [
        _b("SYMBOL LEGEND", 72, 60, w=120, h=12),
        _b("BIG SECTION HEADING", 72, 82, w=200, h=24),  # 2x bigger
        _b("WN", 72, 120, w=20, h=12),
        _b("Wireless Node", 100, 120, w=120, h=12),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    strong = next(c for c in cands if c.layer == "text_rule_strong")
    # Bbox must NOT include the WN row (it sits below the big heading
    # which the new bound will stop expansion at).
    assert strong.bbox[3] < 120, strong.bbox


# ─── 6. Drawing-index drives resolution ───


def test_drawing_index_promotes_legend_sheet(tmp_path: Path) -> None:
    """A page that doesn't carry an explicit see-sheet reference and
    doesn't share a discipline prefix with any legend should still
    resolve to a legend named in the drawing index.
    """
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    # Page 0: drawing index naming the legend sheet
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "DRAWING INDEX", fontsize=14)
    page.insert_text((72, 90), "T0.01 SYMBOLS & LEGENDS", fontsize=10)
    page.insert_text((72, 110), "M1.01 MECHANICAL", fontsize=10)
    page.insert_text((500, 740), "G0.01", fontsize=10)
    # Page 1: legend sheet T0.01
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SYMBOL LEGEND", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Page 2: mechanical page (no legend, different discipline)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "MECHANICAL PLAN", fontsize=14)
    page.insert_text((500, 740), "M1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    targets = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    pages = {a.value.get("page") for a in targets}
    # Page 2 (M1.01) doesn't share discipline with T0.01 — but the
    # drawing index says T0.01 is the legend, so resolver should still
    # apply it. We expect a target_set for page 2.
    assert 2 in pages, f"M1.01 page didn't get a target set; pages with target_sets={pages}"


# ─── 7. Rotated title-block detection ───


def test_rotated_text_is_classified_as_title_block_furniture() -> None:
    """A block tagged as rotated text (e.g., a border label rotated 90°)
    should be added to the title_block exclusion zone regardless of its
    on-page position.
    """
    blocks = [
        TextBlock(text="PROJECT NUMBER", bbox=(20, 200, 35, 600), rotation_deg=90),
        _b("WN", 200, 300, w=20),  # body — should NOT be excluded
    ]
    zones = detect_exclusion_zones(blocks, page_bbox=(0, 0, 612, 792))
    tb = [z for z in zones if z.label == "title_block"]
    assert tb
    # The exclusion bbox must cover the rotated text region.
    bbox = tb[0].bbox
    assert bbox[0] <= 20 and bbox[2] >= 35


# ─── 8. _interpolate_bbox uses glyph metrics for proportional fonts ───
# (Real glyph metrics are only available when reading from a PDF page,
# so we exercise the fallback by checking that the existing token
# detection paths still work for monospaced synthetic input.)


def test_text_tag_detection_still_emits_when_glyph_metrics_unavailable(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOL LEGEND", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((200, 300), "PTZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    ptz = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    # The single PTZ token in the body must produce exactly one detection.
    assert len(ptz) == 1


# ─── 9. Subtype → parent entity rollup on quantity atoms ───


def test_detected_count_atom_carries_parent_entity_keys(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "3", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    for i in range(3):
        page.insert_text((100 + i * 100, 300), "PTZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    qty = [
        a for a in out.atoms
        if a.atom_type == AtomType.quantity
        and a.value.get("schematic_target_key") == "ptz_camera"
    ]
    assert qty
    # Both detected and declared atoms should carry the parent rollups
    # so a BOM line keyed on device:ip_camera or device:camera can
    # cross-reference the schematic count.
    for atom in qty:
        keys = set(atom.entity_keys)
        assert "device:ptz_camera" in keys
        assert "device:ip_camera" in keys, keys
        assert "device:camera" in keys, keys


# ─── 10. _verify_pdf_block handles block_id + bbox together ───


def test_block_id_plus_bbox_locator_both_verify(tmp_path: Path) -> None:
    """When a locator carries both ``block_id`` and ``bbox+crop_sha256``,
    both paths must verify (the bbox crop hash must match AND the
    block text must contain the atom).
    """
    from app.core.schemas import (
        ArtifactType,
        AtomType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )
    from app.core.source_replay import replay_source_ref

    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Hello World", fontsize=14)
    doc.save(str(pdf))
    doc.close()

    art = stable_id("art", str(pdf))
    src = SourceRef(
        id="sr_dual",
        artifact_id=art,
        artifact_type=ArtifactType.pdf,
        filename=pdf.name,
        locator={
            "page": 0,
            "block_id": "blk_does_not_exist",
            "bbox": [50.0, 50.0, 200.0, 80.0],
            "bbox_units": "pdf_points",
            "crop_sha256": "0" * 64,  # wrong hash
        },
        extraction_method="test",
        parser_version="t",
    )
    atom = EvidenceAtom(
        id="atom_dual",
        project_id="p",
        artifact_id=art,
        atom_type=AtomType.schematic_symbol_detection,
        raw_text="x",
        normalized_text="x",
        value={},
        entity_keys=[],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="t",
    )
    receipt = replay_source_ref(atom, src, {art: pdf})
    # The bbox check fails first (wrong hash). The whole receipt
    # must surface that failure rather than silently passing on
    # the block_id path.
    assert receipt.replay_status == "failed"


# ─── 11. End-to-end realistic stress fixture ───


def _build_realistic_drawing_set(path: Path) -> None:
    """A 4-page drawing set that exercises the schematic upgrade:

    Page 0 — DRAWING INDEX naming T0.01 as SYMBOLS & LEGENDS.
    Page 1 — Legend sheet T0.01 with a 5-column legend
             (symbol / description / count / mounting / remarks).
    Page 2 — Floor plan E1.01: ``See sheet T0.01 for legend.`` plus
             3 PTZ markers in the body, a keyed-notes block, and a
             title-block region at the bottom that contains the
             word "PTZ" in the project name.
    Page 3 — Riser sheet E1.02 with no legend reference; should
             resolve via the drawing index → T0.01.
    """
    doc = fitz.open()
    # Page 0: drawing index
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "DRAWING INDEX", fontsize=14)
    page.insert_text((72, 90), "T0.01 SYMBOLS & LEGENDS", fontsize=10)
    page.insert_text((72, 110), "E1.01 FIRST FLOOR PLAN", fontsize=10)
    page.insert_text((72, 130), "E1.02 RISER DIAGRAM", fontsize=10)
    page.insert_text((520, 740), "G0.01", fontsize=10)

    # Page 1: legend
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((360, 90), "MOUNTING", fontsize=10)
    page.insert_text((460, 90), "REMARKS", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "3", fontsize=10)
    page.insert_text((360, 110), '120" AFF', fontsize=10)
    page.insert_text((460, 110), "MFG NVR-X", fontsize=10)
    page.insert_text((72, 128), "CR", fontsize=10)
    page.insert_text((180, 128), "CARD READER", fontsize=10)
    page.insert_text((300, 128), "5", fontsize=10)
    page.insert_text((360, 128), '48" AFF', fontsize=10)
    page.insert_text((460, 128), "BY OWNER", fontsize=10)
    page.insert_text((520, 740), "T0.01", fontsize=10)

    # Page 2: floor plan
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FIRST FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    # 3 PTZ markers in the body matching the declared count
    for i in range(3):
        page.insert_text((100 + i * 130, 320), "PTZ", fontsize=10)
    # 4 CR markers in the body — doesn't match declared count of 5, so
    # a quantity_conflict packet should fire.
    for i in range(4):
        page.insert_text((100 + i * 110, 420), "CR", fontsize=10)
    # KEYED NOTES block — tokens inside the notes must not count as
    # detections.
    page.insert_text((72, 540), "KEYED NOTES", fontsize=11)
    page.insert_text((72, 558), "1. ALL CR AND PTZ DEVICES TO MEET CODE.", fontsize=9)
    page.insert_text((72, 572), "2. VERIFY MOUNTING HEIGHT IN FIELD.", fontsize=9)
    # Title block at the bottom (contains "PTZ" in the project name).
    page.insert_text((400, 700), "Project: PTZ Imaging Suite", fontsize=9)
    page.insert_text((400, 715), "Drafter: ABC / Checker: DEF", fontsize=9)
    page.insert_text((400, 730), "Scale: 1/8\" = 1'-0\"", fontsize=9)
    page.insert_text((520, 740), "E1.01", fontsize=10)

    # Page 3: riser diagram (no explicit reference; relies on drawing-index)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "RISER DIAGRAM", fontsize=14)
    page.insert_text((520, 740), "E1.02", fontsize=10)
    doc.save(str(path))
    doc.close()


def test_realistic_drawing_set_emits_correct_atoms(tmp_path: Path) -> None:
    pdf = tmp_path / "marriott_style.pdf"
    _build_realistic_drawing_set(pdf)
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)

    # 1. Legend atom captures the 5-column structure.
    legend_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_legend]
    assert legend_atoms
    legend = legend_atoms[0]
    ptz_entry = next(
        e for e in legend.value["entries"] if e["symbol"] == "PTZ"
    )
    assert ptz_entry["count_column"] == 3.0

    # 2. PTZ detections: exactly 3 on the body, none from title block / notes.
    ptz_dets = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    assert len(ptz_dets) == 3, [d.value["bbox"] for d in ptz_dets]

    # 3. The security_camera pack doesn't declare a card_reader target,
    # so the CR row in the legend should produce a legend_orphan-style
    # warning (legend entry with no pack target — surfaces as legend_gap
    # only if the pack expected that target).  We don't assert specific
    # CR counts here; CR exercise belongs in the access_control pack
    # tests.

    # 4. Riser sheet E1.02 still gets a target set via drawing-index path.
    target_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    pages_with_targets = {a.value.get("page") for a in target_atoms}
    assert 3 in pages_with_targets, pages_with_targets

    # 5. Quantity atoms carry parent_entity_keys for the cross-artifact join.
    detected_qty = [
        a for a in out.atoms
        if a.atom_type == AtomType.quantity
        and a.value.get("schematic_target_key") == "ptz_camera"
        and a.value.get("schematic_role") == "detected"
    ]
    assert detected_qty
    for atom in detected_qty:
        assert "device:ip_camera" in atom.entity_keys
        assert "device:camera" in atom.entity_keys
