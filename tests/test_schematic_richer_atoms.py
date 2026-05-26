"""Tests for the richer schematic atoms — sheet metadata, rooms,
keyed notes, mounting-height callouts.

Each test exercises one of the new atom types end-to-end through
``OrbitBriefPdfParser.parse_artifact``, so the parser's wiring is
proven alongside the underlying parser module.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from orbitbrief_page_os.segmentation.schematic.callouts import (
    Callout,
    attach_callouts_to_detections,
    detect_callouts,
)
from orbitbrief_page_os.segmentation.schematic.keyed_notes import (
    detect_keyed_notes,
)
from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock
from orbitbrief_page_os.segmentation.schematic.rooms import (
    Room,
    assign_detections_to_rooms,
    detect_rooms,
)
from orbitbrief_page_os.segmentation.schematic.sheet_metadata import (
    parse_sheet_metadata,
)


def _b(text: str, x0: float, y0: float, w: float = 80, h: float = 12) -> TextBlock:
    return TextBlock(text=text, bbox=(x0, y0, x0 + w, y0 + h))


# ─── sheet metadata ───


def test_sheet_metadata_extracts_canonical_fields() -> None:
    blocks = [
        _b("Project Name: Marriott Atlanta Renovation", 50, 700, w=300),
        _b("Sheet Title: First Floor Plan", 50, 720, w=300),
        _b("Scale: 1/8\" = 1'-0\"", 50, 740, w=200),
        _b("Date: 2024-03-15", 50, 760, w=200),
        _b("Rev: 3", 250, 760, w=80),
        _b("Drawn by: ABC", 350, 760, w=120),
        _b("Checked by: DEF", 350, 740, w=120),
        _b("E1.01", 500, 720, w=80),
    ]
    metadata = parse_sheet_metadata(
        page_index=4,
        blocks=blocks,
        sheet_number="E1.01",
        title_block_bbox=(40.0, 690.0, 600.0, 780.0),
    )
    assert metadata is not None
    assert metadata.sheet_number == "E1.01"
    assert metadata.project_name == "Marriott Atlanta Renovation"
    assert metadata.sheet_title == "First Floor Plan"
    assert "1/8" in (metadata.scale or "")
    assert metadata.issue_date == "2024-03-15"
    assert metadata.revision == "3"
    assert metadata.drafter == "ABC"
    assert metadata.checker == "DEF"


def test_sheet_metadata_returns_none_when_no_fields_present() -> None:
    blocks = [_b("just some unrelated body text", 50, 100, w=300)]
    metadata = parse_sheet_metadata(
        page_index=1,
        blocks=blocks,
        sheet_number=None,
        title_block_bbox=None,
    )
    assert metadata is None


def test_sheet_metadata_atom_emitted_per_drawing_page(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((400, 700), "Project: Marriott Renovation", fontsize=9)
    page.insert_text((400, 715), "Sheet Title: First Floor", fontsize=9)
    page.insert_text((400, 730), "Scale: 1/8\" = 1'-0\"", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()

    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    meta_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_sheet_metadata]
    by_page = {a.value["page"]: a for a in meta_atoms}
    assert 1 in by_page, [a.value for a in meta_atoms]
    sheet1 = by_page[1].value
    assert sheet1.get("sheet_number") == "E1.01"
    assert sheet1.get("project_name", "").startswith("Marriott")


# ─── rooms ───


def test_room_detector_finds_name_plus_number() -> None:
    blocks = [
        _b("LOBBY 101", 100, 200, w=80),
        _b("CONFERENCE 204", 200, 300, w=120),
        _b("MDF 1.2", 300, 400, w=80),
    ]
    rooms = detect_rooms(page_index=4, sheet_number="E1.01", blocks=blocks)
    keys = [(r.label, r.number) for r in rooms]
    assert ("LOBBY", "101") in keys
    assert ("CONFERENCE", "204") in keys
    assert ("MDF", "1.2") in keys


def test_room_detector_ignores_unrelated_text() -> None:
    blocks = [
        _b("THIS IS NOT A ROOM", 100, 200, w=200),
        _b("SOME PROSE 12 SENTENCE", 100, 220, w=300),
    ]
    rooms = detect_rooms(page_index=4, sheet_number="E1.01", blocks=blocks)
    assert rooms == []


def test_assign_detections_to_rooms_picks_nearest() -> None:
    from app.parsers.schematic_models import SymbolDetection

    rooms = [
        Room(
            room_id="r1",
            page_index=1,
            sheet_number="E1.01",
            label="LOBBY",
            number="101",
            bbox=(50, 100, 250, 300),
            confidence=0.9,
        ),
        Room(
            room_id="r2",
            page_index=1,
            sheet_number="E1.01",
            label="HALLWAY",
            number="102",
            bbox=(300, 100, 500, 300),
            confidence=0.9,
        ),
    ]
    det_lobby = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="ptz_camera",
        entity_key="device:ptz_camera",
        legend_entry_id=None,
        bbox_pdf=(140, 190, 160, 210),  # center near lobby
        crop_sha256="aaa",
        modality="text_tag",
        confidence=0.9,
    )
    det_hall = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="ptz_camera",
        entity_key="device:ptz_camera",
        legend_entry_id=None,
        bbox_pdf=(390, 190, 410, 210),  # center near hallway
        crop_sha256="bbb",
        modality="text_tag",
        confidence=0.9,
    )
    mapping = assign_detections_to_rooms([det_lobby, det_hall], rooms)
    assert mapping[det_lobby.detection_id] == "r1"
    assert mapping[det_hall.detection_id] == "r2"


def test_detections_carry_room_context_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((100, 300), "LOBBY 101", fontsize=10)
    page.insert_text((120, 320), "PTZ", fontsize=10)  # near lobby
    page.insert_text((400, 500), "CONFERENCE 301", fontsize=10)
    page.insert_text((420, 520), "PTZ", fontsize=10)  # near conference
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()

    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    dets = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    room_ids = [a.value.get("located_in_room_id") for a in dets]
    # Both detections should have a room — and they should NOT be the
    # same room (one near LOBBY 101, one near CONFERENCE 301).
    assert all(r is not None for r in room_ids), [a.value for a in dets]
    assert len(set(room_ids)) == 2, room_ids
    # Room atoms emitted too.
    rooms = [a for a in out.atoms if a.atom_type == AtomType.schematic_room]
    labels = {a.value["label"] for a in rooms}
    assert "LOBBY" in labels and "CONFERENCE" in labels


# ─── keyed notes ───


def test_keyed_notes_parser_extracts_numbered_rows() -> None:
    blocks = [
        _b("KEYED NOTES", 72, 200, w=120),
        _b("1. Provide P/N XYZ-123", 72, 220, w=300),
        _b("2. Coordinate with owner", 72, 236, w=300),
        _b("3. Verify height in field", 72, 252, w=300),
    ]
    notes = detect_keyed_notes(
        page_index=4, sheet_number="E1.01", blocks=blocks
    )
    assert [n.number for n in notes] == ["1", "2", "3"]
    assert notes[0].text.startswith("Provide P/N")


def test_keyed_notes_resolves_body_callouts() -> None:
    blocks = [
        _b("KEYED NOTES", 72, 200, w=120),
        _b("1. Provide P/N XYZ", 72, 220, w=300),
        _b("2. Coordinate with owner", 72, 236, w=300),
        # Body callouts referring to note 1
        _b("1", 300, 400, w=10),
        _b("(1)", 350, 420, w=20),
    ]
    notes = detect_keyed_notes(
        page_index=4, sheet_number="E1.01", blocks=blocks
    )
    note1 = next(n for n in notes if n.number == "1")
    assert len(note1.callout_bboxes) == 2


def test_keyed_note_atoms_emitted_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((72, 500), "KEYED NOTES", fontsize=11)
    page.insert_text((72, 520), "1. Mount at 120 inches AFF", fontsize=9)
    page.insert_text((72, 535), "2. Coordinate with electrician", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    note_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_keyed_note]
    numbers = {a.value["number"] for a in note_atoms}
    assert {"1", "2"}.issubset(numbers), numbers
    note1 = next(a for a in note_atoms if a.value["number"] == "1")
    assert "AFF" in note1.value["text"]


# ─── mounting-height callouts ───


def test_callouts_detector_finds_aff_form() -> None:
    blocks = [
        _b("48\" AFF", 100, 300, w=60),
        _b('120" A.F.F.', 200, 300, w=80),
        _b("CEILING", 300, 300, w=60),
        _b("VERIFY W/ ARCH", 400, 300, w=120),
        _b("Random body prose", 100, 500, w=200),
    ]
    callouts = detect_callouts(blocks)
    texts = [c.text.lower() for c in callouts]
    assert any("aff" in t for t in texts)
    assert any("ceiling" in t for t in texts)
    assert any("verify" in t for t in texts)


def test_callouts_attach_to_nearest_detection() -> None:
    from app.parsers.schematic_models import SymbolDetection

    callouts = [
        Callout(text='48" AFF', bbox=(100, 200, 160, 215)),
        Callout(text="CEILING", bbox=(400, 200, 460, 215)),
    ]
    near_aff = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="cr",
        entity_key="device:cr",
        legend_entry_id=None,
        bbox_pdf=(130, 230, 150, 245),
        crop_sha256="aaa",
        modality="text_tag",
        confidence=0.9,
    )
    near_ceiling = SymbolDetection.make(
        page_index=1,
        sheet_number="E1.01",
        target_key="ptz",
        entity_key="device:ptz",
        legend_entry_id=None,
        bbox_pdf=(430, 230, 450, 245),
        crop_sha256="bbb",
        modality="text_tag",
        confidence=0.9,
    )
    mapping = attach_callouts_to_detections([near_aff, near_ceiling], callouts)
    assert "AFF" in mapping[near_aff.detection_id].text
    assert mapping[near_ceiling.detection_id].text.upper() == "CEILING"


def test_detection_carries_mounting_height_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((200, 300), "PTZ", fontsize=10)
    page.insert_text((230, 310), '120" AFF', fontsize=10)  # near PTZ
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    ptz_dets = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    assert ptz_dets
    has_height = [d for d in ptz_dets if d.value.get("mounting_height")]
    assert has_height, [d.value for d in ptz_dets]
    assert "AFF" in has_height[0].value["mounting_height"]
