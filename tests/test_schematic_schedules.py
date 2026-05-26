"""Tests for ``schematic_schedule_row`` parsing and detection joins."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from app.parsers.schematic_models import SymbolDetection
from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock
from orbitbrief_page_os.segmentation.schematic.schedules import (
    detect_schedules,
    join_schedule_rows_to_detections,
)


def _b(text: str, x0: float, y0: float, w: float = 80, h: float = 12) -> TextBlock:
    return TextBlock(text=text, bbox=(x0, y0, x0 + w, y0 + h))


def test_camera_schedule_parser_extracts_tagged_rows() -> None:
    blocks = [
        _b("CAMERA SCHEDULE", 72, 200, w=160),
        # column header
        _b("TAG", 72, 218, w=40),
        _b("MFG", 130, 218, w=60),
        _b("MODEL", 200, 218, w=80),
        _b("MOUNTING", 290, 218, w=80),
        _b("REMARKS", 380, 218, w=80),
        # data rows
        _b("C-101", 72, 234, w=40),
        _b("Axis", 130, 234, w=60),
        _b("P3245-LV", 200, 234, w=80),
        _b('120" AFF', 290, 234, w=80),
        _b("NIC LENS", 380, 234, w=80),
        _b("C-102", 72, 250, w=40),
        _b("Hanwha", 130, 250, w=60),
        _b("QNV-7080R", 200, 250, w=80),
        _b("CEILING", 290, 250, w=80),
        _b("BY OWNER", 380, 250, w=80),
    ]
    rows = detect_schedules(page_index=4, sheet_number="E1.01", blocks=blocks)
    tags = [r.tag for r in rows]
    assert tags == ["C-101", "C-102"]
    row1 = rows[0]
    fields = row1.fields_dict()
    assert fields.get("mfg") == "Axis"
    assert fields.get("model") == "P3245-LV"
    assert "120" in fields.get("mounting", "") or "120" in fields.get("mounting_height", "")
    assert "NIC" in fields.get("remarks", "")


def test_door_schedule_parser_handles_two_tables_on_one_page() -> None:
    blocks = [
        _b("DOOR SCHEDULE", 72, 200, w=160),
        _b("TAG", 72, 218, w=40),
        _b("HARDWARE", 130, 218, w=80),
        _b("REMARKS", 220, 218, w=80),
        _b("D-1", 72, 234, w=40),
        _b("CARD READER", 130, 234, w=80),
        _b("VERIFY VENDOR", 220, 234, w=80),
        _b("D-2", 72, 250, w=40),
        _b("MAG LOCK", 130, 250, w=80),
        _b("NIC", 220, 250, w=80),
        # Second schedule on the same page
        _b("EQUIPMENT SCHEDULE", 72, 400, w=180),
        _b("TAG", 72, 418, w=40),
        _b("MFG", 130, 418, w=80),
        _b("MODEL", 220, 418, w=80),
        _b("EQ-1", 72, 434, w=40),
        _b("Lutron", 130, 434, w=80),
        _b("HW-25", 220, 434, w=80),
    ]
    rows = detect_schedules(page_index=4, sheet_number="E1.01", blocks=blocks)
    by_kind = {r.tag: r for r in rows}
    assert {"D-1", "D-2", "EQ-1"}.issubset(by_kind.keys())
    door = by_kind["D-1"]
    assert door.schedule_kind == "door"
    eq = by_kind["EQ-1"]
    assert eq.schedule_kind == "equipment"


def test_schedule_join_attaches_row_to_detection_by_tag() -> None:
    blocks = [
        _b("CAMERA SCHEDULE", 72, 200, w=160),
        _b("TAG", 72, 218, w=40),
        _b("MFG", 130, 218, w=60),
        _b("C-101", 72, 234, w=40),
        _b("Axis", 130, 234, w=60),
    ]
    rows = detect_schedules(page_index=4, sheet_number="E1.01", blocks=blocks)
    assert rows
    det = SymbolDetection.make(
        page_index=4,
        sheet_number="E1.01",
        target_key="ptz_camera",
        entity_key="device:ptz_camera",
        legend_entry_id=None,
        bbox_pdf=(300, 400, 320, 414),
        crop_sha256="aaa",
        modality="text_tag",
        confidence=0.9,
        nearby_text="C-101 PTZ",  # tag appears near the detection
    )
    mapping = join_schedule_rows_to_detections(rows, [det])
    assert det.detection_id in mapping
    assert mapping[det.detection_id].tag == "C-101"


def test_schedule_row_atoms_emitted_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    # Legend
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    # Floor plan + camera schedule
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    # Schedule block
    page.insert_text((72, 200), "CAMERA SCHEDULE", fontsize=11)
    page.insert_text((72, 218), "TAG", fontsize=10)
    page.insert_text((130, 218), "MFG", fontsize=10)
    page.insert_text((200, 218), "MODEL", fontsize=10)
    page.insert_text((72, 234), "C-101", fontsize=10)
    page.insert_text((130, 234), "Axis", fontsize=10)
    page.insert_text((200, 234), "P3245-LV", fontsize=10)
    page.insert_text((72, 250), "C-102", fontsize=10)
    page.insert_text((130, 250), "Hanwha", fontsize=10)
    page.insert_text((200, 250), "QNV-7080R", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()

    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    schedule_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_schedule_row]
    tags = {a.value.get("tag") for a in schedule_atoms}
    assert {"C-101", "C-102"}.issubset(tags), tags
    c101 = next(a for a in schedule_atoms if a.value["tag"] == "C-101")
    assert c101.value["schedule_kind"] == "camera"
    assert c101.value["fields"].get("mfg") == "Axis"


def test_schedule_atom_has_replayable_locator(tmp_path: Path) -> None:
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
    page.insert_text((72, 200), "DOOR SCHEDULE", fontsize=11)
    page.insert_text((72, 218), "TAG", fontsize=10)
    page.insert_text((130, 218), "HARDWARE", fontsize=10)
    page.insert_text((72, 234), "D-1", fontsize=10)
    page.insert_text((130, 234), "CARD READER", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    out = parser.parse_artifact("p", art, pdf, domain_pack=pack)
    rows = [a for a in out.atoms if a.atom_type == AtomType.schematic_schedule_row]
    assert rows
    loc = rows[0].source_refs[0].locator
    assert loc.get("bbox") and loc.get("bbox_units") == "pdf_points"
    assert loc.get("crop_sha256")
