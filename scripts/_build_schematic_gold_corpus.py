"""Build the synthetic ``SCHEMATIC_*`` gold corpus under real_data_cases.

Generates a small number of representative drawing-set PDFs plus a
matching ``gold_standard.json`` per case. Each PDF exercises a
different mix of the schematic features so a future change can't
silently degrade a category.

Idempotent: re-running rebuilds the PDFs and gold files in place.

Cases:
  SCHEMATIC_LV_FLOORPLAN
      Low-voltage floor plan with a global legend sheet, two device
      types (PTZ + CR), three rooms, two keyed notes, and a camera
      schedule.
  SCHEMATIC_SECURITY_RISER
      Camera + access-control riser with mixed symbols, an
      explicit "see sheet" reference, and a project-global legend.
  SCHEMATIC_FIRE_RISER
      Fire-alarm riser with smoke / heat / pull-station / horn-
      strobe symbols and a panel + circuits.
  SCHEMATIC_ELECTRICAL_ONELINE
      Electrical one-line with panel, transformer, disconnect, and
      receptacles.
  SCHEMATIC_LV_RASTER_SCAN
      Image-only page (no text layer) — exercises the raster
      ocr_unavailable warning path AND the glyph-template detector
      when text is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-not-found]


REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / "real_data_cases"


def _make_legend_sheet(
    doc: Any,
    *,
    sheet_number: str,
    title: str,
    rows: list[tuple[str, str, int]],
    extra_columns: list[tuple[str, list[str]]] | None = None,
) -> None:
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), f"SHEET {sheet_number} - {title}", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    extras = extra_columns or []
    for i, (header, _values) in enumerate(extras):
        page.insert_text((360 + i * 100, 90), header, fontsize=10)
    for r, (symbol, label, count) in enumerate(rows):
        y = 110 + r * 18
        page.insert_text((72, y), symbol, fontsize=10)
        page.insert_text((180, y), label, fontsize=10)
        page.insert_text((300, y), str(count), fontsize=10)
        for i, (_, values) in enumerate(extras):
            if r < len(values):
                page.insert_text((360 + i * 100, y), values[r], fontsize=10)
    page.insert_text((500, 740), sheet_number, fontsize=10)


def build_lv_floorplan(case_dir: Path) -> dict[str, Any]:
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts / "drawings.pdf"
    doc = fitz.open()
    # T0.01 legend
    _make_legend_sheet(
        doc,
        sheet_number="T0.01",
        title="SYMBOLS & LEGENDS",
        rows=[
            ("PTZ", "PTZ CAMERA", 3),
            ("CR", "CARD READER", 2),
        ],
        extra_columns=[
            ("MOUNTING", ['120" AFF', '48" AFF']),
            ("REMARKS", ["NIC LENS", "BY OWNER"]),
        ],
    )
    # E1.01 first floor plan
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FIRST FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    # Rooms
    page.insert_text((100, 250), "LOBBY 101", fontsize=10)
    page.insert_text((250, 250), "HALLWAY 102", fontsize=10)
    page.insert_text((420, 250), "CONFERENCE 301", fontsize=10)
    # PTZ markers (3 of them)
    page.insert_text((110, 280), "PTZ", fontsize=10)
    page.insert_text((260, 280), "PTZ", fontsize=10)
    page.insert_text((430, 280), "PTZ", fontsize=10)
    # CR markers (2 of them, one near each ground-floor door)
    page.insert_text((180, 400), "CR", fontsize=10)
    page.insert_text((380, 400), "CR", fontsize=10)
    # Keyed notes
    page.insert_text((72, 550), "KEYED NOTES", fontsize=11)
    page.insert_text((72, 568), "1. All devices mounted at 120 AFF unless noted.", fontsize=9)
    page.insert_text((72, 582), "2. Coordinate lens with owner before install.", fontsize=9)
    # Camera schedule
    page.insert_text((72, 620), "CAMERA SCHEDULE", fontsize=11)
    page.insert_text((72, 636), "TAG", fontsize=9)
    page.insert_text((130, 636), "MFG", fontsize=9)
    page.insert_text((200, 636), "MODEL", fontsize=9)
    page.insert_text((280, 636), "MOUNTING", fontsize=9)
    page.insert_text((72, 650), "C-101", fontsize=9)
    page.insert_text((130, 650), "Axis", fontsize=9)
    page.insert_text((200, 650), "P3245-LV", fontsize=9)
    page.insert_text((280, 650), '120" AFF', fontsize=9)
    page.insert_text((72, 664), "C-102", fontsize=9)
    page.insert_text((130, 664), "Hanwha", fontsize=9)
    page.insert_text((200, 664), "QNV-7080R", fontsize=9)
    page.insert_text((280, 664), "CEILING", fontsize=9)
    # Title block
    page.insert_text((400, 700), "Project: Marriott Renovation", fontsize=9)
    page.insert_text((400, 715), "Sheet Title: First Floor Plan", fontsize=9)
    page.insert_text((400, 730), "Scale: 1/8\" = 1'-0\"", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return {
        "case_id": "SCHEMATIC_LV_FLOORPLAN",
        "service_line": "security_camera",
        "recommended_domain_pack": "security_camera",
        "bundle_summary": "Synthetic low-voltage floor plan + legend + schedule + keyed notes.",
        "expected_legend_entries_min": 2,
        "expected_detection_targets_include": ["ptz_camera"],
        "expected_symbol_counts": {"ptz_camera": 3},
        "expected_unknown_symbol_count_max": 2,
        "expected_all_detections_have_bbox": True,
        "expected_all_schematic_atoms_carry_locator": True,
    }


def build_security_riser(case_dir: Path) -> dict[str, Any]:
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts / "drawings.pdf"
    doc = fitz.open()
    _make_legend_sheet(
        doc,
        sheet_number="SC0.01",
        title="SYMBOLS & LEGENDS",
        rows=[
            ("PTZ", "PTZ CAMERA", 2),
            ("DOM", "FIXED DOME CAMERA", 4),
            ("CR", "CARD READER", 3),
        ],
    )
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SECURITY RISER DIAGRAM", fontsize=14)
    page.insert_text((72, 100), "See sheet SC0.01 for legend.", fontsize=10)
    # PTZs
    page.insert_text((150, 280), "PTZ", fontsize=10)
    page.insert_text((350, 280), "PTZ", fontsize=10)
    # Domes
    for i, x in enumerate([100, 250, 400, 500]):
        page.insert_text((x, 360), "DOM", fontsize=10)
    # CRs
    for i, x in enumerate([150, 300, 450]):
        page.insert_text((x, 440), "CR", fontsize=10)
    page.insert_text((500, 740), "SC1.01", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return {
        "case_id": "SCHEMATIC_SECURITY_RISER",
        "service_line": "security_camera",
        "recommended_domain_pack": "security_camera",
        "bundle_summary": "Synthetic security riser: PTZ + dome + card reader across one sheet.",
        "expected_legend_entries_min": 3,
        "expected_detection_targets_include": ["ptz_camera"],
        "expected_symbol_counts": {"ptz_camera": 2, "fixed_dome_camera": 4},
        "expected_all_detections_have_bbox": True,
    }


def build_fire_riser(case_dir: Path) -> dict[str, Any]:
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts / "drawings.pdf"
    doc = fitz.open()
    _make_legend_sheet(
        doc,
        sheet_number="FA0.01",
        title="SYMBOLS & LEGENDS",
        rows=[
            ("SD", "SMOKE DETECTOR", 6),
            ("HD", "HEAT DETECTOR", 2),
            ("PS", "PULL STATION", 2),
            ("HS", "HORN STROBE", 3),
        ],
    )
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FIRE ALARM RISER", fontsize=14)
    page.insert_text((72, 100), "See sheet FA0.01 for legend.", fontsize=10)
    for i, x in enumerate([100, 200, 300, 400, 500, 100]):
        y = 250 if i < 5 else 290
        page.insert_text((x, y), "SD", fontsize=10)
    for i, x in enumerate([200, 400]):
        page.insert_text((x, 330), "HD", fontsize=10)
    for i, x in enumerate([150, 450]):
        page.insert_text((x, 420), "PS", fontsize=10)
    for i, x in enumerate([100, 300, 500]):
        page.insert_text((x, 500), "HS", fontsize=10)
    page.insert_text((500, 740), "FA1.01", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return {
        "case_id": "SCHEMATIC_FIRE_RISER",
        "service_line": "fire_safety",
        "recommended_domain_pack": "fire_safety",
        "bundle_summary": "Synthetic fire-alarm riser with smoke / heat / pull / horn-strobe.",
        "expected_legend_entries_min": 4,
        "expected_detection_targets_include": ["smoke_detector", "heat_detector"],
        "expected_all_detections_have_bbox": True,
    }


def build_electrical_oneline(case_dir: Path) -> dict[str, Any]:
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts / "drawings.pdf"
    doc = fitz.open()
    _make_legend_sheet(
        doc,
        sheet_number="E0.01",
        title="SYMBOLS & LEGENDS",
        rows=[
            ("PNL", "PANELBOARD", 2),
            ("XFR", "TRANSFORMER", 1),
            ("DSC", "DISCONNECT", 1),
            ("RCP", "RECEPTACLE", 4),
        ],
    )
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "ELECTRICAL ONE-LINE", fontsize=14)
    page.insert_text((72, 100), "See sheet E0.01 for legend.", fontsize=10)
    page.insert_text((150, 250), "PNL", fontsize=10)
    page.insert_text((400, 250), "PNL", fontsize=10)
    page.insert_text((275, 350), "XFR", fontsize=10)
    page.insert_text((275, 430), "DSC", fontsize=10)
    for i, x in enumerate([100, 250, 400, 500]):
        page.insert_text((x, 530), "RCP", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return {
        "case_id": "SCHEMATIC_ELECTRICAL_ONELINE",
        "service_line": "electrical",
        "recommended_domain_pack": "electrical",
        "bundle_summary": "Synthetic electrical one-line with panel / transformer / disconnect / receptacles.",
        "expected_legend_entries_min": 4,
        "expected_detection_targets_include": ["electrical_panel"],
        "expected_all_detections_have_bbox": True,
    }


def build_raster_scan(case_dir: Path) -> dict[str, Any]:
    """Simulate an image-only drawing: a page with no text layer plus
    a legend sheet on a separate page so the resolver still picks up
    a legend and the raster page produces an ocr_unavailable warning
    (or, when Tesseract is available, OCR'd blocks).
    """
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts / "drawings.pdf"
    doc = fitz.open()
    _make_legend_sheet(
        doc,
        sheet_number="T0.01",
        title="SYMBOLS & LEGENDS",
        rows=[
            ("PTZ", "PTZ CAMERA", 2),
        ],
    )
    page = doc.new_page(width=612, height=792)
    # Pure shape page: rectangles + triangles, no text layer.
    page.draw_rect(fitz.Rect(50, 50, 562, 700), color=(0, 0, 0), width=2)
    for cx, cy in [(150, 300), (450, 300)]:
        page.draw_polyline(
            [(cx - 10, cy + 15), (cx, cy), (cx + 10, cy + 15), (cx - 10, cy + 15)],
            color=(0, 0, 0),
            fill=(0, 0, 0),
        )
    doc.save(str(pdf_path))
    doc.close()
    return {
        "case_id": "SCHEMATIC_LV_RASTER_SCAN",
        "service_line": "security_camera",
        "recommended_domain_pack": "security_camera",
        "bundle_summary": "Synthetic raster-only drawing: legend on text page, image-only floor plan.",
        "expected_legend_entries_min": 1,
        "expected_all_schematic_atoms_carry_locator": True,
    }


CASES = {
    "SCHEMATIC_LV_FLOORPLAN": build_lv_floorplan,
    "SCHEMATIC_SECURITY_RISER": build_security_riser,
    "SCHEMATIC_FIRE_RISER": build_fire_riser,
    "SCHEMATIC_ELECTRICAL_ONELINE": build_electrical_oneline,
    "SCHEMATIC_LV_RASTER_SCAN": build_raster_scan,
}


def main() -> int:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for case_id, builder in CASES.items():
        case_dir = CASES_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = case_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        gold = builder(case_dir)
        gold_path = labels_dir / "gold_standard.json"
        gold_path.write_text(
            json.dumps(gold, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  WROTE {case_id}")
    print(f"\ntotal cases: {len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
