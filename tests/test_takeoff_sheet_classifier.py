"""Pure unit tests for app.takeoff.sheet_classifier — no PDF required."""
from __future__ import annotations

from app.takeoff.sheet_classifier import (
    classify_page_type,
    classify_sheet,
    is_sheet_in_scope,
    parse_sheet_number_and_name,
)


def test_t001_classifies_as_legend() -> None:
    text = "SHEET NUMBER: T0.01 - SYMBOLS & LEGENDS\nLEGEND DETAILS..."
    sheet = classify_sheet(1, text)
    assert sheet.sheet_number == "T0.01"
    assert sheet.page_type == "legend"
    assert sheet.in_scope is True


def test_t103_classifies_as_floor_plan() -> None:
    text = "SHEET NUMBER: T1.03 - LEVEL 2 BALLROOM FLOOR PLAN\n..."
    sheet = classify_sheet(6, text)
    assert sheet.sheet_number == "T1.03"
    assert sheet.sheet_name == "LEVEL 2 BALLROOM FLOOR PLAN"
    assert sheet.page_type == "floor_plan"


def test_t701_classifies_as_riser() -> None:
    text = "SHEET NUMBER: T7.01 - CABLING RISER DIAGRAM\n..."
    sheet = classify_sheet(19, text)
    assert sheet.page_type == "riser"


def test_t800_classifies_as_equipment_room() -> None:
    text = "SHEET NUMBER: T8.00 - ENLARGED EQUIPMENT ROOM LAYOUTS\n..."
    sheet = classify_sheet(21, text)
    assert sheet.page_type == "equipment_room"


def test_t902_classifies_as_detail() -> None:
    text = "SHEET NUMBER: T9.02 - INSTALLATION DETAILS\n..."
    sheet = classify_sheet(24, text)
    assert sheet.page_type == "detail"


def test_t100_with_not_in_scope_is_out_of_scope() -> None:
    text = (
        "SHEET NUMBER: T1.00 - SERVICE LEVEL PLAN\n"
        "THIS LEVEL NOT IN SCOPE — see general notes.\n"
    )
    sheet = classify_sheet(3, text)
    assert sheet.sheet_number == "T1.00"
    assert sheet.page_type == "floor_plan"
    assert sheet.in_scope is False
    assert sheet.scope_reason is not None


def test_parse_sheet_number_and_name() -> None:
    n, name = parse_sheet_number_and_name(
        "SHEET NUMBER: T1.10 - LEVEL 19-23 FLOOR PLAN\nMORE TEXT"
    )
    assert n == "T1.10"
    assert name == "LEVEL 19-23 FLOOR PLAN"


def test_classify_page_type_strong_keyword_wins_over_prefix() -> None:
    # Even with a T1.xx prefix, a sheet whose name says "INSTALLATION
    # DETAILS" must classify as detail.
    pt = classify_page_type("T1.05", "INSTALLATION DETAILS", "...")
    assert pt == "detail"


def test_is_sheet_in_scope_default_for_legend() -> None:
    in_scope, reason = is_sheet_in_scope(
        "SYMBOLS & LEGENDS", "legend"
    )
    assert in_scope is True
    assert reason is None
