"""Universality regressions — non-Marriott patterns the parser must handle.

Each test pretends to be a different architect's drawing convention. The
core parser modules should produce the same kind of output regardless of
which prefix / phrasing the project chose. Marriott-specific behavior is
covered by ``test_takeoff_*`` modules elsewhere.
"""
from __future__ import annotations

from app.takeoff.multipliers import levels_from_title, multiplier_for_title
from app.takeoff.sheet_classifier import classify_page_type, classify_sheet
from app.takeoff.zones import parse_zones


# ───────────────────────── Sheet classifier ─────────────────────────


def test_classifier_handles_lv_prefix() -> None:
    """LV-set numbering (some low-voltage consultants) — LV1.05 = floor_plan."""
    assert classify_page_type("LV1.05", "LEVEL 5 FLOOR PLAN", "") == "floor_plan"
    assert classify_page_type("LV0.01", "SYMBOLS & LEGENDS", "") == "legend"
    assert classify_page_type("LV9.02", "INSTALLATION DETAILS", "") == "detail"


def test_classifier_handles_e_prefix() -> None:
    """Electrical drawing-set numbering — E1.0 / E2.0 are plan series."""
    assert classify_page_type("E1.0", "POWER PLAN - LEVEL 1", "") == "floor_plan"
    assert classify_page_type("E7.0", "RISER DIAGRAM", "") == "riser"


def test_classifier_handles_it_or_tc_prefix() -> None:
    """IT / TC (technology / telecom) prefixes used by Newcomb & Boyd, CMTA, ..."""
    assert classify_page_type("IT1.05", "FLOOR PLAN", "") == "floor_plan"
    assert classify_page_type("TC8.00", "EQUIPMENT ROOM", "") == "equipment_room"


def test_classifier_keyword_still_wins_over_prefix() -> None:
    """A 1.xx sheet titled SYMBOLS & LEGENDS classifies as legend, not floor_plan."""
    assert classify_page_type("E1.99", "SYMBOLS & LEGENDS", "") == "legend"


def test_classify_sheet_end_to_end_lv_set() -> None:
    text = "SHEET NUMBER: LV1.03 - LEVEL 2 FLOOR PLAN\n  ..."
    sheet = classify_sheet(page_index=0, page_text=text)
    assert sheet.sheet_number == "LV1.03"
    assert sheet.page_type == "floor_plan"


# ─────────────────────────── Multipliers ───────────────────────────


def test_multiplier_floors_synonym() -> None:
    """FLOORS 5-12 == LEVELS 5-12 == 8 floors."""
    levels, mult = multiplier_for_title("FLOORS 5-12 FLOOR PLAN")
    assert levels == ["5", "6", "7", "8", "9", "10", "11", "12"], levels
    assert mult == 8


def test_multiplier_slash_list() -> None:
    """LEVELS 5/8/12 → 3 specific floors."""
    levels = levels_from_title("LEVELS 5/8/12 FLOOR PLAN")
    assert levels == ["5", "8", "12"], levels


def test_multiplier_floor_through_floor() -> None:
    """LEVELS 5 THROUGH 12 → range 5..12."""
    levels = levels_from_title("LEVELS 5 THROUGH 12 FLOOR PLAN")
    assert levels[0] == "5" and levels[-1] == "12" and len(levels) == 8


def test_multiplier_named_mezzanine() -> None:
    """Generic named floors beyond hospitality (MEZZANINE, BASEMENT, ...)."""
    assert levels_from_title("MEZZANINE FLOOR PLAN") == ["Mezzanine"]
    assert levels_from_title("BASEMENT PLAN") == ["Basement"]
    assert levels_from_title("PENTHOUSE LEVEL FLOOR PLAN") == ["Penthouse"]
    assert levels_from_title("GROUND LEVEL FLOOR PLAN") == ["Ground"]


def test_multiplier_marriott_invariant_unchanged() -> None:
    """The Marriott regressions stay green."""
    assert multiplier_for_title("LEVEL 5-12 AND LEVEL 15 FLOOR PLAN")[1] == 9
    assert multiplier_for_title("LEVEL 17-18 FLOOR PLAN")[1] == 2
    assert multiplier_for_title("LEVEL 19-23 FLOOR PLAN")[1] == 5


# ────────────────────────── Zone parsing ───────────────────────────


def test_zone_run_all_cables_to_tr() -> None:
    """NTI / non-Marriott: RUN ALL CABLES to a TR room."""
    text = "RUN ALL CABLES ON THIS LEVEL TO TR-3, ON LEVEL 3."
    zones = parse_zones(text)
    assert len(zones) >= 1
    assert zones[0].target == "TR-3"


def test_zone_cables_back_to_mdf() -> None:
    text = "ALL CABLES BACK TO MDF ROOM, ON THE LOWER LOBBY LEVEL."
    zones = parse_zones(text)
    assert len(zones) >= 1
    assert zones[0].target == "MDF ROOM"


def test_zone_route_cables_to_idf_letter() -> None:
    """IDF with letter suffix (some firms use letters, not numbers)."""
    text = "ROUTE CABLES TO IDF-A, ON THIS LEVEL."
    zones = parse_zones(text)
    assert len(zones) >= 1
    assert zones[0].target == "IDF-A"


def test_zone_marriott_homerun_unchanged() -> None:
    text = "HOMERUN ALL CABLES ON THIS LEVEL TO IDF-2, THIS LEVEL."
    zones = parse_zones(text)
    assert len(zones) >= 1
    assert zones[0].target == "IDF-2"


def test_zone_multiple_phrasings_on_one_page() -> None:
    """A page that mixes HOMERUN + RUN phrasings — both should be captured."""
    text = (
        "HOMERUN ALL CABLES ON THIS LEVEL TO MDF ROOM, THIS LEVEL. "
        "RUN ALL CABLES ON LEVELS 5 & 6 TO IDF-5, ON LEVEL 5."
    )
    zones = parse_zones(text)
    targets = sorted(z.target for z in zones if z.target)
    assert "MDF ROOM" in targets
    assert "IDF-5" in targets
