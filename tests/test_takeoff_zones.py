"""Unit tests for app.takeoff.zones — text-only inputs."""
from __future__ import annotations

from app.takeoff.zones import (
    assign_home_run,
    collect_zone_warnings,
    parse_zones,
)


def test_parse_idf2_this_level() -> None:
    text = (
        "Construction notes\n"
        "HOMERUN ALL CABLES ON THIS LEVEL TO IDF-2, THIS LEVEL.\n"
        "Other content."
    )
    zones = parse_zones(text)
    assert len(zones) == 1
    z = zones[0]
    assert z.target == "IDF-2"
    assert z.applies_to_all_levels is True


def test_parse_mdf_lower_lobby() -> None:
    text = (
        "HOMERUN ALL CABLES ON THIS LEVEL TO MDF ROOM, ON THE LOWER LOBBY LEVEL.\n"
    )
    zones = parse_zones(text)
    assert len(zones) == 1
    z = zones[0]
    assert z.target == "MDF ROOM"
    assert z.target_level == "Lower Lobby"
    assert z.applies_to_all_levels is True


def test_parse_multi_level_zones_t106() -> None:
    text = (
        "HOMERUN ALL CABLES ON LEVELS 5 & 6 TO IDF-5, ON LEVEL 5.\n"
        "HOMERUN ALL CABLES ON LEVELS 7, 8 & 9 TO IDF-8, ON LEVEL 8.\n"
        "HOMERUN ALL CABLES ON LEVELS 10 & 11 TO IDF-11, ON LEVEL 11.\n"
        "HOMERUN ALL CABLES ON LEVEL 15 TO IDF-15, ON LEVEL 15.\n"
    )
    zones = parse_zones(text)
    assert len(zones) == 4
    targets = [z.target for z in zones]
    assert targets == ["IDF-5", "IDF-8", "IDF-11", "IDF-15"]
    assert zones[1].levels == ["7", "8", "9"]


def test_missing_zone_warning_for_t106_missing_level_12() -> None:
    text = (
        "HOMERUN ALL CABLES ON LEVELS 5 & 6 TO IDF-5, ON LEVEL 5.\n"
        "HOMERUN ALL CABLES ON LEVELS 7, 8 & 9 TO IDF-8, ON LEVEL 8.\n"
        "HOMERUN ALL CABLES ON LEVELS 10 & 11 TO IDF-11, ON LEVEL 11.\n"
        "HOMERUN ALL CABLES ON LEVEL 15 TO IDF-15, ON LEVEL 15.\n"
    )
    zones = parse_zones(text)
    warnings = collect_zone_warnings(
        sheet_number="T1.06",
        sheet_name="LEVEL 5-12 AND LEVEL 15",
        sheet_levels=["5", "6", "7", "8", "9", "10", "11", "12", "15"],
        zones=zones,
    )
    assert any(
        "missing_homerun_zone_for_levels" in w and "T1.06" in w and "'12'" in w
        for w in warnings
    )


def test_t110_level_10_typo_warning() -> None:
    text = (
        "HOMERUN ALL CABLES ON LEVEL 19 TO IDF-18, ON LEVEL 18.\n"
        "HOMERUN ALL CABLES ON LEVELS 10, 21, 22, & 23 TO IDF-21, ON LEVEL 21.\n"
    )
    zones = parse_zones(text)
    warnings = collect_zone_warnings(
        sheet_number="T1.10",
        sheet_name="LEVEL 19-23 FLOOR PLAN",
        sheet_levels=["19", "20", "21", "22", "23"],
        zones=zones,
    )
    assert any(
        "possible_zone_note_ocr_or_design_typo" in w and "T1.10" in w
        for w in warnings
    )


def test_assign_home_run_single_zone() -> None:
    text = "HOMERUN ALL CABLES ON THIS LEVEL TO IDF-15, ON LEVEL 15."
    zones = parse_zones(text)
    target, level, notes, flags = assign_home_run(
        zones=zones,
        sheet_levels=["14"],
        sheet_floor_label="LEVEL 14",
        device_level="14",
    )
    assert target == "IDF-15"
    assert level == "15"
    assert flags == []


def test_assign_home_run_ambiguous_when_two_zones_no_device_level() -> None:
    text = (
        "HOMERUN ALL CABLES ON THIS LEVEL TO MDF ROOM, THIS LEVEL.\n"
        "HOMERUN ALL CABLES ON THIS LEVEL TO IDF-1, THIS LEVEL.\n"
    )
    zones = parse_zones(text)
    target, level, notes, flags = assign_home_run(
        zones=zones,
        sheet_levels=["Lower Lobby"],
        sheet_floor_label="Lower Lobby",
        device_level=None,
    )
    assert target is None
    assert "ambiguous_homerun_zone" in flags
    assert len(notes) == 2
