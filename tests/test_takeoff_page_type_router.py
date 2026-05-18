"""Tests for :mod:`app.takeoff.page_type_router`."""
from __future__ import annotations

from app.takeoff.page_type_router import overlay_strategy_for, strategy_table
from app.takeoff.schemas import SheetRecord


def _sheet(page_type: str, in_scope: bool = True) -> SheetRecord:
    return SheetRecord(page_index=0, page_type=page_type, in_scope=in_scope)  # type: ignore[arg-type]


def test_device_overlay_for_floor_plan() -> None:
    assert overlay_strategy_for(_sheet("floor_plan")) == "device_takeoff"


def test_device_overlay_for_typical_plan() -> None:
    assert overlay_strategy_for(_sheet("typical_plan")) == "device_takeoff"


def test_device_overlay_for_equipment_room() -> None:
    assert overlay_strategy_for(_sheet("equipment_room")) == "device_takeoff"


def test_legend_overlay_for_legend_page() -> None:
    assert overlay_strategy_for(_sheet("legend")) == "legend_table_match"


def test_legend_table_match_for_reference_pages() -> None:
    """spec / legend / component_schedule are all tabular reference pages;
    they should share the same table-aware overlay treatment."""
    for page_type in ("spec", "legend", "component_schedule"):
        assert overlay_strategy_for(_sheet(page_type)) == "legend_table_match", page_type


def test_skip_for_riser_detail_unknown() -> None:
    for page_type in ("riser", "detail", "unknown"):
        assert overlay_strategy_for(_sheet(page_type)) == "skip", page_type


def test_out_of_scope_always_skips_regardless_of_page_type() -> None:
    """A floor_plan with in_scope=False (NOT IN SCOPE text) must skip,
    not get a device_takeoff overlay drawn over it."""
    assert overlay_strategy_for(_sheet("floor_plan", in_scope=False)) == "skip"
    assert overlay_strategy_for(_sheet("legend", in_scope=False)) == "skip"


def test_strategy_table_is_stable() -> None:
    table = strategy_table()
    assert ("floor_plan", "device_takeoff") in table
    assert ("legend", "legend_table_match") in table
    assert ("spec", "legend_table_match") in table
    assert ("component_schedule", "legend_table_match") in table
    assert ("riser", "skip") in table
    assert ("detail", "skip") in table
