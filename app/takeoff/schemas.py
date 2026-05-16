"""Pydantic schemas for the low-voltage takeoff pipeline.

Schema version: ``purtera.lowvoltage.takeoff.v1``.

These schemas are the public contract of the takeoff layer. Adding new
fields is fine; renaming or repurposing existing fields is a breaking
change.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "purtera.lowvoltage.takeoff.v1"

# ─────────────────────────── Bounding box ─────────────────────────────


class BBox(BaseModel):
    """A bounding box in either PDF points or image pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    coord_space: Literal["pdf_pt", "image_px"] = "pdf_pt"

    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


# ─────────────────────────── Sheet record ─────────────────────────────


SheetPageType = Literal[
    "spec",
    "legend",
    "component_schedule",
    "floor_plan",
    "typical_plan",
    "riser",
    "equipment_room",
    "detail",
    "unknown",
]


class SheetRecord(BaseModel):
    """Per-page classification + scope + plan viewport."""

    page_index: int
    sheet_number: str | None = None
    sheet_name: str | None = None
    page_type: SheetPageType = "unknown"
    floor_label: str | None = None
    levels_represented: list[str] = Field(default_factory=list)
    multiplier: int = 1
    in_scope: bool = True
    scope_reason: str | None = None
    plan_viewport: BBox | None = None
    excluded_regions: list[BBox] = Field(default_factory=list)


# ─────────────────────────── Legend rule ──────────────────────────────


class LegendRule(BaseModel):
    """Project-level legend mapping a raw symbol to a quote unit family."""

    raw_symbol: str
    normalized_class: str
    system: str
    description: str | None = None
    cable_count: int | None = None
    cable_type: str | None = None
    work_area_termination: str | None = None
    closet_termination: str | None = None
    mounting: str | None = None
    rough_in: str | None = None
    power: str | None = None
    remarks: list[str] = Field(default_factory=list)
    quote_unit: str | None = None
    source_page: int | None = None
    source_bbox: BBox | None = None
    confidence: float = 0.9


# ───────────────────────── Symbol candidate ───────────────────────────


class SymbolCandidate(BaseModel):
    """A single token-shaped symbol detection on a page.

    Candidates are kept even when rejected so reviewers can see why a
    given WN-on-the-legend or out-of-viewport hit didn't become a device.
    """

    id: str
    page_index: int
    raw_symbol: str
    normalized_class: str | None = None
    bbox: BBox
    source_methods: list[str] = Field(default_factory=list)
    confidence: float = 0.94
    rejection_reason: str | None = None
    needs_review: bool = False
    nearby_text: list[str] = Field(default_factory=list)


# ───────────────────────── Device instance ────────────────────────────


class DeviceInstance(BaseModel):
    """An accepted device instance, fused from one or more candidates."""

    id: str
    page_index: int
    sheet_number: str | None = None
    sheet_name: str | None = None
    raw_symbol: str
    normalized_class: str
    system: str | None = None
    bbox: BBox
    floor_label: str | None = None
    levels_represented: list[str] = Field(default_factory=list)
    multiplier: int = 1
    room_guess: str | None = None
    keynote: str | None = None
    keynote_text: str | None = None
    home_run_to: str | None = None
    home_run_level: str | None = None
    zone_notes: list[str] = Field(default_factory=list)
    legend_rule_id: str | None = None
    confidence: float = 0.94
    review_flags: list[str] = Field(default_factory=list)


# ─────────────────────────── Quote line ───────────────────────────────


class QuoteLine(BaseModel):
    """A roll-up quote line (no dollar values — quantity only)."""

    item_key: str
    description: str
    quantity: float
    unit: str
    system: str | None = None
    floor_label: str | None = None
    home_run_to: str | None = None
    source_device_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.9
    notes: list[str] = Field(default_factory=list)


# ────────────────────────── Takeoff document ──────────────────────────


class TakeoffDocument(BaseModel):
    """The full takeoff document for a single PDF source."""

    schema_version: str = SCHEMA_VERSION
    source_pdf: str
    sheets: list[SheetRecord] = Field(default_factory=list)
    legend_rules: list[LegendRule] = Field(default_factory=list)
    candidates: list[SymbolCandidate] = Field(default_factory=list)
    devices: list[DeviceInstance] = Field(default_factory=list)
    quote_lines: list[QuoteLine] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "SCHEMA_VERSION",
    "BBox",
    "SheetRecord",
    "SheetPageType",
    "LegendRule",
    "SymbolCandidate",
    "DeviceInstance",
    "QuoteLine",
    "TakeoffDocument",
]
