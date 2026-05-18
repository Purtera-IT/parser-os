"""Per-page-type dispatcher for takeoff overlay rendering.

Different sheet types need fundamentally different visual treatments:

* ``floor_plan`` / ``typical_plan`` / ``equipment_room`` — show every
  device candidate as a colored dot + tooltips with home-run zone,
  room label, keynote text. This is the standard takeoff overlay.
* ``legend`` / ``spec`` / ``component_schedule`` — reference-layer
  pages. Run the segmentation pipeline and overlay the detected
  BLUE table containers + ORANGE cells + MAGENTA titles + CYAN
  column headers. Same treatment for all three because they're
  structurally the same kind of tabular reference content — the
  parser's job is to show "I saw the table structure", not to
  count devices.
* ``riser`` / ``detail`` — skip overlay rendering by default. These
  carry diagrammatic content the current overlays can't usefully
  annotate.
* Out-of-scope sheets — skip too, regardless of page_type.

The router is the single source of truth for "what should this page
show?". Anything that renders takeoff overlays calls
:func:`overlay_strategy_for` and dispatches based on the result —
keeping the rule in one place so future page types (e.g. ``riser``
with cable-trace extraction) plug in with one entry.
"""
from __future__ import annotations

from typing import Literal

from app.takeoff.schemas import SheetRecord

OverlayStrategy = Literal[
    "skip",
    "device_takeoff",
    "legend_table_match",
]


# Strategy decision table — keyed by page_type. Order in this dict is
# documentation; the lookup is by exact key. Add new page types here
# rather than spreading dispatch logic across overlay modules.
#
# Reference-style pages (spec / legend / component_schedule) all get
# the same table-aware overlay — they're all tabular content where
# the segmentation pipeline's BLUE / ORANGE / CYAN-header / MAGENTA-title
# decomposition is what the operator actually wants to see. A spec
# page's multi-column prose is structurally just as tabular as a
# legend; a component-schedule's part-number listing is too. Routing
# them all to ``legend_table_match`` keeps the visual treatment
# consistent for "the project's reference layer".
_STRATEGY_BY_PAGE_TYPE: dict[str, OverlayStrategy] = {
    "floor_plan":         "device_takeoff",
    "typical_plan":       "device_takeoff",
    "equipment_room":     "device_takeoff",
    "legend":             "legend_table_match",
    "spec":               "legend_table_match",
    "component_schedule": "legend_table_match",
    "riser":              "skip",
    "detail":             "skip",
    "unknown":            "skip",
}


def overlay_strategy_for(sheet: SheetRecord) -> OverlayStrategy:
    """Return the overlay strategy for ``sheet``.

    Out-of-scope sheets always get ``"skip"`` regardless of their
    ``page_type`` — a "LEVEL NOT IN SCOPE" floor plan shouldn't show
    device dots.
    """
    if sheet.in_scope is False:
        return "skip"
    return _STRATEGY_BY_PAGE_TYPE.get(sheet.page_type, "skip")


def strategy_table() -> list[tuple[str, OverlayStrategy]]:
    """Diagnostic accessor — returns the full (page_type, strategy)
    mapping in stable order. Useful for tests + docs."""
    return list(_STRATEGY_BY_PAGE_TYPE.items())


__all__ = [
    "OverlayStrategy",
    "overlay_strategy_for",
    "strategy_table",
]
