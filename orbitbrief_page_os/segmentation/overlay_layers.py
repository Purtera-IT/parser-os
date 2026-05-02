"""Bit flags for QA overlay painting and semantic legend ownership.

Detection output is independent of these flags; they select what gets painted.
The first eight flags are v1 behavior.  The higher bits are reserved semantic
slots so adding a concern is mechanical: pass module + layer bit + legend entry
+ golden tests.

Presets kept for compatibility:
    ``OverlayLayer.ALL`` - current full v1 overlay.
    ``OverlayLayer.BLUE_FAMILY`` - structural blue family only.
"""
from __future__ import annotations

from enum import IntFlag


class OverlayLayer(IntFlag):
    # v1 production layers
    BLUE_WRAPPERS = 1 << 0
    BLUE_TITLE = 1 << 1
    BLUE_BODY = 1 << 2
    ORANGE = 1 << 3
    SUBHDR = 1 << 4
    MINI_TABLE = 1 << 5
    CYAN_COLHDR = 1 << 6
    PURPLE_LOGO = 1 << 7

    # v2 semantic slots.  These are intentionally not included in ALL until a
    # pass draws them and a golden test locks them.
    SYMBOL_TAGS = 1 << 8
    ROW_COL_GROUPS = 1 << 9
    REVISION_CALLOUTS = 1 << 10
    LEGEND_BLOCKS = 1 << 11
    CROSS_REFS = 1 << 12
    MULTI_SCALE_BANDS = 1 << 13
    NOTE_CALLOUTS = 1 << 14
    MARGIN_CHROME = 1 << 15

    ALL = (
        BLUE_WRAPPERS
        | BLUE_TITLE
        | BLUE_BODY
        | ORANGE
        | SUBHDR
        | MINI_TABLE
        | CYAN_COLHDR
        | PURPLE_LOGO
    )

    BLUE_FAMILY = BLUE_WRAPPERS | BLUE_TITLE | BLUE_BODY

    V2_RESERVED = (
        SYMBOL_TAGS
        | ROW_COL_GROUPS
        | REVISION_CALLOUTS
        | LEGEND_BLOCKS
        | CROSS_REFS
        | MULTI_SCALE_BANDS
        | NOTE_CALLOUTS
        | MARGIN_CHROME
    )


def parse_layers_arg(name: str) -> OverlayLayer:
    """CLI/config helper preserving the old ``all`` and ``blue`` presets."""
    n = (name or "all").strip().lower()
    if n in ("all", "full", "*"):
        return OverlayLayer.ALL
    if n in ("blue", "blue_family", "blue-only", "blue_only"):
        return OverlayLayer.BLUE_FAMILY
    raise ValueError(f"unknown overlay layer preset: {name!r} (use all, blue)")
