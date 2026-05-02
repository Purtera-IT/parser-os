"""Single source of truth for overlay legend entries.

The renderer still consumes concrete BGR tuples, but documentation, tests, and
future passes should discover semantics here.  `legend/generate_legend.py` turns
this registry into `LEGEND.md`, preventing color drift between code and docs.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..overlay_layers import OverlayLayer


@dataclass(frozen=True)
class LegendEntry:
    layer: OverlayLayer
    key: str
    label: str
    status: str
    bgr: tuple[int, int, int] | None
    linestyle: str
    hatch: str | None
    meaning: str
    extraction_rule: str
    tests: tuple[str, ...]
    notes: str = ""


LEGEND_ENTRIES: tuple[LegendEntry, ...] = (
    LegendEntry(
        OverlayLayer.BLUE_TITLE,
        "blue_title_wash",
        "Blue title-band highlight",
        "production",
        (220, 80, 0),
        "20% alpha fill",
        None,
        "Section caption/title strip, not a table cell.",
        "Native PDF text spans with centroid inside *_title or *_sublabel become title/sublabel text.",
        ("golden/test5", "golden/Low_Volatge_Test"),
        "Column-header rows should remain eligible for cyan rings and must not be buried by wash.",
    ),
    LegendEntry(
        OverlayLayer.BLUE_WRAPPERS,
        "blue_structural_wrapper",
        "Blue structural outline",
        "production",
        (220, 80, 0),
        "solid 1px outline",
        None,
        "Outer wrapper/panel/schedule footprint.",
        "Text/span/table ownership starts by nearest containing BLUE wrapper.",
        ("golden/test5", "golden/test7"),
    ),
    LegendEntry(
        OverlayLayer.BLUE_BODY,
        "blue_synthetic_body",
        "Blue synthetic body shell",
        "production",
        (220, 80, 0),
        "solid 1px outline",
        None,
        "Inner table/data hull under a wrapper title band.",
        "Body cells and rows must lie inside *_body unless explicitly margin/chrome.",
        ("golden/test7",),
    ),
    LegendEntry(
        OverlayLayer.ORANGE,
        "orange_cell_outline",
        "Orange cell/row outline",
        "production",
        (0, 150, 255),
        "solid 1px outline or 1px micro/repair fill",
        None,
        "Cellular, row, or schedule data region.",
        "Native PDF words whose centroid falls inside ORANGE and not inside CYAN/PURPLE are table/body text.",
        ("golden/test5", "golden/Low_Volatge_Test", "golden/test7"),
    ),
    LegendEntry(
        OverlayLayer.ORANGE,
        "orange_title_separator",
        "Orange title/content separator",
        "production",
        (0, 150, 255),
        "solid 1px line",
        None,
        "Boundary between caption band and data body in spec-style wrappers.",
        "Used as a split cue; text above belongs to title, text below to content.",
        ("golden/Low_Volatge_Test",),
    ),
    LegendEntry(
        OverlayLayer.CYAN_COLHDR,
        "cyan_column_header_ring",
        "Cyan column-header ring",
        "production",
        (255, 220, 0),
        "solid 1px tight outline",
        None,
        "Per-column header words such as TAG, MFGR, DESCRIPTION.",
        "Words inside CYAN become column keys; associated ORANGE row cells inherit nearest header group.",
        ("golden/test5", "golden/test7"),
    ),
    LegendEntry(
        OverlayLayer.PURPLE_LOGO,
        "purple_titleblock_logo",
        "Purple title-block/logo ring",
        "production",
        (200, 48, 200),
        "solid 2px outline",
        None,
        "Compact embedded raster/vector logos, stamps, seals, and artwork in the title block; never whole text panels.",
        "Exclude compact artwork from schedule extraction; normal title-block text/grid remains orange/blue metadata.",
        ("golden/test5", "golden/Low_Volatge_Test", "golden/test7"),
        "False-purple slabs are filtered by semantic_cleanup.",
    ),
    LegendEntry(
        OverlayLayer.SUBHDR,
        "green_subheader_row",
        "Green sub-header row",
        "production",
        (60, 200, 60),
        "solid 1px outline",
        None,
        "Merged sub-heading row inside a table.",
        "Creates row-group label inherited by following body rows until next sub-header.",
        ("golden/test7",),
    ),
    LegendEntry(
        OverlayLayer.MINI_TABLE,
        "green_minitable_data",
        "Green mini-table data cell",
        "production",
        (60, 180, 60),
        "solid 1px outline",
        None,
        "Tiny margin/vendor/note mini-table data cells.",
        "Extract as mini-table body, not main schedule body.",
        ("golden/Low_Volatge_Test",),
    ),
    LegendEntry(
        OverlayLayer.MINI_TABLE,
        "light_green_minitable_header",
        "Light-green mini-table header",
        "production",
        (140, 220, 100),
        "dashed 1px outline",
        None,
        "Mini-table header cell.",
        "Extract as mini-table column key.",
        ("golden/Low_Volatge_Test",),
    ),
    LegendEntry(
        OverlayLayer.SYMBOL_TAGS,
        "symbol_equipment_tag",
        "Symbol/equipment/tag glyph",
        "reserved",
        (40, 40, 220),
        "solid outline + optional dot hatch",
        "dot",
        "Valves, motors, FA devices, equipment IDs, and non-text tag glyphs.",
        "Glyph bbox links to nearest leader/cell/text tag; never collapse into generic ORANGE.",
        ("golden/symbol_fixture",),
        "Reserved color intentionally differs from orange cells.",
    ),
    LegendEntry(
        OverlayLayer.ROW_COL_GROUPS,
        "row_column_grouping",
        "Row/column belongs-with grouping",
        "reserved",
        (180, 110, 255),
        "thin bracket/connector/spine",
        "stripe",
        "Visual ownership of body rows under header bands and column groups.",
        "A body row inherits every intersecting row spine and nearest column band.",
        ("golden/row_group_fixture", "golden/test7"),
    ),
    LegendEntry(
        OverlayLayer.REVISION_CALLOUTS,
        "revision_cloud_note",
        "Revision/cloud/note callout",
        "reserved",
        (0, 210, 210),
        "dash-dot outline",
        None,
        "Revision clouds, delta bubbles, keynotes, and non-body note callouts.",
        "Classify as callout metadata; keep separate from title block revision grid unless overlapping titleblock chrome.",
        ("golden/revision_fixture",),
    ),
    LegendEntry(
        OverlayLayer.LEGEND_BLOCKS,
        "legend_block",
        "Graphic legend block",
        "reserved",
        (90, 170, 255),
        "double outline",
        "cross",
        "Symbol legend or graphic legend area, distinct from drawing grid and schedules.",
        "Map contained symbol glyphs/text pairs to a legend dictionary.",
        ("golden/legend_fixture",),
    ),
    LegendEntry(
        OverlayLayer.CROSS_REFS,
        "cross_reference_bubble",
        "Cross-reference bubble",
        "reserved",
        (255, 120, 120),
        "rounded outline + leader connector",
        None,
        "Detail/sheet/section reference bubbles.",
        "Parse contained text as target sheet/detail and associate with leader endpoint.",
        ("golden/crossref_fixture",),
    ),
    LegendEntry(
        OverlayLayer.MULTI_SCALE_BANDS,
        "multi_scale_band",
        "Multi-scale bands",
        "reserved",
        (120, 120, 255),
        "transparent band + label",
        "light diagonal",
        "Explicit page scale strata: title strip, data body, margin mini-table, sheet chrome.",
        "Selection rules are scale-aware; small margin mini-tables are not promoted to main schedules.",
        ("golden/test5", "golden/test7"),
    ),
)


def entries_by_status(status: str) -> tuple[LegendEntry, ...]:
    return tuple(e for e in LEGEND_ENTRIES if e.status == status)


__all__ = ["LegendEntry", "LEGEND_ENTRIES", "entries_by_status"]
