"""Schematic intelligence layer (PR3+ of the schematic upgrade).

Public surface:

- ``locate_legend_candidates``  — find legend blocks on a single PDF page.
- ``parse_legend``              — turn a candidate block into a ParsedLegend.
- ``LegendResolver``            — document-level legend assignment (PR4).
- ``detect_symbols``            — vector/text symbol detector (PR6).

Everything in this package is deterministic: same input bytes → same
output bytes, no LLM in the hot path, no global state, no time-of-day
dependency.
"""
from __future__ import annotations

from orbitbrief_page_os.segmentation.schematic.debug_overlay import render_overlay
from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    LegendCandidate,
    locate_legend_candidates,
)
from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend
from orbitbrief_page_os.segmentation.schematic.legend_resolver import (
    LegendResolver,
    ResolvedLegend,
    detect_inline_references,
    extract_sheet_number,
    parse_drawing_index,
)
from orbitbrief_page_os.segmentation.schematic.symbol_detector import detect_symbols

__all__ = [
    "LegendCandidate",
    "LegendResolver",
    "ResolvedLegend",
    "detect_inline_references",
    "detect_symbols",
    "extract_sheet_number",
    "locate_legend_candidates",
    "parse_drawing_index",
    "parse_legend",
    "render_overlay",
]
