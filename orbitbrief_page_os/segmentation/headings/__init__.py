"""Heading signals: PDF spans, generic heuristics, optional VLM hook, per-overlay templates."""

from __future__ import annotations

from .builtin import TSC_FOLLOWON_HEADINGS, TSC_MAJOR_BAND_SECTION_TITLES
from .compose import attach_heading_analysis
from .template import load_overlay_heading_template, resolve_effective_headings

__all__ = [
    "TSC_FOLLOWON_HEADINGS",
    "TSC_MAJOR_BAND_SECTION_TITLES",
    "attach_heading_analysis",
    "load_overlay_heading_template",
    "resolve_effective_headings",
]
