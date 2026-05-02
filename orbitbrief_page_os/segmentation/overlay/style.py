"""Central overlay drawing style.

Every visible hue/line-width decision should resolve through this module, not a
hard-coded thickness buried in a detector pass.  New semantic layers add a
legend entry and a style field here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverlayStyle:
    blue_wrapper_px: int = 1
    # Universal outset for BLUE wrapper rectangles, in pixels.  Drawing the
    # wrapper exactly on the box bbox makes the BLUE line coincide with
    # whatever ORANGE inner cell hugs the same bbox — and since ORANGE
    # draws AFTER BLUE in the core pass order, those overlapping pixels
    # become orange, leaving the wrapper invisible.  A small geometric
    # outset (drawing the BLUE rectangle a couple of px outside the bbox)
    # gives the wrapper visible breathing room outside any inner cell that
    # uses the original bbox.  Applied uniformly across all BLUE wrappers
    # so the legend rule "BLUE = outer structural frame, ORANGE = inner
    # cell" reads consistently.
    blue_wrapper_outset_px: int = 2
    blue_body_px: int = 1
    orange_cell_px: int = 1
    orange_separator_px: int = 1
    orange_micro_px: int = 1
    cyan_colhdr_px: int = 1
    purple_logo_px: int = 2
    subhdr_px: int = 1
    green_cell_px: int = 1
    title_alpha: float = 0.20
    sublabel_alpha: float = 0.34
    dashed_dash_px: int = 4
    dashed_gap_px: int = 3
    label_font_scale: float = 0.34
    label_thickness: int = 1


DEFAULT_STYLE = OverlayStyle()
