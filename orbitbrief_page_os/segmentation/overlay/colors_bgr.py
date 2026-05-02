"""OpenCV BGR tuples shared by visible-box overlay passes."""

from __future__ import annotations

# Strong blue — structural wrappers, title washes, ``*_body`` outlines
BLUE = (220, 80, 0)
# Column-header rings (distinct from title blue)
CYAN = (255, 220, 0)
# Title-block logos / embedded raster art
PURPLE = (200, 48, 200)
# Merged sub-header row (e.g. MOTOR DATA)
SUBHDR = (60, 200, 60)
# Contour cells, gap cells, spec rules, mini-table row rings
ORANGE = (0, 150, 255)
# Mini-table data / header cell outlines
GREEN = (60, 180, 60)
GREEN_HDR = (140, 220, 100)
# Reserved for future row-accent use
RED = (40, 40, 220)
# Debug id pill behind wrapper labels (never orange)
LABEL_BG = (55, 55, 55)
