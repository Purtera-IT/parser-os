"""Plan-viewport and excluded-region geometry for a drawing sheet.

A construction drawing sheet is laid out with a large "viewport" on the
left containing the actual plan, and a vertical titleblock / legend
band on the right. Symbol tokens that fall inside the titleblock band
are legend keys / sheet metadata, not real devices on the plan.

This module is intentionally simple: no image analysis, no PDF parsing
— it just produces conservative BBoxes based on the page's PDF-point
rectangle.
"""
from __future__ import annotations

from typing import Any

from app.takeoff.schemas import BBox

# Conservative landscape fallback used when no per-sheet template is
# available. The titleblock-side band starts at 84% width by convention.
PLAN_VIEWPORT_FRACTION = (0.0, 0.0, 0.84, 0.94)
TITLEBLOCK_FRACTION = (0.84, 0.0, 1.0, 1.0)


def page_dimensions(page: Any) -> tuple[float, float]:
    """Return ``(width, height)`` in PDF points for a PyMuPDF Page."""
    rect = page.rect
    return (float(rect.width), float(rect.height))


def default_plan_viewport(page: Any) -> BBox:
    """Conservative plan viewport — left 84% × top 94% of the sheet."""
    w, h = page_dimensions(page)
    return BBox(
        x0=PLAN_VIEWPORT_FRACTION[0] * w,
        y0=PLAN_VIEWPORT_FRACTION[1] * h,
        x1=PLAN_VIEWPORT_FRACTION[2] * w,
        y1=PLAN_VIEWPORT_FRACTION[3] * h,
        coord_space="pdf_pt",
    )


def default_excluded_regions(page: Any) -> list[BBox]:
    """Default excluded regions — currently just the right titleblock band."""
    w, h = page_dimensions(page)
    return [
        BBox(
            x0=TITLEBLOCK_FRACTION[0] * w,
            y0=TITLEBLOCK_FRACTION[1] * h,
            x1=TITLEBLOCK_FRACTION[2] * w,
            y1=TITLEBLOCK_FRACTION[3] * h,
            coord_space="pdf_pt",
        ),
    ]


def is_inside(bbox: BBox, region: BBox) -> bool:
    """Return True iff ``bbox`` is fully contained inside ``region``."""
    return (
        bbox.x0 >= region.x0
        and bbox.y0 >= region.y0
        and bbox.x1 <= region.x1
        and bbox.y1 <= region.y1
    )


def overlaps(bbox: BBox, region: BBox) -> bool:
    """Return True iff ``bbox`` and ``region`` overlap at all."""
    return not (
        bbox.x1 <= region.x0
        or bbox.x0 >= region.x1
        or bbox.y1 <= region.y0
        or bbox.y0 >= region.y1
    )


def is_excluded(bbox: BBox, excluded_regions: list[BBox]) -> bool:
    """Return True iff ``bbox`` overlaps any excluded region."""
    return any(overlaps(bbox, r) for r in excluded_regions)


__all__ = [
    "default_plan_viewport",
    "default_excluded_regions",
    "is_inside",
    "overlaps",
    "is_excluded",
    "page_dimensions",
    "PLAN_VIEWPORT_FRACTION",
    "TITLEBLOCK_FRACTION",
]
