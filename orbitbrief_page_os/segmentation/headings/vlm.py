"""Optional VLM / layout-model hook (disabled by default).

Set ``PARSER_OS_VLM_HEADINGS=1`` and implement ``refine_heading_hints`` in a
downstream integration package if you want real model calls here.
"""

from __future__ import annotations

import os
from typing import Any


def maybe_vlm_heading_hints(
    *,
    page_index: int,
    span_lines: list[dict[str, Any]],
    generic_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Placeholder for vision / layout-model refinement of heading candidates."""
    if os.environ.get("PARSER_OS_VLM_HEADINGS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return {
            "status": "disabled",
            "reason": "Set PARSER_OS_VLM_HEADINGS=1 and wire a model in your integration layer.",
        }
    return {
        "status": "not_implemented",
        "reason": (
            "VLM path is intentionally empty in parser-os core; pass page raster or "
            "layout JSON to your service and merge results into heading_analysis."
        ),
        "page_index": page_index,
        "candidates_seen": len(generic_candidates),
        "span_lines_seen": len(span_lines),
    }
