"""Region proposal generator for vision-based symbol detection.

The vision LLM (qwen2.5-vl) costs ~2s per region. A naive whole-page
scan with 200×200 px tiles on a 4536×3240 page = ~350 tiles per page
= 12+ minutes per page. Unusable.

Region proposals cut that to 20-50 candidate regions per page by
clustering vector strokes near "symbol-shaped" geometry. Pipeline:

  1. ``page.get_drawings()`` → all vector primitives
  2. Filter primitives that look "symbol-sized" (icon glyphs are
     typically 8-40 pt wide / tall — too small = decorative, too
     big = wall/door)
  3. Cluster adjacent primitives by proximity
  4. Reject clusters that fit a known "not-a-symbol" shape
     (long horizontal lines = grid; long vertical lines = walls;
     huge rectangles = title block / drawing frame)
  5. Emit one ``RegionProposal`` per surviving cluster, with the
     bbox inflated by a small margin

The output goes to the vision detector, which runs the LLM only on
those crops.

Deterministic. No LLM. No I/O. Same page → same proposals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

# Tuned defaults — Marriott DD pages are 3024×2160 PDF points.
# Icon glyphs there are roughly 8-30 pt wide. Bumped the upper bound
# slightly for callout balloons + room labels.
SYMBOL_MIN_PT = 6.0
SYMBOL_MAX_PT = 60.0

# Two primitives within this distance get clustered into one region.
CLUSTER_DISTANCE_PT = 12.0

# A region with an aspect ratio outside this band is rejected
# (probably a line / wall / dimension, not a symbol).
MIN_ASPECT = 0.20
MAX_ASPECT = 5.0

# Reject huge regions — likely title blocks, drawing frames, north arrows.
MAX_REGION_PT = 120.0


@dataclass(frozen=True)
class RegionProposal:
    """One candidate region to send to the vision detector."""

    page_index: int
    bbox_pdf: tuple[float, float, float, float]      # in PDF points
    primitive_count: int                              # how many strokes contributed
    reason: str                                       # for debugging — "stroke_cluster" / "text_proximity" / ...


def _bbox_size(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _bbox_union(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Minimum edge-to-edge distance between two bboxes; 0 if overlap."""
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return (dx * dx + dy * dy) ** 0.5


def _is_symbol_sized(bbox: tuple[float, float, float, float]) -> bool:
    w, h = _bbox_size(bbox)
    if w < SYMBOL_MIN_PT or h < SYMBOL_MIN_PT:
        return False
    if w > MAX_REGION_PT or h > MAX_REGION_PT:
        return False
    longest = max(w, h)
    if longest > SYMBOL_MAX_PT * 2:
        return False
    return True


def _passes_aspect_filter(bbox: tuple[float, float, float, float]) -> bool:
    w, h = _bbox_size(bbox)
    if h <= 0:
        return False
    aspect = w / h
    return MIN_ASPECT <= aspect <= MAX_ASPECT


def _drawing_bbox(d: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Extract a bbox from a PyMuPDF ``get_drawings()`` entry."""
    rect = d.get("rect")
    if rect is None:
        return None
    try:
        return (
            float(rect.x0),
            float(rect.y0),
            float(rect.x1),
            float(rect.y1),
        )
    except Exception:                                 # pragma: no cover
        return None


def _cluster_primitives(
    bboxes: Sequence[tuple[float, float, float, float]],
    *,
    distance: float = CLUSTER_DISTANCE_PT,
) -> list[tuple[tuple[float, float, float, float], int]]:
    """Union-find clustering by proximity.

    Returns (bbox, primitive_count) per cluster.
    """
    n = len(bboxes)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # O(n²) but n is bounded (we filtered to symbol-sized before clustering)
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_distance(bboxes[i], bboxes[j]) <= distance:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    out: list[tuple[tuple[float, float, float, float], int]] = []
    for indices in clusters.values():
        combined = bboxes[indices[0]]
        for idx in indices[1:]:
            combined = _bbox_union(combined, bboxes[idx])
        out.append((combined, len(indices)))
    return out


def propose_regions(
    *,
    page: Any,
    page_index: int,
    max_proposals: int = 100,
) -> list[RegionProposal]:
    """Return a list of candidate regions worth sending to the vision LLM.

    Empty list when the page has no vector strokes (raster-only page —
    a different OCR fallback path handles those).
    """
    try:
        drawings = page.get_drawings()
    except Exception:                                 # pragma: no cover
        return []
    if not drawings:
        return []

    # Step 1: filter to symbol-sized + aspect-ratio'd primitives
    primitives: list[tuple[float, float, float, float]] = []
    for d in drawings:
        bbox = _drawing_bbox(d)
        if bbox is None:
            continue
        if not _is_symbol_sized(bbox):
            continue
        if not _passes_aspect_filter(bbox):
            continue
        primitives.append(bbox)

    if not primitives:
        return []

    # Step 2: cluster nearby primitives
    clusters = _cluster_primitives(primitives)

    # Step 3: emit proposals, sorted by (top, left) so the LLM
    # sweeps the page in a deterministic order
    proposals: list[RegionProposal] = []
    for bbox, count in clusters:
        if not _is_symbol_sized(bbox) or not _passes_aspect_filter(bbox):
            continue
        proposals.append(
            RegionProposal(
                page_index=page_index,
                bbox_pdf=bbox,
                primitive_count=count,
                reason="stroke_cluster",
            )
        )

    proposals.sort(key=lambda p: (p.bbox_pdf[1], p.bbox_pdf[0]))
    return proposals[:max_proposals]


__all__ = [
    "CLUSTER_DISTANCE_PT",
    "MAX_REGION_PT",
    "RegionProposal",
    "SYMBOL_MAX_PT",
    "SYMBOL_MIN_PT",
    "propose_regions",
]
