"""Vector-polygon room detection — close wall lines into rooms.

``rooms.py`` already detects rooms by their TEXT label (LOBBY 101,
CONFERENCE 204, etc.). That works when the text exists but tells us
nothing about the room's actual geometry — which device is in which
room is left to a center-distance heuristic.

This module closes the gap: it traces closed polygons in the wall
layer (the long vector strokes that form room boundaries) and
emits a ``RoomPolygon`` per closed region. Each polygon is then
matched against the text-detected ``Room`` labels via point-in-polygon,
giving us:

* The room's true geometric bounding polygon (not just the label bbox)
* Accurate device-in-room attribution (a detection is inside a room
  iff the room's polygon contains its center)

Deterministic. No LLM. PyMuPDF vector primitives only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# Wall strokes are long. Anything shorter than this is almost certainly
# furniture / annotation / dimension, not a wall.
_MIN_WALL_LENGTH_PT = 36.0

# Endpoints within this distance are treated as the same vertex so a
# slightly broken polygon still closes.
_VERTEX_SNAP_PT = 4.0

# A room polygon must have at least this much area. Tiny enclosed
# polygons are usually small mechanical chases, door swings, or
# annotation marks.
_MIN_ROOM_AREA_SQPT = 1200.0   # ~16in × ~10in

# A "room" larger than this is the entire drawing frame / title block.
_MAX_ROOM_AREA_SQPT = 8.0e5    # ~5ft x 5ft real → way too big to be a room


@dataclass(frozen=True)
class RoomPolygon:
    """A closed polygon traced from wall strokes."""

    page_index: int
    polygon: tuple[tuple[float, float], ...]     # CCW vertex list
    bbox_pdf: tuple[float, float, float, float]
    area_sqpt: float
    matched_room_id: str | None = None           # filled when a text Room sits inside
    matched_label: str | None = None
    confidence: float = 0.65


# ── Geometry helpers ─────────────────────────────────────────────


def _seg_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _snap(pt: tuple[float, float], snap_to: tuple[float, float], tol: float) -> tuple[float, float]:
    if _seg_length(pt, snap_to) <= tol:
        return snap_to
    return pt


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace formula. Always returns the absolute area."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += (x0 * y1) - (x1 * y0)
    return abs(total) / 2.0


def _polygon_bbox(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray casting. Polygon must have >= 3 vertices."""
    n = len(polygon)
    if n < 3:
        return False
    x, y = point
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _extract_segments(drawings: Sequence[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pull out all straight-line ``l`` segments from get_drawings()."""
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for d in drawings or []:
        items = d.get("items") or []
        last_pt: tuple[float, float] | None = None
        for item in items:
            if not isinstance(item, (tuple, list)) or not item:
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:
                try:
                    p0 = (float(item[1][0]), float(item[1][1]))
                    p1 = (float(item[2][0]), float(item[2][1]))
                except (TypeError, ValueError, IndexError):
                    continue
                if _seg_length(p0, p1) >= _MIN_WALL_LENGTH_PT:
                    out.append((p0, p1))
                last_pt = p1
            elif op == "m" and len(item) >= 2:
                try:
                    last_pt = (float(item[1][0]), float(item[1][1]))
                except (TypeError, ValueError, IndexError):
                    last_pt = None
    return out


def _build_adjacency(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    snap: float = _VERTEX_SNAP_PT,
) -> dict[tuple[float, float], list[tuple[float, float]]]:
    """Build a vertex→neighbor adjacency map with vertex-snapping."""
    # Canonical vertex registry: any vertex within `snap` of a previously
    # seen vertex becomes that previously-seen vertex.
    canonical: list[tuple[float, float]] = []

    def canonicalize(pt: tuple[float, float]) -> tuple[float, float]:
        for c in canonical:
            if _seg_length(pt, c) <= snap:
                return c
        canonical.append(pt)
        return pt

    adj: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for a, b in segments:
        ca = canonicalize(a)
        cb = canonicalize(b)
        if ca == cb:
            continue
        adj.setdefault(ca, []).append(cb)
        adj.setdefault(cb, []).append(ca)
    return adj


def _find_cycles(
    adj: dict[tuple[float, float], list[tuple[float, float]]],
    *,
    max_cycle_len: int = 12,
) -> list[list[tuple[float, float]]]:
    """Return all minimal cycles (up to length max_cycle_len) in the
    adjacency graph. Used to find closed room boundaries.

    Naive DFS — fine because each wall segment yields ≤ a few hundred
    vertices on a typical floor plan and `max_cycle_len` is small."""
    cycles_set: set[tuple[tuple[float, float], ...]] = set()

    def dfs(start: tuple[float, float], current: tuple[float, float], path: list[tuple[float, float]], visited: set[tuple[float, float]]) -> None:
        if len(path) > max_cycle_len:
            return
        for nxt in adj.get(current, ()):
            if nxt == start and len(path) >= 3:
                cycle = tuple(path)
                # Canonicalize: rotate so smallest vertex is first
                min_idx = cycle.index(min(cycle))
                rotated = cycle[min_idx:] + cycle[:min_idx]
                if rotated not in cycles_set and tuple(reversed(rotated)) not in cycles_set:
                    cycles_set.add(rotated)
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            dfs(start, nxt, path + [nxt], visited)
            visited.remove(nxt)

    vertices = sorted(adj.keys())
    for v in vertices:
        dfs(v, v, [v], {v})

    return [list(c) for c in cycles_set]


# ── Detection ────────────────────────────────────────────────────


def detect_room_polygons(
    *,
    page: Any,
    page_index: int,
    text_rooms: Sequence[Any] = (),
    max_polygons: int = 200,
) -> list[RoomPolygon]:
    """Trace closed polygons in the wall layer and match to text rooms.

    ``text_rooms`` should be the list of ``Room`` objects from
    ``rooms.detect_rooms``. When supplied, each polygon containing a
    room label's center inherits the room_id / label.

    Returns ``[]`` when the page has no drawings.
    """
    if page is None:
        return []
    try:
        drawings = page.get_drawings()
    except Exception:                              # pragma: no cover
        return []
    if not drawings:
        return []

    segments = _extract_segments(drawings)
    if len(segments) < 3:
        return []

    adj = _build_adjacency(segments)
    if not adj:
        return []

    cycles = _find_cycles(adj)
    polygons: list[RoomPolygon] = []
    for cycle in cycles:
        area = _polygon_area(cycle)
        if area < _MIN_ROOM_AREA_SQPT or area > _MAX_ROOM_AREA_SQPT:
            continue
        bbox = _polygon_bbox(cycle)
        polygons.append(
            RoomPolygon(
                page_index=page_index,
                polygon=tuple(cycle),
                bbox_pdf=bbox,
                area_sqpt=area,
            )
        )

    # Match text-detected rooms by point-in-polygon
    out: list[RoomPolygon] = []
    for poly in polygons:
        matched_room_id: str | None = None
        matched_label: str | None = None
        for rm in text_rooms:
            rm_cx = (rm.bbox[0] + rm.bbox[2]) / 2.0
            rm_cy = (rm.bbox[1] + rm.bbox[3]) / 2.0
            if _point_in_polygon((rm_cx, rm_cy), poly.polygon):
                matched_room_id = getattr(rm, "room_id", None)
                matched_label = getattr(rm, "label", None)
                break
        if matched_room_id is not None:
            out.append(
                RoomPolygon(
                    page_index=poly.page_index,
                    polygon=poly.polygon,
                    bbox_pdf=poly.bbox_pdf,
                    area_sqpt=poly.area_sqpt,
                    matched_room_id=matched_room_id,
                    matched_label=matched_label,
                    confidence=0.85,
                )
            )
        else:
            out.append(poly)

    out.sort(key=lambda p: (round(p.bbox_pdf[1], 2), round(p.bbox_pdf[0], 2)))
    return out[:max_polygons]


def assign_detections_to_polygons(
    detections: Sequence[Any],
    polygons: Sequence[RoomPolygon],
) -> dict[str, str]:
    """Return ``{detection_id: room_id}`` using polygon containment.

    Falls back to nothing (None mapping) for detections that aren't
    inside any polygon — caller should use the center-distance method
    in ``rooms.assign_detections_to_rooms`` as the fallback.
    """
    if not detections or not polygons:
        return {}
    mapping: dict[str, str] = {}
    for det in detections:
        if not getattr(det, "bbox_pdf", None):
            continue
        cx = (det.bbox_pdf[0] + det.bbox_pdf[2]) / 2.0
        cy = (det.bbox_pdf[1] + det.bbox_pdf[3]) / 2.0
        for poly in polygons:
            if poly.matched_room_id is None:
                continue
            if _point_in_polygon((cx, cy), poly.polygon):
                mapping[det.detection_id] = poly.matched_room_id
                break
    return mapping


__all__ = [
    "RoomPolygon",
    "assign_detections_to_polygons",
    "detect_room_polygons",
]
