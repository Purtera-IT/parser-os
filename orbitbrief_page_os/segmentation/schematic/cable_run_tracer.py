"""Multi-segment cable run tracer — follow polylines through bends.

``line_runs.py`` extracts straight-line segments and snaps endpoints
to devices. But real cable runs on DD/CD drawings are POLYLINES that
go around walls, take 90° turns, branch at junction boxes. A run like:

    Camera CAM-01 ─┐
                   └──── ACP-3

is a multi-segment path that the straight-line extractor sees as 2-3
separate runs. The PM then doesn't know:

* The TOTAL conduit length needed (sum of all segments)
* Which device pairs are actually connected end-to-end

This module:

1. Pulls vector segments from get_drawings()
2. Builds a vertex adjacency graph (with snapping for slight gaps)
3. For every detection center, finds the shortest path THROUGH the graph
   to every other detection center via BFS/Dijkstra on the segment lengths
4. Emits a typed ``CableRunPath`` per detection-pair with:
   - the full polyline
   - total length
   - count of bends
   - intermediate junction count

Deterministic. No LLM. No I/O.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Sequence


# Segments must be long enough to be a cable run (not annotation tick).
_MIN_SEGMENT_PT = 6.0

# Endpoints within this radius of a vertex collapse into one node.
_VERTEX_SNAP_PT = 4.0

# Detection center within this radius of a graph node is "anchored" to
# that node. Skip detection-to-graph pairs farther than this.
_ANCHOR_RADIUS_PT = 30.0

# Reject paths longer than this — almost certainly noise.
_MAX_PATH_LENGTH_PT = 2000.0


@dataclass(frozen=True)
class CableRunPath:
    """A traced multi-segment path between two detections."""

    page_index: int
    from_detection_id: str
    to_detection_id: str
    polyline: tuple[tuple[float, float], ...]
    total_length_pt: float
    bend_count: int
    bbox_pdf: tuple[float, float, float, float]
    confidence: float = 0.65


# ── Helpers ──────────────────────────────────────────────────────


def _seg_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _extract_segments(drawings: Sequence[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for d in drawings or []:
        items = d.get("items") or []
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
                if _seg_length(p0, p1) >= _MIN_SEGMENT_PT:
                    out.append((p0, p1))
    return out


def _canonicalize_vertices(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    snap: float = _VERTEX_SNAP_PT,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, float]]]:
    """Return (canonical_vertex_list, edges_as_index_pairs_with_weight)."""
    canonical: list[tuple[float, float]] = []
    edges: list[tuple[int, int, float]] = []

    def add(pt: tuple[float, float]) -> int:
        for i, c in enumerate(canonical):
            if _seg_length(pt, c) <= snap:
                return i
        canonical.append(pt)
        return len(canonical) - 1

    for a, b in segments:
        ia = add(a)
        ib = add(b)
        if ia == ib:
            continue
        weight = _seg_length(canonical[ia], canonical[ib])
        edges.append((ia, ib, weight))
    return canonical, edges


def _build_adjacency_indexed(
    n: int,
    edges: Sequence[tuple[int, int, float]],
) -> list[list[tuple[int, float]]]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, w in edges:
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _shortest_path(
    adj: Sequence[Sequence[tuple[int, float]]],
    src: int,
    dst: int,
    *,
    max_length: float = _MAX_PATH_LENGTH_PT,
) -> tuple[float, list[int]] | None:
    """Dijkstra. Returns (total_length, [node indices]) or None if disconnected."""
    n = len(adj)
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[src] = 0.0
    pq: list[tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        if d > max_length:
            return None
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] and nd <= max_length:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] == float("inf") or dist[dst] > max_length:
        return None
    # Reconstruct path
    path: list[int] = []
    u = dst
    while u != -1:
        path.append(u)
        if u == src:
            break
        u = prev[u]
    path.reverse()
    return dist[dst], path


def _anchor_detections(
    detections: Sequence[Any],
    vertices: Sequence[tuple[float, float]],
    *,
    radius: float = _ANCHOR_RADIUS_PT,
) -> dict[str, int]:
    """Map each detection_id → nearest vertex index (within radius)."""
    out: dict[str, int] = {}
    for det in detections:
        bbox = getattr(det, "bbox_pdf", None)
        det_id = getattr(det, "detection_id", None)
        if not bbox or not det_id:
            continue
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        best: tuple[float, int] | None = None
        for i, v in enumerate(vertices):
            d = _seg_length((cx, cy), v)
            if d > radius:
                continue
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            out[det_id] = best[1]
    return out


def _bend_count(path_pts: Sequence[tuple[float, float]]) -> int:
    """Count direction changes > 30 degrees along the polyline."""
    if len(path_pts) < 3:
        return 0
    bends = 0
    for i in range(1, len(path_pts) - 1):
        a, b, c = path_pts[i - 1], path_pts[i], path_pts[i + 1]
        v0 = (b[0] - a[0], b[1] - a[1])
        v1 = (c[0] - b[0], c[1] - b[1])
        m0 = (v0[0] ** 2 + v0[1] ** 2) ** 0.5
        m1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        if m0 == 0 or m1 == 0:
            continue
        dot = (v0[0] * v1[0] + v0[1] * v1[1]) / (m0 * m1)
        dot = max(-1.0, min(1.0, dot))
        import math
        angle = math.degrees(math.acos(dot))
        if angle > 30.0:
            bends += 1
    return bends


# ── Main entry ───────────────────────────────────────────────────


def trace_cable_runs(
    *,
    page: Any,
    page_index: int,
    detections: Sequence[Any],
    max_pairs: int = 200,
) -> list[CableRunPath]:
    """Trace multi-segment paths between every detection pair.

    Returns empty list when no segments or no detections. Output is
    sorted deterministically by (from_id, to_id).
    """
    if page is None or not detections:
        return []
    try:
        drawings = page.get_drawings()
    except Exception:                                 # pragma: no cover
        return []
    if not drawings:
        return []

    segments = _extract_segments(drawings)
    if len(segments) < 2:
        return []

    vertices, edges = _canonicalize_vertices(segments)
    if not vertices or not edges:
        return []

    adj = _build_adjacency_indexed(len(vertices), edges)
    anchors = _anchor_detections(detections, vertices)
    if len(anchors) < 2:
        return []

    paths: list[CableRunPath] = []
    det_ids = sorted(anchors.keys())
    pair_count = 0
    for i in range(len(det_ids)):
        for j in range(i + 1, len(det_ids)):
            if pair_count >= max_pairs:
                break
            src_id = det_ids[i]
            dst_id = det_ids[j]
            src = anchors[src_id]
            dst = anchors[dst_id]
            if src == dst:
                continue
            result = _shortest_path(adj, src, dst)
            if result is None:
                continue
            length, idx_path = result
            poly = [vertices[k] for k in idx_path]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            paths.append(
                CableRunPath(
                    page_index=page_index,
                    from_detection_id=src_id,
                    to_detection_id=dst_id,
                    polyline=tuple(poly),
                    total_length_pt=length,
                    bend_count=_bend_count(poly),
                    bbox_pdf=bbox,
                    confidence=0.7 if len(idx_path) > 2 else 0.55,
                )
            )
            pair_count += 1

    paths.sort(key=lambda p: (p.from_detection_id, p.to_detection_id))
    return paths


__all__ = [
    "CableRunPath",
    "trace_cable_runs",
]
