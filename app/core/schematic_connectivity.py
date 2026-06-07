"""Connectivity extractor — turns grounded symbols + drawn lines into the
symbol-wire GRAPH, then reads cable runs and lengths off it.

This is the gap that produced 0 cable runs on the real Marriott DD and blocks
real bidding ("this camera home-runs 140 ft of Cat6 to IDF-2"). The heuristic
cable tracer never traced anything; this builds the graph deterministically and
correctly.

Pipeline:
  1. line segments       — vector: page.get_drawings() line primitives;
                           raster: HoughLinesP fallback (cv2).
  2. junction graph      — snap segment endpoints within a tolerance into shared
                           nodes (so a polyline / corner becomes one connected
                           path); each segment is a weighted edge (its length).
  3. symbol attachment   — attach each symbol to the graph nodes inside/near its
                           bbox (a device "taps" the wires that touch it).
  4. connections         — two symbols are connected iff their attached nodes are
                           in the same component; run length = shortest path
                           between them along the segments (Dijkstra).

Output ``Connection(a, b, length_units, hops)``. Multiply length_units by the
sheet scale (pt->ft) for cable footage. The graph also feeds a GNN later for
higher-order reasoning (home-run rollups, circuit grouping), but the runs +
lengths come straight from this deterministic build.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


@dataclass
class Symbol:
    id: str
    bbox: tuple[float, float, float, float]   # x0,y0,x1,y1 (same units as segments)
    meaning: str = ""


@dataclass
class Connection:
    a: str
    b: str
    length_units: float
    hops: int


# ── line extraction ───────────────────────────────────────────────────────────


def extract_segments_vector(page) -> list[tuple[float, float, float, float]]:
    """Line segments (PDF points) from a vector page's drawing primitives."""
    segs: list[tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return segs
    for d in drawings:
        for item in d.get("items", []):
            if not item:
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:  # line: ('l', p1, p2)
                p1, p2 = item[1], item[2]
                segs.append((float(p1.x), float(p1.y), float(p2.x), float(p2.y)))
            elif op == "re" and len(item) >= 2:  # rect -> 4 edges
                r = item[1]
                segs += [
                    (r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                    (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0),
                ]
    return segs


def extract_segments_raster(page, dpi: int = 150) -> list[tuple[float, float, float, float]]:
    """Line segments (pixels) via Hough transform for scanned/raster pages."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=60,
                            minLineLength=25, maxLineGap=6)
    if lines is None:
        return []
    return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in lines[:, 0, :]]


# ── junction graph + connections ──────────────────────────────────────────────


def _seg_len(s) -> float:
    return math.hypot(s[2] - s[0], s[3] - s[1])


class _Graph:
    def __init__(self):
        self.adj: dict[int, list[tuple[int, float]]] = {}

    def add_edge(self, u: int, v: int, w: float):
        if u == v:
            return
        self.adj.setdefault(u, []).append((v, w))
        self.adj.setdefault(v, []).append((u, w))

    def dijkstra(self, src: int, targets: set[int]) -> dict[int, float]:
        dist = {src: 0.0}
        pq = [(0.0, src)]
        found: dict[int, float] = {}
        remaining = set(targets)
        while pq and remaining:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            if u in remaining:
                found[u] = d
                remaining.discard(u)
            for v, w in self.adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return found


def _snap_key(x: float, y: float, tol: float) -> tuple[int, int]:
    return (int(round(x / tol)), int(round(y / tol)))


def build_connections(symbols: list[Symbol],
                      segments: list[tuple[float, float, float, float]],
                      *, snap_tol: float = 6.0,
                      attach_margin: float = 8.0) -> list[Connection]:
    """Trace symbol-to-symbol connections + lengths through the wire graph."""
    if not symbols or not segments:
        return []
    # node id per snapped endpoint
    node_of: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []

    def node(x, y):
        k = _snap_key(x, y, snap_tol)
        if k not in node_of:
            node_of[k] = len(coords)
            coords.append((x, y))
        return node_of[k]

    g = _Graph()
    for s in segments:
        u = node(s[0], s[1]); v = node(s[2], s[3])
        g.add_edge(u, v, _seg_len(s))

    # attach symbols to nearby nodes (a device taps wires touching its bbox)
    sym_nodes: dict[str, set[int]] = {}
    for sym in symbols:
        x0, y0, x1, y1 = sym.bbox
        x0 -= attach_margin; y0 -= attach_margin; x1 += attach_margin; y1 += attach_margin
        ns = {i for i, (cx, cy) in enumerate(coords) if x0 <= cx <= x1 and y0 <= cy <= y1}
        if ns:
            sym_nodes[sym.id] = ns

    ids = [s.id for s in symbols if s.id in sym_nodes]
    conns: list[Connection] = []
    seen: set[tuple[str, str]] = set()
    for i, aid in enumerate(ids):
        a_nodes = sym_nodes[aid]
        # union of targets from all other symbols
        target_nodes: dict[int, str] = {}
        for bid in ids[i + 1:]:
            for n in sym_nodes[bid]:
                target_nodes.setdefault(n, bid)
        if not target_nodes:
            continue
        best: dict[str, float] = {}
        for src in a_nodes:
            found = g.dijkstra(src, set(target_nodes))
            for n, d in found.items():
                bid = target_nodes[n]
                if d < best.get(bid, math.inf):
                    best[bid] = d
        for bid, d in best.items():
            key = tuple(sorted((aid, bid)))
            if key in seen:
                continue
            seen.add(key)
            conns.append(Connection(a=aid, b=bid, length_units=round(d, 2), hops=1))
    return conns


def extract_connectivity(page, symbols: list[Symbol], *,
                         prefer_vector: bool = True, **kw) -> list[Connection]:
    """Full connectivity for a page: pick vector segments if present, else raster
    Hough, then trace the symbol-wire graph."""
    segs = extract_segments_vector(page) if prefer_vector else []
    if not segs:
        segs = extract_segments_raster(page)
    return build_connections(symbols, segs, **kw)
