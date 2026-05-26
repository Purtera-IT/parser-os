"""Detect line runs on construction drawings and snap them to devices.

Conduit / cable / riser / homerun trails are vector polylines connecting
devices to panels (e.g. ``card reader CR-101 → ACP-1`` on an access
control drawing).  Without these the parser sees a count of devices
but not the topology that ties them together.

This MVP extracts straight-line segments and short polylines from the
page's drawing primitives, filters them by length and by intersection
with excluded zones, then snaps each endpoint to the nearest
``SymbolDetection`` within a small tolerance.  Endpoints with no
nearby detection are kept on the atom (so a panel implied by the
endpoint but not yet detected can still be inferred downstream).

Deterministic by construction: PyMuPDF's ``get_drawings`` returns
primitives in document order; we sort the result by rounded endpoint
coordinates so the atom IDs are byte-stable across runs.

No LLM, no network. Falls back gracefully when PyMuPDF isn't
available or returns no primitives.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.core.ids import stable_id


_MIN_RUN_LENGTH_PT = 18.0   # 1/4 inch — shorter is decoration, not a run
_MAX_RUN_LENGTH_PT = 720.0  # 10 inches — longer is usually a border / grid line
# Cap per-page run count. On a busy floor plan with no detection
# anchors, every wall segment looks like a candidate. Without a cap,
# a single sheet can emit tens of thousands of atoms and dominate
# the compile output. ``detect_line_runs`` sorts by (top, left), so
# the cap preserves a deterministic prefix.
_MAX_RUNS_PER_PAGE = 200
# Construction drawings often stop a cable run 1/4–1/2 inch short of
# the symbol so the line doesn't visually overlap the device glyph.
# 36 pt = 1/2 inch is a reasonable compromise between catching real
# device-terminating runs and rejecting noise.
_SNAP_TOLERANCE_PT = 36.0


@dataclass(frozen=True)
class LineRun:
    """A polyline detected on a drawing page.

    ``polyline`` is the deterministic list of ``(x, y)`` vertices in
    PDF points; the first and last entries are the run's endpoints.
    ``length_pt`` is the sum of segment lengths.

    ``from_detection_id`` / ``to_detection_id`` are filled in when an
    endpoint snaps within ``_SNAP_TOLERANCE_PT`` of a detection's
    center.  When an endpoint floats free (e.g. it terminates at an
    un-detected panel), the corresponding field stays ``None``.
    """

    line_run_id: str
    page_index: int
    sheet_number: str | None
    polyline: tuple[tuple[float, float], ...]
    length_pt: float
    bbox_pdf: tuple[float, float, float, float]
    from_detection_id: str | None = None
    to_detection_id: str | None = None
    confidence: float = 0.7

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.polyline[0], self.polyline[-1]


def _segment_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        _segment_length(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )


def _polyline_bbox(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return the polyline's bounding rect, padded slightly so a purely
    horizontal or vertical line still has nonzero area (otherwise the
    replayable-locator helper refuses the bbox).
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0, x1, y1 = (min(xs), min(ys), max(xs), max(ys))
    if x1 - x0 < 1.0:
        x0 -= 1.0
        x1 += 1.0
    if y1 - y0 < 1.0:
        y0 -= 1.0
        y1 += 1.0
    return (x0, y0, x1, y1)


def _bbox_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _drawing_items_to_polylines(
    drawings: Sequence[dict],
) -> list[list[tuple[float, float]]]:
    """Reduce PyMuPDF's drawing items to deterministic polyline lists.

    PyMuPDF returns each drawing as ``{"items": [...]}``.  We accept
    only stroke-style straight lines (``"l"`` items) and bezier
    rectangles (``"re"`` items become 4-segment polylines along the
    rectangle perimeter is intentionally rejected — those are usually
    legend swatches / boxes, not runs).
    """
    out: list[list[tuple[float, float]]] = []
    for drawing in drawings or []:
        items = drawing.get("items") or []
        if not items:
            continue
        polyline: list[tuple[float, float]] = []
        for item in items:
            if not item or not isinstance(item, (tuple, list)):
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:
                start = item[1]
                end = item[2]
                try:
                    p0 = (float(start[0]), float(start[1]))
                    p1 = (float(end[0]), float(end[1]))
                except (TypeError, IndexError, ValueError):
                    continue
                if not polyline:
                    polyline.append(p0)
                if polyline[-1] != p0:
                    if polyline:
                        out.append(polyline)
                    polyline = [p0]
                polyline.append(p1)
            elif op in {"m", "c"}:
                # MoveTo or bezier curve — closes the current polyline.
                if polyline:
                    out.append(polyline)
                    polyline = []
        if polyline:
            out.append(polyline)
    return out


def _is_near(
    point: tuple[float, float], bbox: tuple[float, float, float, float],
    tol: float,
) -> bool:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return _segment_length(point, (cx, cy)) <= tol


def _snap_endpoint(
    point: tuple[float, float],
    detections: Sequence[Any],
    tol: float,
) -> str | None:
    """Return the nearest detection_id whose center is within tol of point."""
    best: tuple[float, str] | None = None
    for det in detections:
        bbox = det.bbox_pdf
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        dist = _segment_length(point, (cx, cy))
        if dist > tol:
            continue
        if best is None or dist < best[0] or (dist == best[0] and det.detection_id < best[1]):
            best = (dist, det.detection_id)
    return best[1] if best else None


def detect_line_runs(
    *,
    page: Any,
    page_index: int,
    sheet_number: str | None,
    detections: Sequence[Any] = (),
    excluded_bboxes: Sequence[tuple[float, float, float, float]] = (),
    min_length_pt: float = _MIN_RUN_LENGTH_PT,
    max_length_pt: float = _MAX_RUN_LENGTH_PT,
) -> list[LineRun]:
    """Extract line runs from a PyMuPDF page and snap endpoints to detections.

    Returns ``[]`` when no primitives are available or every candidate
    is filtered out by length / exclusion-zone tests.  Deterministic:
    same page bytes → same atom IDs across runs.
    """
    if page is None:
        return []
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover — fitz can be absent in tests
        return []
    polylines = _drawing_items_to_polylines(drawings)
    out: list[LineRun] = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        length = _polyline_length(poly)
        if length < min_length_pt or length > max_length_pt:
            continue
        bbox = _polyline_bbox(poly)
        if any(_bbox_intersects(bbox, ex) for ex in excluded_bboxes):
            continue
        # Snap both endpoints to nearby detections.
        start, end = poly[0], poly[-1]
        from_id = _snap_endpoint(start, detections, _SNAP_TOLERANCE_PT)
        to_id = _snap_endpoint(end, detections, _SNAP_TOLERANCE_PT)
        # Skip runs whose endpoints both float free AND don't connect
        # anything — those are usually structural lines (walls,
        # partitions) we don't want to count as runs.  When the
        # caller supplies no detections we keep the run unconditionally
        # so downstream consumers can still see the topology.
        if detections and from_id is None and to_id is None:
            continue
        rounded = tuple((round(p[0], 3), round(p[1], 3)) for p in poly)
        line_run_id = stable_id(
            "line_run",
            page_index,
            sheet_number or "",
            rounded,
        )
        out.append(
            LineRun(
                line_run_id=line_run_id,
                page_index=page_index,
                sheet_number=sheet_number,
                polyline=rounded,
                length_pt=length,
                bbox_pdf=bbox,
                from_detection_id=from_id,
                to_detection_id=to_id,
                confidence=0.75 if (from_id and to_id) else 0.55,
            )
        )
    out.sort(
        key=lambda r: (
            round(r.bbox_pdf[1], 2),
            round(r.bbox_pdf[0], 2),
            r.line_run_id,
        )
    )
    # Cap per-page count to avoid drowning consumers in wall segments
    # on raster-heavy floor plans. Prefer runs with snapped endpoints
    # (they carry topology info downstream).
    if len(out) > _MAX_RUNS_PER_PAGE:
        out.sort(
            key=lambda r: (
                -((1 if r.from_detection_id else 0) + (1 if r.to_detection_id else 0)),
                round(r.bbox_pdf[1], 2),
                round(r.bbox_pdf[0], 2),
                r.line_run_id,
            )
        )
        out = out[:_MAX_RUNS_PER_PAGE]
        out.sort(
            key=lambda r: (
                round(r.bbox_pdf[1], 2),
                round(r.bbox_pdf[0], 2),
                r.line_run_id,
            )
        )
    return out
