"""Typed schedule extractors — door, panel, equipment, fixture, cable.

Today schedules collapse to "tabular text." But each schedule type
has a known column shape that downstream PMs need:

* **Door schedule**: door# / type / size / hardware / fire-rating / lockset
* **Panel schedule**: panel# / breaker# / circuit# / load / phase / poles
* **Equipment schedule**: tag / mfr / model# / qty / power / location
* **Fixture schedule**: type / mfr / model# / lamp / mounting
* **Cable schedule**: from / to / type / length / pathway / count

This module classifies tables by header pattern then maps cells
into typed records. Each typed record becomes an atom downstream so
PM_HANDOFF can roll them up (e.g., "82 doors / 14 fire-rated / $48K
hardware total").

Deterministic. No LLM. Operates on text blocks already extracted
by the parent parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence


# ── Schedule kinds + their header keywords ────────────────────────


SCHEDULE_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("door_schedule",       ("DOOR SCHEDULE", "DOOR LIST")),
    ("panel_schedule",      ("PANEL SCHEDULE", "PANEL LIST", "LOAD SCHEDULE")),
    ("equipment_schedule",  ("EQUIPMENT SCHEDULE", "EQUIPMENT LIST", "MECHANICAL EQUIPMENT SCHEDULE", "EQUIPMENT LEGEND")),
    ("fixture_schedule",    ("FIXTURE SCHEDULE", "LIGHT FIXTURE SCHEDULE", "LIGHTING SCHEDULE")),
    ("cable_schedule",      ("CABLE SCHEDULE", "CONDUIT SCHEDULE", "WIRE SCHEDULE", "RACEWAY SCHEDULE")),
    ("room_schedule",       ("ROOM SCHEDULE", "ROOM LIST", "FINISH SCHEDULE", "ROOM FINISH SCHEDULE")),
    ("device_schedule",     ("DEVICE SCHEDULE", "ACCESS CONTROL DEVICE SCHEDULE", "CAMERA SCHEDULE")),
    ("riser_schedule",      ("RISER SCHEDULE", "RISER DIAGRAM")),
)


# Per-kind column header keywords (case-insensitive substring match).
# Used to map detected header cells to typed fields.
COLUMN_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "door_schedule": {
        "door_number":  ("door no", "door #", "no.", "mark", "tag"),
        "door_type":    ("type", "door type"),
        "size":         ("size", "width", "w x h", "wxh"),
        "material":     ("material", "matl"),
        "fire_rating":  ("fire", "ul rating", "rating"),
        "hardware":     ("hardware", "hw", "hw set"),
        "frame":        ("frame", "frame type"),
        "remarks":      ("remarks", "notes", "comments"),
    },
    "panel_schedule": {
        "panel_id":     ("panel", "panel id", "name"),
        "circuit":      ("circuit", "ckt", "ckt #"),
        "breaker_amp":  ("amp", "amps", "breaker"),
        "phase":        ("phase", "ph"),
        "poles":        ("poles", "pole"),
        "load":         ("load", "load (kva)", "kva", "load kva"),
        "description":  ("description", "served", "load served"),
    },
    "equipment_schedule": {
        "tag":          ("tag", "id", "designation", "mark"),
        "manufacturer": ("mfr", "manufacturer", "make"),
        "model":        ("model", "model #", "part #", "catalog"),
        "quantity":     ("qty", "count", "no."),
        "power":        ("kw", "hp", "watts", "amps", "volt", "load"),
        "location":     ("location", "room", "loc"),
        "remarks":      ("remarks", "notes"),
    },
    "fixture_schedule": {
        "type":         ("type", "fixture type"),
        "manufacturer": ("mfr", "manufacturer"),
        "model":        ("model", "catalog", "part #"),
        "lamp":         ("lamp", "lamp type"),
        "mounting":     ("mounting", "mount"),
        "wattage":      ("wattage", "watts", "w"),
        "quantity":     ("qty", "count"),
    },
    "cable_schedule": {
        "tag":          ("tag", "cable id", "cable #", "circuit"),
        "from":         ("from", "source"),
        "to":           ("to", "destination"),
        "cable_type":   ("type", "cable type", "cable"),
        "length_ft":    ("length", "length (ft)", "ft"),
        "pathway":      ("pathway", "conduit", "raceway"),
        "remarks":      ("remarks", "notes"),
    },
    "room_schedule": {
        "room_number":  ("room", "room no", "room #"),
        "room_name":    ("name", "function"),
        "area_sqft":    ("area", "sf", "sqft"),
        "floor":        ("floor", "ceiling", "wall", "base"),
    },
    "device_schedule": {
        "device_id":    ("id", "tag", "device", "device id"),
        "device_type":  ("type", "kind"),
        "location":     ("location", "room"),
        "manufacturer": ("mfr", "manufacturer"),
        "model":        ("model", "part #"),
        "remarks":      ("remarks", "notes"),
    },
    "riser_schedule": {
        "tag":          ("tag", "id", "designation"),
        "system":       ("system", "service"),
        "remarks":      ("remarks", "notes"),
    },
}


@dataclass(frozen=True)
class ScheduleRow:
    """One row of a typed schedule. ``fields`` map normalized
    field-name → cell text. Extra columns landed in ``extras``."""

    page_index: int
    schedule_kind: str                                # "door_schedule" / "panel_schedule" / ...
    fields: dict[str, str]
    extras: dict[str, str] = field(default_factory=dict)
    row_bbox_pdf: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class TypedSchedule:
    """A schedule of typed rows located on one page."""

    page_index: int
    schedule_kind: str
    header_text: str
    bbox_pdf: tuple[float, float, float, float]
    column_map: dict[int, str]                        # column-index → normalized field name
    rows: tuple[ScheduleRow, ...]


# ── Detection + parse ────────────────────────────────────────────


def detect_schedule_kind(header_text: str) -> str | None:
    """Match a header line to a schedule kind. Returns None for non-matches."""
    if not header_text:
        return None
    upper = header_text.upper()
    for kind, keywords in SCHEDULE_KINDS:
        for kw in keywords:
            if kw in upper:
                return kind
    return None


def _block_text(b: Any) -> str:
    t = getattr(b, "text", None)
    if t is None and isinstance(b, dict):
        t = b.get("text")
    return str(t or "").strip()


def _block_bbox(b: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(b, "bbox", None)
    if bbox is None and isinstance(b, dict):
        bbox = b.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None


def map_header_to_columns(
    *,
    schedule_kind: str,
    header_blocks: Sequence[Any],
) -> dict[int, str]:
    """Map a row of header cells (left to right) to typed field names."""
    kw_map = COLUMN_KEYWORDS.get(schedule_kind, {})
    if not kw_map:
        return {}
    sorted_headers = sorted(
        header_blocks,
        key=lambda b: (_block_bbox(b) or (0.0,) * 4)[0],
    )
    out: dict[int, str] = {}
    for i, h in enumerate(sorted_headers):
        text = _block_text(h).lower()
        if not text:
            continue
        best_field: str | None = None
        best_match_len = 0
        for field_name, kws in kw_map.items():
            for kw in kws:
                if kw in text and len(kw) > best_match_len:
                    best_field = field_name
                    best_match_len = len(kw)
        if best_field:
            out[i] = best_field
    return out


def parse_typed_schedule(
    *,
    page_index: int,
    header_block: Any,
    nearby_blocks: Sequence[Any],
) -> TypedSchedule | None:
    """Given a header block matching a schedule kind + a list of
    candidate blocks below it, parse a TypedSchedule.

    Returns None if no typed rows could be extracted.
    """
    header_text = _block_text(header_block)
    kind = detect_schedule_kind(header_text)
    if kind is None:
        return None
    header_bbox = _block_bbox(header_block)
    if header_bbox is None:
        return None

    # Cluster blocks into rows by Y-coordinate
    rows_by_y: dict[float, list[Any]] = {}
    for b in nearby_blocks:
        bbox = _block_bbox(b)
        if bbox is None or bbox[1] < header_bbox[3]:   # must be below header
            continue
        # Bucket by integer Y to handle slight misalignments
        y_bucket = round(bbox[1] / 8.0) * 8.0
        rows_by_y.setdefault(y_bucket, []).append(b)

    if not rows_by_y:
        return None

    sorted_rows = sorted(rows_by_y.items())
    if len(sorted_rows) < 2:
        return None

    # First row = column headers; map to typed fields
    header_row_blocks = sorted_rows[0][1]
    column_map = map_header_to_columns(
        schedule_kind=kind,
        header_blocks=header_row_blocks,
    )
    if not column_map:
        return None

    # Subsequent rows = data
    data_rows: list[ScheduleRow] = []
    for _, row_blocks in sorted_rows[1:]:
        sorted_blocks = sorted(
            row_blocks,
            key=lambda b: (_block_bbox(b) or (0.0,) * 4)[0],
        )
        row_fields: dict[str, str] = {}
        row_extras: dict[str, str] = {}
        for i, b in enumerate(sorted_blocks):
            text = _block_text(b)
            if not text:
                continue
            field_name = column_map.get(i)
            if field_name:
                row_fields[field_name] = text
            else:
                row_extras[f"col_{i}"] = text
        if not row_fields:
            continue
        first_bbox = _block_bbox(sorted_blocks[0])
        last_bbox = _block_bbox(sorted_blocks[-1])
        if first_bbox and last_bbox:
            row_bbox = (first_bbox[0], first_bbox[1], last_bbox[2], last_bbox[3])
        else:
            row_bbox = None
        data_rows.append(
            ScheduleRow(
                page_index=page_index,
                schedule_kind=kind,
                fields=row_fields,
                extras=row_extras,
                row_bbox_pdf=row_bbox,
            )
        )

    if not data_rows:
        return None

    # Box bbox = union of all row bboxes + header
    box_x0 = header_bbox[0]
    box_y0 = header_bbox[1]
    box_x1 = header_bbox[2]
    box_y1 = header_bbox[3]
    for r in data_rows:
        if r.row_bbox_pdf:
            box_x0 = min(box_x0, r.row_bbox_pdf[0])
            box_y0 = min(box_y0, r.row_bbox_pdf[1])
            box_x1 = max(box_x1, r.row_bbox_pdf[2])
            box_y1 = max(box_y1, r.row_bbox_pdf[3])

    return TypedSchedule(
        page_index=page_index,
        schedule_kind=kind,
        header_text=header_text,
        bbox_pdf=(box_x0, box_y0, box_x1, box_y1),
        column_map=column_map,
        rows=tuple(data_rows),
    )


__all__ = [
    "COLUMN_KEYWORDS",
    "SCHEDULE_KINDS",
    "ScheduleRow",
    "TypedSchedule",
    "detect_schedule_kind",
    "map_header_to_columns",
    "parse_typed_schedule",
]
