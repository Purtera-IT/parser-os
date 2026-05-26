"""Detect room / area / zone labels on a floor plan.

Real floor plans label every enclosed area with a name and/or a
room number. Pinning every symbol detection to its containing room
turns ``3 PTZ cameras`` into ``1 PTZ in LOBBY, 1 in HALLWAY 204,
1 in CONFERENCE 301`` — far more useful for review.

Detection is text-rule + bbox-distance: any TextBlock that matches
a room-label shape becomes a candidate Room. A nearby
``schematic_symbol_detection`` is then attributed to the closest
Room when the detection's center falls within the room's vicinity.

Room label shapes (deterministic, no LLM):

- Room name + number: ``LOBBY 101``, ``CONFERENCE 204``, ``MDF 1.2``
- Bare room number: ``101``, ``A-203``, ``B.4`` (when also annotated
  with a known room word elsewhere on the page).
- Name-only: ``LOBBY``, ``RECEPTION``, ``ELEC`` (rejected unless a
  number is part of the same TextBlock OR a room-word denylist
  doesn't apply).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.core.ids import stable_id
from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock


# Canonical room words seen on architectural / low-voltage plans.
_ROOM_WORDS = {
    "lobby", "reception", "office", "conference", "meeting", "huddle",
    "open office", "training", "break", "breakroom", "kitchen",
    "kitchenette", "pantry", "lounge", "lunch", "cafe", "cafeteria",
    "dining", "auditorium", "classroom", "lecture", "seminar",
    "mdf", "idf", "tr", "telecom", "data", "server", "computer",
    "elec", "electrical", "mech", "mechanical", "storage", "stor",
    "janitor", "jc", "closet", "supply", "utility", "boiler",
    "lab", "laboratory", "exam", "patient", "treatment", "imaging",
    "or", "operating", "icu", "pacu", "nursery", "triage",
    "hallway", "corridor", "vestibule", "stair", "stairs", "stairwell",
    "elevator", "lift", "elev", "lobby waiting", "waiting",
    "men", "women", "mens", "womens", "mens room", "womens room",
    "restroom", "toilet", "bath", "bathroom", "shower", "locker",
    "garage", "loading", "dock", "warehouse", "yard", "shop",
    "guard", "security", "dispatch", "control", "monitor",
    "lab", "studio", "library", "vault", "mailroom", "copy",
    "print", "media", "av", "broadcast", "data center", "noc", "soc",
    "ahu", "fcu", "vav", "rtu", "chiller",  # mechanical zones
    "lobby waiting", "reception area", "executive",
}


# Room number patterns (UPPER-CASE-NORMALIZED). Match a leading
# letter+digit code, optionally with a hyphen or dot separator:
#   101 / 1.01 / A-101 / B.4 / 2A / 12-34
_ROOM_NUMBER_RE = re.compile(
    r"\b([A-Z]{0,3}[\-./]?\d{1,4}(?:[\-./][A-Z0-9]{1,4})?)\b"
)
_ROOM_LABEL_PAIR_RE = re.compile(
    r"^([A-Z][A-Z\s/&-]{1,30}?)[\s\-.]+([A-Z]{0,3}[\-./]?\d{1,4}(?:[\-./][A-Z0-9]{1,4})?)$"
)


@dataclass(frozen=True)
class Room:
    room_id: str
    page_index: int
    sheet_number: str | None
    label: str
    number: str | None
    bbox: tuple[float, float, float, float]
    confidence: float


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_room_label_text(text: str) -> tuple[str | None, str | None]:
    """Return ``(label, number)`` if ``text`` looks like a room label.

    Both fields may be None when the text isn't recognizable; if just
    the number is None it's a name-only room (still emitted).
    """
    t = _norm(text)
    if not t or len(t) > 50:
        return (None, None)
    upper = t.upper()
    # Pair: name + number
    m = _ROOM_LABEL_PAIR_RE.match(upper)
    if m:
        name = m.group(1).strip()
        number = m.group(2).strip()
        if any(w.upper() in name for w in _ROOM_WORDS):
            return (name, number)
    # Name only
    if any(w.upper() == upper or w.upper() in upper.split() for w in _ROOM_WORDS):
        # Strip parenthetical etc.
        return (upper, None)
    return (None, None)


def detect_rooms(
    *,
    page_index: int,
    sheet_number: str | None,
    blocks: Sequence[TextBlock],
    excluded_bboxes: Sequence[tuple[float, float, float, float]] = (),
) -> list[Room]:
    """Return the deterministic list of rooms detected on this page.

    Rooms inside excluded zones (title block, schedules, drawing
    index, keyed notes) are skipped — those regions reference rooms
    but aren't themselves rooms.
    """
    out: list[Room] = []
    seen_keys: set[tuple[str, str | None]] = set()
    for blk in blocks:
        if any(
            (
                blk.bbox[0] < ex[2]
                and blk.bbox[2] > ex[0]
                and blk.bbox[1] < ex[3]
                and blk.bbox[3] > ex[1]
            )
            for ex in excluded_bboxes
        ):
            continue
        label, number = _is_room_label_text(blk.text)
        if label is None:
            continue
        key = (label, number)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        confidence = 0.85 if number else 0.6
        room_id = stable_id(
            "room",
            page_index,
            sheet_number or "",
            label,
            number or "",
        )
        out.append(
            Room(
                room_id=room_id,
                page_index=page_index,
                sheet_number=sheet_number,
                label=label,
                number=number,
                bbox=blk.bbox,
                confidence=confidence,
            )
        )
    out.sort(
        key=lambda r: (
            round(r.bbox[1], 2),
            round(r.bbox[0], 2),
            r.label,
            r.number or "",
        )
    )
    return out


def assign_detections_to_rooms(
    detections: Sequence[Any],
    rooms: Sequence[Room],
    *,
    max_distance_pt: float = 144.0,
) -> dict[str, str]:
    """Return a ``{detection_id: room_id}`` mapping.

    For each detection, pick the nearest room whose center is within
    ``max_distance_pt`` of the detection center.  When multiple
    rooms are equidistant, pick the one with the smaller (deterministic)
    ``room_id`` so the assignment is byte-stable.
    """
    if not rooms or not detections:
        return {}
    mapping: dict[str, str] = {}
    for det in detections:
        dx = (det.bbox_pdf[0] + det.bbox_pdf[2]) / 2.0
        dy = (det.bbox_pdf[1] + det.bbox_pdf[3]) / 2.0
        best: tuple[float, str] | None = None
        for room in rooms:
            rx = (room.bbox[0] + room.bbox[2]) / 2.0
            ry = (room.bbox[1] + room.bbox[3]) / 2.0
            dist = ((dx - rx) ** 2 + (dy - ry) ** 2) ** 0.5
            if dist > max_distance_pt:
                continue
            if best is None or dist < best[0] or (dist == best[0] and room.room_id < best[1]):
                best = (dist, room.room_id)
        if best is not None:
            mapping[det.detection_id] = best[1]
    return mapping
