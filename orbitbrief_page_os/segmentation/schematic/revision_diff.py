"""Revision diff — compare two compile results across DD/CD revisions.

Drawing sets revise constantly: a "100% DD" set is followed by "50% CD"
then "100% CD" then "ASI-001" etc. The single most expensive question
a PM asks during review is:

   "What changed between revision X and revision Y?"

Today's parser produces a flat dump per revision — comparing them
means eyeballing thousands of atoms. This module diffs two compile
outputs along five dimensions:

* **Added detections**   — devices in B that don't appear in A
* **Removed detections** — devices in A that don't appear in B
* **Moved detections**   — same symbol+label but bbox moved > tolerance
* **Renamed labels**     — same bbox but the matched label changed
* **Schedule deltas**    — door / panel / equipment rows added/removed

The matching uses ``(page_index, matched_label_text, bbox_center)``
as the natural key with a configurable position tolerance.

Deterministic. No LLM. Operates on the same dataclasses that compile
already produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# Detections that moved within this radius are considered the same atom.
_MOVE_TOLERANCE_PT = 24.0

# Match score below this counts as "different atom" (rename/move neither apply).
_MATCH_THRESHOLD = 0.55


@dataclass(frozen=True)
class DetectionKey:
    page_index: int
    label: str
    symbol_text: str

    @classmethod
    def from_detection(cls, det: Any) -> "DetectionKey":
        return cls(
            page_index=getattr(det, "page_index", -1),
            label=(getattr(det, "matched_label_text", None) or "").strip().upper(),
            symbol_text=(getattr(det, "matched_symbol_text", None) or "").strip().upper(),
        )


@dataclass(frozen=True)
class DetectionMove:
    """A detection that moved (same key, different bbox)."""

    key: DetectionKey
    from_bbox: tuple[float, float, float, float]
    to_bbox: tuple[float, float, float, float]
    distance_pt: float


@dataclass(frozen=True)
class DetectionRename:
    """Detection at same location with a changed label."""

    page_index: int
    bbox_pdf: tuple[float, float, float, float]
    old_label: str
    new_label: str


@dataclass(frozen=True)
class ScheduleDelta:
    """Schedule row added or removed across revisions."""

    schedule_kind: str
    page_index: int
    operation: str                                       # "added" / "removed"
    fields: dict[str, str]


@dataclass(frozen=True)
class RevisionDiff:
    """Full diff between two compile results."""

    revision_a_id: str
    revision_b_id: str
    added_detections: tuple[Any, ...]
    removed_detections: tuple[Any, ...]
    moved_detections: tuple[DetectionMove, ...]
    renamed_detections: tuple[DetectionRename, ...]
    schedule_deltas: tuple[ScheduleDelta, ...]
    summary: dict[str, int] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _index_by_key(detections: Iterable[Any]) -> dict[DetectionKey, list[Any]]:
    out: dict[DetectionKey, list[Any]] = {}
    for d in detections:
        key = DetectionKey.from_detection(d)
        out.setdefault(key, []).append(d)
    return out


# ── Main entry ───────────────────────────────────────────────────


def diff_detections(
    detections_a: Sequence[Any],
    detections_b: Sequence[Any],
    *,
    move_tolerance_pt: float = _MOVE_TOLERANCE_PT,
) -> tuple[list[Any], list[Any], list[DetectionMove], list[DetectionRename]]:
    """Return (added, removed, moved, renamed) lists of detections."""
    a_by_key = _index_by_key(detections_a)
    b_by_key = _index_by_key(detections_b)

    added: list[Any] = []
    removed: list[Any] = []
    moved: list[DetectionMove] = []
    renamed: list[DetectionRename] = []

    # Track which A's have been claimed by a B match
    claimed_a: set[int] = set()
    claimed_b: set[int] = set()

    # Same-key matching: pair up A/B detections with the same key
    for key, b_list in b_by_key.items():
        a_list = a_by_key.get(key, [])
        for b in b_list:
            b_id = id(b)
            if b_id in claimed_b:
                continue
            best_a: tuple[float, Any] | None = None
            for a in a_list:
                a_id = id(a)
                if a_id in claimed_a:
                    continue
                a_center = _bbox_center(getattr(a, "bbox_pdf", (0, 0, 0, 0)))
                b_center = _bbox_center(getattr(b, "bbox_pdf", (0, 0, 0, 0)))
                dist = _distance(a_center, b_center)
                if best_a is None or dist < best_a[0]:
                    best_a = (dist, a)
            if best_a is not None:
                dist, a = best_a
                claimed_a.add(id(a))
                claimed_b.add(id(b))
                if dist > move_tolerance_pt:
                    moved.append(
                        DetectionMove(
                            key=key,
                            from_bbox=getattr(a, "bbox_pdf", (0, 0, 0, 0)),
                            to_bbox=getattr(b, "bbox_pdf", (0, 0, 0, 0)),
                            distance_pt=dist,
                        )
                    )

    # Rename detection: A and B at same location (bbox center close) but
    # different label
    for a in detections_a:
        if id(a) in claimed_a:
            continue
        a_center = _bbox_center(getattr(a, "bbox_pdf", (0, 0, 0, 0)))
        a_page = getattr(a, "page_index", -1)
        a_label = (getattr(a, "matched_label_text", None) or "").strip().upper()
        for b in detections_b:
            if id(b) in claimed_b:
                continue
            b_page = getattr(b, "page_index", -1)
            if a_page != b_page:
                continue
            b_center = _bbox_center(getattr(b, "bbox_pdf", (0, 0, 0, 0)))
            if _distance(a_center, b_center) <= move_tolerance_pt:
                b_label = (getattr(b, "matched_label_text", None) or "").strip().upper()
                if b_label and b_label != a_label:
                    renamed.append(
                        DetectionRename(
                            page_index=a_page,
                            bbox_pdf=getattr(b, "bbox_pdf", (0, 0, 0, 0)),
                            old_label=a_label,
                            new_label=b_label,
                        )
                    )
                    claimed_a.add(id(a))
                    claimed_b.add(id(b))
                    break

    # Leftover A → removed, leftover B → added
    for a in detections_a:
        if id(a) not in claimed_a:
            removed.append(a)
    for b in detections_b:
        if id(b) not in claimed_b:
            added.append(b)

    return added, removed, moved, renamed


def diff_schedules(
    schedules_a: Sequence[Any],
    schedules_b: Sequence[Any],
) -> list[ScheduleDelta]:
    """Diff typed schedules from two revisions row-by-row."""

    def fingerprint(row: Any) -> tuple[str, int, frozenset]:
        kind = getattr(row, "schedule_kind", "")
        page = getattr(row, "page_index", -1)
        fields = getattr(row, "fields", {}) or {}
        # Hash on a canonicalized field dict
        items = frozenset((k, (v or "").strip().upper()) for k, v in fields.items())
        return (kind, page, items)

    a_fp = {fingerprint(r): r for r in schedules_a}
    b_fp = {fingerprint(r): r for r in schedules_b}

    deltas: list[ScheduleDelta] = []
    for fp, row in a_fp.items():
        if fp not in b_fp:
            deltas.append(
                ScheduleDelta(
                    schedule_kind=fp[0],
                    page_index=fp[1],
                    operation="removed",
                    fields=getattr(row, "fields", {}) or {},
                )
            )
    for fp, row in b_fp.items():
        if fp not in a_fp:
            deltas.append(
                ScheduleDelta(
                    schedule_kind=fp[0],
                    page_index=fp[1],
                    operation="added",
                    fields=getattr(row, "fields", {}) or {},
                )
            )
    deltas.sort(key=lambda d: (d.schedule_kind, d.page_index, d.operation))
    return deltas


def diff_revisions(
    *,
    revision_a_id: str,
    revision_b_id: str,
    detections_a: Sequence[Any] = (),
    detections_b: Sequence[Any] = (),
    schedules_a: Sequence[Any] = (),
    schedules_b: Sequence[Any] = (),
    move_tolerance_pt: float = _MOVE_TOLERANCE_PT,
) -> RevisionDiff:
    """Top-level diff entry. Returns a fully-populated RevisionDiff."""
    added, removed, moved, renamed = diff_detections(
        detections_a, detections_b, move_tolerance_pt=move_tolerance_pt
    )
    schedule_deltas = diff_schedules(schedules_a, schedules_b)
    summary = {
        "added": len(added),
        "removed": len(removed),
        "moved": len(moved),
        "renamed": len(renamed),
        "schedules_added": sum(1 for d in schedule_deltas if d.operation == "added"),
        "schedules_removed": sum(1 for d in schedule_deltas if d.operation == "removed"),
    }
    return RevisionDiff(
        revision_a_id=revision_a_id,
        revision_b_id=revision_b_id,
        added_detections=tuple(added),
        removed_detections=tuple(removed),
        moved_detections=tuple(moved),
        renamed_detections=tuple(renamed),
        schedule_deltas=tuple(schedule_deltas),
        summary=summary,
    )


__all__ = [
    "DetectionKey",
    "DetectionMove",
    "DetectionRename",
    "RevisionDiff",
    "ScheduleDelta",
    "diff_detections",
    "diff_revisions",
    "diff_schedules",
]
