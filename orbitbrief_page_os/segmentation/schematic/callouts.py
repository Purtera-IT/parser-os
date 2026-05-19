"""Extract mounting-height / dimension callouts and attach them to
nearby symbol detections.

On real floor plans every device label is paired with a height
callout like ``42" AFF``, ``+120"``, ``CEILING``, ``VERIFY W/ ARCH``.
Without those, a "PTZ detected" atom is half the answer.

The detector finds TextBlocks that match a small set of callout
shapes:

  - ``\\d+(?:\\.\\d+)?\\s*(?:"|in|inches)?\\s*(?:AFF|aff|A\\.F\\.F\\.)``
  - ``\\d+'-\\d+(?:\\s*\\d+/\\d+)?"`` (architectural feet-inches)
  - ``\\+\\s*\\d+(?:\\.\\d+)?\\s*"`` (signed inches)
  - ``CEILING`` / ``CLG``
  - ``VERIFY W/ ARCH`` / ``VIF`` / ``V.I.F.``

Each callout is then matched to the nearest symbol detection within
a small radius. The detection atom's value gains
``mounting_height`` and ``callout_source_atom_id`` fields.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock


_HEIGHT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(\d+(?:\.\d+)?\s*(?:\"|in|inches)?\s*a\.?f\.?f\.?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+\s*'\s*-\s*\d+(?:\s*\d+/\d+)?\s*\")\b"),
    re.compile(r"(\+\s*\d+(?:\.\d+)?\s*(?:\"|in|inches))"),
    re.compile(r"\b(ceiling|clg)\b", re.IGNORECASE),
    re.compile(r"\b(verify\s+w/?\s*arch|v\.?i\.?f\.?)\b", re.IGNORECASE),
    re.compile(r"\b(wall\s+mount|wall-mount|surface\s+mount|pendant)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class Callout:
    text: str
    bbox: tuple[float, float, float, float]


def detect_callouts(
    blocks: Sequence[TextBlock],
    excluded_bboxes: Sequence[tuple[float, float, float, float]] = (),
) -> list[Callout]:
    """Return every mounting-height / dimension callout on the page."""
    out: list[Callout] = []
    for blk in blocks:
        if any(
            blk.bbox[0] < ex[2]
            and blk.bbox[2] > ex[0]
            and blk.bbox[1] < ex[3]
            and blk.bbox[3] > ex[1]
            for ex in excluded_bboxes
        ):
            continue
        for rx in _HEIGHT_PATTERNS:
            m = rx.search(blk.text)
            if m:
                out.append(Callout(text=m.group(1).strip(), bbox=blk.bbox))
                break
    out.sort(key=lambda c: (round(c.bbox[1], 2), round(c.bbox[0], 2), c.text))
    return out


def attach_callouts_to_detections(
    detections: Sequence[Any],
    callouts: Sequence[Callout],
    *,
    max_distance_pt: float = 72.0,
) -> dict[str, Callout]:
    """Return ``{detection_id: Callout}`` for detections that have a
    nearby callout. Deterministic — ties resolve by callout text.
    """
    if not detections or not callouts:
        return {}
    mapping: dict[str, Callout] = {}
    for det in detections:
        dx = (det.bbox_pdf[0] + det.bbox_pdf[2]) / 2.0
        dy = (det.bbox_pdf[1] + det.bbox_pdf[3]) / 2.0
        best: tuple[float, str, Callout] | None = None
        for callout in callouts:
            cx = (callout.bbox[0] + callout.bbox[2]) / 2.0
            cy = (callout.bbox[1] + callout.bbox[3]) / 2.0
            dist = ((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5
            if dist > max_distance_pt:
                continue
            if best is None or dist < best[0] or (dist == best[0] and callout.text < best[1]):
                best = (dist, callout.text, callout)
        if best is not None:
            mapping[det.detection_id] = best[2]
    return mapping
