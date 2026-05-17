"""Spatial-region zone assignment for multi-zone floor-plan sheets.

When a single sheet carries multiple HOMERUN notes that share level
context (e.g., T1.01 has both "HOMERUN … TO MDF ROOM" and "HOMERUN …
TO IDF-1" on the same lobby level), the text parser can identify the
zones but the existing fusion logic falls through to
``ambiguous_homerun_zone`` because the device's level matches both.

This module provides a deterministic spatial-region heuristic for
resolving such cases: each zone gets a polygonal region of the page,
and devices are assigned by which region contains their bbox center.

v0 strategy (simple, robust):
- If a single zone covers the whole page: full-page region.
- If N zones share level context: split the page into N vertical
  strips, assigning each strip to the zone whose HOMERUN sentence
  sits horizontally inside it.

A future refinement could use the closet label position (the literal
"MDF" / "IDF-N" annotations on the plan) to pull region boundaries
toward each closet's actual location.

The output is a list of ``ZoneRegion`` (zone + region bbox) tuples
suitable for ``assign_home_run_spatial`` lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.takeoff.schemas import BBox
from app.takeoff.zones import HomeRunZone


@dataclass
class ZoneRegion:
    """A zone bound to a polygonal region of one sheet.

    For v0 the polygon is just a rectangle.
    """

    zone: HomeRunZone
    region: BBox

    def contains(self, bbox: BBox) -> bool:
        cx, cy = bbox.center()
        return (
            self.region.x0 <= cx <= self.region.x1
            and self.region.y0 <= cy <= self.region.y1
        )


# ─── Sentence-position lookup ──────


def _zone_sentence_bbox(
    page: Any,
    zone: HomeRunZone,
) -> BBox | None:
    """Locate the HOMERUN sentence on the page → bbox.

    Matches the first occurrence of ``HOMERUN`` token whose next ~25
    words reconstruct text close to ``zone.raw_text``. Returns the
    bounding bbox of those words or ``None`` if not found.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return None
    if not words:
        return None
    # Locate every HOMERUN occurrence; score each by how well it
    # matches the zone's raw_text.
    raw_lower = " ".join(zone.raw_text.lower().split())
    best: tuple[float, list] | None = None
    for i, w in enumerate(words):
        if w[4].upper() != "HOMERUN":
            continue
        run_words = words[i : i + 30]
        run_text = " ".join(ww[4].lower() for ww in run_words)
        # Trim run_text to first period.
        if "." in run_text:
            run_text = run_text.split(".", 1)[0] + "."
        # Quick similarity: count overlapping tokens.
        a_toks = set(raw_lower.split())
        b_toks = set(run_text.split())
        if not a_toks:
            continue
        overlap = len(a_toks & b_toks) / max(1, len(a_toks))
        if overlap < 0.7:
            continue
        # Slice the run to the period-terminating word.
        truncated: list = []
        for ww in run_words:
            truncated.append(ww)
            if "." in ww[4]:
                break
        if best is None or overlap > best[0]:
            best = (overlap, truncated)
    if best is None:
        return None
    run = best[1]
    x0 = min(ww[0] for ww in run)
    y0 = min(ww[1] for ww in run)
    x1 = max(ww[2] for ww in run)
    y1 = max(ww[3] for ww in run)
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1, coord_space="pdf_pt")


# ─── Region builder ─────


def build_zone_regions(
    *,
    page: Any,
    zones: list[HomeRunZone],
) -> list[ZoneRegion]:
    """Produce a ZoneRegion per zone for a single sheet.

    Empty/short zone lists produce empty output. For a single zone the
    region is the full page (so ``contains`` is always True). For
    multi-zone pages the page is split into vertical strips at the
    midpoints between each zone-sentence's horizontal position; each
    zone owns the strip nearest its sentence x-position.
    """
    if not zones:
        return []
    try:
        rect = page.rect
        page_w = float(rect.width)
        page_h = float(rect.height)
    except Exception:
        page_w, page_h = 3024.0, 2160.0

    full = BBox(x0=0.0, y0=0.0, x1=page_w, y1=page_h, coord_space="pdf_pt")
    if len(zones) == 1:
        return [ZoneRegion(zone=zones[0], region=full)]

    # Locate each zone sentence; if any zone fails to locate, we fall
    # back to splitting in document-order across equally-sized strips.
    located: list[tuple[HomeRunZone, BBox | None]] = []
    for z in zones:
        bbox = _zone_sentence_bbox(page, z)
        located.append((z, bbox))

    if any(bb is None for _, bb in located):
        # Equal strips, document order.
        strip_w = page_w / len(zones)
        regions: list[ZoneRegion] = []
        for i, (z, _) in enumerate(located):
            region = BBox(
                x0=i * strip_w, y0=0.0, x1=(i + 1) * strip_w, y1=page_h,
                coord_space="pdf_pt",
            )
            regions.append(ZoneRegion(zone=z, region=region))
        return regions

    # Sort zones by their sentence x-position and split at midpoints.
    sorted_zones = sorted(
        located, key=lambda lz: (lz[1].x0 + lz[1].x1) / 2.0,
    )
    centers = [(bb.x0 + bb.x1) / 2.0 for _, bb in sorted_zones]
    regions = []
    for i, (z, _) in enumerate(sorted_zones):
        left = 0.0 if i == 0 else (centers[i - 1] + centers[i]) / 2.0
        right = page_w if i == len(sorted_zones) - 1 else (centers[i] + centers[i + 1]) / 2.0
        regions.append(
            ZoneRegion(
                zone=z,
                region=BBox(
                    x0=left, y0=0.0, x1=right, y1=page_h, coord_space="pdf_pt",
                ),
            )
        )
    return regions


# ─── Spatial assignment ─────


def assign_home_run_spatial(
    *,
    regions: list[ZoneRegion],
    device_bbox: BBox,
) -> tuple[str | None, str | None, list[str], list[str]]:
    """Return (home_run_to, home_run_level, zone_notes, review_flags).

    Mirrors the contract of ``zones.assign_home_run`` so the fusion
    code can swap them in. Returns ``(None, None, [], [])`` when no
    region contains the device bbox.
    """
    if not regions:
        return (None, None, [], [])
    raw_notes = [r.zone.raw_text for r in regions]
    for r in regions:
        if r.contains(device_bbox):
            return (r.zone.target, r.zone.target_level, raw_notes, [])
    return (None, None, raw_notes, ["ambiguous_homerun_zone"])


__all__ = [
    "ZoneRegion",
    "build_zone_regions",
    "assign_home_run_spatial",
]
