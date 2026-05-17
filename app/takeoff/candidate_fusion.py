"""Fuse accepted :class:`SymbolCandidate` objects into :class:`DeviceInstance`.

Fusion v0 was one-to-one: each accepted text candidate became a device.
Fusion v1 (this module) also accepts a list of *shape* candidates and
applies the cross-validation rules from the Phase B spec:

- text + shape near same center (within ``XVAL_RADIUS_PT`` pt) →
  confidence 0.99, source_methods=["pdf_native_text","shape_template"]
- text only → confidence 0.94 (existing behaviour)
- shape only → confidence 0.70, needs_review=True, NOT counted in the
  accepted rollup

Shape-only candidates remain in ``TakeoffDocument.candidates`` for
auditors but are NOT promoted to ``devices`` so they cannot inflate
the rollup counts that downstream packets quote.

Zone assignment uses the home-run logic in :mod:`app.takeoff.zones`.
"""
from __future__ import annotations

from typing import Any

from app.core.ids import stable_id
from app.takeoff.legend_extractor import rules_by_symbol
from app.takeoff.schemas import (
    BBox,
    DeviceInstance,
    LegendRule,
    SheetRecord,
    SymbolCandidate,
)
from app.takeoff.spatial_zones import ZoneRegion, assign_home_run_spatial
from app.takeoff.zones import HomeRunZone, assign_home_run

# Radius (in PDF points) within which a text candidate and a shape
# candidate are considered to refer to the same device.
XVAL_RADIUS_PT = 24.0


def _legend_rule_id(rule: LegendRule) -> str:
    return stable_id("legrule", rule.raw_symbol, rule.normalized_class)


def _center_dist(a: BBox, b: BBox) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def fuse_candidates_to_devices(
    *,
    candidates: list[SymbolCandidate],
    sheet: SheetRecord,
    zones: list[HomeRunZone],
    legend_rules: list[LegendRule],
    shape_candidates: list[SymbolCandidate] | None = None,
    zone_regions: list[ZoneRegion] | None = None,
) -> list[DeviceInstance]:
    """Convert accepted candidates on a sheet into devices.

    ``candidates`` are the native-text candidates produced by
    ``detect_symbol_candidates``. ``shape_candidates`` (optional) are
    OpenCV template-match candidates produced by
    ``shape_candidates_for_page``. Cross-validation happens here:
    a text candidate that has a shape candidate within ``XVAL_RADIUS_PT``
    points of the same symbol family gets its source_methods extended
    and confidence promoted to 0.99.

    Shape-only candidates are NOT fused into devices. They get marked
    needs_review and remain in the candidate list for human audit.

    Rejected candidates are silently skipped.
    """
    rule_index = rules_by_symbol(legend_rules)
    devices: list[DeviceInstance] = []
    shape_candidates = shape_candidates or []

    # Index shape candidates by raw_symbol for fast neighbour lookup.
    shape_index: dict[str, list[SymbolCandidate]] = {}
    for sc in shape_candidates:
        if sc.rejection_reason is not None:
            continue
        shape_index.setdefault(sc.raw_symbol, []).append(sc)

    # Track which shape candidates were consumed by cross-validation.
    matched_shape_ids: set[str] = set()

    for cand in candidates:
        if cand.rejection_reason is not None:
            continue
        rule = rule_index.get(cand.raw_symbol)
        if rule is None:
            # No legend rule means we can't classify the device — skip.
            continue

        # Cross-validate against shape candidates of the same symbol.
        nearby_shapes = shape_index.get(cand.raw_symbol, [])
        crossval_match: SymbolCandidate | None = None
        for sc in nearby_shapes:
            if sc.id in matched_shape_ids:
                continue
            if _center_dist(cand.bbox, sc.bbox) <= XVAL_RADIUS_PT:
                crossval_match = sc
                break
        if crossval_match is not None:
            matched_shape_ids.add(crossval_match.id)
            # Extend the text candidate's source_methods and bump
            # its confidence — these objects are mutable Pydantic
            # models so this is a deliberate side-effect on the
            # candidate that lives in TakeoffDocument.candidates.
            methods = list(cand.source_methods)
            if "shape_template" not in methods:
                methods.append("shape_template")
            cand.source_methods = methods
            cand.confidence = max(cand.confidence, 0.99)
            # Tag the matched shape candidate too so summary code
            # can distinguish "shape-only" from "matched by text".
            sm = list(crossval_match.source_methods)
            if "pdf_native_text" not in sm:
                sm.append("pdf_native_text")
            crossval_match.source_methods = sm
            # Marked as needs_review=False since text validated it.
            crossval_match.needs_review = False

        # Pick a home-run target. For single-floor sheets the device's
        # level is unambiguous; for multi-floor sheets without a
        # device_level we fall through to ambiguity rules — unless
        # ``zone_regions`` is provided, in which case spatial
        # assignment runs FIRST and only ambiguous spatial outcomes
        # fall back to the level-based logic.
        device_level = sheet.levels_represented[0] if len(sheet.levels_represented) == 1 else None
        home_run_to: str | None = None
        home_run_level: str | None = None
        zone_notes: list[str] = []
        review_flags: list[str] = []

        if zone_regions and len(zone_regions) > 1:
            home_run_to, home_run_level, zone_notes, review_flags = (
                assign_home_run_spatial(
                    regions=zone_regions, device_bbox=cand.bbox,
                )
            )
            # If spatial resolved to a specific zone, we're done. If
            # ``ambiguous_homerun_zone`` came back AND we have a
            # device level + level-aware zones, try the level-based
            # logic as a fallback.
            if (
                "ambiguous_homerun_zone" in review_flags
                and device_level is not None
            ):
                fb_to, fb_level, fb_notes, fb_flags = assign_home_run(
                    zones=zones,
                    sheet_levels=sheet.levels_represented,
                    sheet_floor_label=sheet.floor_label,
                    sheet_number=sheet.sheet_number,
                    device_level=device_level,
                )
                if fb_to is not None:
                    home_run_to = fb_to
                    home_run_level = fb_level
                    zone_notes = fb_notes
                    review_flags = fb_flags
        else:
            home_run_to, home_run_level, zone_notes, review_flags = assign_home_run(
                zones=zones,
                sheet_levels=sheet.levels_represented,
                sheet_floor_label=sheet.floor_label,
                sheet_number=sheet.sheet_number,
                device_level=device_level,
            )

        device_id = stable_id(
            "dev",
            sheet.page_index,
            cand.raw_symbol,
            round(cand.bbox.center()[0], 1),
            round(cand.bbox.center()[1], 1),
        )

        devices.append(
            DeviceInstance(
                id=device_id,
                page_index=sheet.page_index,
                sheet_number=sheet.sheet_number,
                sheet_name=sheet.sheet_name,
                raw_symbol=cand.raw_symbol,
                normalized_class=rule.normalized_class,
                system=rule.system,
                bbox=cand.bbox,
                floor_label=sheet.floor_label,
                levels_represented=list(sheet.levels_represented),
                multiplier=sheet.multiplier,
                home_run_to=home_run_to,
                home_run_level=home_run_level,
                zone_notes=list(zone_notes),
                legend_rule_id=_legend_rule_id(rule),
                confidence=cand.confidence,
                review_flags=list(review_flags),
            )
        )

    return devices


__all__ = [
    "fuse_candidates_to_devices",
]
