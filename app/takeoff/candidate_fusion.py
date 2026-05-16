"""Fuse accepted :class:`SymbolCandidate` objects into :class:`DeviceInstance`.

The MVP fusion is intentionally one-to-one: each accepted candidate
becomes a single device. Future passes can group multiple candidates
(text + raster + vector) into a single fused device — the data
structure already supports it via ``source_methods``.

Zone assignment uses the home-run logic in :mod:`app.takeoff.zones`.
"""
from __future__ import annotations

from app.core.ids import stable_id
from app.takeoff.legend_extractor import rules_by_symbol
from app.takeoff.schemas import (
    DeviceInstance,
    LegendRule,
    SheetRecord,
    SymbolCandidate,
)
from app.takeoff.zones import HomeRunZone, assign_home_run


def _legend_rule_id(rule: LegendRule) -> str:
    return stable_id("legrule", rule.raw_symbol, rule.normalized_class)


def fuse_candidates_to_devices(
    *,
    candidates: list[SymbolCandidate],
    sheet: SheetRecord,
    zones: list[HomeRunZone],
    legend_rules: list[LegendRule],
) -> list[DeviceInstance]:
    """Convert accepted candidates on a sheet into devices.

    Rejected candidates are silently skipped — they remain in
    ``TakeoffDocument.candidates`` for the audit trail.
    """
    rule_index = rules_by_symbol(legend_rules)
    devices: list[DeviceInstance] = []

    for cand in candidates:
        if cand.rejection_reason is not None:
            continue
        rule = rule_index.get(cand.raw_symbol)
        if rule is None:
            # No legend rule means we can't classify the device — skip.
            continue

        # Pick a home-run target. For single-floor sheets the device's
        # level is unambiguous; for multi-floor sheets without a
        # device_level we fall through to ambiguity rules.
        device_level = sheet.levels_represented[0] if len(sheet.levels_represented) == 1 else None
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
