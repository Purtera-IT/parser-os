"""Turn :class:`DeviceInstance` records into :class:`QuoteLine` rollups.

For each device's normalized_class we look up its rule in
``rules/quote_units.yaml`` and emit one or more QuoteLines per device.
Quantities use the device's floor multiplier so a single device on a
nine-floor sheet (T1.06) becomes 9 drops, 9 ports, 9 tests, 9 label
pairs, and 90 ft of service loop allowance — exactly what the spec
calls for.

Rollups group by (item_key, system, floor_label, home_run_to). This
keeps the line count manageable for human review while preserving the
zone / floor breakdown that field installers care about.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from app.core.ids import stable_id
from app.takeoff.schemas import DeviceInstance, QuoteLine

_DEFAULT_YAML = Path(__file__).resolve().parent / "rules" / "quote_units.yaml"


def _load_rules(path: Path = _DEFAULT_YAML) -> dict[str, Any]:
    if not path.exists():
        return {"rules": []}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {"rules": []}


def _rules_by_class(rules_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rule in rules_data.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        cls = rule.get("normalized_class")
        if not cls:
            continue
        out[str(cls)] = rule
    return out


def quote_lines_for_devices(devices: list[DeviceInstance]) -> list[QuoteLine]:
    """Roll devices up into QuoteLines.

    The output is sorted (by item_key, floor_label, home_run_to) so the
    same input always produces the same output.
    """
    rules_data = _load_rules()
    rules_by_class = _rules_by_class(rules_data)

    # Group key: (item_key, system, floor_label or "", home_run_to or "")
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "quantity": 0.0,
            "unit": "each",
            "description": "",
            "system": None,
            "device_ids": [],
            "notes": [],
        }
    )

    for device in devices:
        rule = rules_by_class.get(device.normalized_class)
        if rule is None:
            # Generic fallback — one per device.
            item_key = f"{device.normalized_class}_drop"
            description = f"{device.normalized_class} (generic device drop)"
            unit = "each"
            key = (
                item_key,
                device.system or "",
                device.floor_label or "",
                device.home_run_to or "",
            )
            slot = grouped[key]
            slot["description"] = description
            slot["unit"] = unit
            slot["system"] = device.system
            slot["quantity"] += float(device.multiplier)
            slot["device_ids"].append(device.id)
            continue

        # Each rule line emits ``quantity_per_device * multiplier`` units.
        for line in rule.get("lines") or []:
            item_key = str(line.get("item_key"))
            description = str(line.get("description") or item_key)
            unit = str(line.get("unit") or "each")
            qpd = float(line.get("quantity_per_device") or 1)
            qty = qpd * float(device.multiplier)
            key = (
                item_key,
                device.system or rule.get("system") or "",
                device.floor_label or "",
                device.home_run_to or "",
            )
            slot = grouped[key]
            slot["description"] = description
            slot["unit"] = unit
            slot["system"] = device.system or rule.get("system")
            slot["quantity"] += qty
            slot["device_ids"].append(device.id)
            for note in rule.get("notes") or []:
                if note not in slot["notes"]:
                    slot["notes"].append(note)

    quote_lines: list[QuoteLine] = []
    for (item_key, _system, floor_label, home_run_to), slot in grouped.items():
        line_id = stable_id(
            "qline",
            item_key,
            floor_label or "*",
            home_run_to or "*",
        )
        quote_lines.append(
            QuoteLine(
                item_key=item_key,
                description=slot["description"],
                quantity=float(slot["quantity"]),
                unit=slot["unit"],
                system=slot["system"],
                floor_label=floor_label or None,
                home_run_to=home_run_to or None,
                source_device_ids=list(slot["device_ids"]),
                confidence=0.9,
                notes=list(slot["notes"]),
            )
        )
        # Reference the line_id via item_key so future readers can find
        # it (the public schema doesn't need a separate id field — the
        # group key already disambiguates).
        del line_id

    quote_lines.sort(
        key=lambda q: (q.item_key, q.floor_label or "", q.home_run_to or "")
    )
    return quote_lines


__all__ = ["quote_lines_for_devices"]
