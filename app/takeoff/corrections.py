"""Lightweight human-correction loop for takeoff candidates / devices.

The corrections file lives at::

    <PDF_STEM>.derived/takeoff_corrections.json

and (when present) is applied at the end of the pipeline. Supported
operations:

* ``delete_candidate``: drop a SymbolCandidate by ``id``.
* ``add_device``: insert a manually-authored DeviceInstance.
* ``reclassify_candidate``: change a candidate's ``normalized_class``
  (and re-fuse it into a device using the legend rule).

The file is not required — its absence is the normal state. We do not
ship a UI; this is a data-only schema so the takeoff stays
deterministic and replayable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.takeoff.legend_extractor import rules_by_symbol
from app.takeoff.schemas import (
    DeviceInstance,
    LegendRule,
    SheetRecord,
    SymbolCandidate,
)


def _corrections_path(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}.derived") / "takeoff_corrections.json"


def apply_corrections_if_present(
    *,
    pdf_path: Path,
    candidates: list[SymbolCandidate],
    devices: list[DeviceInstance],
    sheets: list[SheetRecord],
    legend_rules: list[LegendRule],
) -> tuple[list[SymbolCandidate], list[DeviceInstance], list[dict[str, Any]]]:
    """Return the post-correction (candidates, devices, applied_log).

    If no corrections file exists the inputs are returned unchanged.
    """
    path = _corrections_path(pdf_path)
    if not path.exists():
        return (candidates, devices, [])

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return (candidates, devices, [])

    entries = payload.get("corrections") or []
    if not isinstance(entries, list):
        return (candidates, devices, [])

    rule_index = rules_by_symbol(legend_rules)
    sheet_index = {s.page_index: s for s in sheets}
    applied: list[dict[str, Any]] = []
    cand_index = {c.id: c for c in candidates}

    out_candidates = list(candidates)
    out_devices = list(devices)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        op = entry.get("op")
        if op == "delete_candidate":
            target_id = entry.get("candidate_id")
            if not target_id:
                continue
            out_candidates = [c for c in out_candidates if c.id != target_id]
            out_devices = [d for d in out_devices if not d.id.startswith("dev_")
                           or _candidate_matches(d, cand_index.get(target_id))]
            applied.append(entry)
        elif op == "add_device":
            try:
                out_devices.append(DeviceInstance(**entry["device"]))
                applied.append(entry)
            except Exception:
                continue
        elif op == "reclassify_candidate":
            target_id = entry.get("candidate_id")
            new_symbol = entry.get("new_symbol")
            if not target_id or not new_symbol:
                continue
            rule = rule_index.get(new_symbol)
            if not rule:
                continue
            for c in out_candidates:
                if c.id == target_id:
                    c.raw_symbol = new_symbol
                    c.normalized_class = rule.normalized_class
                    c.rejection_reason = None
                    applied.append(entry)
                    break

    return (out_candidates, out_devices, applied)


def _candidate_matches(
    device: DeviceInstance, candidate: SymbolCandidate | None
) -> bool:
    """Helper for delete_candidate — return True if device is unrelated."""
    if candidate is None:
        return True
    # Same page + same approximate bbox center => same source.
    if device.page_index != candidate.page_index:
        return True
    cx_d, cy_d = device.bbox.center()
    cx_c, cy_c = candidate.bbox.center()
    return abs(cx_d - cx_c) > 1.0 or abs(cy_d - cy_c) > 1.0


__all__ = ["apply_corrections_if_present"]
