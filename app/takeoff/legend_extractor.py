"""Load default legend rules and (eventually) extract per-project ones.

For the v0 MVP this module simply reads ``rules/low_voltage_symbols.yaml``
and exposes the seeded LegendRules. A future pass can extract the
project's own legend by parsing the T0.01 sheet's table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.takeoff.schemas import LegendRule

_DEFAULT_YAML = Path(__file__).resolve().parent / "rules" / "low_voltage_symbols.yaml"


def _load_rules_yaml(path: Path = _DEFAULT_YAML) -> dict[str, Any]:
    if not path.exists():
        return {"defaults": [], "ignore_tokens": []}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def load_default_legend_rules(
    source_page: int | None = None,
    confidence: float = 0.9,
) -> list[LegendRule]:
    """Return the seeded default legend rules.

    Each rule's ``source_page`` is left ``None`` unless the caller
    found a legend page in the PDF (typically T0.01). The
    ``confidence`` field defaults to 0.9 for seeded rules; a
    project-extracted rule would carry a higher value.
    """
    data = _load_rules_yaml()
    defaults = data.get("defaults") or []
    out: list[LegendRule] = []
    for raw in defaults:
        if not isinstance(raw, dict):
            continue
        # Drop YAML-only fields and coerce remarks into a list.
        payload = dict(raw)
        payload.setdefault("remarks", [])
        if not isinstance(payload["remarks"], list):
            payload["remarks"] = [str(payload["remarks"])]
        payload["source_page"] = source_page
        payload["confidence"] = confidence
        out.append(LegendRule(**payload))
    return out


def load_ignore_tokens() -> set[str]:
    """Tokens that look like symbols but must never become devices."""
    data = _load_rules_yaml()
    tokens = data.get("ignore_tokens") or []
    return {str(t).strip() for t in tokens if str(t).strip()}


def rules_by_symbol(rules: list[LegendRule]) -> dict[str, LegendRule]:
    """Index legend rules by ``raw_symbol`` for fast device-fusion lookup."""
    return {r.raw_symbol: r for r in rules}


__all__ = [
    "load_default_legend_rules",
    "load_ignore_tokens",
    "rules_by_symbol",
]
