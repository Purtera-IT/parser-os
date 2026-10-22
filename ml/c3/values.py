"""Structured consumers for extract tabs and open value strings.

These grammars match the parser's own readers (task hours, commercial terms,
commercial-term lines). This module does not import ``app``; tests lock the
two copies together. A string that does not parse is not guessed.
"""
from __future__ import annotations

import math
import re
from typing import Any

# Every pack in app/core/atom_types.json service_packs. A name outside this
# list is rejected, not folded into "other".
DEAL_SERVICES = (
    "audio_visual", "low_voltage_cabling", "wireless", "staff_augmentation",
    "security_camera", "camera_vms_operations", "security_access", "fire_safety",
    "paging_mass_notification", "das", "electrical", "telecom", "network_maintenance",
    "building_management_systems", "datacenter", "rack_and_stack", "hardware",
    "imac", "itad", "data_migration", "msp", "alm", "audit", "professional_services",
    "other",
)

#: What each primary service means: the answer texts of deal:primary_service.
#: OWNER-CHECK: alm, camera_vms_operations and msp were drafted from the value
#: names alone (no training deal uses them); confirm them with the owner.
DEAL_SERVICE_DESC = {
    "audio_visual": "Installing displays, projectors, video-conferencing systems, kiosks or speakers and making them work; any cabling is only there to serve them.",
    "low_voltage_cabling": "Running, terminating, labelling and testing data or voice cable (copper or fibre), with the racks, cabinets, patch panels and backboards that hold it.",
    "wireless": "Wireless networks: surveying, installing or moving access points and the runs that feed them.",
    "staff_augmentation": "Technicians supplied by the day or the hour for recurring or on-call on-site support under the customer's direction, not a defined install.",
    "security_camera": "Installing and connecting security cameras, with their mounts, power and runs.",
    "camera_vms_operations": "Operating or maintaining a video system after the install: configuration, health checks, footage requests.",  # OWNER-CHECK
    "security_access": "Access control: card readers, door hardware, controllers and intercom entry.",
    "fire_safety": "Fire alarm, detection or suppression work.",
    "paging_mass_notification": "Paging, overhead speakers and mass-notification systems.",
    "das": "A distributed antenna system or cellular signal boosting inside a building.",
    "electrical": "Line-voltage electrical work: circuits, outlets, panels.",
    "telecom": "Telephony and carrier work: phone systems, handsets, voice migrations, circuits and demarcation points.",
    "network_maintenance": "Ongoing upkeep of an existing network: switch and router changes, troubleshooting, scheduled maintenance.",
    "building_management_systems": "Building controls: HVAC, lighting or sensor controllers and their networks.",
    "datacenter": "Work inside a data centre: build-outs, moves, structured power and cooling for computer rooms.",
    "rack_and_stack": "Mounting servers and network gear in racks and cabling them, with no wider install.",
    "hardware": "Supplying equipment is the main thing sold; labour is incidental.",
    "imac": "Install, move, add or change of end-user and point-of-sale devices: computers, Chromebooks, registers, printers, peripherals, including removing and packing old ones.",
    "itad": "Retiring IT assets: decommission, data wipe, recycling or resale.",
    "data_migration": "Moving data or systems from one platform to another.",
    "msp": "A standing managed-services contract: remote monitoring and support for a monthly fee.",  # OWNER-CHECK
    "alm": "Asset lifecycle management: tagging, tracking and refresh logistics for a fleet of devices.",  # OWNER-CHECK
    "audit": "The deliverable is a survey, inventory or assessment of what is on site, not an install.",
    "professional_services": "Design, consulting or project management with no field install.",
    "other": "A field service none of the others names.",
}
assert tuple(DEAL_SERVICE_DESC) == DEAL_SERVICES

BILLING_TYPES = ("fixed", "t_and_m", "milestone", "per_site")
COMMERCIAL_NUMBERS = ("pm_hours", "pc_hours", "travel_days")

TERM_KINDS = (
    "payment_term", "contract_term", "change_order_rule", "pricing_assumption",
    "commercial_total", "not_a_term",
)
TERM_ENUMS = {
    "renewal": ("auto", "manual", "none"),
    "timing": ("arrears", "advance", "on_completion", "milestone"),
    "tax": ("included", "excluded", "at_invoice"),
    "approval": ("written", "verbal"),
}
TERM_NUMBERS = {
    "term_months": False,
    "notice_days": True,
    "net_days": True,
    "min_hours": True,
    "increment_min": True,
    "after_hours_multiplier": True,
    "rate_lock_months": True,
}
TERM_KIND_KEYS = {
    "contract_term": ("term_months", "renewal", "notice_days"),
    "payment_term": ("net_days", "timing", "tax"),
    "pricing_assumption": ("min_hours", "increment_min", "after_hours_multiplier", "rate_lock_months"),
    "change_order_rule": ("notice_days", "approval"),
    "commercial_total": (),
    "not_a_term": (),
}
TERM_LINE_TYPES = frozenset({
    "payment_term", "contract_term", "change_order_rule", "commercial_total", "pricing_assumption",
})

UNSTATED = "unstated"
HOURS_BASES = ("stated", "estimate", "deal_kit")
LITERAL_HEADS = frozenset({"norm", "terminology"})

_WORD = re.compile(r"[a-z0-9$\"]+|\[mask\]")


def _finite(value: float) -> bool:
    return math.isfinite(value)


def parse_commercial_verdict(verdict: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for part in str(verdict or "").split(";"):
        key, sep, val = part.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if key == "billing":
            if val.lower() in BILLING_TYPES:
                out["billing"] = val.lower()
        elif key in COMMERCIAL_NUMBERS:
            try:
                number = float(val)
            except ValueError:
                continue
            if _finite(number) and number >= 0:
                out[key] = number
    return out or None


def _clean_term(key: str, val: Any) -> Any:
    if key in TERM_ENUMS:
        text = str(val if val is not None else "").strip().lower()
        return text if text in TERM_ENUMS[key] else None
    if key in TERM_NUMBERS:
        if isinstance(val, bool) or val is None:
            return None
        try:
            number = float(str(val).strip())
        except (TypeError, ValueError):
            return None
        if not _finite(number) or number < 0 or (number == 0 and not TERM_NUMBERS[key]):
            return None
        return int(number) if number.is_integer() else number
    return None


def parse_term_verdict(verdict: Any) -> dict[str, Any] | None:
    pairs: list[tuple[str, str]] = []
    for part in str(verdict or "").split(";"):
        key, sep, val = part.partition("=")
        if sep:
            pairs.append((key.strip().lower(), val.strip()))
    kinds = [v.lower() for k, v in pairs if k == "kind"]
    kind = kinds[-1] if kinds else ""
    if kind not in TERM_KIND_KEYS:
        return None
    out: dict[str, Any] = {"kind": kind}
    for key, val in pairs:
        if key not in TERM_KIND_KEYS[kind]:
            continue
        cleaned = _clean_term(key, val)
        if cleaned is not None:
            out[key] = cleaned
    return out


def canonical_term_verdict(verdict: Any) -> str | None:
    parsed = parse_term_verdict(verdict)
    if parsed is None:
        return None
    kind = str(parsed.pop("kind"))
    parts = [f"kind={kind}"]
    for key in TERM_KIND_KEYS[kind]:
        cleaned = parsed.get(key)
        if cleaned is None:
            continue
        parts.append(f"{key}={cleaned:g}" if isinstance(cleaned, float) else f"{key}={cleaned}")
    return ";".join(parts)


def parse_hours_verdict(verdict: str) -> dict[str, Any] | None:
    text = str(verdict or "").strip()
    if not text or text.lower() == UNSTATED:
        return None
    out: dict[str, Any] = {}
    for part in text.split(";"):
        key, sep, val = part.partition("=")
        if not sep:
            continue
        key, val = key.strip().lower(), val.strip()
        if key == "hours":
            try:
                number = float(val)
            except ValueError:
                return None
            if not _finite(number) or number < 0:
                return None
            out["hours"] = number
        elif key == "qty":
            try:
                qty = float(val)
            except ValueError:
                continue
            if _finite(qty) and qty > 0:
                out["qty"] = qty
        elif key == "per" and val:
            out["per"] = val
        elif key == "role" and val:
            out["role"] = val
        elif key == "basis":
            out["basis"] = val.lower()
    if "hours" not in out:
        return None
    if "qty" in out and "per" not in out:
        out.pop("qty", None)
    return out


def hours_judgment_basis(verdict: str) -> str | None:
    text = str(verdict or "").strip()
    if text.lower() == UNSTATED:
        return UNSTATED
    parsed = parse_hours_verdict(text)
    if parsed is None:
        return None
    return str(parsed.get("basis") or "").strip().lower()


def hours_canonical(parsed: dict[str, Any]) -> str:
    """The stated value, without the basis that only chooses the layer."""
    parts = [f"hours={float(parsed['hours']):g}"]
    if parsed.get("per"):
        parts.append(f"per={parsed['per']}")
        if parsed.get("qty"):
            parts.append(f"qty={float(parsed['qty']):g}")
    if parsed.get("role"):
        parts.append(f"role={parsed['role']}")
    return ";".join(parts)


def canonical_commercial(parsed: dict[str, Any]) -> str:
    """Parsed kit shape only. Keys the grammar does not know are left out."""
    parts = []
    if parsed.get("billing"):
        parts.append(f"billing={parsed['billing']}")
    for key in COMMERCIAL_NUMBERS:
        if key in parsed:
            parts.append(f"{key}={float(parsed[key]):g}")
    return ";".join(parts)


def judged_atom_type(judgment: dict[str, Any]) -> str:
    target = judgment.get("target") if isinstance(judgment.get("target"), dict) else {}
    summary = target.get("a") if isinstance(target.get("a"), dict) else {}
    return str(summary.get("atomType") or "").strip()


def word_indexes(line: str, value: str) -> tuple[int, ...] | None:
    """Unique contiguous word span of ``value`` inside ``line``, else None.

    The word pattern matches ``text.word_spans``. A value that is not a unique
    span is not given a guessed location.
    """
    words = [m.group(0) for m in _WORD.finditer(str(line or "").lower())]
    wanted = [m.group(0) for m in _WORD.finditer(str(value or "").lower())]
    if not words or not wanted or len(wanted) > len(words):
        return None
    hits = [i for i in range(len(words) - len(wanted) + 1) if words[i:i + len(wanted)] == wanted]
    if len(hits) != 1:
        return None
    start = hits[0]
    return tuple(range(start, start + len(wanted)))


def site_count(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if _finite(number) and number >= 0 else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text.split()[0])
    except (ValueError, IndexError):
        return None
    return number if _finite(number) and number >= 0 else None
