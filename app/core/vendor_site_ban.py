"""Deterministic ban on PurTera corporate / vendor addresses as job sites.

PurTera LLC's Alpharetta office appears on every SOW letterhead, signature
block, and ACCEPTANCE CRITERIA vendor row. Regex and roster heuristics mint it
as ``physical_site:alpharetta_ga_30009`` even when the deal has zero customer
install locations. This module is the hard gate: known vendor-address patterns
and header/letterhead section paths are dropped *before* LLM vendor
suppression and even when they are the deal's only ``physical_site``.

Aligned with ``feedback_store.seed_default_corrections`` global exemplar
``global_purtera_self_address``.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical markers from feedback_store global_purtera_self_address seed.
_PURTERA_ADDRESS_MARKERS = (
    "amber park",
    "11720",
    "purtera llc",
    "purtera hq",
    "purtera headquarters",
)

# Section / block contexts where a street address is almost never a job site.
_VENDOR_SECTION_HINTS = (
    "acceptance criteria",
    "letterhead",
    "header",
    "footer",
    "services by",
    "agreed by",
    "signature",
    "quoted by",
    "prepared by",
    "billing address",
    "vendor",
)

_GLOBAL_PURTERA_CORRECTION_ID = "global_purtera_self_address"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _section_path_from_atom(atom: Any) -> list[str]:
    for ref in getattr(atom, "source_refs", None) or []:
        loc = getattr(ref, "locator", None) or {}
        if isinstance(loc, dict):
            sp = loc.get("section_path")
            if sp:
                return [str(x) for x in sp]
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        sp = val.get("section_path")
        if sp:
            return [str(x) for x in sp]
    return []


def site_text_blob(
    atom: Any | None = None,
    *,
    text: str = "",
    section_path: list[str] | None = None,
    value: dict[str, Any] | None = None,
) -> str:
    """Concatenate address-bearing fields for pattern matching."""
    parts: list[str] = []
    if atom is not None:
        parts.append(str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""))
        val = getattr(atom, "value", None) or {}
        if isinstance(val, dict):
            value = val
        sp = _section_path_from_atom(atom)
        if sp:
            parts.extend(sp)
    if text:
        parts.append(text)
    if section_path:
        parts.extend(str(x) for x in section_path)
    if isinstance(value, dict):
        for key in (
            "address",
            "street_address",
            "name",
            "facility_name",
            "city",
            "state",
            "zip",
            "city_state",
            "source_context",
        ):
            v = value.get(key)
            if v:
                parts.append(str(v))
    return _norm(" ".join(parts))


def section_path_suggests_vendor_block(
    atom: Any | None = None,
    *,
    text: str = "",
    section_path: list[str] | None = None,
) -> bool:
    """True when headings/locators place content in letterhead or vendor rows."""
    sp = list(section_path or [])
    if atom is not None and not sp:
        sp = _section_path_from_atom(atom)
    sp_text = _norm(" ".join(sp))
    blob = site_text_blob(atom, text=text, section_path=sp)
    for hint in _VENDOR_SECTION_HINTS:
        if hint in sp_text or hint in blob:
            return True
    return False


def is_purtera_vendor_address(
    atom: Any | None = None,
    *,
    text: str = "",
    section_path: list[str] | None = None,
    value: dict[str, Any] | None = None,
) -> bool:
    """True when text matches known PurTera corporate office patterns."""
    blob = site_text_blob(atom, text=text, section_path=section_path, value=value)
    if not blob:
        return False

    if any(m in blob for m in _PURTERA_ADDRESS_MARKERS):
        if "11720" in blob or "amber park" in blob:
            return True
        if "purtera llc" in blob or "purtera hq" in blob or "purtera headquarters" in blob:
            return True
        if "purtera" in blob and "alpharetta" in blob and "30009" in blob:
            return True

    if "11720" in blob and "alpharetta" in blob:
        return True
    if "amber park" in blob and "alpharetta" in blob:
        return True
    if "purtera" in blob and "alpharetta" in blob and "30009" in blob:
        return True

    return False


def is_banned_vendor_physical_site(atom: Any) -> bool:
    """True when this ``physical_site`` atom must never ship as a customer job site."""
    if _atom_type_str(atom) != "physical_site":
        return False

    if is_purtera_vendor_address(atom):
        return True

    if section_path_suggests_vendor_block(atom):
        blob = site_text_blob(atom)
        if "purtera" in blob and (
            "alpharetta" in blob or "11720" in blob or "amber park" in blob or "30009" in blob
        ):
            return True
        if "11720" in blob and "30009" in blob:
            return True

    return False


def _stamp_banned(atom: Any) -> None:
    try:
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            val["_decision"] = {
                "source": "deterministic",
                "correction_id": _GLOBAL_PURTERA_CORRECTION_ID,
                "confidence": 1.0,
            }
    except Exception:  # pragma: no cover - provenance must never break compile
        pass


def drop_banned_vendor_physical_sites(atoms: list[Any]) -> tuple[list[Any], int]:
    """Remove deterministic vendor-ban ``physical_site`` atoms.

    Unlike LLM vendor suppression, this runs even when the banned site is the
    deal's only locational anchor — PurTera HQ is never a valid job site.
    """
    drop_ids: set[str] = set()
    for atom in atoms:
        if not is_banned_vendor_physical_site(atom):
            continue
        aid = getattr(atom, "id", None)
        if aid:
            drop_ids.add(aid)
            _stamp_banned(atom)

    if not drop_ids:
        return atoms, 0

    kept = [a for a in atoms if getattr(a, "id", None) not in drop_ids]
    return kept, len(drop_ids)


__all__ = [
    "drop_banned_vendor_physical_sites",
    "is_banned_vendor_physical_site",
    "is_purtera_vendor_address",
    "section_path_suggests_vendor_block",
    "site_text_blob",
]
