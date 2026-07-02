"""Anchor orphan field-work atoms to the deal's sole confirmed job site.

Single-facility deals (common for install/SOW packages) often parse every task
with ``device:*`` keys but no ``site:*`` anchor because section headings never
repeat the city name. After ``semantic_dedup`` the roster is final; this pass
links tasks and site notes when exactly one ``physical_site`` canonical key
exists — never on multi-site deals.
"""

from __future__ import annotations

from typing import Any

_ANCHOR_TYPES = frozenset({"task", "site_implementation_note"})


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _canonical_site_keys(atoms: list[Any]) -> list[str]:
    """Distinct ``site:*`` keys from surviving ``physical_site`` atoms."""
    seen: list[str] = []
    for atom in atoms:
        if _atom_type_str(atom) != "physical_site":
            continue
        keys = [str(k) for k in (getattr(atom, "entity_keys", None) or []) if str(k).startswith("site:")]
        if not keys:
            val = getattr(atom, "value", None) or {}
            if isinstance(val, dict):
                sid = str(val.get("site_id") or val.get("id") or "").strip()
                if sid:
                    import re

                    slug = re.sub(r"[^a-z0-9]+", "_", sid.lower()).strip("_")
                    if slug:
                        keys = [f"site:{slug}"]
        for k in keys:
            if k not in seen:
                seen.append(k)
    return seen


def anchor_orphan_atoms_to_confirmed_site(atoms: list[Any]) -> tuple[list[Any], int]:
    """When exactly one job site is confirmed, attach it to site-less tasks/notes."""
    site_keys = _canonical_site_keys(atoms)
    if len(site_keys) != 1:
        return atoms, 0

    target = site_keys[0]
    linked = 0
    for atom in atoms:
        if _atom_type_str(atom) not in _ANCHOR_TYPES:
            continue
        existing = list(getattr(atom, "entity_keys", None) or [])
        if any(str(k).startswith("site:") for k in existing):
            continue
        atom.entity_keys = sorted(set(existing) | {target})
        flags = list(getattr(atom, "review_flags", None) or [])
        if "single_site_anchor" not in flags:
            flags.append("single_site_anchor")
            atom.review_flags = flags
        linked += 1
    return atoms, linked


__all__ = ["anchor_orphan_atoms_to_confirmed_site"]
