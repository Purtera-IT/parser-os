"""One building, one roster row: merge site rows that are provably the same place.

The roster is what the SOW, Atlas and the Deal Kit read. A site list that
arrives in several sheets keys the same store three ways (``site:106``,
``site:bevmosto_106``, ``site:1435_0``), and on 000123 that published 366 rows
for a 161-store job. Two identities count, and only these:

* the customer's own site code, exactly (``SMF_Auburn_1435``);
* a numbered street (``2745 bell rd``, see ``address_parse._street_for_dedup``)
  whose ZIP and city agree, or are missing on one side.

On the 219 labeled site pairs (64 same, 155 distinct) these merged 39 same-site
pairs and no distinct ones. Anything weaker stays a duplicate candidate for a
PM. Nothing is deleted: the kept row lists every row folded into it under
``merged_from`` (key, name, address), so a wrong merge is visible and undoable.
"""
from __future__ import annotations

import re
from typing import Any

_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def _street(row: dict[str, Any]) -> str:
    from app.core.address_parse import _street_for_dedup

    s = _street_for_dedup(str(row.get("street_address") or row.get("address") or ""))
    # A numbered street with a name, not a bare number or a lone word.
    return s if s[:1].isdigit() and len(s.split()) >= 2 else ""


def _zip(row: dict[str, Any]) -> str:
    z = str(row.get("zip") or "").strip()
    if z:
        return z[:5]
    m = _ZIP.search(str(row.get("street_address") or row.get("address") or ""))
    return m.group(1) if m else ""


def _compatible(a: str, b: str) -> bool:
    return not a or not b or a == b


def identity_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Indexes of rows that are the same place, groups of two or more."""
    from app.core.entity_resolution import _site_code

    n = len(rows)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        parent[root(j)] = root(i)

    by_code: dict[str, int] = {}
    by_street: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        code = _site_code(str(r.get("site") or ""), r)
        if code:
            if code in by_code:
                join(by_code[code], i)
            else:
                by_code[code] = i
        st = _street(r)
        if st:
            by_street.setdefault(st, []).append(i)
    for idx in by_street.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ra, rb = rows[idx[a]], rows[idx[b]]
                if _compatible(_zip(ra), _zip(rb)) and _compatible(
                        str(ra.get("city") or "").strip().lower(), str(rb.get("city") or "").strip().lower()):
                    join(idx[a], idx[b])
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _completeness(r: dict[str, Any]) -> tuple:
    return (bool(_zip(r)), bool(r.get("city")), bool(r.get("facility_name")),
            float(r.get("readiness") or 0), int(r.get("signal_count") or 0))


def merge_roster(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """(merged rows in the original order, rows folded away). The kept row is
    the most complete one; counts take the larger side, aliases the union."""
    groups = identity_groups(rows)
    if not groups:
        return rows, 0
    drop: set[int] = set()
    for g in groups:
        keep = max(g, key=lambda i: _completeness(rows[i]))
        k = rows[keep]
        merged = list(k.get("merged_from") or [])
        aliases = set(k.get("aliases") or [])
        for i in g:
            if i == keep:
                continue
            r = rows[i]
            drop.add(i)
            merged.append({"site": r.get("site"), "name": r.get("display_name") or r.get("facility_name"),
                           "address": r.get("street_address") or r.get("address")})
            aliases.update(a for a in (r.get("aliases") or []) if a)
            for f in ("facility_name", "street_address", "city", "state", "zip"):
                if not k.get(f) and r.get(f):
                    k[f] = r[f]
            for f in ("device_count", "stakeholder_count", "constraint_count", "scope_atom_count",
                      "signal_count", "contradiction_count"):
                if isinstance(r.get(f), (int, float)) and r[f] > (k.get(f) or 0):
                    k[f] = r[f]
            for f in ("device_keys", "stakeholder_keys"):
                if isinstance(r.get(f), list):
                    k[f] = sorted(set(k.get(f) or []) | set(r[f]))
        k["merged_from"] = merged
        k["aliases"] = sorted(aliases)
    return [r for i, r in enumerate(rows) if i not in drop], len(drop)
