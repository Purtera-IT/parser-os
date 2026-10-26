"""Where each roster site is, when the site list itself does not say.

A customer site list often names its sites by id only ("Site No.": dc 5841,
plus y/n work columns) and keeps where they are on another tab of the same
workbook, keyed by the same id (010447: "2027_145_Sites" for the scope,
"annette source" for Site Location / State; the customer's email says to match
them by Site No.). Without the join every site has no city, so nothing can be
routed, travelled to or priced for travel.

Two passes, both extraction (the parser pulls values; no decision is made):
  * ``join_sheet_locations`` -- inside one workbook: a roster row takes the
    city / state / country of the row with the same id on a sibling sheet whose
    header carries the same id column.
  * ``share_roster_locations`` -- across the deal: the same roster sent twice
    (two workbooks) shares what either copy found, so the copy that survives
    dedup has it too.
Pure functions, no I/O.
"""

from __future__ import annotations

import re
from typing import Any

_CITY_HEADER = re.compile(r"^\s*(city|town|site\s*location|location\s*name|site\s*city|store\s*city)\s*$", re.I)
_STATE_HEADER = re.compile(r"^\s*(state|province|state\s*/\s*prov(ince)?|st|prov|state/province)\s*$", re.I)
_COUNTRY_HEADER = re.compile(r"^\s*(country|location)\s*$", re.I)
_TYPE_HEADER = re.compile(r"^\s*(site\s*type|type)\s*$", re.I)
_COUNTRIES = {"usa": "US", "us": "US", "united states": "US", "canada": "CA", "ca": "CA", "mexico": "MX", "mx": "MX"}
_HEADER_SCAN = 10


def _norm_id(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(v or "").lower())


def _cell(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip() if v is not None else ""


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _roster_ref(atom: Any) -> Any:
    for s in getattr(atom, "source_refs", None) or []:
        if getattr(s, "extraction_method", None) == "xlsx_site_roster_v1":
            return s
    return None


def _is_roster(atom: Any) -> bool:
    return _roster_ref(atom) is not None


def _clean_city(place: str, site_type: str, state: str = "") -> str:
    """The town in a facility's location name: "Keasbey BDC" (type BDC) -> "Keasbey",
    "Carlisle Crown Bolt" (type CB) -> "Carlisle", "Hialeah - Miami DFC" -> "Hialeah"."""
    words = place.split()
    for t in [x for x in re.split(r"[/\s]+", site_type or "") if x]:
        words = [w for w in words if w.lower().strip(",/") != t.lower()]
        # an acronym type spelled out at the end ("CB" -> "Crown Bolt")
        n = len(t)
        if t.isupper() and 1 < n <= 4 and len(words) > n and "".join(w[0] for w in words[-n:]).upper() == t:
            words = words[:-n]
    p = " ".join(words).split(" - ")[0].strip(" ,-/")
    if state and p.upper().endswith(" " + state.upper()):
        p = p[: -len(state) - 1].strip(" ,-/")
    return p or place


def _location_index(rows: list[list[Any]], id_header: str) -> dict[str, dict[str, str]]:
    """{normalized id: {city, state, country}} from a sheet whose header has ``id_header`` and a city column."""
    want = id_header.strip().lower()
    for h in range(min(_HEADER_SCAN, len(rows))):
        header = [_cell(c) for c in rows[h]]
        lower = [c.lower() for c in header]
        if want not in lower:
            continue
        idc = lower.index(want)
        city = next((i for i, c in enumerate(header) if _CITY_HEADER.match(c)), None)
        if city is None:
            return {}
        state = next((i for i, c in enumerate(header) if _STATE_HEADER.match(c)), None)
        country = next((i for i, c in enumerate(header) if _COUNTRY_HEADER.match(c)), None)
        # "Site Type" over a bare "Type" (which may be a phase: "Pilot", "2027")
        stype = next((i for i, c in enumerate(header) if re.match(r"^\s*site\s*type\s*$", c, re.I)), None)
        if stype is None:
            stype = next((i for i, c in enumerate(header) if _TYPE_HEADER.match(c)), None)
        out: dict[str, dict[str, str]] = {}
        for r, row in enumerate(rows[h + 1:], start=h + 1):
            get = lambda i: _cell(row[i]) if i is not None and i < len(row) else ""  # noqa: E731
            key = _norm_id(get(idc))
            place = get(city)
            if not key or not place or key in out:
                continue
            loc = {"city": _clean_city(place, get(stype), get(state)), "site_location": place, "row": str(r + 1)}
            if get(state):
                loc["state"] = get(state)
            c = _COUNTRIES.get(get(country).lower())
            if c:
                loc["country"] = c
            out[key] = loc
        return out
    return {}


def _fill(value: dict, loc: dict[str, str], source: dict[str, str]) -> bool:
    if value.get("city") or value.get("street_address") or value.get("address"):
        return False
    value["city"] = loc["city"]
    if loc.get("state") and not value.get("state"):
        value["state"] = loc["state"]
    if loc.get("country") and not value.get("country"):
        value["country"] = loc["country"]
    if loc.get("site_location") and loc["site_location"] != loc["city"]:
        value.setdefault("site_location", loc["site_location"])
    value["location_from"] = source
    return True


def join_sheet_locations(atoms: list[Any], sheets: list[dict[str, Any]]) -> int:
    """Fill roster atoms' city/state from a sibling sheet of the same workbook keyed by the same id column."""
    filled = 0
    cache: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for atom in atoms:
        if not _is_roster(atom):
            continue
        value = _value(atom)
        cells = value.get("cells") if isinstance(value.get("cells"), dict) else {}
        own_sheet = str((getattr(_roster_ref(atom), "locator", None) or {}).get("sheet") or "")
        sid = value.get("site_id") or value.get("id")
        id_header = next((h for h, v in cells.items() if _norm_id(v) and _norm_id(v) == _norm_id(sid)), None)
        if not id_header or not sid:
            continue
        for sh in sheets:
            name = str(sh.get("name") or "")
            if name == own_sheet:
                continue
            idx = cache.get((name, id_header))
            if idx is None:
                idx = cache[(name, id_header)] = _location_index(sh.get("rows") or [], id_header)
            loc = idx.get(_norm_id(sid))
            if loc and _fill(value, loc, {"sheet": name, "row": loc["row"], "by": id_header}):
                filled += 1
                break
    return filled


def share_roster_locations(atoms: list[Any]) -> int:
    """The same roster site in two files: a copy without a city takes the other copy's."""
    found: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        if not _is_roster(atom):
            continue
        v = _value(atom)
        key = _norm_id(v.get("site_id") or v.get("id"))
        if key and v.get("city") and key not in found:
            found[key] = v
    filled = 0
    for atom in atoms:
        if not _is_roster(atom):
            continue
        v = _value(atom)
        src = found.get(_norm_id(v.get("site_id") or v.get("id")))
        if src is None or src is v:
            continue
        loc = {k: str(src[k]) for k in ("city", "state", "country", "site_location") if src.get(k)}
        if _fill(v, loc, dict(src.get("location_from") or {}, copied=True)):
            filled += 1
    return filled
