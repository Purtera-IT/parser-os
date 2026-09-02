"""One document, one site: read the site out of a SOW's own header block.

`physical_site` atoms are only emitted when a table looks like a site ROSTER --
a header row of Site ID / Facility / Address and one site per data row. A
per-site SOW is the opposite shape: one document IS one site, and it says so in
a labelled property block:

    Cost Center/Loc Name | Marion County School District Johnakin Middle School
    Address Line 1       | 601 Gurley St
    City | Marion | State | SC | Zip Code | 29571

That correctly fails the roster test, so ten per-site SOWs contributed zero
sites. On deal 010215 the emails said "10 sites" eight times, ten per-site scope
documents arrived, and the site layer resolved two -- because the only sites it
could see were the ones scraped out of email prose, arriving fused
("601 gurley street 1205 south main street") and truncated ("1123" -> "123").

The document already answers the question. Nothing had to infer it; something
had to read it.

Requires the merged-header cell fix: before it, `dict(zip(header, cells))`
collapsed these four-cell rows to one value and the address was destroyed.
"""

from __future__ import annotations

import re
from typing import Any

_LABELS: dict[str, tuple[str, ...]] = {
    "name": ("cost center/loc name", "location name", "site name", "facility name",
             "loc name", "store name", "building name"),
    "address": ("address line 1", "address", "street address", "site address",
                "address 1", "street"),
    "address2": ("address line 2", "address 2", "suite", "unit"),
    "city": ("city", "town"),
    "state": ("state", "province", "region"),
    "zip": ("zip code", "zip", "postal code", "postcode"),
    "site_id": ("cost center/loc #", "cost center", "loc #", "site id", "store #",
                "location #", "site #"),
}

_NOISE = re.compile(r"^(n/?a|none|tbd|-|—|)$", re.I)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _match_label(text: str) -> str | None:
    t = _norm(text).rstrip(":").lower()
    for field, names in _LABELS.items():
        if t in names:
            return field
    return None


def fields_from_property_row(cells: dict | None) -> dict[str, str]:
    """Read label→value pairs out of one property-block row.

    Cells arrive positionally ordered as label, value, label, value… The keys
    are useless here -- they are the merged section header repeated, which is
    why they collide -- so pairing is done on ORDER, and only labels we
    recognise are taken. A value is never inferred from position alone.
    """
    if not isinstance(cells, dict) or not cells:
        return {}
    values = [_norm(v) for v in cells.values()]
    out: dict[str, str] = {}
    for i, cell in enumerate(values):
        field = _match_label(cell)
        if not field:
            continue
        value = values[i + 1] if i + 1 < len(values) else ""
        # The next cell must be a value, not another label, or a blank field
        # would silently swallow the label that follows it.
        if not value or _match_label(value) or _NOISE.match(value):
            continue
        out.setdefault(field, value)
    return out


def site_from_property_rows(rows: list[dict | None]) -> dict[str, str] | None:
    """Merge a document's property rows into one site, or None if it is not one.

    Requires a name or an address: a block carrying only a city and state names
    no place, and emitting it would create a site the document never asserted.
    """
    merged: dict[str, str] = {}
    for cells in rows or []:
        for k, v in fields_from_property_row(cells).items():
            merged.setdefault(k, v)
    if not merged.get("name") and not merged.get("address"):
        return None
    return merged


def site_display_name(site: dict[str, str]) -> str:
    """A human label for the site, preferring what the document called it."""
    name = _norm(site.get("name", "")).replace("\n", " ")
    addr = " ".join(x for x in (site.get("address", ""), site.get("city", ""),
                                site.get("state", ""), site.get("zip", "")) if x)
    return name or addr or "site"
