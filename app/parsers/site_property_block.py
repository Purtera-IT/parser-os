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
    # NOT "site_id". This code is frequently the ACCOUNT's, not the site's: all
    # ten 010215 SOWs carry 94575001, the district's. Naming it site_id made
    # semantic_dedup — which reads `site_id` before `id` — collapse ten schools
    # into one. A field that claims to identify a site must actually do so.
    "cost_center": ("cost center/loc #", "cost center", "loc #", "site id",
                    "store #", "location #", "site #"),
    # Not identity, but stated per-site and commercially material. A raw-vs-
    # captured check against the ten 010215 SOWs showed the site atom was
    # carrying 6 of the 9 labelled fields; these are the other three. Tax-exempt
    # status changes what may be billed, and segment routes the work — dropping
    # them means a downstream reader has to re-open the document to find facts
    # the parser already had in hand.
    "country": ("country",),
    "segment": ("segment", "business segment", "vertical"),
    "tax_exempt": ("is location tax exempt?", "is location tax exempt",
                   "tax exempt", "tax exempt?"),
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


def _shape_fields(rows: list[dict | None]) -> dict[str, str]:
    """Fields read from the VALUES, with no label vocabulary at all.

    A label whitelist is one vendor wide: the next customer writes "Site
    Address" or leaves the cell unlabelled and the reader silently finds
    nothing. Shape survives that -- "601 Gurley St" is an address in any
    document, under any label.

    Measured on the ten 010215 SOWs: shape alone finds 10/10 addresses with
    exactly one candidate each. Labels are still consulted afterwards, but only
    for what shape cannot decide (a site's NAME looks like ordinary text), and
    never to overrule a value that typed itself.
    """
    from app.parsers.value_shapes import classify_value, street_addresses

    flat: list[str] = []
    for cells in rows or []:
        if isinstance(cells, dict):
            flat.extend(str(v or "") for v in cells.values())

    out: dict[str, str] = {}
    addrs = street_addresses(flat)
    if addrs:
        out["address"] = addrs[0]
    else:
        # No English street suffix — "4820 Camino Del Rio", "100 Broadway".
        # A number-led phrase is accepted ONLY when the block also carries a
        # postal code or a state, which is what separates a place from a
        # quantity like "1 UKG DX Clock".
        from app.parsers.value_shapes import candidate_addresses, looks_like_site_block as _lsb
        has_geo = any(classify_value(c) in ("postal", "state") for c in flat)
        if has_geo:
            cands = candidate_addresses(flat)
            if cands:
                out["address"] = cands[0]
    for cell in flat:
        kind = classify_value(cell)
        if kind == "postal":
            out.setdefault("zip", cell.strip())
        elif kind == "state":
            out.setdefault("state", cell.strip())
    return out


def site_from_property_rows(rows: list[dict | None]) -> dict[str, str] | None:
    """Merge a document's property rows into one site, or None if it is not one.

    Requires a name or an address: a block carrying only a city and state names
    no place, and emitting it would create a site the document never asserted.
    """
    # Shape first: a value that types itself needs no label to be trusted, and
    # cannot be missed because a vendor renamed the field.
    merged: dict[str, str] = dict(_shape_fields(rows))
    # Labels second, and only where shape was silent — a site's NAME, its
    # segment, its tax status all look like ordinary text and cannot be typed by
    # shape. setdefault means a label never overrules a typed value.
    for cells in rows or []:
        for k, v in fields_from_property_row(cells).items():
            merged.setdefault(k, v)
    # A vendor who labels the field differently still gets a name: structure
    # finds it where vocabulary cannot.
    if not merged.get("name"):
        structural = site_name_from_block(rows)
        if structural:
            merged["name"] = structural
    if not merged.get("name") and not merged.get("address"):
        return None
    return merged


def site_name_from_block(rows: list[dict | None]) -> str | None:
    """The site's name, found by STRUCTURE rather than by a label whitelist.

    In a property block the labels are short and the values are not: "City",
    "Address Line 1", "Cost Center/Loc #" against "Marion County School District
    Johnakin Middle School". The name is the longest run of text that types as
    nothing (not an address, postal code, state, phone or email), carries
    several words, and does not read as a label itself.

    Measured on the ten 010215 SOWs: this picks the school every time, with no
    knowledge that the field is called "Cost Center/Loc Name". A vendor calling
    it "Site", "Location" or nothing at all is read identically.
    """
    from app.parsers.value_shapes import classify_value

    seen: list[str] = []
    for cells in rows or []:
        if not isinstance(cells, dict):
            continue
        for v in cells.values():
            t = _norm(v)
            if t and t not in seen:
                seen.append(t)

    cands = [
        t for t in seen
        if not classify_value(t)
        and len(t.split()) >= 3
        # A label announces a field; it does not name a place.
        and not t.rstrip().endswith(("#", ":", "?"))
    ]
    return max(cands, key=len) if cands else None


def site_key(site: dict[str, str]) -> str:
    """A stable per-site key, from the identity the DOCUMENT states.

    site_readiness keys on ``value["id"] or value["site_id"]`` and skips a site
    that has neither -- deliberately, because an address-derived slug produced
    ghost sites like "site:address_1180_peachtree_st". That guard is right.

    But the cost-centre number is not a site id here. All ten 010215 SOWs carry
    the same one, 94575001, because it identifies the DISTRICT, not the school.
    Keying on it collapses ten schools into one, which is the same information
    loss as dropping nine of them.

    The location NAME is different: it is a labelled field the document fills in
    per site ("Cost Center/Loc Name | Marion County School District Johnakin
    Middle School"), not a string a regex guessed out of prose. Two schools
    sharing an address -- Marion HS and Marion Intermediate both sit at 1205 S
    Main St -- stay distinct under it, correctly, because they are distinct
    schools.

    Falls back to the cost centre, then the address, so a block that names no
    location still keys on something the document asserted.
    """
    slug = lambda t: re.sub(r"[^a-z0-9]+", "_", str(t or "").lower()).strip("_")
    name = _norm(site.get("name", ""))
    addr = _norm(site.get("address", ""))

    # Address FIRST, then a name tail.
    #
    # Two reasons, both learned the hard way. semantic_dedup only treats an id as
    # canonical when it contains a digit (`_looks_complete_site_id`, written for
    # enterprise codes like ATL-HQ-01); a purely name-derived key has none, so
    # all ten 010215 sites fell out of the canonical index and collapsed back to
    # two. An address carries its house number, which satisfies that AND is the
    # most site-identifying thing a block contains.
    #
    # The name tail still matters: Marion HS and Marion Intermediate share
    # 1205 S Main St and are two schools. Only the last words are used, because
    # the leading words are the account's ("Marion County School District…") and
    # identical across every site in the deal.
    tail = "_".join(slug(name).split("_")[-3:]) if name else ""
    if addr and tail:
        return f"loc_{slug(addr)}_{tail}"[:90]
    if addr:
        return f"loc_{slug(addr)}"[:90]
    if name:
        return f"loc_{slug(name)}"[:90]

    if site.get("cost_center"):
        return "loc_" + re.sub(r"[^a-z0-9]+", "_", str(site["cost_center"]).lower()).strip("_")
    return ""


def site_display_name(site: dict[str, str]) -> str:
    """A human label for the site, preferring what the document called it."""
    name = _norm(site.get("name", "")).replace("\n", " ")
    addr = " ".join(x for x in (site.get("address", ""), site.get("city", ""),
                                site.get("state", ""), site.get("zip", "")) if x)
    return name or addr or "site"
