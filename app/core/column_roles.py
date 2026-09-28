"""What a column holds, read from its values rather than its name.

The person who ran deal 010264 used a 5,450-row inventory workbook for
exactly one thing: getting the site addresses out of it for the SOW. He did
it by hand, and the hard part was reconciling a list of thousands of device
rows down to the set of distinct places a crew has to go.

Nothing in the parser knew the sheet could answer that. It could say what
the table stated and what it left open, but not what it could be ASKED for
-- and "where are the sites" is not a finding or a question, it is a
capability. A retrieval layer that has to be told, per workbook, that Cost
Center means a site is not a retrieval layer; it is a pile of per-deal rules
waiting to rot.

So a sheet declares what it can supply, and the declaration is derived from
the values. Names cannot do this job: "Cost Center" is named like money,
holds a ten-digit account code, and reading it by its name summed it to $77
billion on a deal for 1,563 time clocks. Values can. A column of five-digit
strings beside a column of two-letter uppercase tokens beside a column whose
cells start with a house number is a postal address -- in any workbook, from
any tool, in any industry, with the columns called whatever the exporter
felt like calling them.

Two of the roles here are positional rather than shaped, and deliberately so.
A city is just a word; what makes it a city is that it sits between a street
and a state. That is also how a person reads a strange spreadsheet, and
pretending otherwise would mean shipping a gazetteer and still missing
Henrico.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: How much of a column's filled values must match a shape before the column
#: is said to hold that thing. Real exports carry typos, "N/A" and the odd
#: merged cell, so demanding all of them would detect nothing; demanding half
#: would call a serial-number column a postcode on a bad day.
ROLE_SHARE = 0.75

#: How many values are sampled per column to decide its role. A column's kind
#: is settled long before the hundredth row, and reading more of a
#: 115,000-row export to learn the same answer is wasted time.
ROLE_SAMPLE = 200

_POSTAL_US = re.compile(r"^\d{5}(-\d{4})?$")
_POSTAL_CA = re.compile(r"^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$")
_POSTAL_UK = re.compile(r"^[A-Za-z]{1,2}\d[A-Za-z\d]?[ ]?\d[A-Za-z]{2}$")
_REGION = re.compile(r"^[A-Z]{2,3}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_MAC = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$|^[0-9A-Fa-f]{12}$")
_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_URL = re.compile(r"^(https?://|www\.)\S+$", re.I)
#: A phone number a person could dial. The separator or the leading + is
#: required, and that is the whole point: "0010058001" is a ten-digit string
#: and so is a phone number, and without this a column of Sodexo cost-centre
#: codes was declared as somebody to call.
_PHONE = re.compile(r"^\+\d[\d\s().-]{6,17}$|^[\d(][\d]*[\s().-]+[\d\s().-]{5,}\d$")
_DATE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})")

#: The words a street address ends in, across the anglophone postal systems
#: an export is likely to carry. A street is also recognisable without them
#: -- a house number followed by words -- which is what catches "5500
#: AUDUBON DR" in a file that abbreviates differently.
_STREET_WORD = re.compile(
    r"\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|"
    r"ct|court|cir|circle|pl|place|pkwy|parkway|hwy|highway|ter|terrace|"
    r"sq|square|trl|trail|loop|route|rte|suite|ste|unit|apt|floor|fl|bldg)\b",
    re.I)
_HOUSE_NUMBER = re.compile(r"^\d{1,6}[A-Za-z]?\s+\S")

#: A ZIP is five digits and so is a cost-centre code, a store number and a
#: part number. What separates them is that a postcode column is not the
#: column that identifies the rows -- and that it sits beside a state.
_POSTAL_NEEDS_NEIGHBOUR = True


@dataclass
class Role:
    name: str
    confidence: float
    why: str


@dataclass
class Supply:
    """Something this sheet can be asked for, and how to get it."""
    what: str                        # "site_address", "contact", ...
    columns: dict[str, str]          # part -> column name
    distinct: int = 0                # how many different ones it holds
    rows: int = 0
    key: str | None = None           # the column that names each one
    notes: list[str] = field(default_factory=list)


def _share(values: list[str], test) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if test(v)) / len(values)


def _looks_postal(values: list[str]) -> bool:
    return max(
        _share(values, lambda v: bool(_POSTAL_US.match(v))),
        _share(values, lambda v: bool(_POSTAL_CA.match(v))),
        _share(values, lambda v: bool(_POSTAL_UK.match(v))),
    ) >= ROLE_SHARE


def _looks_street(values: list[str]) -> bool:
    return _share(
        values,
        lambda v: bool(_HOUSE_NUMBER.match(v)) or bool(_STREET_WORD.search(v)),
    ) >= ROLE_SHARE


def classify(
    name: str, values: list[str], *, is_identifier: bool = False
) -> Role | None:
    """What one column holds, from up to ROLE_SAMPLE of its values.

    ``is_identifier`` says the column names the rows. A column that gives
    every row a different value is a key, and a key that happens to be
    numeric is not a phone number or a postcode however much it looks like
    one -- which is how a cost-centre column came to be declared as both.
    """
    vals = [v for v in values if v][:ROLE_SAMPLE]
    if len(vals) < 5:
        return None
    for role, test, why in (
        ("email", lambda v: bool(_EMAIL.match(v)), "addresses with a domain"),
        ("mac", lambda v: bool(_MAC.match(v)), "six hex octets"),
        ("ip", lambda v: bool(_IP.match(v)), "dotted quads"),
        ("url", lambda v: bool(_URL.match(v)), "web addresses"),
        ("date", lambda v: bool(_DATE.match(v)), "calendar dates"),
    ):
        if _share(vals, test) >= ROLE_SHARE:
            return Role(role, _share(vals, test), why)
    if not is_identifier and _looks_postal(vals):
        return Role("postal_code", 1.0, "postal codes")
    if _looks_street(vals):
        return Role("street", 1.0, "house numbers and street types")
    if _share(vals, lambda v: bool(_REGION.match(v))) >= ROLE_SHARE:
        # Two uppercase letters is also a country code, a currency and a
        # status flag. What makes it a region is a small vocabulary over
        # many rows, which a flag ("OK"/"NO") does not reach.
        if len({v.upper() for v in vals}) >= 3:
            return Role("region", 0.8, "two-letter region codes")
    if not is_identifier and _share(vals, lambda v: bool(_PHONE.match(v))) >= ROLE_SHARE:
        return Role("phone", 0.8, "dialable numbers")
    return None


def site_supply(
    columns: list[str],
    roles: dict[str, Role],
    rows: list[list[Any]],
    header_idx: int,
    *,
    key_column: str | None = None,
) -> Supply | None:
    """Can this sheet say where the work is, and at how many places?

    This is the question a SOW actually needs answered, and it is not
    answered by any single column. A street alone is not a site; a state
    alone is a hundred of them. So the sheet supplies sites when it carries
    a street or a postcode with something to place it in, and what it
    supplies is the count of DISTINCT addresses -- not the row count, which
    on this deal was 5,448 devices standing at far fewer doors.
    """
    idx = {c: i for i, c in enumerate(columns)}
    part: dict[str, str] = {}
    for col, role in roles.items():
        if role.name in ("street", "postal_code", "region") and role.name not in part:
            part[role.name] = col

    if "street" not in part and "postal_code" not in part:
        return None
    if len(part) < 2:
        return None

    # Everything left is read POSITIONALLY, between the two parts whose
    # shape is unmistakable. A city is just a word -- what makes it a city
    # is sitting between a street and a postcode -- and a gazetteer big
    # enough to know Henrico and Willcox would still miss the next one.
    #
    # A region is two uppercase letters, which is also a country code, a
    # currency and a yes/no flag, so on its own it is only trusted when the
    # sheet holds several. A deal in one state holds exactly one, and the
    # column is still a state if it sits inside the address -- which is the
    # case this recovers.
    ends = [idx[c] for c in (part.get("street"), part.get("postal_code"))
            if c and c in idx]
    if len(ends) == 2:
        lo, hi = sorted(ends)
        for i in range(lo + 1, hi):
            cand = columns[i]
            if not cand.strip() or cand in part.values():
                continue
            if "region" not in part:
                vals = [str(r[i]).strip() for r in rows[header_idx + 1:header_idx + 201]
                        if i < len(r) and str(r[i] or "").strip()]
                if vals and _share(vals, lambda v: bool(_REGION.match(v))) >= ROLE_SHARE:
                    part["region"] = cand
                    continue
            if "locality" not in part:
                part["locality"] = cand
    if "locality" not in part:
        anchor = part.get("street") or part.get("postal_code")
        region = part.get("region")
        if region and anchor and anchor in idx and region in idx:
            lo, hi = sorted((idx[anchor], idx[region]))
            for i in range(lo + 1, hi):
                cand = columns[i]
                if cand in part.values() or not cand.strip():
                    continue
                part["locality"] = cand
                break

    cols = [part[k] for k in ("street", "locality", "region", "postal_code") if k in part]
    positions = [idx[c] for c in cols if c in idx]
    if not positions:
        return None

    seen: set[tuple] = set()
    counted = 0
    for r in rows[header_idx + 1:]:
        r = r or []
        tup = tuple(
            str(r[i]).strip().upper() if i < len(r) and r[i] is not None else ""
            for i in positions
        )
        if not any(tup):
            continue
        counted += 1
        seen.add(tup)

    if not seen:
        return None
    notes = []
    if "locality" not in part:
        notes.append("no town or city column was found beside the address")
    if "street" not in part:
        notes.append("postcodes only -- a postcode is an area, not a door")
    return Supply(
        what="site_address",
        columns=part,
        distinct=len(seen),
        rows=counted,
        key=key_column,
        notes=notes,
    )


def contact_supply(
    roles: dict[str, Role], distinct: dict[str, int], rows: int
) -> Supply | None:
    """Can this sheet say who to call AT EACH SITE?

    A column of addresses is not automatically a way to reach the customer.
    This workbook's only email column is "Modified By": nine Sodexo staff
    over 5,448 rows, the people who edited the spreadsheet. Declaring them
    as site contacts would send a crew's questions to whoever last touched
    a cell.

    The tell is cardinality. A genuine per-site contact varies with the
    site; an audit trail is a handful of names repeated thousands of times.
    """
    part: dict[str, str] = {}
    for col, role in roles.items():
        if role.name not in ("email", "phone"):
            continue
        d = distinct.get(col, 0)
        if rows >= 50 and 0 < d <= max(20, rows * 0.01):
            continue    # a few names over many rows: staff, not contacts
        part[role.name] = col
    if not part:
        return None
    return Supply(what="contact", columns=part)
