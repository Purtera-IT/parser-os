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

SCAFFOLDING = """
Which of the rules in this module are meant to survive, and which are not.

STRUCTURE -- keep. These do not guess; they forbid, and they are true by
definition of what a table is:

  * rows that share a street share its town  (``_determined_by``)
  * a town holds more than one thing, and fewer things than there are
    streets inside it                        (``_fits_locality``)
  * a column that gives every row its own value names rows, and a name is
    not a measurement                        (``Column.is_identifier``)

VOCABULARY -- delete. Everything below is a list of English words or a
threshold picked by hand this afternoon, and it exists only to label the
first few thousand columns so a head can learn the job properly:

  * ``_STREET_WORD``          an anglophone word list
  * ``_POSTAL_US/_CA/_UK``    three countries out of about two hundred
  * ``_ADVERSE``              another English word list
  * ``_MONEY_NAME``, ``_PLACE_NAME``, ``_STATUS_NAME``  -- header-NAME
    regexes, and the worst of them: reading a column by its name is what
    summed a ten-digit account code to $77,041,410,354 on a deal for
    1,563 time clocks
  * ``ROLE_SHARE`` = 0.75, ``FULL_ADDRESS_SHARE`` = 0.55, and the length
    and token floors. A real datacenter site list scored 0.73 and declared
    no sites at all. Nothing chose 0.75 over 0.73; I did, and I was wrong.

The replacement is a ``column_role`` head trained on the corrections these
produce, gated by ``_eval_gate.py``. When it wins, the vocabulary goes --
the structure above stays, because a head is better off being told what is
impossible than having to learn it from examples.
"""


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

#: Most of the world writes the number AFTER the street: Burgstrasse 9,
#: Gammel Gugvej 39, Via Roma 12. A global datacenter site list -- a sheet
#: with a column literally called STREET_ADDR -- declared no sites at all,
#: because every rule here had been written from American examples.
_TRAILING_NUMBER = re.compile(r"\D\s+\d{1,6}\s*[A-Za-z]?$")

#: The shape an address has in any language, once the vocabulary is given
#: up on: several words, letters AND digits together, and short. A city is
#: words with no digits; a serial number is digits with no space; a
#: description runs long. Used only where the column is not a small
#: category, because "InTouch 9000" fits this too and is a device model.
_ADDRESSY = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[^\n]{4,70}$")

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


def _strong_street(values: list[str]) -> float:
    """Evidence that survives on its own: a house number in front of the
    name, or a word that only appears in street names."""
    return _share(
        values,
        lambda v: bool(_HOUSE_NUMBER.match(v)) or bool(_STREET_WORD.search(v)),
    )


def _weak_street(values: list[str]) -> float:
    """Evidence that is only evidence in company.

    A number trailing the name is how most of Europe writes an address --
    and it is also how a product line is written. "Burgstrasse 9" and
    "InTouch 9000" are the same shape, and trusting this signal alone made
    a device-model column the street of a site address on a deal for time
    clocks. It counts only where the column holds thousands of different
    values, which a model column never does.
    """
    return max(
        _share(values, lambda v: bool(_TRAILING_NUMBER.search(v))),
        _share(values,
               lambda v: bool(_ADDRESSY.match(v)) and 2 <= len(v.split()) <= 8),
    )


#: A whole address in one cell needs less shape evidence than a street
#: column, because it is carrying more: the street signal is diluted by the
#: town, the country and the postcode sitting in the same string.
FULL_ADDRESS_SHARE = 0.55


def _looks_full_address(values: list[str]) -> bool:
    """One cell holding the entire address, not a part of it.

    A global datacenter list writes "Burgstrasse 9 RAEREN Liege Belgium
    4730" in a single column called STREET_ADDR. There is no city column
    to pair it with and there never will be, so a rule that only knows how
    to assemble an address out of parts finds nothing at all -- which is
    what happened: the sheet declared no sites, and it is a sheet of 5,544
    of them.

    Either the column shows plain street evidence, or it has the shape a
    written address has in any language: long, several words, a number in
    it somewhere, and mostly proper nouns. That last condition is what
    separates an address from a description -- "Replace the failed clock
    at the Houston site" is long and has a number too, but it is a
    sentence, and sentences are mostly lower case.
    """
    vals = [v for v in values if v]
    if len(vals) < 5:
        return False
    lens = sorted(len(v) for v in vals)
    toks = sorted(len(v.split()) for v in vals)
    if lens[len(lens) // 2] < 25 or toks[len(toks) // 2] < 4:
        return False
    if _strong_street(vals) >= FULL_ADDRESS_SHARE:
        return True

    def _postal_like(v: str) -> bool:
        """Does a token in here look like a postcode or a house number?

        This is what separates an address from any other string of proper
        nouns with a number in it. "0010058001 AR Common" is a cost-centre
        label: it opens with a ten-digit account number and carries nothing
        that places it. "Burgstrasse 9 RAEREN Liege Belgium 4730" carries
        both a house number and a postcode, and neither is the first thing
        in the string, because an address is a place and not an ID.
        """
        parts = v.split()
        if len(parts) < 3:
            return False
        for t in parts[1:]:
            t = t.strip(",.;")
            if _POSTAL_US.match(t) or _POSTAL_CA.match(t) or _POSTAL_UK.match(t):
                return True
            if re.match(r"^[A-Za-z]{1,3}-\d{3,5}$", t):
                return True
            if t.isdigit() and 1 <= len(t) <= 5:
                return True
        return False

    def _proper(v: str) -> bool:
        parts = v.split()
        if not _postal_like(v):
            return False
        capped = sum(1 for t in parts if t[:1].isupper() or t[:1].isdigit())
        return capped >= len(parts) * 0.6

    return _share(vals, _proper) >= ROLE_SHARE


def _looks_street(values: list[str], *, many_valued: bool = True) -> bool:
    if _strong_street(values) >= ROLE_SHARE:
        return True
    return many_valued and _weak_street(values) >= ROLE_SHARE


def street_score(values: list[str]) -> float:
    """How strongly a column reads as street addresses.

    Needed because a sheet often has more than one column that could be a
    street. URI's location analysis has LOCATION DESCRIPTION ("URI Kingston
    Campus 7") and ADDRESS1 ("45 Upper College Rd"), and taking the first
    match in column order made the campus name the street and demoted the
    actual street to the city. Named signals score above shape ones for
    exactly that reason: a house number is evidence, looking vaguely
    address-shaped is not.
    """
    vals = [v for v in values if v][:ROLE_SAMPLE]
    if len(vals) < 5:
        return 0.0
    return _strong_street(vals) + _weak_street(vals) * 0.25


def classify(
    name: str,
    values: list[str],
    *,
    is_identifier: bool = False,
    many_valued: bool = True,
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
    if _looks_full_address(vals):
        return Role("full_address", 0.9, "a whole address in one cell")
    if _looks_street(vals, many_valued=many_valued):
        return Role("street", 1.0, "addresses: a number against a street name")
    if _share(vals, lambda v: bool(_REGION.match(v))) >= ROLE_SHARE:
        # Two uppercase letters is also a country code, a currency and a
        # status flag. What makes it a region is a small vocabulary over
        # many rows, which a flag ("OK"/"NO") does not reach.
        seen = {}
        for v in vals:
            seen[v.upper()] = seen.get(v.upper(), 0) + 1
        top = max(seen.values()) / len(vals)
        # Three or more codes, none of them dominating. "OK" is two upper
        # case letters over 87% of a device list, and read as a region it
        # put Status into a site address.
        if len(seen) >= 3 and top < 0.6:
            return Role("region", 0.8, "two-letter region codes")
    if not is_identifier and _share(vals, lambda v: bool(_PHONE.match(v))) >= ROLE_SHARE:
        return Role("phone", 0.8, "dialable numbers")
    return None


def _determined_by(
    rows: list[list[Any]], header_idx: int, key: int, other: int,
    *, sample: int = 3000, agree: float = 0.85,
) -> bool:
    """Do rows that share a street agree on this other column?

    An address is not a set of columns that each look right; it is a set of
    columns that describe ONE place. Two rows at 5500 Audubon Dr are in the
    same town, the same state and the same postcode. They are not in the
    same firmware state, and they were not last contacted at the same
    minute -- those vary within the door, which is exactly what makes them
    not part of the address.

    This is the test that shape rules cannot do, and it is what stops a
    site address being assembled out of whatever columns happened to look
    plausible in column order.
    """
    groups: dict[str, str] = {}
    checked = agreed = 0
    for r in rows[header_idx + 1: header_idx + 1 + sample]:
        r = r or []
        k = str(r[key]).strip().upper() if key < len(r) and r[key] is not None else ""
        v = str(r[other]).strip().upper() if other < len(r) and r[other] is not None else ""
        if not k or not v:
            continue
        if k not in groups:
            groups[k] = v
            continue
        checked += 1
        if groups[k] == v:
            agreed += 1
    if checked < 20:
        # Almost every street occurs once, so there is nothing to disagree
        # about and the test cannot say anything either way.
        return True
    return agreed >= checked * agree


def site_supply(
    columns: list[str],
    roles: dict[str, Role],
    rows: list[list[Any]],
    header_idx: int,
    *,
    key_column: str | None = None,
    distinct: dict[str, int] | None = None,
    samples: dict[str, list[str]] | None = None,
) -> Supply | None:
    """Can this sheet say where the work is, and at how many places?

    This is the question a SOW actually needs answered, and no single
    column answers it. A street alone is not a site; a state alone is a
    hundred of them. So the sheet supplies sites when it carries a street
    or a postcode with something to place it in, and what it supplies is
    the count of DISTINCT addresses -- not the row count, which on deal
    010264 was 5,448 devices standing at far fewer doors.
    """
    idx = {c: i for i, c in enumerate(columns)}
    distinct = distinct or {}
    samples = samples or {}
    part: dict[str, str] = {}

    # The best street, not the leftmost one.
    streets = [col for col, r in roles.items() if r.name == "street"]
    if streets:
        part["street"] = max(streets, key=lambda col: street_score(samples.get(col, [])))
    for col, role in roles.items():
        if role.name in ("postal_code", "region") and role.name not in part:
            part[role.name] = col

    # One column carrying the whole address answers the question by itself.
    whole = next((col for col, r in roles.items() if r.name == "full_address"), None)
    if whole is not None and "street" not in part:
        i = idx.get(whole, -1)
        if i >= 0:
            seen: set[str] = set()
            counted = 0
            for r in rows[header_idx + 1:]:
                r = r or []
                v = (str(r[i]).strip().upper()
                     if i < len(r) and r[i] is not None else "")
                if not v:
                    continue
                counted += 1
                seen.add(v)
            if seen:
                return Supply(
                    what="site_address",
                    columns={"address": whole},
                    distinct=len(seen),
                    rows=counted,
                    key=key_column,
                    notes=["the address is one free-text column, so the town "
                           "and the postcode are not separately readable"],
                )

    if "street" not in part and "postal_code" not in part:
        return None
    if len(part) < 2:
        return None

    n_body = sum(1 for r in rows[header_idx + 1:]
                 if any(str(cell or "").strip() for cell in (r or [])))

    def _repeats(col: str) -> bool:
        """A town has many devices in it; a rack code has one.

        The datacenter site list's SPACE column holds BEWA16, DKNO3, DKCE1
        -- one per row. Taken as the city it made every row its own site
        and the distinct-address count became the row count, which is the
        one number this whole supply exists to avoid reporting.
        """
        d = distinct.get(col, 0)
        return 0 < d <= max(2, n_body * 0.6)

    # Everything left is read POSITIONALLY, between the two parts whose
    # shape is unmistakable. A city is just a word -- what makes it a city
    # is sitting between a street and a postcode -- and a gazetteer big
    # enough to know Henrico and Willcox would still miss the next one.
    # Every part other than the street has to be settled BY the street.
    anchor_col = part.get("street") or part.get("postal_code")
    anchor_i = idx.get(anchor_col, -1) if anchor_col else -1

    def _belongs(col: str) -> bool:
        if anchor_i < 0 or col not in idx or col == anchor_col:
            return True
        return _determined_by(rows, header_idx, anchor_i, idx[col])

    for _k in ("region", "postal_code"):
        if _k in part and part[_k] != anchor_col and not _belongs(part[_k]):
            part.pop(_k)

    def _fits_locality(col: str) -> bool:
        """A town holds several streets, and it holds more than one thing.

        Two properties, both of them about what a town IS. A column with a
        single value ("Cond: Recycle") cannot say WHICH place a row is at,
        however faithfully it follows the street. And a town cannot have
        more distinct values than the streets inside it -- a column that
        does is varying per row, like a rack code, which is how SPACE came
        to be read as the city of a datacenter and made every row its own
        site.
        """
        d = distinct.get(col, 0)
        st = distinct.get(part.get("street") or "", 0)
        if d < 2 or not _repeats(col):
            return False
        if st > 0 and d > st:
            return False
        return _belongs(col)

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
            if "locality" not in part and _fits_locality(cand):
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
                if not _fits_locality(cand):
                    continue
                part["locality"] = cand
                break

    # A region that does not repeat is not a region either -- it is a code
    # that happens to be two letters, and folding it into the address makes
    # every row its own place.
    if "region" in part and not _repeats(part["region"]):
        part.pop("region")
    if len(part) < 2:
        return None

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
