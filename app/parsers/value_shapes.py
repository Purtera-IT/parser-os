"""Type a cell by what its VALUE looks like, never by what its label says.

The first version of the site reader matched labels: "address line 1",
"cost center/loc name", and so on. That is a vocabulary, and a vocabulary is one
vendor wide. The next customer writes "Site Address", "Location", "Adresse", or
puts the address in an unlabelled cell, and the reader goes blind with no
warning -- it simply finds nothing, which looks identical to a document that
contains nothing.

Shape survives that. "601 Gurley St" is a street address in any document, under
any label, in any language of form design. Measured on the ten 010215 SOWs,
shape alone finds 10/10 addresses with exactly one candidate each.

Two things that measurement taught, both kept here:

  Apostrophes are part of street names. "305 O'Neal St" was missed by a
  character class that allowed word characters and hyphens, and the row fell
  through to a looser pattern that matched "1 UKG DX Clock" -- a hardware
  quantity read as an address. A near-miss on a strict pattern is more dangerous
  than no match, because something else always fills the gap.

  Never accept a loose match. The strict form requires a street suffix; anything
  that merely starts with a number is not an address, however address-shaped it
  looks in isolation.
"""

from __future__ import annotations

import re

_SUFFIX = (
    r"(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|blvd|boulevard|hwy|highway"
    r"|ct|court|pl|place|pkwy|parkway|ter|terrace|cir|circle|way|alt|route|rte)"
)

# House number, street words (apostrophes included), a real suffix, and an
# optional trailing route number: "6641 South Hwy 41".
STREET_ADDRESS = re.compile(
    rf"^\d{{1,6}}\s+[\w.\-/'’ ]+\b{_SUFFIX}\b\.?(?:\s+\d+)?$", re.I
)
POSTAL_CODE = re.compile(r"^\d{5}(?:-\d{4})?$")
STATE_CODE = re.compile(r"^[A-Z]{2}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.\w{2,}$")
# A bare run of digits is NOT a phone number: the cost-centre code 94575001
# typed as one under a permissive pattern, which would have put an account
# number in a contact field. Require either separator punctuation or a full
# 10-11 digit national number.
PHONE = re.compile(
    r"^(?=.*[-().+ ])[\d\-().+ ]{7,}(?:\s*x\s*\d+)?$|^\+?\d{10,11}$"
)


# A street SUFFIX is an English vocabulary. Real addresses often have none:
# "4820 Camino Del Rio", "100 Broadway", "1 Infinite Loop". Requiring one makes
# the reader monolingual, which is the same brittleness as a label whitelist.
#
# So a number-led phrase is a CANDIDATE address, promoted only by co-location
# with a postal code or state in the same block. That structural check is what
# keeps "1 UKG DX Clock" — a hardware quantity — from being read as a place.
CANDIDATE_ADDRESS = re.compile(r"^\d{1,6}\s+[\w.\-/'’ ]{3,}$")


def classify_value(value: object) -> str | None:
    """What this cell IS, from its own text. None when nothing is certain.

    Order matters: an email contains dots and letters that a loose address
    pattern would happily eat, so the unambiguous shapes are tested first.
    """
    s = str(value or "").strip()
    if not s:
        return None
    if EMAIL.match(s):
        return "email"
    if POSTAL_CODE.match(s):
        return "postal"
    if STATE_CODE.match(s):
        return "state"
    if STREET_ADDRESS.match(s):
        return "address"
    if PHONE.match(s) and sum(c.isdigit() for c in s) >= 7:
        return "phone"
    return None


def candidate_addresses(cells: object) -> list[str]:
    """Number-led phrases that MIGHT be addresses, in order, de-duplicated.

    Only meaningful alongside looks_like_site_block: on its own this matches
    quantities. Co-location is what makes it safe.
    """
    out: list[str] = []
    try:
        it = list(cells or [])
    except TypeError:
        return out
    for c in it:
        s = str(c or "").strip()
        if classify_value(s) in (None, "address") and CANDIDATE_ADDRESS.match(s) and s not in out:
            out.append(s)
    return out


def street_addresses(cells: object) -> list[str]:
    """Every distinct street address in an iterable of cells, in order.

    Merged cells repeat their text, so duplicates are collapsed while order is
    preserved -- the first is the one the block leads with.
    """
    out: list[str] = []
    try:
        it = list(cells or [])
    except TypeError:
        return out
    for c in it:
        s = str(c or "").strip()
        if STREET_ADDRESS.match(s) and s not in out:
            out.append(s)
    return out


def looks_like_site_block(cells: object) -> bool:
    """True when these cells carry an address AND a postal code or state.

    Co-location is the structural signal that this is a place, not a passing
    mention. It is what separates a site block from a line item that happens to
    start with a number.
    """
    kinds = set()
    try:
        it = list(cells or [])
    except TypeError:
        return False
    for c in it:
        k = classify_value(c)
        if k:
            kinds.add(k)
    return "address" in kinds and bool(kinds & {"postal", "state"})
