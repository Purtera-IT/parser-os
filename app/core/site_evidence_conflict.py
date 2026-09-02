"""When two documents give a site two different addresses, say so.

Deal 010215 carries both:

    Academy of Early Learning -> 600 E Northside Ave, Marion 29571   (per-site SOW, Aug 12)
    Academy of Early Learning -> 111 Academy St,      Mullins 29574  (locations list, Aug 21)

Eight of the ten sites agree exactly. Two do not, and in both cases the SOW
carries the address of the school in the PRECEDING SOW -- the signature of a
copy-paste that was never updated.

Which one is right is not ours to decide. Both are customer documents, and
picking a winner silently is how a technician ends up at the wrong school with
the system reporting full confidence. The honest output is the disagreement
itself, with both values and the document that asserts each, so a human resolves
it before dispatch.

This is deliberately NOT deduplication. Entity resolution merges things that are
the same; this reports things that claim to be the same and are not.
"""

from __future__ import annotations

import re
from typing import Any

_DIRECTIONAL = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}
_SUFFIX = {
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "rd": "road", "dr": "drive", "ln": "lane", "blvd": "boulevard",
    "hwy": "highway", "ct": "court", "pl": "place", "pkwy": "parkway",
    "ter": "terrace", "cir": "circle", "alt": "alternate",
}


def normalize_address(value: Any) -> str:
    """Compare addresses on meaning, not spelling.

    "600 E Northside Ave" and "600 East Northside Avenue" are one address; a
    checker that called them a conflict would cry wolf and get ignored, which is
    the failure mode that matters most for a warning.
    """
    s = re.sub(r"[^\w\s]", " ", str(value or "").lower())
    out = []
    for tok in s.split():
        tok = _DIRECTIONAL.get(tok, tok)
        tok = _SUFFIX.get(tok, tok)
        out.append(tok)
    return " ".join(out).strip()


def _norm_name(value: Any) -> str:
    s = re.sub(r"[^\w\s]", " ", str(value or "").lower())
    # Drop the account prefix every SOW repeats, so "Marion County School
    # District Johnakin Middle School" and "Johnakin MS" can meet.
    s = re.sub(r"\b(marion county school district|school district)\b", " ", s)
    s = re.sub(r"\b(school|academy|elementary|primary|middle|high|ms|hs)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_site_address_conflicts(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sites asserted with more than one address.

    `sites` is a list of {name, address, source, authored_at} — one per
    assertion, not one per site. Sites with no name cannot be matched to each
    other and are skipped rather than guessed at.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for s in sites or []:
        key = _norm_name(s.get("name"))
        addr = str(s.get("address") or "").strip()
        if not key or not addr:
            continue
        by_name.setdefault(key, []).append(s)

    conflicts: list[dict[str, Any]] = []
    for key, rows in by_name.items():
        distinct: dict[str, dict[str, Any]] = {}
        for r in rows:
            distinct.setdefault(normalize_address(r.get("address")), r)
        if len(distinct) < 2:
            continue
        conflicts.append({
            "site": rows[0].get("name"),
            "address_count": len(distinct),
            "addresses": [
                {
                    "address": r.get("address"),
                    "source": r.get("source"),
                    "authored_at": r.get("authored_at"),
                }
                # Oldest first, so the reader sees which claim came later.
                for r in sorted(distinct.values(), key=lambda x: str(x.get("authored_at") or ""))
            ],
            "reason": f"{len(distinct)} different addresses are asserted for this site",
        })
    return sorted(conflicts, key=lambda c: str(c.get("site") or ""))
