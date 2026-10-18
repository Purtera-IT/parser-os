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

import os

import re
from typing import Any

_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def _street(row: dict[str, Any]) -> str:
    """The row's numbered street: from its street field, else from its display
    name or its key ("6125 Tyvola Centre Drive" was only in display_name on
    010043; "site:loc_1030_delta_blvd_delta_admin" only in the key on 010115)."""
    from app.core.address_parse import _street_for_dedup

    key_text = str(row.get("site") or "").split(":", 1)[-1].replace("_", " ")
    for raw in (row.get("street_address"), row.get("address"), row.get("display_name"),
                row.get("facility_name"), key_text):
        text = re.sub(r"^\s*(loc|site)\s+", "", str(raw or ""), flags=re.I)
        text = re.split(r"\s+[—–-]\s+", text)[-1] if " — " in text else text
        m = re.search(r"\b\d{1,6}\s+[a-z][^,|]*", text, re.I)
        s = _street_for_dedup(m.group(0)) if m else ""
        # A numbered street with a name, not a bare number or a lone word.
        if s[:1].isdigit() and len(s.split()) >= 2:
            return s
    return ""


def _town(row: dict[str, Any]) -> tuple[str, str]:
    """(city, zip) a row names, from its fields or its display name."""
    text = " ".join(str(row.get(k) or "") for k in ("display_name", "facility_name", "address"))
    z = _zip(row) or (_ZIP.search(text).group(1) if _ZIP.search(text) else "")
    city = str(row.get("city") or "").strip().lower()
    if not city:
        m = re.search(r"([A-Za-z][A-Za-z .]+?),?\s+[A-Z]{2}\b", text)
        city = m.group(1).strip().lower() if m else ""
    return city, z


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
    # A row that names only a town (no street, no site code) is the town of the
    # deal's one site there: it folds into the single street row that shares
    # its ZIP (else its city). Two street rows in that town: it stays apart.
    streeted = [i for i in range(n) if _street(rows[i])]
    for i in range(n):
        if _street(rows[i]) or _site_code(str(rows[i].get("site") or ""), rows[i]):
            continue
        city, z = _town(rows[i])
        if not city and not z:
            continue
        hits = [j for j in streeted if (z and _town(rows[j])[1] == z) or (not z and city and _town(rows[j])[0] == city)]
        if len({root(j) for j in hits}) == 1:
            join(hits[0], i)
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


#: A roster site survives when at least one line that names it reads as
#: physical_site at this bar. On 010082 / 010115 a shared call transcript
#: ("whether it's the Seattle Mariners Stadium or the Las Vegas Convention
#: center") put five such places on both rosters; the base read those lines as
#: small_talk / stakeholder / deal_metadata (<= 0.65 site), and every real
#: address line as physical_site at 0.73-1.00.
# A site line at this bar; at MENTION_SITE_FLOOR when nothing on the row reads
# more surely as something else. 0.7 alone overrode the head's own top answer on
# 000132's "Delphos, OH" (0.56) and "Troy, MI" (0.53).
MENTION_SITE_BAR = float(os.environ.get("SOWSMITH_MENTION_SITE_BAR", "0.7"))
#: Between this and MENTION_SITE_BAR a site read counts when no other line of
#: the row reads more surely as something else (010082's "Easterling, 600":
#: physical_site 0.65 beside small_talk 0.71 stays out).
MENTION_SITE_FLOOR = float(os.environ.get("SOWSMITH_MENTION_SITE_FLOOR", "0.5"))
#: A line a rule flagged as a vendor's or sender's address (email signature)
#: counts as a site line only above this. On 010038 four of six roster rows
#: were signature addresses (CDW Chicago, SG Buford ...): rows that had a
#: street, so the street exemption kept them; the base read CDW's as a
#: stakeholder at 0.97.
VENDOR_SITE_BAR = 0.9


def filter_roster(rows: list[dict[str, Any]], site_atoms: dict[str, list[str]],
                  line_types: dict[str, list], flags: dict[str, list[str]] | None = None
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept rows, set-aside rows). A row is judged when the base read at
    least one of its lines, street address or not, and never all rows. A
    set-aside row keeps its evidence and the reason."""
    flags = flags or {}
    kept, aside = [], []
    for r in rows:
        ids = [i for i in site_atoms.get(str(r.get("site") or ""), []) if line_types.get(i)]
        if not ids:
            kept.append(r)
            continue
        # A rule can copy a signature's address out as a site line of its own
        # ("625 W. Adams |, Chicago" on 010038): read alone it is an address
        # (physical_site 1.00), while the signature line it came from reads as
        # a stakeholder (0.96). A flagged site line on a row with a sure
        # stakeholder line is that signature.
        signature = any(line_types[i][0] == "stakeholder" and float(line_types[i][1] or 0) >= 0.9
                        for i in ids)
        best = 0.0
        for i in ids:
            t, p = line_types[i]
            if t != "physical_site":
                continue
            # A bare "City, ST" in a list of the deal's locations has no street
            # to be a signature's: 000132's internal "Locations" note was flagged
            # vendor and held to the signature bar.
            _fl = flags.get(i) or []
            vendor = "vendor_address_suspected" in _fl and "city_site_list" not in _fl
            # The vendor flag comes and goes between runs (010038's CDW Chicago
            # signature had it on one compile and not the next); an address read
            # out of an email body on a row with a sure person line is the same
            # signature either way. Across 17 deals this matches that one row.
            if (vendor or "email_note_physical_site" in _fl) and signature:
                continue
            bar = VENDOR_SITE_BAR if vendor else MENTION_SITE_BAR
            if float(p or 0) >= bar:
                best = max(best, float(p or 0))
            elif not vendor and float(p or 0) >= MENTION_SITE_FLOOR:
                other = max((float(q or 0) for j in ids for (u, q) in [line_types[j]] if u != "physical_site"),
                            default=0.0)
                if float(p or 0) > other:
                    best = max(best, float(p or 0))
        if best:
            kept.append(r)
        else:
            reads = [line_types[i] for i in ids]
            top = max(reads, key=lambda x: float(x[1] or 0))
            aside.append({**r, "set_aside": "mention_only",
                          "reason": f"no line naming it reads as a job site ({len(reads)} read; "
                                    f"most sure: {top[0]} {float(top[1] or 0):.2f})"})
    if not kept:
        # Nothing reads as a site (010387: the real address lines read "keep
        # the current type" / entity). Keep every row but the signatures: a row
        # whose read lines are all sure stakeholders is a name block
        # ("Bill Ort, AVP - End User Compute and Engineering Services").
        sig = [r for r in aside if _all_stakeholder(r, site_atoms, line_types)]
        if sig and len(sig) < len(rows):
            gone = {str(r.get("site")) for r in sig}
            return [r for r in rows if str(r.get("site")) not in gone], sig
        return rows, []
    return kept, aside


def _all_stakeholder(row: dict[str, Any], site_atoms: dict[str, list[str]], line_types: dict[str, list]) -> bool:
    reads = [line_types[i] for i in site_atoms.get(str(row.get("site") or ""), []) if line_types.get(i)]
    return bool(reads) and all(t == "stakeholder" and float(p or 0) >= 0.9 for t, p in reads)
