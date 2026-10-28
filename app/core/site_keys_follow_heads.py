"""Site keys follow the heads.

The parser mints ``site:`` keys while it extracts (column values, code shapes,
geo mentions, entity backfill), long before anything knows which lines are job
sites. The C3 type head decides that. This pass runs once the heads' final
types are on the atoms (after ``apply_authority(final=True)`` and
``own_site_keys``) and makes the keys agree with them:

1. One address written over several lines is one site. Adjacent site lines of
   one document where the later lines are only the street or the
   "City, ST ZIP" of the first get the first line's key (2026-10-28 audit:
   "1000 Waterford, Suite 950 / 1000 NW 57th Court / Miami, Florida 33126"
   was three keys, one of them none).
2. A line the heads typed ``physical_site`` names its own place: when it has
   no key (a heading such as "JFK / New York"), it gets one from its own words,
   so the heads' answer reaches the roster.
3. A key no site line backs is dropped from every line, and counted. A key is
   backed when a line carrying it in its own right (not copied by document
   provenance or the single-site anchor) is a site line the parser extracted
   from a document and the heads did not re-type, a site-roster row, or a line
   the heads read as ``physical_site``. A site line the parser INFERRED
   (entity backfill, geo mention) backs its key only when the heads read it as
   a site. Live 2026-10-28: "site:join_url" (a meeting's "Join URL" field,
   backfilled onto "Brian said he was going to be joining"), "site:tech_1"
   (a Deal Kit "Site: Tech #1" staffing column), "site:after_tax" and
   "site:reno_nv" sat on hundreds of lines; join_url was 010200's only roster
   site.

Nothing is decided here that the heads did not answer: the pass only runs when
the heads read the deal, and every dropped key is listed in the record it
returns (the brief carries it as ``site_keys``).
"""

from __future__ import annotations

import re
from typing import Any

_PREFIX = "site:"
#: Review flag on a site line whose keys no site line backs: no roster row, no quote site.
NOT_BACKED = "site_not_backed_by_heads"

#: Flags/values that mark a site line the parser inferred rather than read.
_INFERRED_FLAGS = frozenset({
    "site_entity_backfill", "geo_mention_site", "note_provenance_backfill",
})

_STREET_SUFFIX = re.compile(
    r"\b(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ct|court|ln|lane|way|hwy|highway|"
    r"pkwy|parkway|pl|place|cir|circle|ter|terrace|trl|trail|sq|square|loop|row|plaza|pike|route|rte)\b\.?",
    re.I)
_STREET_LINE = re.compile(r"^\s*\d{1,6}[A-Za-z]?\s+\S")
_CITY_LINE = re.compile(
    r"^\s*[A-Z][A-Za-z.' -]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\.?(?:\s+[A-Z0-9][A-Z0-9 -]{2,9})?\s*$")
_SUITE_LINE = re.compile(r"^\s*(?:suite|ste\.?|unit|floor|fl\.?|bldg\.?|building|room)\s*#?\s*\w+\s*$", re.I)
_LABEL = re.compile(r"^\s*(?:[-*•·]\s*)?(?:(?:[A-Za-z]+\s+){0,2}(?:site|location|address|facility)"
                    r"(?:\s+(?:name|address|no\.?|number))?\s*[:#-]\s*)", re.I)


_FIELD_ROW = re.compile(r"^\s*[^:|]{1,40}:\s*\S|\|")


def _type_of(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def _keys(atom: Any) -> list[str]:
    return [str(k) for k in (getattr(atom, "entity_keys", None) or [])]


def _site_keys(atom: Any) -> list[str]:
    return [k for k in _keys(atom) if k.startswith(_PREFIX) and len(k) > len(_PREFIX)]


def _value(atom: Any) -> dict[str, Any]:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _head(atom: Any, types: dict[str, Any]) -> tuple[str, float] | None:
    """The heads' answer for this line: the read (``types``) or the label it left on the value."""
    r = types.get(str(getattr(atom, "id", "") or ""))
    if isinstance(r, (list, tuple)) and r and r[0]:
        return str(r[0]), float(r[1] or 0.0)
    lab = _value(atom).get("c3_label")
    if isinstance(lab, dict) and lab.get("type"):
        if lab.get("p") is not None:
            return str(lab["type"]), float(lab["p"])
        # A fill label (the head typed it, the LLM filled values) recorded no p before 10-28: the
        # head's own reading carries it.
        says = _value(atom).get("c3_base_says")
        if isinstance(says, dict) and says.get("type") == lab["type"] and says.get("p") is not None:
            return str(lab["type"]), float(says["p"])
        return str(lab["type"]), 1.0
    return None


def _copied(atom: Any, key: str) -> bool:
    """The key reached this line from its document or the deal's one site, not from its words."""
    prov = getattr(atom, "decision_provenance", None) or {}
    if isinstance(prov, dict) and prov.get("source") == "document_scope" and prov.get("site_key") == key:
        return True
    return "single_site_anchor" in (getattr(atom, "review_flags", None) or [])


def _inferred(atom: Any) -> bool:
    if _value(atom).get("inferred") is True:
        return True
    return bool(_INFERRED_FLAGS & set(getattr(atom, "review_flags", None) or []))


def _roster_row(atom: Any) -> bool:
    if "note_site_roster_v1" in (getattr(atom, "review_flags", None) or []):
        return True
    return any("site_roster" in str(getattr(r, "extraction_method", "") or "")
               for r in (getattr(atom, "source_refs", None) or []))


def _ref(atom: Any) -> Any:
    refs = getattr(atom, "source_refs", None) or []
    return refs[0] if refs else None


def _order(atom: Any) -> int | None:
    ref = _ref(atom)
    loc = getattr(ref, "locator", None) if ref is not None else None
    if not isinstance(loc, dict):
        return None
    for f in ("line_start", "line", "paragraph_index", "block_index", "row"):
        v = loc.get(f)
        if isinstance(v, int):
            return v
    return None


def _text(atom: Any) -> str:
    return str(getattr(atom, "raw_text", "") or "").strip()


def _part(text: str) -> str:
    """What part of an address one line is: street, city, suite or name."""
    t = _LABEL.sub("", text).strip()
    if _SUITE_LINE.match(t):
        return "suite"
    if _CITY_LINE.match(t) and not _STREET_LINE.match(t):
        return "city"
    if _STREET_LINE.match(t) and _STREET_SUFFIX.search(t) and "," not in t:
        return "street"
    return "name"


def slug_of_line(text: str) -> str:
    """A site key from a site line's own words: its label, notes in brackets and anything after a
    dash or colon go ("JFK / New York {(may not be required)}" -> ``jfk_new_york``)."""
    t = _LABEL.sub("", str(text or ""))
    t = re.split(r"[({\[]", t, maxsplit=1)[0]
    t = re.split(r"\s[-–—:]\s|:\s", t, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z0-9]+", t)[:8]
    return "_".join(w.lower() for w in words)


def _is_site_line(atom: Any, types: dict[str, Any]) -> bool:
    if _type_of(atom) != "physical_site":
        return False
    h = _head(atom, types)
    return h is None or h[0] in ("physical_site", "_keep")


def _join_split_addresses(atoms: list[Any], types: dict[str, Any]) -> int:
    by_doc: dict[str, list[tuple[int, Any]]] = {}
    for a in atoms:
        if not _is_site_line(a, types):
            continue
        o = _order(a)
        if o is None:
            continue
        by_doc.setdefault(str(getattr(a, "artifact_id", "") or ""), []).append((o, a))
    changed = 0
    for rows in by_doc.values():
        rows.sort(key=lambda x: x[0])
        group: list[tuple[int, Any]] = []
        parts: set[str] = set()

        def flush() -> int:
            if len(group) < 2:
                return 0
            lead = next((k for _, a in group for k in _site_keys(a)), "")
            if not lead:
                return 0
            n = 0
            for _, a in group:
                if _site_keys(a) != [lead]:
                    a.entity_keys = [k for k in _keys(a) if not k.startswith(_PREFIX)] + [lead]
                    flags = list(getattr(a, "review_flags", None) or [])
                    if "address_lines_joined" not in flags:
                        a.review_flags = flags + ["address_lines_joined"]
                    n += 1
            return n

        for o, a in rows:
            p = _part(_text(a))
            adjacent = bool(group) and o - group[-1][0] <= 1
            if adjacent and p != "name" and p not in parts:
                group.append((o, a))
                parts.add(p)
                continue
            changed += flush()
            group, parts = [(o, a)], {p}
        changed += flush()
    return changed


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _LABEL.sub("", str(text or "")).lower()))


def _street_head(text: str) -> str:
    m = re.match(r"(\d{1,6}[a-z]?(?: [a-z0-9]+){1,2})", _norm(text))
    return m.group(1) if m else ""


def _fold_street_copies(atoms: list[Any], types: dict[str, Any]) -> int:
    """A site line that is only a street ("55 Main Rd") written whole inside another keyed site line
    (the roster row "facility: Northgate | address: 55 Main Rd | city: ...") is that site: one key,
    not two (010309: 3 sites sent to the quote as 6)."""
    rows = [(a, _norm(_text(a)), _site_keys(a)[0]) for a in atoms
            if _is_site_line(a, types) and _site_keys(a) and _part(_text(a)) != "street"]
    n = 0
    for a in atoms:
        if not _is_site_line(a, types) or _part(_text(a)) != "street":
            continue
        me = _norm(_text(a))
        if not me or not _street_head(_text(a)):
            continue
        owners = {k for b, t, k in rows if b is not a and re.search(r"(?:^| )" + re.escape(me) + r"(?: |$)", t)}
        if len(owners) != 1:
            continue
        key = next(iter(owners))
        if _site_keys(a) != [key]:
            a.entity_keys = [k for k in _keys(a) if not k.startswith(_PREFIX)] + [key]
            a.review_flags = list(getattr(a, "review_flags", None) or []) + ["street_copy_of_site"]
            n += 1
    return n


def _mint_for_head_sites(atoms: list[Any], types: dict[str, Any]) -> int:
    # A second copy of a keyed site line (the same words, or the same street
    # start) is that site: "Customer Address: 117 Main Way, Town, Washington"
    # quoted again in a reply takes the note's key, not one of its own.
    known: dict[str, str] = {}
    for a in atoms:
        if _type_of(a) == "physical_site" and _site_keys(a):
            for form in (_norm(_text(a)), _street_head(_text(a))):
                if form:
                    known.setdefault(form, _site_keys(a)[0])
    n = 0
    for a in atoms:
        if _type_of(a) != "physical_site" or _site_keys(a):
            continue
        h = _head(a, types)
        if not h or h[0] != "physical_site":
            continue
        key = known.get(_norm(_text(a))) or known.get(_street_head(_text(a)) or "\0")
        if not key and _FIELD_ROW.match(_LABEL.sub("", _text(a))):
            # "City: Morehead", "Alabama: Request |": a field of a table row, not a place's name.
            # The row (or the site it belongs to) carries the key; a field minting one made a
            # state-coverage table 48 sites.
            continue
        slug = key[len(_PREFIX):] if key else slug_of_line(_text(a))
        if not slug:
            continue
        a.entity_keys = _keys(a) + [_PREFIX + slug]
        v = _value(a)
        if isinstance(getattr(a, "value", None), dict) and not (v.get("site_id") or v.get("id")):
            name = re.split(r"[({\[]", _LABEL.sub("", _text(a)), maxsplit=1)[0].strip(" ,;-")
            a.value = {**v, "site_id": slug, "name": v.get("name") or name}
        flags = list(getattr(a, "review_flags", None) or [])
        a.review_flags = flags + ["site_key_from_head"]
        n += 1
    return n


def _backed(atoms: list[Any], types: dict[str, Any], floor: float) -> set[str]:
    out: set[str] = set()
    for a in atoms:
        keys = _site_keys(a)
        if not keys:
            continue
        own = [k for k in keys if not _copied(a, k)]
        if not own:
            continue
        h = _head(a, types)
        head_site = bool(h and h[0] == "physical_site" and h[1] >= floor)
        if head_site or _roster_row(a):
            out.update(own)
        elif _type_of(a) == "physical_site" and not _inferred(a) and not (
                h and h[0] not in ("physical_site", "_keep") and h[1] >= floor):
            # Read off a document as a site and not re-typed by the heads ("_keep": the parser's
            # type stands). A line the heads read as something else (a signature's address read
            # stakeholder 0.99: the reseller's HQ was 010307's only roster site) backs nothing.
            out.update(own)
    return out


def settle(atoms: list[Any], types: dict[str, Any] | None) -> dict[str, Any]:
    """Make the deal's site keys agree with the heads' types. ``types`` is the heads' read
    (atom id -> [type, p]); with none, nothing changes. Returns the record of what changed."""
    if not types:
        return {"by": "none (heads did not read)"}
    try:
        from app.core.site_identity import MENTION_SITE_FLOOR as floor
    except Exception:  # pragma: no cover
        floor = 0.5
    joined = _join_split_addresses(atoms, types)
    joined += _fold_street_copies(atoms, types)
    minted = _mint_for_head_sites(atoms, types)
    backed = _backed(atoms, types, float(floor))
    dropped: dict[str, int] = {}
    for a in atoms:
        keys = _keys(a)
        bad = [k for k in keys if k.startswith(_PREFIX) and k not in backed]
        if not bad:
            continue
        for k in bad:
            dropped[k] = dropped.get(k, 0) + 1
        a.entity_keys = [k for k in keys if k not in bad]
        if _type_of(a) == "physical_site" and not _site_keys(a):
            # The roster and the quote's site list must not rebuild it from the line's own id.
            flags = list(getattr(a, "review_flags", None) or [])
            if NOT_BACKED not in flags:
                a.review_flags = flags + [NOT_BACKED]
        prov = getattr(a, "decision_provenance", None)
        if isinstance(prov, dict) and prov.get("source") == "document_scope" and prov.get("site_key") in bad:
            try:
                a.decision_provenance = None
            except Exception:
                pass
    return {
        "by": "heads:physical_site",
        "kept": sorted(backed),
        "address_lines_joined": joined,
        "keys_minted_from_head": minted,
        "dropped": [{"site": k, "lines": n} for k, n in sorted(dropped.items(), key=lambda x: -x[1])],
    }


__all__ = ["NOT_BACKED", "settle", "slug_of_line"]
