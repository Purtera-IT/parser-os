"""The one representation a scope router sees — LLM and head alike.

Routing decides `project_mode`, which decides which questions the PM asks the
customer. Whatever the router reads is therefore the highest-leverage text in
the system, and today it is thin. Clayton — 1,770 atoms, 437 sites — reaches the
router as 2,000 characters, of which six of nineteen lines are the same roster
row repeated and three are the word "Quantity":

    FILES: … | Clayton_Dispatch_Readiness | Exhibit A - Retail Locations
    - HC: 77  | Zone: 3 | Address: … | City: Chester | ST: VA
    - HC: 365 | Zone: 2 | Address: … | City: Pelham  | ST: AL      (x4 more)
    - Quantity Yes
    - Quantity No
    - Networking L1 Technician 2 hr. min: 73.5 | 4hr: 66 | 8hr: 61   <- the signal
    - Workdays: 21 | Two-site bundles: 22 | Peak crews/day: 19       <- and this

It labelled correctly, but on a quarter of its budget, having spent the rest on
duplicates. The fix is density, not volume: the full atom blob for that deal is
479,232 characters, and handing over 120k tokens would cost more and read worse.

Four changes, each aimed at a specific waste:

  * repeated row SHAPES collapse to one exemplar plus a count, so 437 roster
    rows cost one line and still say "there are 437 of these";
  * label-only rows ("Quantity Yes") are dropped;
  * aggregates the router cannot infer from a sample are stated outright — the
    atom-type histogram, the site count, and the COMMERCIAL SHAPE, which is the
    load-bearing one: staff-augmentation deals price people (a technician rate
    card) where install deals price materials (a BOM). Mined over the 90 gold
    labels, "tech rate" appears 2.06 times per staff-aug deal and 0.00 times
    across every other class;
  * exclusions get their own section, because "out of scope" lines are short and
    unusually discriminative, and none of Clayton's reached the router at all.

Sampling is stratified by atom type rather than strided over a flat list, so the
sample reflects the deal's shape instead of whatever the parser emitted first.

**Versioned on purpose.** ``SCOPE_SUMMARY_VERSION`` is stamped into every
training row. A head distilled from a teacher must see the identical
representation at inference — it cannot learn a function of inputs it does not
have — and a corpus that silently mixes two formats is a corpus that rots.

Pure functions, no I/O.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

SCOPE_SUMMARY_VERSION = "scope_summary/2"

# Pricing rows dominate a parse and drown the scope: a TV install reads as 313
# material rows against 52 scope atoms. Kept out of the SCOPE section, but still
# counted in the histogram and read for commercial shape.
#
# `bom_line` is deliberately NOT here, matching v1. Excluding it was measured
# and it costs real signal: `cables`, `runs`, `floor`, `feet` and `surface` are
# cabling-domain vocabulary that lives in BOM rows, and dropping them lost a
# discriminative token on 84 of 86 gold deals. Round-robin stratification below
# already stops BOM from owning the budget, which was the only reason to
# exclude it — so it can stay and carry its vocabulary with it.
NOISE_TYPES = frozenset({
    "pricing_assumption", "commercial_total", "rate_card", "line_item",
})

_MAX_SCOPE_LINES = 40
_MAX_LINE_CHARS = 200
_MIN_LINE_CHARS = 8
_COLLAPSE_AT = 3          # this many rows of one shape before collapsing
_MAX_EXCLUSIONS = 8
_MAX_TYPES_SHOWN = 10

_KV = re.compile(r"([A-Za-z][\w /.\-]{0,28}?)\s*:")
_DIGITS = re.compile(r"\d+")
_LABEL_ONLY = re.compile(
    r"^[\s\-|]*[\w /.\-]{0,30}?:?\s*(yes|no|n/?a|tbd|none|null|true|false)\s*$", re.I
)
_EXCLUSION_CUE = re.compile(
    r"\b(out\s+of\s+scope|not\s+included|excludes?|exclusion|by\s+others|"
    r"customer\s+(?:to\s+)?provide[sd]?|not\s+in\s+scope)\b", re.I
)
_RATE_CARD_CUE = re.compile(
    r"\b(?:tech(?:nician)?|labou?r|weekend|holiday|after[\s\-]?hours|ot)\s+rate\b"
    r"|\brate\s+card\b|\bper[\s\-]?(?:site|visit|ticket|day|diem)\b"
    r"|\b\d+\s*hr\.?\s*min\b|\bhourly\s+rate\b|\bday\s+rate\b",
    re.I,
)
_BOM_CUE = re.compile(
    r"\b(?:part\s*(?:#|number|no)|sku|mfg|manufacturer|model\s*(?:#|number)|"
    r"unit\s+price|extended\s+price|qty\s+\d+)\b", re.I
)


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    if at is None and isinstance(atom, dict):
        at = atom.get("atom_type")
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_text(atom: Any) -> str:
    for attr in ("raw_text", "normalized_text", "text"):
        v = getattr(atom, attr, None)
        if v:
            return str(v).strip()
    if isinstance(atom, dict):
        for k in ("text", "raw_text", "normalized_text", "body"):
            if atom.get(k):
                return str(atom[k]).strip()
    return ""


def _row_shape(line: str) -> str:
    """A stable identity for "rows that look the same".

    Two roster rows differing only in address and ZIP share a shape; two
    different scope sentences do not. Keyed on the label set when the line is
    ``key: value`` pairs, else on the line with numbers masked.
    """
    keys = _KV.findall(line)
    if len(keys) >= 2:
        return "kv:" + "|".join(k.strip().lower() for k in keys[:8])
    return "tx:" + _DIGITS.sub("#", line.lower())[:80]


def _is_noise(line: str) -> bool:
    if len(line) < _MIN_LINE_CHARS:
        return True
    if _LABEL_ONLY.match(line):
        return True
    return not any(c.isalpha() for c in line)


def _commercial_shape(atoms: Sequence[Any]) -> str:
    """People-priced or materials-priced. The single most discriminative
    aggregate, and invisible in a 40-row sample of scope prose."""
    rate = bom = 0
    for atom in atoms:
        text = _atom_text(atom)
        if not text:
            continue
        if _RATE_CARD_CUE.search(text):
            rate += 1
        if _BOM_CUE.search(text):
            bom += 1
    parts = []
    if rate:
        parts.append(f"labour rate card present ({rate} rows)")
    if bom:
        parts.append(f"materials BOM present ({bom} rows)")
    if not parts:
        return "no explicit rate card or BOM found"
    return "; ".join(parts)


def build_scope_summary(
    atoms: Sequence[Any],
    documents: Iterable[dict] | None = None,
    *,
    max_scope_lines: int = _MAX_SCOPE_LINES,
) -> str:
    """Return the routing representation for this deal."""
    documents = list(documents or [])
    names = " | ".join(
        str(d.get("filename") or "").rsplit(".", 1)[0]
        for d in documents if d.get("filename")
    )[:300]

    by_type: dict[str, list[str]] = defaultdict(list)
    type_counts: Counter[str] = Counter()
    exclusions: list[str] = []
    for atom in atoms:
        atype = _atom_type(atom) or "unknown"
        type_counts[atype] += 1
        text = _atom_text(atom)
        if not text or _is_noise(text):
            continue
        text = text[:_MAX_LINE_CHARS]
        if _EXCLUSION_CUE.search(text):
            exclusions.append(text)
            continue
        if atype not in NOISE_TYPES:
            by_type[atype].append(text)

    # Collapse repeated row shapes WITHIN a type, keeping one exemplar + count.
    collapsed: dict[str, list[tuple[str, int]]] = {}
    for atype, lines in by_type.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for line in lines:
            groups[_row_shape(line)].append(line)
        out: list[tuple[str, int]] = []
        for members in groups.values():
            out.append((members[0], len(members)))
        out.sort(key=lambda kv: -kv[1])
        collapsed[atype] = out

    # Stratified round-robin across types, so one chatty type cannot own the
    # budget the way roster rows did.
    scope_lines: list[str] = []
    cursors = {t: 0 for t in collapsed}
    while len(scope_lines) < max_scope_lines:
        progressed = False
        for atype in sorted(collapsed):
            i = cursors[atype]
            if i >= len(collapsed[atype]):
                continue
            line, count = collapsed[atype][i]
            cursors[atype] = i + 1
            suffix = f"   [x{count} similar rows]" if count >= _COLLAPSE_AT else ""
            scope_lines.append(f"- ({atype}) {line}{suffix}")
            progressed = True
            if len(scope_lines) >= max_scope_lines:
                break
        if not progressed:
            break

    sites = type_counts.get("physical_site", 0)
    hist = " | ".join(
        f"{t} {n}" for t, n in type_counts.most_common(_MAX_TYPES_SHOWN) if t
    )

    parts = [
        f"SCOPE SUMMARY {SCOPE_SUMMARY_VERSION}",
        f"FILES: {names}",
        f"SHAPE: atoms {sum(type_counts.values())} | sites {sites} | documents {len(documents)}",
        f"COMMERCIAL: {_commercial_shape(atoms)}",
        f"ATOM TYPES: {hist}",
    ]
    if exclusions:
        seen: set[str] = set()
        picked: list[str] = []
        for line in exclusions:
            shape = _row_shape(line)
            if shape in seen:
                continue
            seen.add(shape)
            picked.append(line)
            if len(picked) >= _MAX_EXCLUSIONS:
                break
        parts.append("EXCLUSIONS:")
        parts.extend(f"- {line}" for line in picked)
    parts.append("SCOPE:")
    parts.extend(scope_lines)
    return "\n".join(parts)


__all__ = ["SCOPE_SUMMARY_VERSION", "NOISE_TYPES", "build_scope_summary"]
