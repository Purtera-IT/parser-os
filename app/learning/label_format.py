"""The label format of the C3 head design: checks, and the one transform that
moves labels written before it onto it.

The storage did not change. A label is still one ``atom_labels`` row: the
columns, ``reads_set`` and the note, plus its ``atom_label_links``. What the
design adds is the map from each field to the head it trains
(``app/core/label_heads.json``) and a few rules about shape:

* the note is the universal WHY, then one line starting ``[<company>]`` with
  the company's rule (``human_labels.split_note``);
* a multi reading (``train_for``, ``needed_by``, ``location_tier``) is a
  list, not a comma, pipe or prose string;
* ``about``, ``supplier`` and ``entity_keys`` are columns, never readings;
* a reading written under an older name moves to its registered name
  (``KEY_ALIASES``, ``HOURS_ALIASES``);
* a closed reading holds one of its registered values.

``transform_row`` makes exactly those moves and nothing else: every value it
touches ends up somewhere, under the name the registry gives it. It never
changes a type, a verdict, or the words of a note. ``format_checks`` names
what a person still has to decide; the label API runs the same checks on
every save (Platform-infra ``shared/label-format.js``) and shows them on the
card, so the two lists must stay identical.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Iterable

from app.core.atom_type_registry import KEEP, load_registry
from app.core.label_heads import head_of_read, heads
from app.learning.human_labels import split_note

#: Types that are noise for every company: only these carry `noise_class`.
NOISE_TYPES = frozenset({KEEP, "small_talk"})

#: Old spellings of registered values, mapped to the registered one.
VALUE_ALIASES: dict[str, dict[str, str]] = {
    "sow_section": {"purtera_responsibilities": "provider_responsibilities"},
    "billing_type": {
        "time_and_materials": "t_and_m", "time_materials": "t_and_m", "tm": "t_and_m", "t&m": "t_and_m",
        "fixed_price": "fixed", "fixed_fee": "fixed", "per site": "per_site",
    },
    # The first needed_by vocabulary named documents, not consumers.
    "needed_by": {"quote": "quoting", "quotes": "quoting", "dispatch": "delivery"},
    "scope_side": {"ours": "provider", "us": "provider", "vendor": "provider", "theirs": "customer",
                   "client": "customer", "both": "shared", "third party": "third_party"},
}

#: Readings written under an older name before the registry had one. Each
#: moves to the registered name only when that name is empty or already says
#: the same; otherwise the old key stays and the dry run reports it.
KEY_ALIASES: dict[str, str] = {
    "qty": "equipment_qty",
    "tech_qty": "crew_size",
    "visit_frequency": "cadence",
}

#: Labor-hours readings under older names. A true/false value says whether the
#: text states hours (hours_stated); a number is the stated hours
#: (labor_hours), or our estimate (co_hours_estimate) when the row says the
#: text states none or the value calls itself an estimate.
HOURS_ALIASES = ("hours", "loe_hours", "tech_hours")

#: Readings that duplicate a column. They move into it when it is empty or
#: equal; a reading that differs stays where it is.
COLUMN_READS = ("about", "supplier", "entity_keys")

#: How older labels joined several values of one reading.
_MULTI_SPLIT = re.compile(r"\s*(?:,|\||/|;|\band\b)\s*", re.I)

#: Words that make a universal WHY company-specific (portable-labels.md,
#: note-split cleanup rules). Checked outside quoted source text only, since a
#: quote stays verbatim.
#: Our own pricing workbook is the Deal Kit, so it belongs on the company line;
#: a customer's pricing workbook is a universal source and stays allowed.
POLICY_WORDS = re.compile(r"\b(reject(?:s|ed)?|deal kit|atlas|hubspot|gantt|(?:internal|our) pricing workbook)\b", re.I)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{3,}'")


def _reads(row: dict[str, Any]) -> dict[str, Any]:
    r = row.get("reads_set")
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except ValueError:
            r = None
    return r if isinstance(r, dict) else {}


def _read_defs() -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in load_registry()["reads"]}


def closed_values(key: str) -> tuple[str, ...] | None:
    """A reading's registered values when they are a fixed set, else None."""
    r = _read_defs().get(key)
    vals = str((r or {}).get("values") or "")
    if "|" not in vals:
        return None
    return tuple(v.strip() for v in vals.split("|") if v.strip())


def company_of(row: dict[str, Any]) -> str:
    return str(_reads(row).get("co_company") or "purtera").strip().lower() or "purtera"


def _marker(company: str) -> str:
    return f"[{company}]"


def format_checks(row: dict[str, Any]) -> list[dict[str, str]]:
    """What still needs a person on this label, one entry per problem.

    Each entry is ``{"check", "head", "message"}``. A check never blocks a
    save; it says which head is thin or inconsistent.
    """
    out: list[dict[str, str]] = []

    def add(check: str, head_key: str, message: str) -> None:
        out.append({"check": check, "head": head_key, "message": message})

    reads = _reads(row)
    # A policy reject still parked on _keep during the backfill carries its
    # real type in `universal_type`: it is a fact, not noise.
    ltype = str(reads.get("universal_type") or row.get("label_type") or "")
    action = str(reads.get("co_action") or "").strip().lower()
    note = str(row.get("note") or "")
    company = company_of(row)
    marker = _marker(company)
    defs = _read_defs()

    if ltype in NOISE_TYPES and not reads.get("noise_class"):
        add("noise_without_class", "content.type", "Noise needs a noise class: which kind of line carries no fact.")
    if ltype in NOISE_TYPES and action:
        add("noise_with_action", "conduct.action",
            "Noise is dropped for every company, so it takes no keep/reject/ignore.")
    if ltype and ltype not in NOISE_TYPES and reads.get("noise_class"):
        add("class_on_a_fact", "content.type", "A noise class is only for _keep and small_talk lines.")
    if action in {"reject", "ignore"} and not reads.get("co_reason"):
        add("reason_missing", "conduct.action", f"A {action} needs a reason code.")
    lines = note.splitlines()
    has_line = any(ln.lstrip().lower().startswith(marker) for ln in lines)
    if action == "reject" and not has_line:
        add("policy_line_missing", "conduct.action",
            f"A reject needs a {marker} line saying the rule and its effect on this line.")
    if marker in note.lower() and not has_line:
        add("marker_inline", "conduct.action", f"{marker} must start its own line, or the WHY and the rule cannot be told apart.")
    universal, _ = split_note(note, company)
    if POLICY_WORDS.search(_QUOTED.sub("", universal)):
        add("policy_words_in_why", "rationale.why",
            f"The WHY names a verdict or one of our systems; move that part to the {marker} line.")
    if (row.get("coarse") == "site" or ltype.endswith("site")) and str(row.get("rejected") or "") == "true":
        add("site_rejected", "conduct.action", "A site is never rejected.")
    for k, v in reads.items():
        if k not in defs:
            add("unregistered_read", "meta.bookkeeping", f"{k} is not a registered reading, so no head learns it.")
            continue
        allowed = closed_values(k)
        if allowed is None or v is True:
            continue
        vals = v if isinstance(v, list) else [v]
        bad = [str(x) for x in vals if str(x) not in allowed and str(x).lower() not in {"true", "false"}]
        if bad:
            add("value_outside_vocab", head_of_read(k) or "",
                f"{k}: {', '.join(bad)} is not one of {' | '.join(allowed)}.")
    return out


def _split_multi(key: str, v: Any) -> list[str] | None:
    """A multi reading's items, or None when splitting would lose something:
    an item outside a closed set ("mixed: 4 major_metro ...") keeps the
    whole value as it was, for a person to rewrite."""
    items = v if isinstance(v, list) else _MULTI_SPLIT.split(str(v))
    aliases = VALUE_ALIASES.get(key, {})
    out: list[str] = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        x = aliases.get(x.lower(), x.lower() if closed_values(key) else x)
        if x not in out:
            out.append(x)
    allowed = closed_values(key)
    if not out or (allowed and any(x not in allowed for x in out)):
        return None
    return out


def _move_read(reads: dict[str, Any], src: str, dst: str, moved: list[str]) -> None:
    val = reads[src]
    if dst in reads and str(reads[dst]) != str(val):
        moved.append(f"{src}: kept, {dst} already says {reads[dst]!r}")
        return
    reads.pop(src)
    reads.setdefault(dst, val)
    moved.append(f"{src} -> {dst}")


def _is_flag(v: Any) -> bool:
    return isinstance(v, bool) or str(v).strip().lower() in {"true", "false"}


def transform_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """(the row in the head format, what moved). Lossless and idempotent."""
    new = deepcopy(row)
    reads = deepcopy(_reads(row))
    moved: list[str] = []
    reg = load_registry()
    column_keys = {"about": {a["key"] for a in reg.get("about") or []},
                   "supplier": {a["key"] for a in reg.get("suppliers") or []}}

    for col in COLUMN_READS:
        if col not in reads:
            continue
        if col == "entity_keys":
            v = reads["entity_keys"]
            items = v if isinstance(v, list) else [x for x in re.split(r"\s*,\s*", str(v)) if x]
            have = row.get("entity_keys")
            have = have if isinstance(have, list) else []
            reads.pop("entity_keys")
            union = have + [str(x) for x in items if str(x) not in have]
            if union != have:
                new["entity_keys"] = union
                moved.append("entity_keys: reading -> column")
            else:
                moved.append("entity_keys: duplicate reading dropped (column already has them)")
            continue
        val = str(reads.get(col) or "").strip()
        have = str(row.get(col) or "").strip()
        if val in column_keys[col] and (not have or have == val):
            reads.pop(col)
            if not have:
                new[col] = val
                moved.append(f"{col}: reading -> column")
            else:
                moved.append(f"{col}: duplicate reading dropped (column already says it)")

    for src, dst in KEY_ALIASES.items():
        if src in reads:
            _move_read(reads, src, dst, moved)

    for src in HOURS_ALIASES:
        if src not in reads:
            continue
        v = reads[src]
        if _is_flag(v):
            dst = "hours_stated"
        elif str(reads.get("hours_stated")).strip().lower() == "false" or "estimat" in str(v).lower():
            dst = "co_hours_estimate"
        else:
            dst = "labor_hours"
        _move_read(reads, src, dst, moved)

    for key, aliases in VALUE_ALIASES.items():
        v = reads.get(key)
        if isinstance(v, str) and v.strip().lower() in aliases:
            reads[key] = aliases[v.strip().lower()]
            moved.append(f"{key}: {v} -> {reads[key]}")

    for key, d in _read_defs().items():
        v = reads.get(key)
        if not d.get("multi") or v is None or v is True or v == "":
            continue
        items = _split_multi(key, v)
        if items is not None and items != v:
            reads[key] = items
            moved.append(f"{key}: {v!r} -> list")

    note = row.get("note")
    if isinstance(note, str):
        marker = _marker(company_of(row))
        lines = note.splitlines()
        if not any(ln.lstrip().lower().startswith(marker) for ln in lines):
            for i, ln in enumerate(lines):
                at = ln.lower().find(marker)
                if at > 0:
                    lines[i:i + 1] = [ln[:at].rstrip(), ln[at:]]
                    new["note"] = "\n".join(lines)
                    moved.append(f"note: {marker} moved to the start of its own line")
                    break

    if reads != _reads(row):
        new["reads_set"] = reads
    return new, moved


def transform_link(link: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A `context` or `supports` link written as a stand-in for derived_from
    (note starting "[derived_from]" or "derived_from:") becomes one."""
    note = str(link.get("note") or "")
    m = re.match(r"^\s*(\[derived_from\]|derived_from:)\s*", note, re.I)
    rel = link.get("relation")
    if rel in ("context", "supports") and m:
        return {**link, "relation": "derived_from", "note": note[m.end():].strip() or None}, \
            [f"{rel} -> derived_from"]
    return dict(link), []


def coverage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """How many rows give each head something to learn from."""
    out = {h["key"]: 0 for h in heads() if h["status"] != "self_supervised"}
    for row in rows:
        reads = _reads(row)
        for h in heads():
            if h["key"] not in out:
                continue
            filled = any(k in reads for k in h["reads"]) or any(
                row.get(c) not in (None, "", [], {}) for c in h["columns"] if c != "note")
            if "note" in h["columns"]:
                why, pol = split_note(str(row.get("note") or ""), company_of(row))
                filled = filled or bool(why if h.get("note_part") == "universal" else pol)
            out[h["key"]] += int(filled)
    return out


__all__ = [
    "NOISE_TYPES",
    "VALUE_ALIASES",
    "closed_values",
    "coverage",
    "format_checks",
    "transform_link",
    "transform_row",
]
