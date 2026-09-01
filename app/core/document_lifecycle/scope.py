"""How WIDE a document is, and what may be read from it on THIS deal.

See docs/CROSS_DEAL_KNOWLEDGE.md for the whole design. The short version:

A document already carries two axes -- stage says WHEN it arrived, direction
says WHOSE it is. Neither says how much of the world it covers. Octavian's
``Sodexo Breakdown.xlsx`` is attached to deal 010215 and covers the entire
Sodexo programme; its covering email says so out loud ("the attached breakdown
for all Sodexo sites"), and nothing reads it.

Its ten atoms on 010215 include:

    [deal_metadata]    Oppty: PO# 00034150      <- not this deal
    [scope_item]       Oppty: PO# 00033068      <- not this deal either
    [scope_item]       Total | Total: 4750      <- aggregate, unattributed
    [commercial_total] Total | Total: 535       <- aggregate

010215's own PO is PO-00034965. Two totals spanning at least three POs sit in
the evidence set as though they were this deal's commercial figures.

THE AGGREGATE RULE
------------------
From a document whose scope exceeds the deal, rollups are STRUCTURALLY
inadmissible -- not low-confidence, inadmissible. A total from a multi-deal
document is the most dangerous atom there is: perfectly plausible, precisely
wrong, and it lands in a price. A missing row is a gap somebody notices; a wrong
total is a number somebody quotes.

WHEN UNCERTAIN
--------------
Demote aggregates, keep rows. Detection will over-fire on documents that merely
mention another deal in passing, and that is the safe direction to be wrong:
losing a rollup costs a re-read, admitting a wrong one costs a mispriced quote.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

#: Purpulse deal numbers: six digits, leading zero (010215, 000131).
DEAL_NUMBER_RE = re.compile(r"\b(0\d{5})\b")
#: PO / opportunity keys as they appear in these workbooks: "Oppty: PO# 00034150".
PO_RE = re.compile(r"\bPO[#\s-]*0*(\d{6,8})\b", re.IGNORECASE)
#: An explicit Oppty label carrying a bare number: "Oppty: 10131".
#
# The Sodexo breakdown writes deal numbers WITHOUT their leading zero, so
# DEAL_NUMBER_RE (which requires 0#####) saw nothing and every section header in
# the workbook was invisible -- line items then inherited whichever PO number
# happened to appear earlier, attributing Boston's rows to a Decom PO.
OPPTY_LABEL_RE = re.compile(r"\boppty\s*[#:]?\s*0*(\d{4,6})\b", re.IGNORECASE)

#: Language a document uses when it is speaking for more than one deal.
#
# Patterns, not literals: people write "all Sodexo sites" and "all 10 sites",
# never the bare "all sites" a literal list would need. The first version of
# this missed the exact sentence that prompted the whole feature.
PROGRAM_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\ball\b(?:\s+\S+){0,3}?\s+(?:sites?|locations?|stores?|schools?)\b",
        r"\beach\s+(?:site|location|store|school)\b",
        r"\bper\s+(?:site|location|store|school)\b",
        r"\bnationwide\b",
        r"\bprogram(?:me)?\s+(?:plan|wide|level)?\b",
        r"\bovertarching\b|\boverarching\b",
        r"\bmaster\s+(?:agreement|list|schedule)\b",
    )
)

#: Document types that are about the CUSTOMER, not about one deal.
ACCOUNT_TYPES = frozenset({"RATE_CARD", "MSA", "NDA", "COI", "INSURANCE", "W9", "CREDIT_APPLICATION"})
GLOBAL_TYPES = frozenset({"INSTALL_INSTRUCTIONS", "TECHNICAL_STANDARDS", "VENDOR_DATASHEET", "SOW_TEMPLATE", "THIRD_PARTY_POLICY"})

SCOPE_DEAL = "deal"
SCOPE_PROGRAM = "program"
SCOPE_ACCOUNT = "account"
SCOPE_GLOBAL = "global"


def _norm_key(key: str) -> str:
    """One spelling for a key, so 010215 and 10215 are the same deal.

    Both sides MUST go through this. The first version normalised only the
    caller's keys, so a document naming its own deal as "010215" was compared
    against "10215" and counted as foreign -- a deal's own paperwork would have
    been flagged as cross-deal.
    """
    return str(key or "").strip().lstrip("0")


def _keys_in(text: str) -> set[str]:
    """Every deal key a piece of text names: deal numbers and PO numbers."""
    out = {_norm_key(m.group(1)) for m in DEAL_NUMBER_RE.finditer(text)}
    out |= {_norm_key(m.group(1)) for m in PO_RE.finditer(text)}
    out |= {_norm_key(m.group(1)) for m in OPPTY_LABEL_RE.finditer(text)}
    return {k for k in out if k}


def detect_scope(
    *,
    texts: Iterable[str],
    document_type: str | None = None,
    this_deal_keys: Iterable[str] = (),
    delivering_text: str = "",
) -> dict[str, Any]:
    """How wide this document is.

    ``texts`` are the document's atom texts (content, never the filename).
    ``this_deal_keys`` are the keys that mean THIS deal -- its number and its
    PO -- so a document naming only those stays ``deal``.
    """
    mine = {_norm_key(k) for k in this_deal_keys if _norm_key(k)}
    blob = "\n".join(t for t in texts if t)
    found = _keys_in(blob)
    foreign = sorted(k for k in found if k not in mine)

    dtype = str(document_type or "").upper()
    signals: list[str] = []

    if dtype in GLOBAL_TYPES:
        return {"scope": SCOPE_GLOBAL, "foreign_keys": [], "signals": [f"type {dtype} is standing reference"]}
    if dtype in ACCOUNT_TYPES:
        return {"scope": SCOPE_ACCOUNT, "foreign_keys": foreign, "signals": [f"type {dtype} is customer-level"]}

    if foreign:
        signals.append(f"names {len(foreign)} key(s) that are not this deal: {', '.join(foreign[:4])}")

    haystack = f"{blob}\n{delivering_text}"
    phrase_hits = [m.group(0).strip() for pat in PROGRAM_PATTERNS for m in [pat.search(haystack)] if m]
    if phrase_hits:
        signals.append(f"scope language: {', '.join(phrase_hits[:3])}")

    # A single foreign key alone is weak -- documents mention other deals in
    # passing. Two or more, or one plus scope language, is a programme document.
    if len(foreign) >= 2 or (foreign and phrase_hits):
        return {"scope": SCOPE_PROGRAM, "foreign_keys": foreign, "signals": signals}
    if phrase_hits and not mine:
        return {"scope": SCOPE_PROGRAM, "foreign_keys": foreign, "signals": signals}

    return {"scope": SCOPE_DEAL, "foreign_keys": foreign, "signals": signals}


#: Atom types that carry a rollup rather than a line of scope.
AGGREGATE_TYPES = frozenset({"commercial_total"})
#: A bare "Total" row: no unit, no rate, just a sum.
_BARE_TOTAL_RE = re.compile(r"^\s*(grand\s+)?totals?\b", re.IGNORECASE)


def is_aggregate(atom: Mapping[str, Any]) -> bool:
    """True when this atom is a sum rather than a line item.

    Deliberately generous. A line item wrongly called an aggregate is demoted in
    one multi-deal document and can be re-read; an aggregate wrongly admitted is
    a wrong number in a quote.
    """
    if str(atom.get("atom_type") or "") in AGGREGATE_TYPES:
        return True
    text = str(atom.get("text") or "")
    head = text.split("|", 1)[0]
    return bool(_BARE_TOTAL_RE.match(head))


def admit_atom(
    atom: Mapping[str, Any],
    *,
    scope: str,
    this_deal_keys: Iterable[str] = (),
) -> tuple[str, str]:
    """``(verdict, why)`` for one atom of a document with the given scope.

    Verdicts: ``admit`` (evidence for this deal), ``context`` (readable, never
    quotable as this deal's scope).
    """
    if scope == SCOPE_DEAL:
        return "admit", "document is scoped to this deal"

    if is_aggregate(atom):
        return (
            "context",
            f"aggregate in a {scope}-scoped document; a rollup spanning deals is never this deal's number",
        )

    mine = {_norm_key(k) for k in this_deal_keys if _norm_key(k)}
    keys = _keys_in(str(atom.get("text") or ""))
    if keys & mine:
        return "admit", "row names this deal"
    if keys:
        return "context", f"row names another deal: {', '.join(sorted(keys)[:3])}"

    if scope == SCOPE_GLOBAL:
        return "admit", "standing reference; not deal-specific by nature"
    if scope == SCOPE_ACCOUNT:
        return "admit", "customer-level document; applies to this deal"

    # program, unattributed. Silence in a multi-deal document is not neutral --
    # the same reason a silent zero and a real zero must never look alike.
    return "context", "unattributed row in a multi-deal document"


def summarise(atoms: Iterable[Mapping[str, Any]], *, scope: str, this_deal_keys: Iterable[str] = ()) -> dict[str, Any]:
    """What Part 1 shows on a deal: how much of a document actually applies."""
    admitted = demoted = aggregates = 0
    reasons: list[str] = []
    for a in atoms:
        verdict, why = admit_atom(a, scope=scope, this_deal_keys=this_deal_keys)
        if verdict == "admit":
            admitted += 1
        else:
            demoted += 1
            if is_aggregate(a):
                aggregates += 1
            if why not in reasons:
                reasons.append(why)
    return {
        "scope": scope,
        "atoms_admitted": admitted,
        "atoms_demoted": demoted,
        "aggregates_withheld": aggregates,
        "reasons": reasons[:4],
    }


def narrow_rows(
    atoms: list[Mapping[str, Any]],
    *,
    scope: str,
    this_deal_keys: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Resolve each row to the deal it actually belongs to.

    Part 2 of docs/CROSS_DEAL_KNOWLEDGE.md. Part 1 marks a whole document;
    this reads it row by row.

    A multi-deal workbook is blocked by opportunity, and the line items sit
    under their deal's header rather than repeating it:

        r2   Oppty: 10131 | Site: Boston, MA        <- header
        r3     Additional Onsite Hourly | 115 | 2       belongs to 10131
        r10  Oppty: 10132 | Site: Avon CT           <- header
        r11    Additional Onsite Hourly | 115 | 2       belongs to 10132

    Judging each row on its own text calls every line item "unattributed", which
    is true and useless. Walking back to the nearest preceding key in document
    order says which deal it is -- so a row that IS this deal's can be admitted
    instead of demoted with the rest.

    Document order comes from (sheet, row), which is only meaningful now that
    locators carry real worksheet rows rather than a block counter.

    A row before ANY header stays unattributed. In a multi-deal document that is
    still a demotion: silence there is not neutral, and inheriting a key
    backwards would be inventing one.
    """
    mine = {_norm_key(k) for k in this_deal_keys if _norm_key(k)}

    def order(atom: Mapping[str, Any]) -> tuple:
        loc = atom.get("locator") or {}
        return (str(loc.get("sheet") or ""), int(loc.get("row") or 0))

    out: list[dict[str, Any]] = []
    carried: str | None = None
    for atom in sorted(atoms, key=order):
        text = str(atom.get("text") or "")
        own = _keys_in(text)
        if own:
            # A row naming a key IS the header for what follows.
            carried = sorted(own)[0] if not (own & mine) else sorted(own & mine)[0]

        verdict, why = admit_atom(atom, scope=scope, this_deal_keys=this_deal_keys)
        belongs = sorted(own)[0] if own else carried

        # Only ever RESCUES a row the flat rule demoted; it never admits an
        # aggregate, and never overrides a document scoped to this deal.
        if verdict == "context" and not is_aggregate(atom) and belongs and belongs in mine:
            verdict, why = "admit", f"row sits under this deal's section ({belongs})"

        out.append({
            "text": text[:200],
            "verdict": verdict,
            "why": why,
            "belongs_to": belongs,
            "inherited": bool(belongs and not own),
        })
    return out
