"""A receipt for the proposed site count: the number, the facts behind it, and
where each fact came from (PUR-23 / PUR-62).

`site_count_reconcile` decides *whether* the stated and resolved counts agree.
This module does not decide anything new. It carries the reasoning that already
happened through to the payload in a shape a PM can audit in one click:

    {value, status: read|derived|abstained, rule, reasoning,
     abstention_needs, facts: [{atom_id, text, document_id, document_name,
                                page, span, quote}]}

* ``read``      -- a document states the count outright; facts are the sentences.
* ``derived``   -- no document states it; we counted the resolved site roster.
                   ``rule`` names the rule that produced the number.
* ``abstained`` -- neither exists. ``value`` is None on purpose and
                   ``abstention_needs`` says what would let us propose one.
                   Abstaining is a decision, not a missing field.

Additive only: nothing here changes the reconciliation verdict.
"""

from __future__ import annotations

from typing import Any

from app.core.site_count_reconcile import _COUNT_NEAR_SITE, _as_int

MAX_FACTS = 3

RULE_STATED = "stated_count_most_repeated"
RULE_ROSTER = "resolved_site_roster_count"

ABSTENTION_NEEDS = [
    "a sentence that states the number of sites or locations (e.g. '10 sites')",
    "or a site list / roster with one row per location",
]


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _source(atom: Any) -> tuple[str | None, str | None, Any]:
    refs = _get(atom, "source_refs") or []
    ref = refs[0] if refs else None
    doc_id = _get(atom, "artifact_id") or (_get(ref, "artifact_id") if ref is not None else None)
    name = _get(ref, "filename") if ref is not None else None
    locator = (_get(ref, "locator") if ref is not None else None) or {}
    page = locator.get("page") if isinstance(locator, dict) else None
    if doc_id == "unknown_artifact":
        doc_id = None
    return doc_id, name, page


def _fact(atom: Any, text: str, span: tuple[int, int] | None) -> dict[str, Any]:
    doc_id, name, page = _source(atom)
    return {
        "atom_id": _get(atom, "id") or _get(atom, "atom_id"),
        "text": text.strip()[:240],
        "document_id": doc_id,
        "document_name": name,
        "page": page,
        "span": {"start": span[0], "end": span[1]} if span else None,
        "quote": text[span[0]:span[1]] if span else None,
    }


def stated_count_facts(atoms: list[Any], count: int) -> list[dict[str, Any]]:
    """Facts (one per atom) whose text asserts ``count`` sites, with the matched span."""
    out: list[dict[str, Any]] = []
    for a in atoms or []:
        text = str(_get(a, "raw_text") or "")
        if not text:
            continue
        for m in _COUNT_NEAR_SITE.finditer(text):
            if _as_int(m.group(1)) == count:
                out.append(_fact(a, text, (m.start(), m.end())))
                break
    return out


def build_site_count_receipt(
    atoms: list[Any],
    reconciliation: dict[str, Any],
    site_atoms: list[Any] | None = None,
) -> dict[str, Any]:
    stated = reconciliation.get("stated")
    resolved = int(reconciliation.get("resolved") or 0)

    if stated is not None:
        facts = stated_count_facts(atoms, int(stated))
        mentions = reconciliation.get("stated_mentions") or len(facts)
        reasoning = f"The documents state {stated} sites ({mentions} mention(s))."
        others = {k: v for k, v in (reconciliation.get("all_counts_seen") or {}).items() if k != stated}
        if others:
            reasoning += " Other counts seen: " + ", ".join(f"{k} ({v}x)" for k, v in others.items()) + "; the most-repeated count wins."
        if reconciliation.get("agrees") is False:
            reasoning += f" The site roster resolved {resolved}, which does not match."
        return {
            "value": int(stated),
            "status": "read",
            "rule": RULE_STATED,
            "reasoning": reasoning,
            "abstention_needs": [],
            "facts": facts[:MAX_FACTS],
            "total_facts": len(facts),
        }

    if resolved > 0:
        facts = [_fact(a, str(_get(a, "raw_text") or ""), None) for a in (site_atoms or []) if _get(a, "raw_text")]
        return {
            "value": resolved,
            "status": "derived",
            "rule": RULE_ROSTER,
            "reasoning": (
                f"No document states a site count. Derived by counting the {resolved} "
                "distinct sites resolved from the documents."
            ),
            "abstention_needs": [],
            "facts": facts[:MAX_FACTS],
            "total_facts": len(facts),
        }

    return {
        "value": None,
        "status": "abstained",
        "rule": None,
        "reasoning": "No document states a site count and no sites were resolved, so no count is proposed.",
        "abstention_needs": list(ABSTENTION_NEEDS),
        "facts": [],
        "total_facts": 0,
    }
