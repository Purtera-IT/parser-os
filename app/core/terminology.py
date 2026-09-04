"""House wording, learned from the PM and applied only where we do the writing.

"I prefer SLO instead of SLA, because we guarantee objectives, not agreements"
is a correction like any other: it has a target, a verdict and a reason. What
makes it different is where it applies. A classification head decides something
about a document; this one decides how WE write.

So the rule here is narrow and absolute: **preferences rewrite text this system
composes, never text it quotes.** A question we generate, a summary line, a
conflict note — those are ours. An atom's `raw_text`, a document's own words, a
site's name as printed: never. Rewriting quoted evidence would break the
receipt that proves the sentence exists, which is the one thing the whole
pipeline is built to guarantee.

Resolution is deterministic, not semantic. A preference names an exact term, so
matching it is a whole-word replacement with the PM's capitalisation respected;
there is no similarity judgment to make and no embedder to be offline.
Conditions still apply: "when Chase is on it" holds the preference back on
everyone else's deals.
"""

from __future__ import annotations

import re
from typing import Any

RELATION = "preferred_term"

#: A preference is a term, not an essay. Anything longer is somebody pasting.
_MAX_TERM_CHARS = 60


def _match_case(source: str, replacement: str) -> str:
    """Write the replacement the way the sentence was written."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def preferred_terms(
    store: Any,
    *,
    deal_id: str = "",
    facts: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every wording preference in force for this deal, narrowest scope last.

    Deal-scoped preferences are returned after global ones so a caller applying
    them in order lets the deal override the house.
    """
    if store is None:
        return []
    try:
        from app.core.feedback_store import SCOPE_DEAL, SCOPE_GLOBAL, condition_holds

        corrections = store.all_corrections(active_only=True)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for c in corrections:
        if getattr(c, "relation", "") != RELATION:
            continue
        if not condition_holds(getattr(c, "relations", None), facts):
            continue
        scope = getattr(c, "scope", SCOPE_GLOBAL)
        if scope == SCOPE_DEAL and str(getattr(c, "scope_key", "")) != str(deal_id or ""):
            continue
        exemplars = list(getattr(c, "exemplars", None) or [])
        instead_of = ""
        for ex in exemplars:
            head = ex.split("\n[ctx]")[0].strip()
            if head and len(head) <= _MAX_TERM_CHARS:
                instead_of = head
                break
        prefer = str(getattr(c, "verdict", "") or "").strip()
        if not prefer or not instead_of or len(prefer) > _MAX_TERM_CHARS:
            continue
        rows.append(
            {
                "prefer": prefer,
                "instead_of": instead_of,
                "why": str(getattr(c, "instruction", "") or ""),
                "scope": scope,
                "correction_id": str(getattr(c, "id", "")),
                "condition": (getattr(c, "relations", None) or {}).get("when") or {},
            }
        )
    rows.sort(key=lambda r: 0 if r["scope"] != SCOPE_DEAL else 1)
    return rows


def apply_preferred_terms(text: str, terms: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Rewrite authored text. Returns ``(text, ids of preferences applied)``.

    Whole words only, so "SLA" never eats "SLAB" and a preference for "site"
    does not rewrite "onsite".
    """
    out = str(text or "")
    if not out or not terms:
        return out, []
    applied: list[str] = []
    for term in terms:
        old = str(term.get("instead_of") or "").strip()
        new = str(term.get("prefer") or "").strip()
        if not old or not new or old.lower() == new.lower():
            continue
        pattern = re.compile(rf"(?<![\w-]){re.escape(old)}(?![\w-])", re.IGNORECASE)
        if not pattern.search(out):
            continue
        out = pattern.sub(lambda m: _match_case(m.group(0), new), out)
        applied.append(str(term.get("correction_id") or ""))
    return out, applied


def apply_to_authored_rows(
    rows: list[dict[str, Any]],
    terms: list[dict[str, Any]],
    *,
    keys: tuple[str, ...] = ("summary", "text", "question", "message"),
) -> int:
    """Rewrite the authored fields of a list of dict rows in place.

    Returns how many fields changed. Only the named keys are touched, and only
    when the caller has already established that those fields are ours to
    write.
    """
    if not rows or not terms:
        return 0
    changed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if not isinstance(value, str) or not value:
                continue
            rewritten, applied = apply_preferred_terms(value, terms)
            if applied and rewritten != value:
                row[key] = rewritten
                row.setdefault("pm_preferences_applied", []).extend(applied)
                changed += 1
    return changed


__all__ = [
    "RELATION",
    "apply_preferred_terms",
    "apply_to_authored_rows",
    "preferred_terms",
]
