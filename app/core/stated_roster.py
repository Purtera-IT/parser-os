"""When the deal states its site count and one of its site lists has exactly that many sites, that list is the
roster: sites that only a LARGER list names (a superset tab of every store) are set aside from the quote.

010447 (2026-11-05): the deal says 145 sites nine times and its "2027_145_Sites" tab lists them; the workbook's
"annette source" tab lists 244 real Home Depot stores. The type head reads every row right (they are places), and no
head says which places are in this job, so the quote priced 239-366 sites. Upstream of the quote, ledgered as
``roster.stated_count_wins`` (no head answers scope membership: every fire is ``no_head``). Shadow by default (owner,
2026-11-05: the heads decide the sites; when the data does not settle it, the brief asks the PM): ``apply`` only logs
unless the mode is set active, and ``question`` puts the conflict in the brief.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

RULE_ID = "roster.stated_count_wins"
#: The matching list may differ from the stated count by this share (a header row, a closed site).
TOLERANCE = 0.1
#: Act only when the quote holds clearly more sites than the deal states.
OVER = 1.2


def _loc(a: Any) -> dict[str, Any]:
    loc = getattr(a, "locator", None)
    if loc is None and isinstance(a, dict):
        loc = a.get("locator")
    return loc if isinstance(loc, dict) else {}


def _art(a: Any) -> str:
    return str(getattr(a, "artifact_id", None) or (a.get("artifact_id") if isinstance(a, dict) else "") or "")


def _list_of(a: Any) -> tuple[str, str] | None:
    """(file, sheet) of a line a site list holds, or None for any other line (an email, a document)."""
    loc = _loc(a)
    if not loc.get("sheet") or not (loc.get("roster_rank") is not None or "roster" in str(loc.get("extraction") or "")):
        return None
    return (_art(a), str(loc["sheet"]))


def stated_count(atoms: list[Any]) -> int | None:
    """The site count the documents say most often (ties to the larger), or None."""
    from app.core.site_count_reconcile import stated_site_counts

    c = Counter(n for n, _ in stated_site_counts(atoms))
    if not c:
        return None
    return max(c, key=lambda n: (c[n], n))


def apply(sites: list[dict[str, Any]], atoms: list[Any]) -> list[dict[str, Any]]:
    """``sites``: the quote roster (each with ``atom_ids``). Returns it with superset-only sites set aside."""
    stated = stated_count(atoms)
    if not stated or len(sites) <= OVER * stated:
        return sites
    by_id = {str(getattr(a, "id", "") or (a.get("id") if isinstance(a, dict) else "")): a for a in atoms or []}
    lists_of = [{lst for i in s.get("atom_ids") or [] if (lst := _list_of(by_id.get(str(i)))) is not None} for s in sites]
    size = Counter(lst for ls in lists_of for lst in ls)
    match = {lst for lst, n in size.items() if abs(n - stated) <= TOLERANCE * stated}
    if not match:
        return sites
    from app.core import rule_ledger

    keep: list[dict[str, Any]] = []
    for s, ls in zip(sites, lists_of):
        only_bigger = bool(ls) and not (ls & match) and all(size[x] > (1 + TOLERANCE) * stated for x in ls)
        if not only_bigger:
            keep.append(s)
            continue
        ids = s.get("atom_ids") or []
        rule_ledger.fire(RULE_ID, by_id.get(str(ids[0])) if ids else None, None, "not:in_scope",
                         note=f"deal states {stated} sites; a {stated}-row list names them; this site is only on a "
                              f"larger list")
        if rule_ledger.shadow(RULE_ID):
            keep.append(s)
    return keep


def question(atoms: list[Any], type_of) -> dict[str, Any] | None:
    """The brief's question when the deal's site lists disagree with its stated count: the deal states N sites, one
    list the heads read as places holds about N and another holds clearly more. ``type_of(atom)`` is the heads' type.
    The PM's answer (a site fix on the deal page) settles it and becomes a label; nothing is dropped here."""
    stated = stated_count(atoms)
    if not stated:
        return None
    rows: dict[tuple[str, str], list[str]] = {}
    for a in atoms or []:
        lst = _list_of(a)
        if lst is not None and type_of(a) == "physical_site":
            rows.setdefault(lst, []).append(str(getattr(a, "id", "") or (a.get("id") if isinstance(a, dict) else "")))
    match = {k: v for k, v in rows.items() if abs(len(v) - stated) <= TOLERANCE * stated}
    bigger = {k: v for k, v in rows.items() if len(v) > (1 + TOLERANCE) * stated and k not in match}
    if not match or not bigger:
        return None
    m_key, m_ids = max(match.items(), key=lambda kv: len(kv[1]))
    b_key, b_ids = max(bigger.items(), key=lambda kv: len(kv[1]))
    from app.core import rule_ledger

    rule_ledger.fire(RULE_ID, None, None, "ask", note=f"stated {stated}; lists {len(m_ids)} vs {len(b_ids)}")
    return {"id": "q:site_list_scope", "family": "sites", "layer": "base", "owner": "project_manager",
            "by": "rule:" + RULE_ID,
            "text": (f"The deal states {stated} sites. The site list \"{m_key[1]}\" has {len(m_ids)} rows and "
                     f"\"{b_key[1]}\" has {len(b_ids)}. Which list is this job's?"),
            "evidence": [m_ids[0], b_ids[0]]}
