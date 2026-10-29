"""A repeated address takes the heads' answer from its full-context copy (upstream, at the heads' read).

Live 010358 (2026-10-29): the heads read "Scott Morgan | 1361 N Glenville Dr | scott@… | 214…" as ``stakeholder``,
and every bare copy of that address -- a stand-alone "1361 N Glenville Dr" in an email signature, its
"Richardson TX 75081" line, the same two lines in a scrambled invite -- as ``physical_site`` at 0.84-0.997. Six
locked refits (round 2, X1-X6) never moved the bare street off site .99: a bare address line carries nothing the
heads can see beyond "an address". The heads already decided that address where they could see it.

So, where the heads' answers first come in (``c3_type_base._remote_types`` and the document read), before any stage
or rule reads them: a bare address line takes the answer the heads gave its full-context copy.

* Full-context copy: a line that holds the same street (number + words) and shows whose it is -- an email, a phone,
  or ``|`` cells on the line itself (a contact line), or a table/sheet row (a roster row is its own context).
* Bare copy: not a table row, at most 80 characters, no contact mark, and its own street sits inside full-context
  copies. A bare "City ST ZIP" line right after a bare street that took a copy's answer (same document, next
  line in reading order) is the rest of that address and takes the same answer.
* Only address answers move (physical_site / stakeholder / signatory / deal_metadata; a signatory copy gives
  ``stakeholder``: an address is not a signer). Copies that disagree decide nothing: no fire, the line keeps its
  own read. The answer's p is the highest p among the agreeing copies.

Every bare copy with a full-context copy fires ``repeat_address.follow_full_copy`` (agree when the bare line's own
read already said it), so the ledger shows when the heads can do it alone. In shadow it logs and changes nothing.
The bare line's own read is kept in ``_DEAL["own_reads"]``, and the copy it followed in ``_DEAL["follows_copy"]``.
Owner rule 2026-10-29: rules run upstream and are marked; this replaces the downstream
``party_address_veto.follow_party_copies`` (now shadow by default).
"""
from __future__ import annotations

import re
from typing import Any

from app.core import rule_ledger

RULE_ID = "repeat_address.follow_full_copy"

_CONTACT_MARK = re.compile(r"@|\b\d{3}[-. )]+\d{3}[-. ]\d{4}\b|\|")
_CITY_LINE = re.compile(r"^[A-Za-z][A-Za-z .'-]*,?\s+[A-Z]{2}\.?\s+\d{5}(?:-\d{4})?$")
_ADDRESS_TYPES = frozenset({"physical_site", "stakeholder", "signatory", "deal_metadata"})
_BARE_MAX = 80
_FULL_MAX = 400


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _street(text: str) -> str:
    """The street of a bare line ("1361 N Glenville Dr"), normalised: a number and at least two words, so a bare
    city or a ZIP never counts."""
    s = _norm(str(text or "").split(",")[0])
    words = s.split()
    return s if len(words) >= 3 and words[0].isdigit() and any(w.isalpha() for w in words[1:]) else ""


def _table_row(row: dict[str, Any]) -> bool:
    return bool(row.get("table_id") or row.get("sheet") or row.get("table_row_index") is not None)


def _answer(read: Any) -> tuple[str, float] | None:
    if not isinstance(read, (list, tuple)) or not read or read[0] is None:
        return None
    try:
        return str(read[0]), float(read[1] or 0.0) if len(read) > 1 else 0.0
    except (TypeError, ValueError):
        return str(read[0]), 0.0


def follow_full_copies(types: dict[str, Any], rows: list[dict[str, Any]],
                       deal: dict[str, Any] | None = None) -> list[str]:
    """Give each bare address line in ``types`` (``{row key: [type, p, ...]}``, edited in place) the heads' answer
    on its full-context copy. ``rows`` are the rows the heads read (``key``, ``text``, ``doc_id``, ``order``,
    ``atom_id``, table fields). ``deal`` (``c3_type_base._DEAL``) keeps the bare lines' own reads. Returns the keys
    whose answer changed."""
    if not types or not rows:
        return []
    full: list[tuple[str, str, float, str]] = []  # (normalised text, answer, p, atom id)
    for r in rows:
        text = str(r.get("text") or "")
        got = _answer(types.get(r.get("key")))
        if not got or got[0] not in _ADDRESS_TYPES or len(text) > _FULL_MAX:
            continue
        if not (_CONTACT_MARK.search(text) or _table_row(r)):
            continue
        ans = "stakeholder" if got[0] == "signatory" else got[0]
        full.append((_norm(text), ans, got[1], str(r.get("atom_id") or r.get("key"))))
    if not full:
        return []
    shadow = rule_ledger.shadow(RULE_ID)
    changed: list[str] = []
    taken: dict[str, tuple[str, float, str]] = {}  # row key -> (answer, p, copy atom id): the street lines that followed

    def _take(r: dict[str, Any], ans: str, p: float, copy_id: str, note: str) -> None:
        key = r.get("key")
        own = _answer(types.get(key))
        rule_ledger.fire(RULE_ID, str(r.get("atom_id") or key), own[0] if own else None, ans, note)
        taken[key] = (ans, p, copy_id)
        if shadow or (own and own[0] == ans):
            return
        if deal is not None:
            deal.setdefault("own_reads", {})[key] = list(types.get(key) or [])
            deal.setdefault("follows_copy", {})[key] = {"copy": copy_id, "rule": RULE_ID}
        types[key] = [ans, p]
        changed.append(key)

    def _bare(r: dict[str, Any]) -> bool:
        text = str(r.get("text") or "")
        if _table_row(r) or len(text) > _BARE_MAX or _CONTACT_MARK.search(text):
            return False
        own = _answer(types.get(r.get("key")))
        return bool(own) and own[0] in _ADDRESS_TYPES

    for r in rows:
        if not _bare(r):
            continue
        street = _street(r.get("text") or "")
        if not street:
            continue
        # A copy that is only the street again (a bare table cell) shows no more than this line does.
        copies = [c for c in full if f" {street} " in f" {c[0]} " and c[0] != street
                  and c[3] != str(r.get("atom_id") or r.get("key"))]
        answers = {c[1] for c in copies}
        if len(answers) != 1:
            continue
        ans = answers.pop()
        best = max(copies, key=lambda c: c[2])
        _take(r, ans, best[2], best[3], "bare copy of an address the heads read in full context")

    # The "City ST ZIP" line of a street that followed its copy is the same address.
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(str(r.get("doc_id") or ""), []).append(r)
    for doc_rows in by_doc.values():
        doc_rows = sorted(doc_rows, key=lambda r: float(r.get("order") or 0.0))
        for prev, r in zip(doc_rows, doc_rows[1:]):
            if prev.get("key") not in taken or r.get("key") in taken or not _bare(r):
                continue
            if not _CITY_LINE.match(str(r.get("text") or "").strip()):
                continue
            ans, p, copy_id = taken[prev["key"]]
            _take(r, ans, p, copy_id, "City ST ZIP line of a street that follows its full-context copy")
    return changed


__all__ = ["RULE_ID", "follow_full_copies"]
