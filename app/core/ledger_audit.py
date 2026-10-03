"""Ledger audit: atoms a dedup or fold stage hid, found after the compile.

Report only. Given the atoms a compile kept and its suppression ledger (the
``suppressed_atoms`` sidecar; see :mod:`app.core.suppression_ledger`), this
lists the folds that hide content instead of merging it. It never changes
which atoms are kept.

Rules
-----

``unresolved_survivor``
    A FOLD (not a deliberate drop) whose survivor is null, missing, or an id
    that no kept atom answers to, even after following the ledger chain (a
    survivor that was itself folded), ``via`` aliases and cross-document copy
    pointers. A kept atom that ``settle_ledger`` put back because its fold
    ended nowhere (``_restored``), and a cross-document copy whose canonical
    atom is missing, are listed here too: each is a fold that named no
    standing survivor.

``merged_survivor``
    The survivor's normalized text strictly contains the suppressed text plus
    more words: the clean line was folded into a longer, merged one.

``lossy_survivor``
    The other direction: the suppressed text strictly contains the survivor's
    text plus more words, so the fold lost those words (a contact row folded
    into a name-and-title atom that has no email).

``partial_list``
    A source list where only some items were suppressed. A list is one source
    document plus one list parent, found from whatever locator fields the atom
    carries: the items one mail line was split into and a run of bulleted
    mail lines (the lists ``list_whole`` keeps whole; a folded item it gives
    back as a cross-document copy stands, so it counts as kept), the parent
    paragraph a prose split came from, the bullet_path
    parent, the list label / lead_in, a table, or the run of whole paragraphs
    under one docx section path. An item counts as suppressed only when no
    kept atom stands at the same source position with the same words (two
    extraction passes over one line are one item), and a fold whose survivor
    is another item of the same list (a duplicate inside the list) does not
    count.

Inputs may be ``EvidenceAtom`` objects, ``CompileResult.model_dump()`` rows
(``value`` / ``source_refs``) or envelope rows (``structured`` / ``locator``,
and the envelope's ``suppressed`` rows with ``stage`` and ``survivor``).

Pure function, no I/O.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.core.suppression_ledger import (
    DROP_STAGES,
    DROPPED_NOT_FOLDED_KEY,
    SUPPRESSION_FLAG_PREFIX,
    SURVIVOR_KEY,
)

SCHEMA = "ledger_audit/v1"
RULES = ("unresolved_survivor", "merged_survivor", "lossy_survivor", "partial_list")
DEFAULT_EXAMPLES = 20

_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•·▪◦‣‧∙⁃–—+>]+|\(?\d{1,3}[.)])\s+")
#: Drops that remove document structure (a heading, a routed sheet, a count
#: banner, mail chrome), never a list item: not counted as a hidden item.
_STRUCTURE_STAGES = frozenset({"section_heading", "sheet_router", "pricing_rollup_rows_emitted", "chrome"})
#: A docx "section" run counts as a list only when its paragraphs read as items.
_SECTION_ITEM_MAX_WORDS = 30


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def norm_text(text: Any) -> str:
    """Lowercase words, list marker and punctuation removed."""
    s = str(text or "")
    for _ in range(3):
        t = _LIST_MARKER_RE.sub("", s, count=1)
        if t == s or not t.strip():
            break
        s = t
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


class _View:
    """One atom or ledger row, read the same way whatever shape it came in."""

    __slots__ = ("id", "artifact_id", "text", "key", "value", "loc", "flags", "stage", "kind",
                 "survivor", "atom_type", "env_survivor")

    def __init__(self, atom: Any, *, suppressed: bool) -> None:
        self.id = str(_get(atom, "id", "") or "")
        self.artifact_id = str(_get(atom, "artifact_id", "") or "")
        self.text = str(_get(atom, "raw_text", None) or _get(atom, "text", None)
                        or _get(atom, "normalized_text", None) or "")
        self.key = norm_text(self.text)
        val = _get(atom, "value", None)
        if not isinstance(val, dict):
            val = _get(atom, "structured", None)
        self.value = val if isinstance(val, dict) else {}
        loc = _get(atom, "locator", None)
        if not isinstance(loc, dict):
            refs = _get(atom, "source_refs", None) or []
            loc = _get(refs[0], "locator", None) if refs else None
        self.loc = loc if isinstance(loc, dict) else {}
        self.flags = [str(f) for f in (_get(atom, "review_flags", None) or [])]
        t = _get(atom, "atom_type", None)
        self.atom_type = str(getattr(t, "value", t) or "")
        sup = self.value.get("_suppression") if isinstance(self.value.get("_suppression"), dict) else {}
        stage = str(sup.get("stage") or "") or str(_get(atom, "stage", "") or "")
        if not stage:
            stage = next((f[len(SUPPRESSION_FLAG_PREFIX):] for f in self.flags
                          if f.startswith(SUPPRESSION_FLAG_PREFIX)), "")
        self.stage = stage
        kind = str(sup.get("kind") or "")
        if not kind and suppressed and isinstance(atom, dict):
            kind = str(atom.get("kind") or "")  # an envelope `suppressed` row
        if kind not in ("", "drop", "fold"):
            kind = ""  # an atom's own `kind` field (e.g. "email_body_line"), not a ledger kind
        if not kind and suppressed:
            kind = "drop" if (self.value.get(DROPPED_NOT_FOLDED_KEY) or stage in DROP_STAGES) else "fold"
        self.kind = kind
        rec = self.value.get(SURVIVOR_KEY) or self.value.get("duplicate_of")
        self.survivor = str(rec.get("atom_id") or "") if isinstance(rec, dict) else ""
        # An envelope `suppressed` row names its survivor only when one stands.
        env = _get(atom, "survivor", None) if isinstance(atom, dict) else None
        self.env_survivor = env if isinstance(env, dict) else None
        if not self.survivor and self.env_survivor:
            self.survivor = str(self.env_survivor.get("id") or "")

    def field(self, name: str) -> Any:
        v = self.loc.get(name)
        if v is None or v == "" or v == []:
            v = self.value.get(name)
        return v


def _strictly_within(inner: str, outer: str) -> bool:
    """``inner`` is a whole-word part of ``outer`` and ``outer`` says more."""
    if not inner or not outer or inner == outer:
        return False
    return f" {inner} " in f" {outer} " and len(outer.split()) > len(inner.split())


def _tup(v: Any) -> tuple:
    if isinstance(v, (list, tuple)):
        return tuple(str(x) for x in v)
    return (str(v),) if v not in (None, "") else ()


def _list_key(a: _View) -> tuple | None:
    """Which source list ``a`` is an item of, or None."""
    art = a.artifact_id
    msg = a.field("message_index")
    parent = a.value.get("_parent_atom_id") or a.loc.get("parent_atom_id")
    if parent:
        return ("split", art, str(parent))
    bp = a.loc.get("bullet_path")
    if isinstance(bp, (list, tuple)) and len(bp) >= 1:
        return ("bullet", art, str(a.field("page") or ""), _tup(a.field("section_path")), _tup(bp[:-1]))
    if a.field("table_index") not in (None, "") and a.field("row") not in (None, ""):
        return ("table", art, str(a.field("page") or ""), str(a.field("table_index")))
    if a.value.get("list_item"):
        label = a.value.get("list_label") or a.value.get("parent_field") or a.value.get("intro")
        lead = _tup(a.field("lead_in"))
        if label or lead:
            return ("list", art, str(msg if msg is not None else ""), str(label or ""), lead)
    try:
        n_sent = int(a.field("sentence_count") or 2)
    except (TypeError, ValueError):
        n_sent = 2
    sentence_split = a.field("sentence_index") not in (None, "") and n_sent > 1
    lead = _tup(a.field("lead_in"))
    if lead and not sentence_split:
        return ("lead_in", art, str(msg if msg is not None else ""), _tup(a.field("section_path")), lead)
    section = _tup(a.field("section_path"))
    if (section and a.field("paragraph_index") not in (None, "") and not sentence_split
            and len(a.key.split()) <= _SECTION_ITEM_MAX_WORDS):
        return ("section", art, section)
    return None


def _unit(a: _View, lkey: tuple) -> tuple:
    """One item of a list: its source position, plus its words unless a table row."""
    if lkey[0] == "table":
        return ("row", str(a.field("row")))
    pos = tuple(str(a.field(k) if a.field(k) is not None else "") for k in (
        "paragraph_index", "line", "line_start", "row", "prose_split_sub_idx"))
    sub = a.value.get("_sub_idx")
    return pos + (str(sub if sub is not None else ""), a.key)


def _ex(a: _View, **extra: Any) -> dict[str, Any]:
    out = {"atom_id": a.id, "artifact_id": a.artifact_id, "stage": a.stage, "text": a.text[:200]}
    out.update(extra)
    return out


def _mail_lists(raw: list[Any]) -> dict[str, tuple]:
    """Atom id -> list key for the lists ``list_whole`` reads in mail: the
    items one line was split into, and a run of bulleted lines with its
    lead-in. Same structure the fix keeps whole, so the audit and the fix
    agree on what a list is. Object atoms only (an envelope row has no value
    to read it from)."""
    objs = [a for a in raw if not isinstance(a, dict)]
    if not objs:
        return {}
    try:
        from app.core.list_whole import _line_lists, _line_runs
    except Exception:  # pragma: no cover - the audit stands on its own
        return {}
    out: dict[str, tuple] = {}
    for by, groups in (("mail_line", _line_lists(objs)), ("mail_run", _line_runs(objs))):
        for members in groups:
            first = members[0]
            v = _get(first, "value", None) or {}
            key = (by, str(_get(first, "artifact_id", "") or ""), str(v.get("message_index", "")),
                   str(min(str(_get(m, "id", "")) for m in members)))
            for m in members:
                out.setdefault(str(_get(m, "id", "") or ""), key)
    return out


def audit_ledger(
    kept: Iterable[Any],
    suppressed: Iterable[Any],
    *,
    max_examples: int = DEFAULT_EXAMPLES,
) -> dict[str, Any]:
    """Counts and up to ``max_examples`` examples per rule (see module doc)."""
    kept = list(kept or [])
    suppressed = list(suppressed or [])
    kept_v = [_View(a, suppressed=False) for a in kept]
    supp_v = [_View(a, suppressed=True) for a in suppressed]
    mail_lists = _mail_lists(kept + suppressed)
    kept_by_id = {a.id: a for a in kept_v if a.id}
    supp_by_id = {a.id: a for a in supp_v if a.id}
    # A re-pointed fold keeps the first name it had under ``via``.
    alias: dict[str, str] = {}
    for a in supp_v + kept_v:
        for key in (SURVIVOR_KEY, "duplicate_of"):
            rec = a.value.get(key)
            if isinstance(rec, dict) and rec.get("via") and rec.get("atom_id"):
                alias.setdefault(str(rec["via"]), str(rec["atom_id"]))

    def standing(sid: str) -> _View | None:
        cur, seen = sid, set()
        while cur and cur not in seen and len(seen) < 64:
            seen.add(cur)
            if cur in kept_by_id:
                return kept_by_id[cur]
            nxt = supp_by_id.get(cur)
            if nxt is not None and nxt.survivor:
                cur = nxt.survivor
                continue
            cur = alias.get(cur, "")
        return None

    found: dict[str, list[dict[str, Any]]] = {r: [] for r in RULES}

    for s in supp_v:
        if s.kind == "drop":
            continue
        if not s.survivor:
            found["unresolved_survivor"].append(_ex(s, survivor_id=None, why="no survivor named"))
            continue
        w = standing(s.survivor)
        if w is not None:
            w_key = w.key
        elif s.env_survivor and s.env_survivor.get("text"):
            # Envelope row: the envelope names a survivor only when it stands.
            w_key = norm_text(s.env_survivor.get("text"))
        else:
            found["unresolved_survivor"].append(
                _ex(s, survivor_id=s.survivor, why="survivor is not kept and resolves to no kept atom"))
            continue
        w_id = w.id if w is not None else s.survivor
        w_text = (w.text if w is not None else str((s.env_survivor or {}).get("text") or ""))[:200]
        if _strictly_within(s.key, w_key):
            found["merged_survivor"].append(_ex(s, survivor_id=w_id, survivor_text=w_text))
        elif _strictly_within(w_key, s.key):
            found["lossy_survivor"].append(_ex(s, survivor_id=w_id, survivor_text=w_text))

    for a in kept_v:
        if isinstance(a.value.get("_restored"), dict):
            r = a.value["_restored"]
            found["unresolved_survivor"].append(_ex(
                a, stage=str(r.get("stage") or ""), survivor_id=None,
                why="restored: " + str(r.get("reason") or "fold named no standing survivor")))
            continue
        dup = a.value.get("duplicate_of")
        if isinstance(dup, dict) and (dup.get("canonical_missing") or (
                dup.get("atom_id") and standing(str(dup["atom_id"])) is None)):
            found["unresolved_survivor"].append(_ex(
                a, stage=str(dup.get("stage") or ""), survivor_id=dup.get("atom_id") or None,
                why="cross-document copy whose canonical atom does not stand"))

    # Partial lists.
    groups: dict[tuple, dict[str, Any]] = {}
    for a, is_kept in [(k, True) for k in kept_v] + [(s, False) for s in supp_v]:
        if not is_kept and (a.stage in _STRUCTURE_STAGES or a.loc.get("block_kind") == "heading"):
            continue
        lk = mail_lists.get(a.id) or _list_key(a)
        if lk is None:
            continue
        g = groups.setdefault(lk, {"units": {}, "members": set()})
        u = g["units"].setdefault(_unit(a, lk), {"kept": [], "supp": []})
        (u["kept"] if is_kept else u["supp"]).append(a)
        g["members"].add(a.id)
    for lk, g in groups.items():
        units = g["units"]
        if len(units) < 2:
            continue
        lost: list[_View] = []
        n_kept = 0
        for u in units.values():
            if u["kept"]:
                n_kept += 1
                continue
            hidden = [s for s in u["supp"]
                      if not (s.survivor and (s.survivor in g["members"]
                                              or (standing(s.survivor) or s).id in g["members"]))]
            if hidden:
                lost.append(hidden[0])
        if lost and n_kept:
            found["partial_list"].append({
                "artifact_id": lk[1],
                "list": {"by": lk[0], "key": [list(x) if isinstance(x, tuple) else x for x in lk[2:]]},
                "items": len(units),
                "suppressed_items": len(lost),
                "suppressed": [_ex(s, kind=s.kind, survivor_id=s.survivor or None) for s in lost[:10]],
                "kept_ids": sorted({k.id for u in units.values() for k in u["kept"]})[:10],
            })

    counts = {r: len(found[r]) for r in RULES}
    return {
        "schema": SCHEMA,
        "kept_atoms": len(kept_v),
        "suppressed_atoms": len(supp_v),
        "folds": sum(1 for s in supp_v if s.kind != "drop"),
        "drops": sum(1 for s in supp_v if s.kind == "drop"),
        "counts": counts,
        "findings": sum(counts.values()),
        "examples_cap": max_examples,
        "examples": {r: found[r][:max_examples] for r in RULES},
    }


__all__ = ["audit_ledger", "norm_text", "RULES", "SCHEMA"]
