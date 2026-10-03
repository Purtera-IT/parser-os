"""One list folds as a unit, or not at all.

Live 000132: a quoted reply carried the customer's request twice -- once as
one line of " - "-separated items, once as a bullet list -- and two HubSpot
notes carried the same items. Each item was matched on its own: some folded
onto one note, some onto the other, the rest stayed. The email then showed a
list with holes in it ("End-user support ...", "24/7 on-call availability",
"Coordination with local stakeholders" and nothing above them), and the
missing items could only be found by reading the suppression ledger.

Whether a document holds a list is a fact about the list, not about each item.
So after every dedup stage has run: a list whose items are partly standing and
partly folded onto ANOTHER document gets the folded ones back, each as its
document's copy of the line it was folded into (``cross_doc_copies``), so the
list reads whole and nothing is counted twice. A list every item of which
folded is a copy as a unit and stays folded.

A "list" is the structure the parser read:

* the items one source line was split into (an inline " - " list), and
* a run of adjacent bulleted lines of one message, each one item, with the
  lead-in line just above it.

Structure only: no atom's type, text or id changes.
"""
from __future__ import annotations

from typing import Any

#: Stage name recorded on a copy this pass gives back.
STAGE = "list_kept_whole"


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _locator(atom: Any) -> dict:
    refs = getattr(atom, "source_refs", None) or []
    loc = getattr(refs[0], "locator", None) if refs else None
    return loc if isinstance(loc, dict) else {}


def _line(atom: Any) -> int | None:
    loc = _locator(atom)
    for raw in (loc.get("line_start"), _value(atom).get("line")):
        try:
            if raw is not None:
                return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _seq(atom: Any) -> int:
    loc = _locator(atom)
    try:
        return int(loc.get("sentence_index") or 0)
    except (TypeError, ValueError):
        return 0


def _place(atom: Any) -> tuple[str, str]:
    """The document and message an atom was read from."""
    v = _value(atom)
    msg = v.get("message_index", _locator(atom).get("message_index"))
    return str(getattr(atom, "artifact_id", "") or ""), "" if msg is None else str(msg)


def _is_mail(atom: Any) -> bool:
    """An email message's own line (not a HubSpot note's)."""
    v = _value(atom)
    if "hubspot_note_parser" in (getattr(atom, "review_flags", None) or []):
        return False
    return bool(v.get("email_thread") or v.get("message_index") is not None
                or str(v.get("kind") or "").startswith("email_"))


def _bare(text: Any) -> str:
    t = " ".join(str(text or "").split())
    return t[2:] if t[:2] in ("- ", "– ") else t


def _line_lists(atoms: list[Any]) -> list[list[Any]]:
    """The atoms of each source line that the parser split as an inline list."""
    from app.core.sentences import split_inline_dash_list

    by_line: dict[tuple, list[Any]] = {}
    for a in atoms:
        line = _line(a)
        if line is None or _scaffolding(a):
            continue
        by_line.setdefault((*_place(a), line), []).append(a)
    out: list[list[Any]] = []
    for members in by_line.values():
        if len(members) < 3:
            continue
        members = sorted(members, key=_seq)
        texts = [_bare(getattr(m, "raw_text", "")) for m in members]
        parts = split_inline_dash_list(" - ".join(texts))
        # Every atom of the line is one segment of the list, and nothing else.
        if parts and [p for p in parts if p] == texts:
            out.append(members)
    return out


def _scaffolding(atom: Any) -> bool:
    """A header, or chatter (a signature block is not a list of items)."""
    kind = str(_value(atom).get("kind") or "")
    if kind.endswith("_header") or kind.endswith("_meta"):
        return True
    return "chatter" in (getattr(atom, "review_flags", None) or [])


def _line_runs(atoms: list[Any]) -> list[list[Any]]:
    """A list written one item per line: adjacent bulleted lines of one
    message (``list_marker``), each one item, with the lead-in line just above.

    Only a line the parser read as a bullet counts. A run of short unmarked
    lines -- "Locations" and the site names under it -- is not this list,
    even when it sits right above it.
    """
    by_line: dict[tuple, dict[int, list[Any]]] = {}
    for a in atoms:
        line = _line(a)
        if line is None or _scaffolding(a):
            continue
        by_line.setdefault(_place(a), {}).setdefault(line, []).append(a)
    out: list[list[Any]] = []

    def _close(lines: dict[int, list[Any]], run: list[int]) -> None:
        if len(run) < 2:
            return
        group = [m for ln in run for m in lines[ln]]
        lead = lines.get(run[0] - 1, [])
        if lead and len({_bare(getattr(m, "raw_text", "")) for m in lead}) == 1 \
                and not any(_value(m).get("list_marker") for m in lead):
            group = list(lead) + group
        out.append(group)

    for lines in by_line.values():
        run: list[int] = []
        for line in sorted(lines):
            members = lines[line]
            texts = {_bare(getattr(m, "raw_text", "")) for m in members}
            bullet = len(texts) == 1 and all(_value(m).get("list_marker") for m in members)
            if bullet and run and line == run[-1] + 1:
                run.append(line)
                continue
            _close(lines, run)
            run = [line] if bullet else []
        _close(lines, run)
    return out


def keep_lists_whole(atoms: list[Any], suppressed: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    """Give a partly folded list its folded items back, as copies.

    Run once, after the ledger is settled (every fold names a standing
    survivor). Returns ``(atoms, suppressed, restored)``.
    """
    from app.core.cross_doc_copies import QUOTED_IN_KEY, _is_quoted_mail_echo, _mark_copy, is_cross_doc_copy
    from app.core.suppression_ledger import SURVIVOR_KEY, SUPPRESSION_FLAG_PREFIX, suppression_kind

    if not suppressed:
        return atoms, suppressed, []
    standing = {id(a) for a in atoms}
    by_id = {str(getattr(a, "id", "") or ""): a for a in atoms}
    # A copy the own-copy sweep minted is not one of the document's own
    # lines: it neither makes a list partial nor belongs to one.
    pool = [a for a in list(atoms) + list(suppressed)
            if (_value(a).get("duplicate_of") or {}).get("stage") != "own_copy_sweep"]
    groups = _line_lists(pool) + _line_runs(pool)

    def _canonical(atom_id: str) -> Any | None:
        w = by_id.get(atom_id)
        hops = 0
        while w is not None and is_cross_doc_copy(w) and hops < 16:
            nxt = str((_value(w).get("duplicate_of") or {}).get("atom_id") or "")
            w, hops = by_id.get(nxt), hops + 1
        return w

    # A document that already shows its own copy of a survivor (the own-copy
    # sweep made one) holds that item: giving the folded atom back too would
    # show it twice.
    held = {
        (str(getattr(a, "artifact_id", "") or ""), str((_value(a).get("duplicate_of") or {}).get("atom_id") or ""))
        for a in atoms if is_cross_doc_copy(a)
    }
    back: dict[int, tuple[Any, Any]] = {}
    for group in groups:
        if not any(id(m) in standing for m in group):
            continue  # the whole list folded: one copy, folded as a unit
        for m in group:
            if id(m) in standing or id(m) in back or suppression_kind(m) == "drop":
                continue
            surv = _value(m).get(SURVIVOR_KEY) or {}
            w = _canonical(str(surv.get("atom_id") or ""))
            own = str(getattr(m, "artifact_id", "") or "")
            # Only a fold onto ANOTHER document's line leaves this document
            # without its item; a fold inside the document still shows it.
            if w is None or str(getattr(w, "artifact_id", "") or "") == own:
                continue
            if (own, str(getattr(w, "id", "") or "")) in held:
                continue
            # A reply quoting another EMAIL's line: that message's section
            # shows it (010003). Only a line another kind of document owns --
            # a note the message quoted -- leaves this list a hole.
            if _is_quoted_mail_echo(m) and _is_mail(w):
                continue
            back[id(m)] = (m, w)
    if not back:
        return atoms, suppressed, []

    restored: list[Any] = []
    for m, w in back.values():
        v = dict(_value(m))
        v.pop("_suppression", None)
        v.pop(SURVIVOR_KEY, None)
        m.value = v
        m.review_flags = [f for f in (getattr(m, "review_flags", None) or [])
                          if not str(f).startswith(SUPPRESSION_FLAG_PREFIX)]
        _mark_copy(m, w, STAGE)
        own = str(getattr(m, "artifact_id", "") or "")
        quoting = _value(w).get(QUOTED_IN_KEY)
        if isinstance(quoting, list) and own in quoting:
            # The document now shows its own copy: it no longer only quotes it.
            rest = [x for x in quoting if x != own]
            if rest:
                w.value[QUOTED_IN_KEY] = rest
            else:
                w.value.pop(QUOTED_IN_KEY, None)
        restored.append(m)

    # Each item goes back next to its own list's standing items -- after the
    # nearest one that precedes it, or before the first when none does -- so
    # the list reads in its own order.
    back_ids = {id(m) for m in restored}
    placed: set[int] = set()
    after: dict[int, list[Any]] = {}
    before: dict[int, list[Any]] = {}
    for group in groups:
        first = next((m for m in group if id(m) in standing), None)
        anchor = None
        for m in group:
            if id(m) in standing:
                anchor = m
            elif id(m) in back_ids and id(m) not in placed and first is not None:
                placed.add(id(m))
                if anchor is not None:
                    after.setdefault(id(anchor), []).append(m)
                else:
                    before.setdefault(id(first), []).append(m)
    out: list[Any] = []
    for a in atoms:
        out.extend(before.get(id(a), ()))
        out.append(a)
        out.extend(after.get(id(a), ()))
    out.extend(m for m in restored if id(m) not in placed)
    return out, [s for s in suppressed if id(s) not in back_ids], restored


__all__ = ["keep_lists_whole", "STAGE"]
