"""One line, many documents: the earliest owns it, the rest keep a copy.

A sentence that appears in two documents is one fact for counting, scope and
pricing -- but it is a line in BOTH documents, and a person labelling the
later one has to see it there. Live 000132:

* a HubSpot note written in May was pasted into a June email; the dedup kept
  the email's copy (the later, larger document) and the note showed 5 of its
  13 lines;
* the v2 SOW repeats v1's clauses; 30 of v2's lines were folded onto v1 and
  vanished from v2, and 19 of the Deal Kit's went the other way.

Two rules, applied by every dedup stage that can fold across documents:

1. **The earliest document owns the canonical atom.** Earliest by the
   document's own date (the note's date, the email's sent date, the file's
   stated or authored date), falling back to the order the documents were
   read in. :func:`document_order` builds that order once per compile, and
   the stages pick their survivor from the earliest document first.

2. **A cross-document fold leaves a copy, not a hole.** The folded atom is
   kept -- with its own document and locator -- flagged
   :data:`COPY_FLAG` and pointing at the canonical atom through
   ``value["duplicate_of"]``. :func:`split_copies` finds them after a stage.
   The compiler holds copies out of every later stage (exactly as it holds
   admission chatter) and puts them back only for the result, so no count,
   scope roll-up or price ever reads one twice.

Within one document nothing changes: two copies of a line in the same file
are still folded into one atom and recorded in the suppression ledger.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

#: Marks an atom kept only so its document shows its own copy of a line that
#: another (earlier) document owns. Consumers that count, price or roll up
#: scope must skip it; the labelling walk lists it under its own document.
COPY_FLAG = "cross_doc_copy"

#: An order key for a document nothing could date or place: after everything.
_LAST = (2, 0.0, 10**9)


def is_cross_doc_copy(atom: Any) -> bool:
    return COPY_FLAG in (getattr(atom, "review_flags", None) or [])


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _parse_date(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    dt = None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(text)
        except Exception:
            try:
                from dateutil import parser as _dp

                dt = _dp.parse(text, fuzzy=True)
            except Exception:
                return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _own_date(items: list[Any]) -> datetime | None:
    """The date a document carries about ITSELF, read off its atoms.

    A note's CRM header; an email's own (outermost, unquoted) message; a
    transcript's header. A quoted message's date is the date of the message
    it quotes, not of this document, so it is never read here.
    """
    for a in items:
        v = _value(a)
        if str(v.get("kind") or "") == "hubspot_note_meta":
            d = _parse_date(v.get("date"))
            if d is not None:
                return d
    for a in items:
        v = _value(a)
        if v.get("quoted"):
            continue
        if str(v.get("kind") or "") == "email_header":
            d = _parse_date(v.get("date"))
            if d is not None:
                return d
    dates = []
    for a in items:
        v = _value(a)
        if v.get("quoted"):
            continue
        et = v.get("email_thread") if isinstance(v.get("email_thread"), dict) else {}
        msg = et.get("message") if isinstance(et.get("message"), dict) else {}
        if msg.get("quoted"):
            continue
        for raw in (msg.get("sent_at"), v.get("authored_at"), et.get("date"), v.get("note_date")):
            d = _parse_date(raw)
            if d is not None:
                dates.append(d)
                break
    if dates:
        # The outermost message is the newest one the file holds.
        return max(dates)
    for a in items:
        v = _value(a)
        if str(v.get("kind") or "").endswith("_header") and v.get("document_date"):
            d = _parse_date(v.get("document_date"))
            if d is not None:
                return d
    return None


def _filename(items: list[Any]) -> str:
    for a in items:
        for r in getattr(a, "source_refs", None) or []:
            name = str(getattr(r, "filename", "") or "")
            if name:
                return name
    return ""


def document_order(
    atoms: Iterable[Any],
    *,
    provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, tuple]:
    """``artifact_id -> sort key``; a smaller key is an earlier document.

    Dated documents sort by their own date; undated ones follow, in the order
    the compile read them. ``provenance`` is the manifest's per-filename
    record (``authored_at``), used when the atoms state no date of their own.
    """
    by_doc: dict[str, list[Any]] = {}
    first_seen: dict[str, int] = {}
    for i, a in enumerate(atoms):
        aid = str(getattr(a, "artifact_id", "") or "")
        if not aid:
            continue
        by_doc.setdefault(aid, []).append(a)
        first_seen.setdefault(aid, i)
    out: dict[str, tuple] = {}
    for aid, items in by_doc.items():
        when = _own_date(items)
        if when is None and provenance:
            rec = provenance.get(_filename(items)) or {}
            when = _parse_date(rec.get("authored_at"))
        if when is None:
            out[aid] = (1, 0.0, first_seen[aid])
        else:
            try:
                ts = when.timestamp()
            except Exception:
                ts = 0.0
            out[aid] = (0, ts, first_seen[aid])
    return out


def doc_key(atom: Any, order: Mapping[str, tuple] | None) -> tuple:
    """The atom's document's position in ``order`` (all equal when no order)."""
    if not order:
        return (0, 0.0, 0)
    return order.get(str(getattr(atom, "artifact_id", "") or ""), _LAST)


def _ref_key(ref: Any) -> tuple:
    loc = getattr(ref, "locator", None) or {}
    try:
        loc_key = tuple(sorted((str(k), repr(v)) for k, v in dict(loc).items()))
    except Exception:
        loc_key = ()
    return (str(getattr(ref, "id", "") or ""), str(getattr(ref, "filename", "") or ""), loc_key)


#: A leading list marker: a bullet glyph ("- ", "* ", "• ") or an
#: ordinal ("1. ", "2) ", "(3) "). Needs whitespace after it, so "1.5 hours",
#: "-48V" and "*required" are left alone.
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-*•·▪◦‣‧∙⁃–—+>]+|\(?\d{1,3}[.)])\s+"
)


def strip_list_marker(text: Any) -> str:
    """The line without its leading list marker(s), for COMPARISON keys only.

    A note bullet "- Install 4 APs" and the email line "Install 4 APs" are
    one line; so are "1. Install 4 APs" and "2) Install 4 APs" in two pastes
    of a list renumbered differently. Folding to ``[a-z0-9]`` drops a glyph
    but keeps an ordinal, and a key truncated BEFORE folding shifts by the
    marker's width -- either way the copies used to key apart. Displayed atom
    text is never touched.
    """
    s = str(text or "")
    for _ in range(3):  # "- 1. item" nests
        t = _LIST_MARKER_RE.sub("", s, count=1)
        if t == s or not t.strip():
            break
        s = t
    return s


def _text_key(atom: Any) -> str:
    t = getattr(atom, "raw_text", "") or getattr(atom, "normalized_text", "") or ""
    return re.sub(r"[^a-z0-9]+", " ", strip_list_marker(t).lower()).strip()


def _same_words(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    wa, wb = set(a.split()), set(b.split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.6


def _survivor(dropped: Any, kept_by_ref: dict[tuple, list[Any]], kept_by_text: dict[str, list[Any]]) -> Any | None:
    """The kept atom ``dropped`` was folded into, when that is another document's.

    A fold carries the loser's source refs onto the winner, so the winner is
    the kept atom holding the loser's primary ref. Every atom of a document can
    share that document's ref (a note's atoms all cite the note), so a
    candidate in the loser's own document is not evidence of a cross-document
    fold, and the words must agree. With no ref trail, the same words in a
    kept atom of another document stand in.
    """
    own = str(getattr(dropped, "artifact_id", "") or "")
    words = _text_key(dropped)
    if not words:
        return None
    # The same line survives in its own document: that was an in-document
    # fold, which stays a suppression.
    for k in kept_by_text.get(words, ()):
        if str(getattr(k, "artifact_id", "") or "") == own:
            return None
    refs = list(getattr(dropped, "source_refs", None) or [])
    if refs:
        for cand in kept_by_ref.get(_ref_key(refs[0]), ()):
            if str(getattr(cand, "artifact_id", "") or "") == own:
                continue
            if _same_words(words, _text_key(cand)):
                return cand
    for cand in kept_by_text.get(words, ()):
        if str(getattr(cand, "artifact_id", "") or "") != own:
            return cand
    return None


def split_copies(before: list[Any], after: list[Any], *, stage: str) -> list[Any]:
    """The atoms a stage folded onto ANOTHER document's atom, marked as copies.

    Each returned atom keeps its own ``artifact_id`` and locator, gains
    :data:`COPY_FLAG`, and records the canonical atom in
    ``value["duplicate_of"]``; the canonical atom lists the documents that
    also carry the line in ``value["also_in_documents"]``.

    Not copies, and left to the suppression ledger: an in-document fold, a
    quoted echo of an earlier message (the thread already shows that
    message), held chatter, and anything whose survivor cannot be found.
    """
    after_ids = {id(a) for a in after}
    dropped = [a for a in before if id(a) not in after_ids]
    if not dropped:
        return []
    kept_by_ref: dict[tuple, list[Any]] = {}
    kept_by_text: dict[str, list[Any]] = {}
    for k in after:
        for r in getattr(k, "source_refs", None) or []:
            kept_by_ref.setdefault(_ref_key(r), []).append(k)
        kept_by_text.setdefault(_text_key(k), []).append(k)

    copies: list[Any] = []
    for d in dropped:
        v = _value(d)
        if v.get("quoted") or is_cross_doc_copy(d):
            continue
        if "admission_regex" in (getattr(d, "review_flags", None) or []):
            continue
        w = _survivor(d, kept_by_ref, kept_by_text)
        if w is None:
            continue
        v = dict(v)
        v["duplicate_of"] = {
            "atom_id": str(getattr(w, "id", "") or ""),
            "artifact_id": str(getattr(w, "artifact_id", "") or ""),
            "stage": stage,
        }
        d.value = v
        flags = list(getattr(d, "review_flags", None) or [])
        if COPY_FLAG not in flags:
            flags.append(COPY_FLAG)
        d.review_flags = flags
        wv = _value(w)
        if isinstance(getattr(w, "value", None), dict):
            docs = list(wv.get("also_in_documents") or [])
            aid = str(getattr(d, "artifact_id", "") or "")
            if aid and aid not in docs:
                docs.append(aid)
                wv["also_in_documents"] = docs
        copies.append(d)
    return copies


def resolve_canonical(copies: list[Any], final_atoms: list[Any]) -> None:
    """Point every copy at a canonical atom that is still in the result.

    A canonical atom can itself be folded by a later stage (a note line that
    owned a pasted copy, then retyped away by semantic dedup); its own copy
    entry, or a kept atom of another document with the same words, takes its
    place. A copy whose canonical cannot be found says so rather than pointing
    at nothing.
    """
    final_ids = {str(getattr(a, "id", "") or ""): a for a in final_atoms}
    copy_by_id = {str(getattr(c, "id", "") or ""): c for c in copies}
    by_text: dict[str, list[Any]] = {}
    for a in final_atoms:
        by_text.setdefault(_text_key(a), []).append(a)
    for c in copies:
        v = _value(c)
        ref = v.get("duplicate_of") if isinstance(v.get("duplicate_of"), dict) else None
        if not ref:
            continue
        target = str(ref.get("atom_id") or "")
        hops = 0
        while target and target not in final_ids and target in copy_by_id and hops < 16:
            nxt = _value(copy_by_id[target]).get("duplicate_of") or {}
            target = str(nxt.get("atom_id") or "")
            hops += 1
        if target in final_ids:
            if target != ref.get("atom_id"):
                w = final_ids[target]
                v["duplicate_of"] = {**ref, "atom_id": target,
                                     "artifact_id": str(getattr(w, "artifact_id", "") or "")}
            continue
        own = str(getattr(c, "artifact_id", "") or "")
        alt = next((a for a in by_text.get(_text_key(c), ())
                    if str(getattr(a, "artifact_id", "") or "") != own), None)
        if alt is not None:
            v["duplicate_of"] = {**ref, "atom_id": str(getattr(alt, "id", "") or ""),
                                 "artifact_id": str(getattr(alt, "artifact_id", "") or "")}
        else:
            v["duplicate_of"] = {**ref, "canonical_missing": True}


__all__ = [
    "COPY_FLAG",
    "resolve_canonical",
    "doc_key",
    "document_order",
    "is_cross_doc_copy",
    "split_copies",
]
