"""A CRM note that is a pasted email is not a second source.

PMs paste the customer's mail into a HubSpot note so the deal record carries
it. The note and the mail then both parse, and every sentence arrives twice --
once as email, once as note -- which reads as two independent people saying
the same thing. Live 010289: 6 of 49 atoms were one message counted twice.

The email is the original: it has a sender, a timestamp and a thread. The note
copy folds into it and leaves its provenance behind, so "this is also in the
note AJ pasted" survives without a second card to label.

COPY-OF IS A FACT ABOUT A DOCUMENT, NOT ABOUT A SENTENCE. This matched atom
by atom once, and a document could come out half folded: live 010288 left
three of Alec's sentences alone in a file of their own, at the bottom of the
atom list, attributed to the man who pasted them. Nobody chose that -- it is
what per-atom matching produces when a few atoms miss, and it is worse than
either folding the document or keeping it whole.

So the question is asked once per document pair. Above ``FOLD_BAR`` every
matching atom folds; below ``SEPARATE_BAR`` nothing does; in between the pair
is reported rather than guessed at, because an ambiguous copy is a card for a
person and not a coin flip made in private.

See ``_LABELING_DOCTRINE.md`` -> "One message, one record".
"""
from __future__ import annotations

import re
from typing import Any

#: The note's own scaffolding (``note_id=…``, a field label) is not a paste of
#: anything; only body prose and list items can be duplicates of a mail line.
_NOTE_KINDS = ("hubspot_note_body", "note_field_item", "note_field", "note_field_image")

#: A document's own scaffolding: the note header the CRM wrote, a quoted
#: routing header. Nobody pasted these from anywhere, so they are neither a
#: copy nor evidence that the rest is not one -- they must stay out of the
#: share on BOTH sides. Counting them is how a note whose body duplicates an
#: email entirely still scored 1/2 and outlived the thing it copied.
_PLUMBING_KINDS = (
    "hubspot_note_meta", "quoted_message_header", "email_header", "conversation_meta",
)


def _type_of(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _key(atom: Any) -> str:
    """Text, folded. Type is deliberately NOT in the key: the same sentence
    typed ``scope_item`` in the mail and ``deal_metadata`` in the note is one
    sentence."""
    t = getattr(atom, "normalized_text", "") or getattr(atom, "raw_text", "") or ""
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def _link_key(atom: Any) -> str:
    """The document a line points at, with every gateway peeled off.

    One picture reached 010288 twice: bare in the email Alec sent, and wrapped
    in a safelink inside the note AJ pasted -- 700 characters of Outlook and
    Proofpoint around the same PNG. By text they are two unrelated strings, so
    the note never looked like a copy. By destination they are one fact.
    """
    t = str(getattr(atom, "raw_text", "") or getattr(atom, "normalized_text", "") or "")
    m = re.search(r"https?://[^\s<>\"']+", t)
    if not m:
        return ""
    try:
        from app.core.link_unwrap import unwrap_link

        return str(unwrap_link(m.group(0)) or "").strip().lower()
    except Exception:
        return ""


def _is_note_atom(atom: Any) -> bool:
    v = _value(atom)
    if str(v.get("kind") or "") in _NOTE_KINDS:
        return True
    return "hubspot_note_parser" in list(getattr(atom, "review_flags", None) or [])


def _is_email_atom(atom: Any) -> bool:
    v = _value(atom)
    return bool(v.get("email_thread") or v.get("message_index") is not None or str(v.get("kind") or "").startswith("email_"))


#: Short lines collide by accident ("Relay" in a note list and in a mail list
#: ARE the same item, but "Yes" is not). Keep the floor low enough for real
#: bill-of-material items, high enough to skip acknowledgements.
_MIN_KEY_LEN = 4

#: Long enough that one sentence inside another is the same sentence, not a
#: coincidence of common words.
_MIN_CONTAINS_LEN = 40


#: Share of a copy's foldable atoms that must be found in the original before
#: the whole document folds. High: folding a document that merely quotes
#: another loses a real source.
FOLD_BAR = 0.8

#: Below this the two are simply different documents.
SEPARATE_BAR = 0.4

#: What a document is worth as an original. An email has a sender, a
#: timestamp and a thread; a note has whoever pasted it. Never decided by
#: similarity -- that is how the man who pasted a sentence became the man who
#: said it.
ORIGINALITY = {"email": 3, "note": 1}


def _doc_kind(atoms: list[Any]) -> str:
    """What a whole document is, by what most of its atoms look like."""
    mail = sum(1 for a in atoms if _is_email_atom(a) and not _is_note_atom(a))
    note = sum(1 for a in atoms if _is_note_atom(a))
    if mail and mail >= note:
        return "email"
    return "note" if note else "other"


def _foldable(atom: Any) -> bool:
    """Whether this line can be somebody's copy of another line.

    A document's own scaffolding cannot: nobody pasted the CRM's note header
    in from an email, so it is not a copy -- and, just as importantly, it is
    not evidence that the rest of the document is not one.
    """
    if str(_value(atom).get("kind") or "") in _PLUMBING_KINDS:
        return False
    return len(_key(atom)) >= _MIN_KEY_LEN


def _twin(note_key: str, by_key: dict[str, Any], link_key: str = "") -> Any | None:
    """The original of this line: the same text, the same text with the note's
    own title glued on the front (a paste lands under a heading), or the same
    document pointed at through a different gateway."""
    hit = by_key.get(note_key)
    if hit is not None:
        return hit
    if link_key:
        hit = by_key.get("\x00link:" + link_key)
        if hit is not None:
            return hit
    if len(note_key) < _MIN_CONTAINS_LEN:
        return None
    for key, atom in by_key.items():
        if len(key) >= _MIN_CONTAINS_LEN and not key.startswith("\x00link:") and key in note_key:
            return atom
    return None


def _fold(copy_atom: Any, original: Any) -> None:
    """Record where else this sentence appeared, and keep its evidence."""
    val = _value(original)
    notes = list(val.get("also_in_note") or [])
    ref = str(getattr(copy_atom, "artifact_id", "") or "")
    if ref and ref not in notes:
        notes.append(ref)
    val["also_in_note"] = notes
    original.value = val
    try:
        refs = list(getattr(original, "source_refs", None) or [])
        for r in list(getattr(copy_atom, "source_refs", None) or []):
            if r not in refs:
                refs.append(r)
        original.source_refs = refs
    except Exception:
        pass


def _index(items: list[Any]) -> dict[str, Any]:
    """A document's foldable lines, by text AND by what they link to.

    Only mail lines can be the original a note was pasted from. A document
    counted as "email" can still carry lines that are not mail (an attachment
    stamped into the thread); "Delphos, OH" in a SOW table is not the line a
    PM pasted, so it must not be what the note's city line folds onto.
    """
    out: dict[str, Any] = {}
    for a in items:
        if not _foldable(a) or not _is_email_atom(a) or _is_note_atom(a):
            continue
        out.setdefault(_key(a), a)
        link = _link_key(a)
        if link:
            out.setdefault("\x00link:" + link, a)
    return out


def _parse_date(raw: Any):
    """A timestamp as an aware datetime, or None. Notes write ISO, mail headers
    write RFC 2822, a quoted "Sent:" line writes prose."""
    if not raw:
        return None
    from datetime import datetime, timezone

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


def _note_date(items: list[Any]):
    """When the note was written: the CRM header carries it."""
    for a in items:
        v = _value(a)
        if str(v.get("kind") or "") == "hubspot_note_meta":
            d = _parse_date(v.get("date"))
            if d is not None:
                return d
    for a in items:
        d = _parse_date(_value(a).get("note_date") or _value(a).get("date"))
        if d is not None:
            return d
    return None


def _mail_line_date(atom: Any):
    """When this mail line was written: the quoted message's own "Sent:" if it
    is a quote, else the message's date."""
    v = _value(atom)
    et = v.get("email_thread") if isinstance(v.get("email_thread"), dict) else {}
    msg = et.get("message") if isinstance(et.get("message"), dict) else {}
    for raw in (v.get("authored_at"), msg.get("sent_at"), v.get("date"), et.get("date")):
        d = _parse_date(raw)
        if d is not None:
            return d
    return None


def _note_is_the_original(note_items: list[Any], twins: list[Any]) -> bool:
    """Which way a shared text runs between a note and a mail.

    The default is that the PM pasted the mail into the note (010289). But
    the reverse happens too: 000132's note was written first, and a later
    email QUOTED it -- folding the note onto that email credited the note's
    scope and its city list to the reply that quoted them, and left the note
    with only its title. So, in order:

    1. the mail's copy sits in its quoted/forwarded region -> the mail is the
       copy; the note keeps the text;
    2. both are dated -> the earlier one is the original;
    3. otherwise the mail is the original (it has a sender and a timestamp).
    """
    if not twins:
        return False
    quoted = sum(1 for t in twins if _value(t).get("quoted"))
    if quoted * 2 > len(twins):
        return True
    note_d = _note_date(note_items)
    if note_d is None:
        return False
    mail_dates = [d for d in (_mail_line_date(t) for t in twins) if d is not None]
    if not mail_dates:
        return False
    try:
        return note_d < min(mail_dates)
    except Exception:
        return False


def _contains(container: Any, contained: Any) -> bool:
    """Whether every word of ``contained`` survives in ``container``."""
    a, b = _key(container), _key(contained)
    return bool(b) and b in a


def collapse_pasted_note_duplicates(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Fold a document that is a copy of another onto the original.

    Returns ``(kept, dropped)``. Each surviving original records
    ``also_in_note``; a line the copy ADDED keeps its own document and is
    flagged ``added_when_filed``, because whoever typed it said it.
    """
    by_doc: dict[str, list[Any]] = {}
    for atom in atoms:
        by_doc.setdefault(str(getattr(atom, "artifact_id", "") or ""), []).append(atom)

    kinds = {doc: _doc_kind(items) for doc, items in by_doc.items()}
    originals = {
        doc: _index(items)
        for doc, items in by_doc.items()
        if ORIGINALITY.get(kinds[doc], 0) > 0
    }

    folded: set[int] = set()
    dropped: list[Any] = []
    for doc, items in by_doc.items():
        rank = ORIGINALITY.get(kinds[doc], 0)
        candidates = [a for a in items if _foldable(a)]
        if not candidates:
            continue

        # Ask once, per document pair, against every document that outranks
        # this one as an original.
        # At an equal share, the mail that AUTHORED the lines beats a reply
        # that merely quotes them: the quote is a copy too.
        best, best_share, best_pairs, best_quoted = None, 0.0, {}, 1.0
        for other, by_key in originals.items():
            if other == doc or ORIGINALITY.get(kinds[other], 0) <= rank:
                continue
            pairs = {id(a): _twin(_key(a), by_key, _link_key(a)) for a in candidates}
            pairs = {k: v for k, v in pairs.items() if v is not None}
            share = len(pairs) / len(candidates)
            quoted = (
                sum(1 for t in pairs.values() if _value(t).get("quoted")) / len(pairs)
                if pairs else 1.0
            )
            if share > best_share or (share == best_share and share > 0 and quoted < best_quoted):
                best, best_share, best_pairs, best_quoted = other, share, pairs, quoted

        if best is None or best_share < SEPARATE_BAR:
            continue
        if best_share < FOLD_BAR:
            # Neither one document nor two. Say so; do not split it.
            for a in items:
                v = _value(a)
                v["maybe_copy_of"] = {"document": best, "share": round(best_share, 2)}
                a.value = v
            continue

        twins = [t for t in best_pairs.values() if t is not None]
        if _note_is_the_original(items, twins):
            # The mail quoted the note. The note is the source: it keeps every
            # line, and only the mail's copies that it fully contains fold
            # onto it (a mail line with words the note lacks stays).
            for a in candidates:
                twin = best_pairs.get(id(a))
                if twin is None or twin is a or id(twin) in folded:
                    continue
                if not _contains(a, twin):
                    continue
                v = _value(a)
                refs = list(v.get("also_in_email") or [])
                ref = str(getattr(twin, "artifact_id", "") or "")
                if ref and ref not in refs:
                    refs.append(ref)
                v["also_in_email"] = refs
                a.value = v
                try:
                    srcs = list(getattr(a, "source_refs", None) or [])
                    for r in list(getattr(twin, "source_refs", None) or []):
                        if r not in srcs:
                            srcs.append(r)
                    a.source_refs = srcs
                except Exception:
                    pass
                folded.add(id(twin))
                dropped.append(twin)
            continue

        for a in candidates:
            original = best_pairs.get(id(a))
            if original is None or original is a:
                # The copy said something the original never did: it belongs
                # to whoever typed it, at this document's time.
                v = _value(a)
                v["added_when_filed"] = True
                v["copy_of_document"] = best
                a.value = v
                continue
            _fold(a, original)
            folded.add(id(a))
            dropped.append(a)

    kept = [a for a in atoms if id(a) not in folded]
    return kept, dropped


__all__ = ["collapse_pasted_note_duplicates"]
