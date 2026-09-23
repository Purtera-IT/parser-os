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
_NOTE_KINDS = ("hubspot_note_body", "note_field_item", "note_field")


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
    return len(_key(atom)) >= _MIN_KEY_LEN


def _twin(note_key: str, by_key: dict[str, Any]) -> Any | None:
    """The original of this line: the same text, or the same text with the
    note's own title glued on the front (a paste lands under a heading)."""
    hit = by_key.get(note_key)
    if hit is not None:
        return hit
    if len(note_key) < _MIN_CONTAINS_LEN:
        return None
    for key, atom in by_key.items():
        if len(key) >= _MIN_CONTAINS_LEN and key in note_key:
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
        doc: {_key(a): a for a in items if _foldable(a)}
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
        best, best_share, best_pairs = None, 0.0, {}
        for other, by_key in originals.items():
            if other == doc or ORIGINALITY.get(kinds[other], 0) <= rank:
                continue
            pairs = {id(a): _twin(_key(a), by_key) for a in candidates}
            pairs = {k: v for k, v in pairs.items() if v is not None}
            share = len(pairs) / len(candidates)
            if share > best_share:
                best, best_share, best_pairs = other, share, pairs

        if best is None or best_share < SEPARATE_BAR:
            continue
        if best_share < FOLD_BAR:
            # Neither one document nor two. Say so; do not split it.
            for a in items:
                v = _value(a)
                v["maybe_copy_of"] = {"document": best, "share": round(best_share, 2)}
                a.value = v
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
