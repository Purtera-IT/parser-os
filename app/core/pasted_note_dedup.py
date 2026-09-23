"""A CRM note that is a pasted email is not a second source.

PMs paste the customer's mail into a HubSpot note so the deal record carries
it. The note and the mail then both parse, and every sentence arrives twice --
once as email, once as note -- which reads as two independent people saying
the same thing. Live 010289: 6 of 49 atoms were one message counted twice.

The email is the original: it has a sender, a timestamp and a thread. The note
copy folds into it and leaves its provenance behind, so "this is also in the
note AJ pasted" survives without a second card to label.
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


def collapse_pasted_note_duplicates(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Fold note atoms onto the identical email atom. Returns ``(kept,
    dropped)``; the email survivor records ``also_in_note``."""
    by_key: dict[str, Any] = {}
    for atom in atoms:
        if not _is_email_atom(atom) or _is_note_atom(atom):
            continue
        k = _key(atom)
        if len(k) >= _MIN_KEY_LEN:
            by_key.setdefault(k, atom)

    # A CRM note pastes the mail UNDER its own title, so its first line reads
    # "The Ask w diagram link Hey AJ, Here are the details for the small job
    # I was discussing earlier." -- the same sentence with a prefix glued on,
    # which an exact key misses. Live 010288 showed it as a second card.
    def _contained(note_key: str) -> Any | None:
        if len(note_key) < _MIN_CONTAINS_LEN:
            return None
        for key, atom in by_key.items():
            if len(key) >= _MIN_CONTAINS_LEN and key in note_key:
                return atom
        return None

    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        original = by_key.get(_key(atom)) if _is_note_atom(atom) else None
        if original is None and _is_note_atom(atom):
            original = _contained(_key(atom))
        if original is None or original is atom:
            kept.append(atom)
            continue
        val = _value(original)
        notes = list(val.get("also_in_note") or [])
        ref = str(getattr(atom, "artifact_id", "") or "")
        if ref and ref not in notes:
            notes.append(ref)
        val["also_in_note"] = notes
        original.value = val
        try:
            refs = list(getattr(original, "source_refs", None) or [])
            for r in list(getattr(atom, "source_refs", None) or []):
                if r not in refs:
                    refs.append(r)
            original.source_refs = refs
        except Exception:
            pass
        dropped.append(atom)
    return kept, dropped


__all__ = ["collapse_pasted_note_duplicates"]
