"""Find the picture in ANY line, in any document, not just a note field.

010288's drawing arrived as a note field whose whole value was a link, and
that is the only shape the parser could see. So the same drawing pasted into
an email body, or written as "Diagram: <link> (rev 1)", produced no picture at
all -- the PM got 500 characters of Outlook wrapper and no way to know a
drawing existed.

This runs over every atom from every parser after extraction: pull the URLs
out of the text, peel the gateway wrappers, and if one of them points at an
image, say so on the atom. Downstream (the labeler, the source pane) already
knows what to do with ``image_url``; it just never had one to work with.

The picture is content, always: a line that carries one is never small talk,
however chatty its words are.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from app.core.link_unwrap import is_image_url, unwrap_link

# A URL as it appears in prose. The trailing class deliberately excludes the
# punctuation a sentence puts AFTER a link -- ")", ",", ".", ">" -- which is
# otherwise swallowed into the path and breaks the fetch.
_URL_RE = re.compile(r"https?://[^\s<>\"'\]]+", re.I)
_TRAILING = ".,;:!?)]}>\"'"

#: Where an atom might be keeping its text.
_TEXT_KEYS = ("value", "raw_text", "text", "url", "href", "link")


def _urls_in(text: str) -> list[str]:
    out: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        raw = m.group(0).rstrip(_TRAILING)
        if raw and raw not in out:
            out.append(raw)
    return out


def _atom_value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    if isinstance(v, dict):
        return v
    v = getattr(atom, "structured", None)
    return v if isinstance(v, dict) else {}


def _candidate_text(atom: Any, value: dict) -> str:
    parts = [str(getattr(atom, "text", "") or ""), str(getattr(atom, "raw_text", "") or "")]
    for key in _TEXT_KEYS:
        got = value.get(key)
        if isinstance(got, str):
            parts.append(got)
    return "\n".join(p for p in parts if p)


def stamp_linked_pictures(atoms: Iterable[Any]) -> int:
    """Mark every atom whose text points at an image. Returns how many.

    Never raises: a malformed link must not cost the deal its parse.
    """
    found = 0
    for atom in atoms or []:
        try:
            value = _atom_value(atom)
            if not isinstance(value, dict) or value.get("image_url"):
                continue
            for raw in _urls_in(_candidate_text(atom, value)):
                direct = unwrap_link(raw)
                if not is_image_url(direct):
                    continue
                value["image_url"] = direct
                value["media_type"] = "image"
                if direct != raw:
                    value.setdefault("wrapped_url", raw)
                # A picture is content. Whatever the words around it looked
                # like, this line is not banter.
                value.pop("chatter", None)
                found += 1
                break
        except Exception:
            continue
    return found


__all__ = ["stamp_linked_pictures"]
