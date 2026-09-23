"""Things the deal points at but does not contain.

"the small job I was discussing earlier" says there was a call. "Diagram:"
says there is a drawing. "as we discussed", "per my last email", "see
attached" -- every one of them is the author telling us the deal is not all
here, and every one of them died in prose.

The phrase is easy to spot. Whether the thing is actually MISSING is the part
that needs the deal: a "diagram" is only missing when no drawing is filed. So
this marks the reference, resolves what it can, and leaves the rest as a
chase item a PM can work or dismiss.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id

#: A conversation that is not in the thread.
_CONVERSATION_RE = re.compile(
    r"\b(?:"
    r"(?:as |we |i |)(?:was |were |had |)discuss(?:ed|ing)(?: earlier| before| last week| on the call| yesterday|)|"
    r"per (?:our|my|the) (?:call|conversation|chat|discussion|last email)|"
    r"(?:on|during) (?:our|the) (?:call|meeting)|"
    r"(?:spoke|talked) (?:to|with) \w+|"
    r"(?:as|like) (?:i|we) (?:mentioned|said|explained)(?: earlier| before|)"
    r")\b",
    re.I,
)

#: A file the sender believes we have.
_ATTACHMENT_RE = re.compile(
    r"\b(?:see |please see |find |)(?:attached|enclosed)\b|"
    r"\b(?:i|we) (?:sent|forwarded|shared) (?:over |you |)(?:the|a|our)\b|"
    r"\battach(?:ment|ed)\b",
    re.I,
)

#: The counterpart is someone we already know: the "earlier" conversation may
#: predate this deal entirely. AJ had worked with this rep before, so chasing
#: an email inside 010288 would have found nothing.
_PRIOR_RELATIONSHIP_RE = re.compile(
    r"\b(?:"
    r"as always|like last time|the usual|same as (?:the |)last|"
    r"(?:we|you|i) (?:have|'ve|) (?:done|run|worked on) (?:a few|several|these|this) (?:of these|before|)|"
    r"(?:as|like) (?:we|you) know|you(?:'ll| will) remember|"
    r"(?:our|the) (?:usual|standard) (?:setup|kit|approach)|"
    r"(?:good|great) (?:working|to work) with you again|"
    r"(?:discussing|discussed|spoke) (?:earlier|before|previously)"
    r")\b",
    re.I,
)

CONVERSATION = "conversation"
ATTACHMENT = "attachment"


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _text(atom: Any) -> str:
    return " ".join(str(getattr(atom, "raw_text", "") or "").split())


def find_dangling_references(atoms: list[Any], *, project_id: str, filenames: list[str] | None = None) -> int:
    """Mark the sentences that point at something the deal does not hold.

    Nothing is minted. "Alecandrich refers to a conversation the deal does not
    hold" was my sentence, not his -- an atom nobody said, which is the same
    mistake as "the sender calls this a small job". The reading rides on the
    line that points at the missing thing, and the deal's chase list is every
    atom carrying one.

    One per kind per deal: "as discussed" shows up in six mails of one thread
    and it is the same missing call.
    """
    marked = 0
    seen: set[str] = set()
    # An attachment the deal actually holds is not missing. Anything that is
    # not a mail or a note is a document somebody sent us.
    has_files = any(
        not str(f).lower().endswith((".eml", ".msg", ".txt"))
        for f in (filenames or [])
    )

    for atom in atoms:
        text = _text(atom)
        if not text or _value(atom).get("chatter"):
            continue
        head = text[:400]
        for kind, pattern in ((CONVERSATION, _CONVERSATION_RE), (ATTACHMENT, _ATTACHMENT_RE)):
            if kind in seen or not pattern.search(head):
                continue
            if kind == ATTACHMENT and has_files:
                continue
            seen.add(kind)
            v = _value(atom)
            said_by = v.get("said_by") if isinstance(v.get("said_by"), dict) else {}
            who = str(said_by.get("name") or said_by.get("email") or "the sender")
            prior = bool(_PRIOR_RELATIONSHIP_RE.search(head))
            from app.core.deal_signals import _reads

            _reads(
                atom,
                "chase",
                kind,
                why=(
                    f"{who} refers to a conversation the deal does not hold"
                    + (" and has worked with us before, so it may predate this deal" if prior else "")
                    if kind == CONVERSATION
                    else f"{who} refers to an attachment the deal does not hold"
                ),
                confidence=0.6,
            )
            val = _value(atom)
            val["chase"] = {
                "kind": kind,
                "prior_relationship": prior,
                "ask": (
                    "ask them what was agreed rather than hunting this thread"
                    if prior and kind == CONVERSATION
                    else ("find it or ask what was agreed" if kind == CONVERSATION else "ask for the file")
                ),
            }
            atom.value = val
            marked += 1
    return marked


__all__ = ["find_dangling_references", "CONVERSATION", "ATTACHMENT"]
