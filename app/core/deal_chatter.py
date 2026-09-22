"""Relationship talk is not scope.

A deal thread is mostly people being people: thanks, banter, pipeline hopes,
"sending this to my solutions team". Those lines are real — they belong in the
record and they are evidence of who said what — but they are not statements
about the work, and a PM asked to label them is being asked a question with no
answer. Live 010289: 11 of 49 atoms were this.

So we mark, never delete. ``mark_chatter`` sets ``chatter`` on the value and
flags the atom; the labeling walk hides it, the heads skip it, and the audit
trail still has it.
"""
from __future__ import annotations

import re
from typing import Any

CHATTER_FLAG = "chatter"

#: Wishing the deal well. Pipeline talk about the opportunity ITSELF (more of
#: these, a bigger relationship, a conversation with the owner) never
#: describes the job being quoted.
_PIPELINE_RE = re.compile(
    r"\b(?:"
    r"lead to (?:many )?more|more of the same|much more (?:business|work)|"
    r"(?:get|start) a conversation going|if this is a success|"
    r"successful implementation|opportunit(?:y|ies)|"
    r"(?:new|another) opty|looking forward to|excited to"
    r")\b",
    re.I,
)

#: Handing the thread along: who is doing the replying, not what is being built.
_HANDOFF_RE = re.compile(
    r"\b(?:"
    r"sending (?:it|this) (?:over|along)|"
    r"(?:will|ill|i'll) (?:get back|come back|revert|follow up|circle back)|"
    r"(?:looping|adding|copying) (?:in |)\w+|"
    r"(?:passing|handing) (?:it|this) (?:to|over)|"
    r"(?:lmk|let me know) if (?:there are |you have |)(?:any |)(?:follow[- ]?up |)questions|"
    r"(?:my|our) solutions team|"
    r"here are the details|"
    r"(?:sorry|apologies),? (?:left|missed|forgot)"
    r")\b",
    re.I,
)

#: Gratitude and greetings that carry no object.
_SOCIAL_RE = re.compile(
    r"^(?:"
    r"thank(?:s| you)\b|thanks!|much appreciated|appreciate it|no worries|"
    r"happy to help|sounds good|will do|got it|perfect|great|awesome|"
    r"hope (?:you|all)|good (?:morning|afternoon|evening)"
    r")",
    re.I,
)

#: A number, a spec, a product, a date, a site: the line is about the work.
_SUBSTANCE_RE = re.compile(
    r"\b(?:\d|ft\b|feet\b|cat5|cat6|rs232|usb|relay|lock|door|maglock|cable|"
    r"switch|rack|camera|licen[cs]e|install|mount|cutover|site|floor|suite)\b",
    re.I,
)


def is_chatter(text: str, *, entity_keys: list[str] | None = None) -> bool:
    """Is this line relationship talk rather than a statement about the work?

    Conservative on purpose: anything carrying a number, a product word or a
    resolved device/site/quantity entity is work, whatever else it says. A
    pipeline sentence that also names the hardware stays.
    """
    t = " ".join(str(text or "").split())
    if not t:
        return False
    for k in entity_keys or []:
        if str(k).startswith(("device:", "vendor:", "quantity:", "site:", "req")):
            return False
    if _SUBSTANCE_RE.search(t):
        return False
    return bool(_PIPELINE_RE.search(t) or _HANDOFF_RE.search(t) or _SOCIAL_RE.match(t))


def mark_chatter(atoms: list[Any]) -> int:
    """Flag relationship talk in place. Returns how many were marked.

    Only prose atoms are considered: a list item, a BOM line, a site or a
    person is never chatter, however chatty the sentence around it was.
    """
    from app.core.schemas import AtomType

    prose_types = {"scope_item", "deal_metadata", "customer_instruction"}
    marked = 0
    for atom in atoms:
        at = getattr(atom, "atom_type", None)
        at = at.value if hasattr(at, "value") else str(at or "")
        if at not in prose_types:
            continue
        val = getattr(atom, "value", None)
        if not isinstance(val, dict) or val.get("list_item") or val.get("chatter"):
            continue
        text = getattr(atom, "raw_text", "") or ""
        if not is_chatter(text, entity_keys=list(getattr(atom, "entity_keys", None) or [])):
            continue
        val["chatter"] = True
        val["retagged_from"] = val.get("retagged_from") or at
        try:
            atom.atom_type = AtomType.deal_metadata
        except Exception:
            atom.atom_type = "deal_metadata"
        flags = list(getattr(atom, "review_flags", None) or [])
        for f in (CHATTER_FLAG, "head_exclude"):
            if f not in flags:
                flags.append(f)
        atom.review_flags = flags
        marked += 1
    return marked


__all__ = ["is_chatter", "mark_chatter", "CHATTER_FLAG"]
