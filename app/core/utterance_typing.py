"""Every spoken turn gets a type, even when no pattern recognises it.

The transcript parser types a turn by shape: a question mark, an action verb,
a constraint word. A turn none of those fire on used to come out as
``raw_utterance`` -- a type the labeling taxonomy does not have, so a long call
reached the labeler as hundreds of atoms with no type at all (one 582-turn
call: every turn an atom, none typed).

Email does not do that. A body line no pattern recognises falls back to a
coarse prose type, and a pleasantry goes down the admission chatter path. A
turn is the same kind of evidence as an email sentence, so it is typed the same
way:

* banter ("Hope you had a great 4th of July!") -> ``deal_metadata``, kept as an
  admission-chatter atom (flagged ``chatter`` + ``admission_regex``), so the
  admission head sees the negative and no other head reads it;
* anything else -> :func:`app.core.atom_typing.classify_prose`, the one coarse
  prose typer every format shares, flagged :data:`FALLBACK_TYPED_FLAG`.

The flag is what keeps this from being a quality regression. A fallback type
is a GUESS for a labeler to confirm, not a claim: atoms carrying it are
treated everywhere downstream exactly as an untyped turn was -- context, not
review work; never folded into a longer turn by word overlap; never a packet
anchor; never one side of a cross-document conflict. :func:`is_untyped_speech`
is the one test every such place uses.
"""
from __future__ import annotations

import re
from typing import Any

#: A spoken turn whose type came from the fallback, not from a pattern.
FALLBACK_TYPED_FLAG = "utterance_fallback_typed"
#: Set by the substance gate on a shape-typed turn it found ungrounded.
SMALLTALK_DEMOTED_FLAG = "transcript_smalltalk_demoted"


_QUESTION_LEAD_RE = re.compile(
    r"^(?:who|what|when|where|why|how|can|could|should|would)\b", re.I)


def fallback_utterance_type(text: str) -> tuple[Any, str | None]:
    """``(AtomType, chatter_reason)`` for a turn no pattern typed.

    ``chatter_reason`` is ``"banter"`` when the turn should be minted as an
    admission-chatter ``deal_metadata`` atom, else ``None``.
    """
    from app.core.atom_typing import classify_prose
    from app.core.schemas import AtomType
    from app.core.sentences import sentence_kind

    t = " ".join(str(text or "").split())
    if t and sentence_kind(t) == "banter":
        return AtomType.deal_metadata, "banter"
    # Speech drops its question marks: "could we swap to a nearby ready site"
    # is a question with or without one. Same lead-word rule the email body
    # typer uses, held to four words so "How about that." stays a remark.
    if _QUESTION_LEAD_RE.match(t) and len(t.split()) >= 4 and not t.endswith("."):
        return AtomType.open_question, None
    return classify_prose(t), None


#: Words that make a line about a scope / site / quantity / schedule /
#: staffing fact. A question carrying one asks for something the job needs
#: ("How many devices per school?", "Can we swap to a nearby ready site?"),
#: and a statement carrying one states it ("There is no stack coordinator").
#: Not a deal vocabulary -- the nouns any field-services job is made of.
DEAL_FACT_RE = re.compile(
    r"\b(?:"
    r"how\s+many|how\s+much|number\s+of|quantit(?:y|ies)|qty|"
    r"per\s+(?:school|site|room|floor|building|location|classroom|campus|store|office)|"
    r"sites?|locations?|schools?|buildings?|campus(?:es)?|stores?|offices?|"
    r"address(?:es)?|swap|devices?|units?|drops?|"
    r"deadline|timeline|go[- ]live|cutover|kick-?off|start\s+date|"
    r"scope|coordinators?|technicians?|techs?|crews?|staff(?:ing|ed)?|onsite|on-site"
    r")\b",
    re.I,
)


def asks_or_states_deal_fact(text: str) -> bool:
    """True when the line is about a scope/site/quantity/schedule/staffing fact."""
    return bool(DEAL_FACT_RE.search(str(text or "")))


def _atype(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    if at is None and isinstance(atom, dict):
        at = atom.get("atom_type")
    return str(getattr(at, "value", at) or "")


def _flags(atom: Any) -> list[str]:
    if isinstance(atom, dict):
        return [str(f) for f in (atom.get("review_flags") or [])]
    return [str(f) for f in (getattr(atom, "review_flags", None) or [])]


def is_untyped_speech(atom: Any) -> bool:
    """A spoken turn no pattern typed: ``raw_utterance`` (legacy), a turn typed
    by the fallback, or a typed turn the substance gate found ungrounded."""
    if _atype(atom) == "raw_utterance":
        return True
    flags = _flags(atom)
    return FALLBACK_TYPED_FLAG in flags or SMALLTALK_DEMOTED_FLAG in flags


__all__ = [
    "FALLBACK_TYPED_FLAG",
    "SMALLTALK_DEMOTED_FLAG",
    "DEAL_FACT_RE",
    "asks_or_states_deal_fact",
    "fallback_utterance_type",
    "is_untyped_speech",
]
