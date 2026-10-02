"""Relationship talk is a JUDGEMENT, so it is a head, not a rule.

A deal thread is mostly people being people, and a PM should not be asked to
type "Thank you for bringing this our way". But deciding what counts is
exactly the judgement a pattern cannot make: this module once hid "I will get
a conversation going with the club owner" -- the sentence naming the deal's
decision maker -- because it contained the phrase "get a conversation going".

So nothing here hides anything any more. The rule leaves a PREDICTION on the
atom (``reads: small_talk``) that a labeler confirms or drops, and those
labels train the head that replaces it. Until the head exists, an atom is
hidden from the queue only when a HUMAN has labeled it ``small_talk``.
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
    r"(?:will|ill|i'll|we'll|we will|i will) (?:get back|come back|revert|follow up|circle back|be in touch|keep you (?:posted|updated|in the loop))|"
    r"^will be in touch|"
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
    r"^(?:(?:well|ok(?:ay)?|oh|and)[,!]?\s+)?(?:"
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


#: "I will get a conversation going with the club owner" is a PROMISE. It
#: happens to contain a phrase that also shows up in pipeline chatter, and the
#: rule hid the sentence that named the deal's decision maker. A first-person
#: undertaking is never small talk, whatever else it sounds like.
#:
#: The apostrophe is required: ``we'?ll`` read "Well, thanks again for your
#: time" as an undertaking, so the pleasantry could never be chatter (the same
#: rule app.core.sentences already applies).
_PROMISE_RE = re.compile(
    r"\b(?:i|we)\s*(?:['’]ll|will|can|shall)\s+\w+|\b(?:i|we)['’]ll\b|"
    r"\b(?:i|we)\s+(?:am|are)\s+going\s+to\b|\blet me\s+\w+",
    re.I,
)


def _is_undertaking(text: str) -> bool:
    """A first-person promise about the work.

    "We'll get back to you" / "I'll circle back" promise only to keep talking:
    with the hand-off phrase removed nothing is promised and nothing about the
    job is left, so they are not protected. "We'll schedule the techs once the
    floor is done" (a PM/PC/SA commitment) promises work and always is.
    """
    if not _PROMISE_RE.search(text):
        return False
    rest = _HANDOFF_RE.sub(" ", text)
    if rest == text or _PROMISE_RE.search(rest):
        return True
    try:
        from app.core.sentences import _WORK_CUE_RE
    except Exception:  # pragma: no cover
        return True
    return bool(_WORK_CUE_RE.search(rest))


#: A wait or a dependency is a fact about the job's timeline, never small
#: talk: "we are waiting for the tv to arrive (it is with the shipping carrier
#: now)" says delivery gates the install (010003), whatever chatty clause
#: rides along with it.
_DEPENDENCY_RE = re.compile(
    r"\b(?:"
    r"waiting (?:for|on)|wait(?:ing)? until|pending|on hold|in transit|back-?order\w*|"
    r"(?:shipping )?carrier|tracking (?:number|status)|delivery|deliver(?:ed|s)?|"
    r"arriv(?:e|es|ed|al)|ship(?:s|ped|ping|ment)?|once (?:we|they|it|the)\b|"
    r"until (?:we|they|it|the)\b|depends on|dependent on|blocked (?:on|by)|"
    r"need(?:s|ed)? (?:the |a |an |)po\b|purchase order"
    r")\b",
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
    # A greeting is judged as a greeting and the rest of the line on its own:
    # "Hello, we already have wall mounts, and I believe parking is not free"
    # (010003) was hidden behind "Show small talk" for its first word. A line
    # that is nothing BUT a greeting is small talk.
    from app.core.greetings import starts_with_greeting, strip_leading_greeting

    if starts_with_greeting(t):
        rest = strip_leading_greeting(t)
        if not rest or not re.search(r"[A-Za-z0-9]", rest):
            return True
        t = rest
    for k in entity_keys or []:
        if str(k).startswith(("device:", "vendor:", "quantity:", "site:", "req")):
            return False
    if _SUBSTANCE_RE.search(t):
        return False
    if _is_undertaking(t):
        return False
    if _DEPENDENCY_RE.search(t):
        return False
    return bool(_PIPELINE_RE.search(t) or _HANDOFF_RE.search(t) or _SOCIAL_RE.match(t))


def _under_exclusion_heading(atom: Any) -> bool:
    try:
        from app.parsers.sow_sections import under_exclusion_heading
    except Exception:
        return False
    for ref in list(getattr(atom, "source_refs", None) or [])[:1]:
        loc = getattr(ref, "locator", None)
        if isinstance(loc, dict) and under_exclusion_heading(loc.get("section_path") or []):
            return True
    return False


def mark_chatter(atoms: list[Any]) -> int:
    """Leave a small-talk PREDICTION on prose that reads as relationship talk.

    Returns how many were marked. Nothing is hidden and no type changes: the
    labeler sees the guess as a chip and has the last word.

    Only prose atoms are considered: a list item, a BOM line, a site or a
    person is never chatter, however chatty the sentence around it was.
    """
    prose_types = {"scope_item", "deal_metadata", "customer_instruction"}
    marked = 0
    for atom in atoms:
        at = getattr(atom, "atom_type", None)
        at = at.value if hasattr(at, "value") else str(at or "")
        if at not in prose_types:
            continue
        val = getattr(atom, "value", None)
        if not isinstance(val, dict) or val.get("list_item"):
            continue
        if any(r.get("key") == "small_talk" for r in val.get("reads") or ()):
            continue  # already predicted
        # A sentence a fact was read out of is never small talk, whatever it
        # sounds like: "Here are the details for the small job" says the job
        # is small, and hiding it throws that away.
        if val.get("signals"):
            continue
        # A line under an "Out of Scope" / "Exclusions" heading is a thing we
        # will not do, however chatty or short it reads ("Long-term
        # warehousing...", "Chromebook imaging...").
        if _under_exclusion_heading(atom):
            continue
        text = getattr(atom, "raw_text", "") or ""
        if not is_chatter(text, entity_keys=list(getattr(atom, "entity_keys", None) or [])):
            continue
        # A prediction, not a verdict: the atom stays in the queue, keeps its
        # type, and carries the guess for a human to confirm or drop.
        from app.core.deal_signals import _reads

        _reads(atom, "small_talk", True, why="reads as relationship talk, not a statement about the work",
               confidence=0.5)
        flags = list(getattr(atom, "review_flags", None) or [])
        if CHATTER_FLAG not in flags:
            flags.append(CHATTER_FLAG)
        atom.review_flags = flags
        marked += 1
    return marked


__all__ = ["is_chatter", "mark_chatter", "CHATTER_FLAG"]
