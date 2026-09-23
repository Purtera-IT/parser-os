"""What each head is allowed to see.

Every head used to be served one string: the atom's text plus its table,
section and lead-in (``DECIDE_TEXT_VERSION`` 2). That is the right context for
deciding a TYPE, and it is the wrong context for almost everything else a
labeler records.

The clearest case is ``blocked_on``. "If you all would be able to do something
like this, I will get a conversation going with the club owner" is blocked on
US when a reseller writes it to us, and blocked on THEM when we write the same
words to them. Identical text, opposite answer, and the only thing that
separates them is who was speaking to whom -- which v2 does not include. A
head trained on that label could not learn it; it could only memorise the
sentence.

So context is routed per head: each relation declares the parts it needs, and
nothing more. Less is deliberate. A part a head cannot use is noise it has to
learn to ignore, and every extra token is budget spent away from the words.

``atom_type`` keeps EXACTLY the v2 string, byte for byte, so the rows already
in ``_training_*.db`` stay in distribution against new ones. Only the heads
that never had gold get the wider view.
"""
from __future__ import annotations

from typing import Any, Iterable

#: Bumped when the rendering of any part below changes. Rows carry it so a
#: trainer can refuse to mix two spellings of the same context.
CONTEXT_VERSION = 1

#: Exactly what DECIDE_TEXT_VERSION 2 appends, in its order. Do not reorder.
V2_PARTS: tuple[str, ...] = ("table", "section", "lead_in")

#: relation -> the parts appended after the atom's own words.
HEAD_CONTEXT: dict[str, tuple[str, ...]] = {
    # The type head's gold predates this file. Its string must not move.
    "atom_type": V2_PARTS,
    "atom_type_coarse": V2_PARTS,
    "facet": V2_PARTS,
    # Is this about the job, the account, a partner, or how we work? The
    # sender's company decides it more often than the words do: the same
    # sentence from cdw.com and from one of our own people is `partner` and
    # `internal`.
    "about": V2_PARTS + ("doc", "from"),
    # What does it want from us? Who is asking, and what follows it -- an ask
    # with its answer underneath wants nothing.
    "wants": V2_PARTS + ("from", "below"),
    # A conditional's gate is a person, and the parties name them.
    "reads:blocked_on": ("from",),
    # An announcement is only an announcement if something follows.
    "reads:opens_block": ("lead_in", "below"),
    # Sizing is in the words; the list underneath says what is being sized.
    "reads:job_scale": ("lead_in", "below"),
    # "As discussed earlier" is a gap only when the thing is not in the deal,
    # and the neighbours are the cheapest evidence of that.
    "reads:chase": ("above", "below", "doc"),
    "reads:introduces_party": ("from",),
    "reads:commitment": ("from",),
    "reads:small_talk": ("from", "below"),
}

#: Any reading without an entry above. Wide rather than wrong.
DEFAULT_READ_CONTEXT: tuple[str, ...] = V2_PARTS + ("from", "below")

_NEIGHBORS = 2


def _as_list(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v] if v else []
    return [str(x) for x in (v or []) if x]


def _party(p: Any) -> str:
    """A party as the head should see it: company and role, not a name.

    A name is a memorisable token -- "Alec" appears in one deal. "cdw.com
    (reseller, theirs)" is the thing that generalises to the next reseller.
    """
    if not isinstance(p, dict):
        return ""
    bits = [str(p.get("company") or "").strip()]
    role = str(p.get("role") or "").strip()
    side = str(p.get("side") or "").strip()
    inner = ", ".join(x for x in (role, side) if x)
    if inner:
        bits.append(f"({inner})")
    out = " ".join(b for b in bits if b).strip()
    return out


def _from_part(label: dict[str, Any]) -> str:
    by = _party(label.get("said_by"))
    to = [_party(p) for p in (label.get("said_to") or []) if isinstance(p, dict)]
    to = [t for t in to if t][:2]
    if not by and not to:
        return ""
    if label.get("internal_only"):
        return f"{by or 'unknown'} -> our own people".strip()
    return f"{by or 'unknown'} -> {', '.join(to) if to else 'unknown'}"


def _render(part: str, label: dict[str, Any]) -> str:
    if part == "table":
        v = str(label.get("table_ref") or "").strip()
        return f" [table: {v}]" if v else ""
    if part == "section":
        v = " > ".join(_as_list(label.get("section")))[:200]
        return f" [section: {v}]" if v else ""
    if part == "lead_in":
        v = " › ".join(_as_list(label.get("lead_in")))[:300]
        return f" [intro: {v}]" if v else ""
    if part == "doc":
        v = str(label.get("doc_type") or "").strip()
        return f" [doc: {v}]" if v else ""
    if part == "from":
        v = _from_part(label)
        return f" [from: {v}]" if v else ""
    if part == "above":
        v = " | ".join(_as_list(label.get("neighbors_above"))[:_NEIGHBORS])[:300]
        return f" [above: {v}]" if v else ""
    if part == "below":
        v = " | ".join(_as_list(label.get("neighbors_below"))[:_NEIGHBORS])[:300]
        return f" [below: {v}]" if v else ""
    raise KeyError(f"unknown context part {part!r}")


def parts_for(relation: str) -> tuple[str, ...]:
    if relation in HEAD_CONTEXT:
        return HEAD_CONTEXT[relation]
    if relation.startswith("reads:"):
        return DEFAULT_READ_CONTEXT
    return V2_PARTS


def context_text(relation: str, label: dict[str, Any], *, text: str | None = None) -> str:
    """The exact string this head is served for this label."""
    body = " ".join(str(text if text is not None else label.get("text") or "").split())
    out = body
    for part in parts_for(relation):
        out += _render(part, label)
    return out


def context_note(relation: str) -> dict[str, Any]:
    """What went into it, so a row can say why it looks the way it does."""
    return {"context_version": CONTEXT_VERSION, "context_parts": list(parts_for(relation))}


__all__ = [
    "CONTEXT_VERSION",
    "HEAD_CONTEXT",
    "DEFAULT_READ_CONTEXT",
    "V2_PARTS",
    "parts_for",
    "context_text",
    "context_note",
]
