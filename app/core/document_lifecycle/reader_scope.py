"""Which documents a given model is allowed to read, and why.

THE PROBLEM
-----------
Training a Deal Kit model on a deal that is already finished is only honest if
the model sees what the *person* saw when they built the kit. Feed it documents
that arrived afterwards and it learns from its own answer: it scores brilliantly
on history and is useless on a live deal. That is temporal leakage, and on a
manually-worked corpus it is the default outcome, not an edge case.

THE CUT
-------
A deal's own pipeline stages, on the dates HubSpot recorded, decide it.

    Open / Awaiting Scope     the customer is telling us what they need
    Submitted for Quoting     Chase is quoting -- the Deal Kit is being built
    Decision Pending          the quote exists; negotiation over OUR output
    Closed Won / Lost         delivery, or nothing

So a Deal Kit model reads everything up to the END of Submitted for Quoting.
That window is working time, not output time: the kit is produced during it, not
before it. On deal 010215 the kit was authored 15:50:31 and the deal moved to
Decision Pending at 15:53:00 -- two and a half minutes later.

A SOW model reads that, plus Decision Pending, because a SOW is negotiated after
the quote goes out.

WHY THERE IS NO TIME BUFFER
---------------------------
The thing a buffer would be trying to exclude is our own output produced inside
the window -- the Deal Kit itself. A buffer is the wrong tool: arbitrary, and it
would drop real customer mail that happened to land late in the window. Our own
output is already excluded on a different axis, by TYPE: a ``label`` is never
evidence, whenever it arrived. The Deal Kit is excluded because it is ours, not
because of its timestamp.

WHAT THIS DOES NOT DO
---------------------
It does not decide whether a document is ours. That is ``admissibility``, and it
is a harder problem than it looks -- ten of deal 010215's customer documents are
named "SOW" and were classified as our output until direction was inferred from
who originated them. This module trusts that answer and gates on time.
"""
from __future__ import annotations

from typing import Any, Mapping

from app.core.document_lifecycle.deal_stage import _norm, _stage_index

#: Stage a consumer may read THROUGH, inclusive. ``None`` means no limit.
#: Ordering comes from the deal's own timeline, never from this list.
CONSUMERS: dict[str, dict[str, Any]] = {
    "deal_kit": {
        "through": "Submitted for Quoting",
        "why": "the Deal Kit is built during quoting; anything later postdates it",
    },
    "sow": {
        "through": "Decision Pending",
        "why": "a SOW is negotiated after the quote goes out",
    },
    "atlas": {
        "through": None,
        "reads": ("evidence", "reference", "atlas"),
        "why": "delivery reads the whole deal, delivery material included",
    },
}

#: Classes a consumer may read when it does not say otherwise. A label is our
#: own answer; letting a model read it while producing the same artifact is the
#: leak this exists to stop.
#:
#: Atlas is the exception and says so above: delivery material IS what it reads,
#: so applying the default here silently gave it the same 27 documents as the
#: SOW model and called that "the whole deal".
_READABLE = ("evidence", "reference")


def consumers() -> tuple[str, ...]:
    return tuple(CONSUMERS)


def visible_to(
    consumer: str,
    *,
    stage: str | None,
    admissible_for: str | None,
    timeline: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """May ``consumer`` read a document that arrived in ``stage``?

    Returns ``(visible, why)``. The reason is not decoration: every call is a
    claim about what a model was allowed to know, and a training set nobody can
    audit is one that has to be taken on faith.
    """
    spec = CONSUMERS.get(consumer)
    if spec is None:
        return False, f"unknown consumer {consumer!r}"

    # 1. Our own output is never input, whenever it arrived.
    readable = spec.get("reads") or _READABLE
    if admissible_for is not None and admissible_for not in readable:
        return False, f"admissible_for={admissible_for}; not readable by {consumer}"

    # 2. An unplaced document has no position to compare. Excluding it loses
    #    real material; including it silently asserts a date we do not have.
    #    Excluded, and said out loud, so an auditor can see the size of the gap.
    if not stage:
        return False, "no stage: undated artifact or no deal timeline; cannot place it before the cut"

    through = spec["through"]
    if through is None:
        return True, f"{consumer} reads the whole deal"

    cut_idx = _stage_index(through, timeline)
    here_idx = _stage_index(stage, timeline)

    # 3. Pre-deal material predates every stage and is always discovery context.
    if here_idx < 0 and _norm(stage) == _norm("Before deal created"):
        return True, "predates the deal; discovery context"

    # 4. A stage absent from this deal's timeline cannot be ordered. Same rule
    #    as an unplaced document: refuse, and say why.
    if cut_idx < 0:
        return False, f"{through!r} is not in this deal's timeline; cannot place the cut"
    if here_idx < 0:
        return False, f"{stage!r} is not in this deal's timeline; cannot place the document"

    if here_idx <= cut_idx:
        return True, f"arrived in {stage}, at or before {through}"
    return False, f"arrived in {stage}, after {through}; postdates what {consumer} produced"


def audit(
    documents: list[Mapping[str, Any]],
    *,
    consumer: str,
    timeline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Every document, with whether ``consumer`` may read it and why.

    Built for reading by a person before any training data is generated from it.
    """
    visible: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for doc in documents:
        block = doc.get("deal_stage") or {}
        stage = block.get("stage_at_arrival")
        adm = block.get("admissible_for") or (doc.get("lifecycle") or {}).get("admissible_for")
        ok, why = visible_to(consumer, stage=stage, admissible_for=adm, timeline=timeline)
        row = {
            "filename": doc.get("filename"),
            "stage": stage,
            "admissible_for": adm,
            "direction": doc.get("direction"),
            "authored_at": doc.get("authored_at"),
            "why": why,
        }
        (visible if ok else hidden).append(row)
    return {
        "consumer": consumer,
        "through": CONSUMERS.get(consumer, {}).get("through"),
        "visible_count": len(visible),
        "hidden_count": len(hidden),
        "visible": visible,
        "hidden": hidden,
    }
