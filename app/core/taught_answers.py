"""A question somebody has already answered must stop being asked.

`resolve_open_questions` closes a question when a fact elsewhere in the corpus
shares an answer-bearing entity key with it. That catches "how many drops?"
answered by an atom carrying `count:212`. It cannot catch an answer that names
nothing in common with the question, and those are most of them.

Live 010180, in one thread three messages apart:

    "Can you try listening to the recording below, and see if we can get
     budgetary numbers together?"
    "Any chance you have another way of sharing the recording? It's not
     letting me pull it up due to access restraints."
    "I think the best bet is reviewing the notes I sent over, that has
     everything."

The first is resolved -- its answer is on the next line. The second is still
`needs_review` on every compile, so a PM is still being asked to chase a
recording the customer withdrew two messages later. "The notes I sent over,
that has everything" shares no entity key with "the recording", and key
overlap is blind to it.

A labeler had already said so. Fifteen `answers` edges on this deal, one of
them exactly this pair. They reach Postgres and the training blob, they train
`edge_relation` -- and they reach no compile, because links are the one thing
the labelling API does not forward as a correction. Teaching it changed
nothing.

So this asks the decide() STORE instead: has anybody said this question is
answered? Store-only, no LLM, guess-free -- an abstain leaves the question
exactly as it was. The same contract as every other taught seam.
"""
from __future__ import annotations

from typing import Any, Optional

from app.core import telemetry as _telemetry

RELATION = "question_answered"
CANDIDATES: tuple[str, str] = ("answered", "open")
INSTRUCTION = (
    "Has this question already been answered somewhere in this deal "
    "(answered), or is it still something a PM must ask (open)?"
)

#: Set on the atom and its flags, matching what key-overlap resolution writes,
#: so every downstream reader treats both the same way.
TAUGHT_FLAG = "answered_by_teacher"


def _atom_text(atom: Any) -> str:
    return str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")


def _atom_type(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def resolve_taught_answers(atoms: list[Any], *, project_id: str = "") -> int:
    """Close any open question a teacher has said is answered. Returns the count.

    Mutates in place, the same fields `resolve_open_questions` sets, so a
    consumer cannot tell which route closed a question -- only that it is
    closed and by whom.
    """
    try:
        from app.core.decide import DecisionScope, decide, get_store
        from app.core.schemas import ReviewStatus
    except Exception:  # pragma: no cover
        return 0
    if get_store() is None:
        return 0

    scope = DecisionScope(deal_id=str(project_id or ""))

    # SAY HOW FAR THROUGH THIS IS.
    #
    # Measured over nine live compiles, `open_question_resolution` is the
    # LARGEST stage in the pipeline -- 204s median, 278s max, 36.8% of the
    # compile -- and larger than `typed_atom_classification`, which was the
    # only stage with a counter. The cost is one decide() per open question,
    # right below.
    #
    # The candidates are gathered first so the denominator is the work this
    # stage will actually do. Counting all atoms would put the bar at a few
    # percent on a corpus where only the questions cost anything, and a
    # denominator the loop never approaches cannot produce a usable rate.
    _candidates = []
    for atom in atoms:
        if _atom_type(atom) != "open_question":
            continue
        value = getattr(atom, "value", None)
        if isinstance(value, dict) and value.get("answered") is True:
            continue  # key overlap already closed it
        if not _atom_text(atom):
            continue
        _candidates.append(atom)
    _total = len(_candidates)
    if _total:
        _telemetry.set_stage_progress(0, _total)

    resolved = 0
    for _i, atom in enumerate(_candidates, 1):
        value = getattr(atom, "value", None)
        text = _atom_text(atom)
        try:
            d = decide(RELATION, text[:600], list(CANDIDATES),
                       instruction=INSTRUCTION, scope=scope, model=None)
        except Exception:
            # In the `finally` below the tick still happens: a question whose
            # decision RAISED is still a question this stage is done with, and
            # counting only the successes would stall the bar on exactly the
            # compile that is going wrong.
            continue
        finally:
            if _total:
                _telemetry.set_stage_progress(_i, _total)
        # Only a taught verdict closes a question. A model guess does not: the
        # cost of wrongly closing one is a fact nobody ever chases, and that is
        # the failure this whole deal has been about.
        if getattr(d, "verdict", None) != "answered":
            continue
        if getattr(d, "source", "") != "store":
            continue
        if isinstance(value, dict):
            value["answered"] = True
            value["answered_by"] = "teacher"
        flags = list(getattr(atom, "review_flags", None) or [])
        for flag in ("answered_in_corpus", TAUGHT_FLAG):
            if flag not in flags:
                flags.append(flag)
        try:
            atom.review_flags = sorted(set(flags))
        except Exception:
            pass
        if getattr(atom, "review_status", None) == ReviewStatus.needs_review:
            atom.review_status = ReviewStatus.auto_accepted
        resolved += 1
    return resolved


__all__ = ["RELATION", "CANDIDATES", "INSTRUCTION", "TAUGHT_FLAG",
           "resolve_taught_answers"]
