"""Ask the learned store whether a question is worth asking.

A PM who rejects a question is teaching, not tidying: "the customer cannot
answer this", "this is boilerplate", "the SOW already says it". The store has
held a ``gap`` head since it was built — relation ``gap_valid`` in
:data:`app.core.pm_feedback.HEAD_REGISTRY` — but nothing ever consulted it, so
a rejection banked a correction that could never fire and the same question
came back on the next deal.

This module is the consumer. It is deliberately thin:

* **Semantic, not lexical.** The store resolves through an embedding
  prototype, so "Who signs site acceptance at Palo Alto Office?" fires on the
  correction a PM made against "Who signs acceptance at the Macon site?" —
  a phrasing that shares no rule id and few words.
* **Scoped.** A rejection tagged as a fact about one deal only fires inside
  that deal; a judgment about the rule itself fires everywhere. The store
  resolves narrowest-first, so the caller just passes the deal.
* **Silent when unsure.** No store, no embedder, no confident hit — the
  verdict is ``None`` and the caller shows the question. A learned system
  that hides a real question when it is offline is worse than one that
  occasionally repeats itself.
"""

from __future__ import annotations

from typing import Any, Iterable

# The relation the `gap` head governs. Single source of truth is
# pm_feedback.HEAD_REGISTRY["gap"].relation; asserted in tests.
GAP_RELATION = "gap_valid"

#: The store only fires on a verdict inside this set.
GAP_VERDICTS: tuple[str, ...] = ("valid", "invalid")

#: What a rejection teaches: this ask is not worth putting to the customer.
VERDICT_INVALID = "invalid"


def _store(explicit: Any = None) -> Any:
    if explicit is not None:
        return explicit
    try:
        from app.core.decide import get_store

        return get_store()
    except Exception:
        return None


def _scope(deal_id: str) -> Any:
    from app.core.decide import DecisionScope

    return DecisionScope(deal_id=str(deal_id or ""))


def screen_question(
    text: str,
    *,
    deal_id: str = "",
    context: str = "",
    store: Any = None,
) -> dict[str, Any]:
    """One question → ``{verdict, correction_id, confidence}``.

    ``verdict`` is ``"invalid"`` when a PM's rejection covers this question,
    ``"valid"`` when a PM explicitly kept one like it, and ``None`` when
    nothing in the store speaks to it.
    """
    out: dict[str, Any] = {"verdict": None, "correction_id": "", "confidence": 0.0}
    probe = " ".join(str(text or "").split())
    if not probe:
        return out
    st = _store(store)
    if st is None:
        return out
    try:
        hit = st.resolve(
            relation=GAP_RELATION,
            text=probe,
            candidates=list(GAP_VERDICTS),
            context=str(context or ""),
            scope=_scope(deal_id),
            instruction="",
            relations=None,
        )
    except Exception:
        return out
    if hit is None or getattr(hit, "verdict", None) not in GAP_VERDICTS:
        return out
    out["verdict"] = hit.verdict
    out["correction_id"] = str(getattr(hit, "correction_id", "") or "")
    try:
        out["confidence"] = round(float(getattr(hit, "confidence", 0.0) or 0.0), 4)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out


def screen_questions(
    questions: Iterable[dict[str, Any]],
    *,
    deal_id: str = "",
    store: Any = None,
) -> list[dict[str, Any]]:
    """Screen many. Each item needs ``text``; ``id`` and ``rule_id`` ride along.

    Never raises and never reorders: the result is one row per input, in order.
    """
    st = _store(store)
    results: list[dict[str, Any]] = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        row = {
            "id": str(q.get("id") or ""),
            "rule_id": str(q.get("rule_id") or ""),
            "text": " ".join(str(q.get("text") or "").split()),
        }
        row.update(
            screen_question(
                row["text"],
                deal_id=deal_id or str(q.get("deal_id") or ""),
                context=str(q.get("context") or ""),
                store=st,
            )
        )
        results.append(row)
    return results


def suppressed_ids(results: Iterable[dict[str, Any]]) -> list[str]:
    """The ids a PM has already taught us not to ask."""
    return [
        str(r.get("id") or r.get("rule_id") or r.get("text") or "")
        for r in results or []
        if r.get("verdict") == VERDICT_INVALID
    ]


def drop_learned_bad_questions(
    questions: list[dict[str, Any]],
    *,
    text_key: str = "summary",
    deal_id: str = "",
    store: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition generated questions into (asked, suppressed).

    Used at the source: the SRL gap generator mints questions from what is
    missing, and a PM's standing judgment about an ask should stop it being
    minted at all, not just hidden downstream.
    """
    if not questions:
        return [], []
    st = _store(store)
    if st is None:
        return list(questions), []
    asked: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for q in questions:
        text = str((q or {}).get(text_key) or "")
        verdict = screen_question(text, deal_id=deal_id, store=st).get("verdict")
        if verdict == VERDICT_INVALID:
            dropped.append(q)
        else:
            asked.append(q)
    return asked, dropped


__all__ = [
    "GAP_RELATION",
    "GAP_VERDICTS",
    "VERDICT_INVALID",
    "screen_question",
    "screen_questions",
    "suppressed_ids",
    "drop_learned_bad_questions",
]
