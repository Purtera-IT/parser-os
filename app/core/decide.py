"""The single meaning-judgment chokepoint.

Every semantic decision the pipeline makes — "is this address a job site or the
selling party's letterhead?", "is this sheet an ordered BOM or a price book?",
"what typed atom is this scope_item?" — used to be answered by a scatter of
~270 lexical gates and a handful of hand-tuned LLM prompts. Each one was its
own brittle judgment with no shared memory, so the same mistake recurred deal
after deal and a PM's correction had nowhere to live.

``decide()`` funnels all of those judgments through one resolver with a fixed
precedence:

    1. STORE   — a confident hit in the feedback store (Phase 3) decides
                 instantly and deterministically, citing a correction_id. This
                 is where a PM's past correction (e.g. "PurTera is our company,
                 never a site") is enforced forever, with zero LLM cost.
    2. LLM     — only when the store is undecided: ask the model, optionally
                 primed with the nearest stored corrections as few-shot
                 examples (Phase 3). Hard discriminations route to a stronger
                 model; cheap ones keep the tiny default.
    3. FALLBACK — model disabled / unreachable / unparseable → ``verdict=None``
                 ("undecided"). Callers MUST treat this as "change nothing":
                 keep the atom, flag it ``needs_review``. We never act on a
                 guess and never silently drop.

Phase 2 contract: the store is not yet wired (``_STORE is None``), so
``decide()`` is a *transparent* pass-through to the existing
``semantic_role.classify_role`` primitive. Output is identical to calling that
primitive directly — this module adds the seam, not new behavior. Phase 3
registers a store via :func:`set_store` and the precedence above goes live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import app.core.semantic_role as semantic_role


@dataclass(frozen=True)
class DecisionScope:
    """Where a decision is allowed to draw corrections from.

    The store (Phase 3) resolves narrowest-first: a ``deal``-scoped correction
    overrides a ``pack`` one, which overrides ``global``. ``pack`` lets domain
    knowledge accrete per vertical (datacenter, healthcare, retail) without
    polluting the global layer. Carried through Phase 2 unused so call sites
    don't change again when the store lands.
    """

    deal_id: str = ""
    pack: str = ""


@dataclass
class Decision:
    """The resolved judgment for one ``decide()`` call.

    Attributes:
        verdict: one of ``candidates`` (or ``None`` when undecided — callers
            apply their own safe default, typically keep + flag).
        confidence: ``0.0``–``1.0``.
        source: ``"store"`` | ``"llm"`` | ``"fallback"`` — which tier decided.
        correction_id: set iff a stored correction decided it; this is what a
            downstream envelope cites so a PM can trace *why*. ``None`` until
            the store is wired (Phase 3).
        rationale: short human-readable explanation.
        neighbors: store hits that informed the decision (audit / few-shot
            trace). Empty in Phase 2.
    """

    verdict: str | None
    confidence: float = 0.0
    source: str = "fallback"
    correction_id: str | None = None
    rationale: str = ""
    neighbors: list[Any] = field(default_factory=list)


class FeedbackStore(Protocol):
    """Phase-3 contract. A store resolves a decision from past corrections.

    Returns a :class:`Decision` with ``source="store"`` on a confident hit, or
    ``None`` when it has nothing confident to say (so ``decide()`` falls through
    to the LLM). Must never raise and must be safe offline.
    """

    def resolve(
        self,
        *,
        relation: str,
        text: str,
        candidates: list[str],
        context: str,
        scope: DecisionScope,
        instruction: str,
        relations: dict | None,
    ) -> Decision | None: ...

    def few_shot(
        self,
        *,
        relation: str,
        text: str,
        scope: DecisionScope,
        k: int = 3,
    ) -> list[Any]: ...


# Process-wide store handle. ``None`` in Phase 2 (transparent pass-through);
# Phase 3 calls ``set_store`` to wire the real feedback store.
_STORE: FeedbackStore | None = None


def set_store(store: FeedbackStore | None) -> None:
    """Register (or clear) the process-wide feedback store. Phase-3 hook; also
    the test seam for injecting a deterministic fake store."""
    global _STORE
    _STORE = store


def get_store() -> FeedbackStore | None:
    return _STORE


# ── decision telemetry ───────────────────────────────────────────────
# Per-process counters proving the central claim: the LLM is consulted ONLY on
# the genuinely hard decisions. ``store_hits`` (incl. neural head) + confident
# resolutions answer the easy ones; ``llm_calls`` should fall as the head
# learns. ``llm_call_rate`` = llm_calls / decisions.
_STATS = {"decisions": 0, "store_hits": 0, "llm_calls": 0, "fallback": 0, "teacher_writes": 0}


def reset_decide_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def get_decide_stats() -> dict[str, float]:
    d = dict(_STATS)
    n = max(d["decisions"], 1)
    d["llm_call_rate"] = d["llm_calls"] / n
    d["store_hit_rate"] = d["store_hits"] / n
    return d


def _dump_stats_atexit() -> None:
    """If SOWSMITH_DECIDE_STATS_OUT is set, write the decision telemetry on
    process exit. Lets a subprocess compile report its LLM call-rate without an
    in-process handle — used by the live multi-deal validation harness."""
    path = os.environ.get("SOWSMITH_DECIDE_STATS_OUT", "").strip()
    if not path or _STATS["decisions"] == 0:
        return
    try:
        import json as _json
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(get_decide_stats(), fh)
    except Exception:  # pragma: no cover
        pass


import atexit as _atexit

_atexit.register(_dump_stats_atexit)


# Teacher-cache: when the LLM *does* fire (a hard case), persist its confident
# verdict as a weak correction so the head learns that region and stops paying
# for the LLM there next time. Opt-in (a compile that writes to the store is
# mutating shared state) — enabled for live validation via the env flag.
_TEACHER_MIN_CONF = float(os.environ.get("SOWSMITH_TEACHER_MIN_CONF", "0.85"))


def _teacher_cache_enabled() -> bool:
    return os.environ.get("SOWSMITH_TEACHER_CACHE", "") not in ("", "0", "false")


def decide(
    relation: str,
    text: str,
    candidates: list[str],
    *,
    instruction: str,
    context: str = "",
    scope: DecisionScope | None = None,
    relations: dict | None = None,
    model: str | None = None,
    timeout: int | None = None,
    llm: bool = True,
) -> Decision:
    """Resolve the role/type of ``text`` from ``candidates``.

    Precedence: store hit → LLM (few-shot primed) → safe fallback. See the
    module docstring. Never raises.

    Args:
        relation: the decision family (e.g. ``"physical_site"``,
            ``"atom_type"``). Corrections are grounded on the relation, so the
            store only applies a learned rule to the decision it was made for.
        text: the snippet under judgment.
        candidates: the closed verdict set.
        instruction: one-line neutral description of the decision (handed to the
            LLM, and to the store as the few-shot framing).
        context: surrounding source text that disambiguates ``text``.
        scope: deal/pack scope for store resolution (Phase 3).
        relations: structured grounding signals (e.g. ``{"owner": ...}``).
        model: optional LLM override for a hard discrimination.
        timeout: optional per-call LLM timeout.
        llm: when ``False``, skip the LLM tier entirely — only a confident store
            hit can decide, otherwise return the safe fallback (``verdict=None``).
            This is the store-fronts-regex seam: a caller that already owns a
            lexical fallback uses ``decide(..., llm=False)`` (or
            :func:`resolve_or`) to let the store *override* on a confident,
            context-rich hit without ever paying for an LLM round-trip. With no
            store wired this is always a transparent ``None`` — byte-identical
            to not calling ``decide`` at all.

    Returns:
        A :class:`Decision`. ``verdict is None`` means "undecided" — the caller
        keeps the atom and flags it for review.
    """
    scope = scope or DecisionScope()

    if not text or not candidates:
        return Decision(verdict=None, source="fallback", rationale="empty input")

    _STATS["decisions"] += 1

    # 1) STORE — a confident learned correction decides instantly (Phase 3).
    store = _STORE
    if store is not None:
        try:
            hit = store.resolve(
                relation=relation,
                text=text,
                candidates=candidates,
                context=context,
                scope=scope,
                instruction=instruction,
                relations=relations,
            )
        except Exception:  # pragma: no cover - store must never break decide()
            hit = None
        if hit is not None and hit.verdict is not None:
            _STATS["store_hits"] += 1
            return hit

    # 2) LLM — undecided by the store; ask the model, primed with nearest
    #    corrections as few-shot examples when a store is present (Phase 3).
    #    Skipped entirely when llm=False: the store-fronts-regex seam wants the
    #    store to *override* a lexical fallback, never to add LLM latency.
    if not llm:
        return Decision(
            verdict=None,
            confidence=0.0,
            source="fallback",
            rationale="store undecided; llm disabled, caller applies lexical fallback",
        )

    few_shot_examples: list[Any] = []
    if store is not None:
        try:
            few_shot_examples = store.few_shot(
                relation=relation, text=text, scope=scope
            )
        except Exception:  # pragma: no cover
            few_shot_examples = []

    _STATS["llm_calls"] += 1
    role, conf = semantic_role.classify_role(
        text,
        candidates,
        instruction=_with_examples(instruction, few_shot_examples),
        context=context,
        model=model,
        timeout=timeout,
    )

    # 3) FALLBACK — model disabled / unreachable / off-menu → undecided.
    if role is None:
        _STATS["fallback"] += 1
        return Decision(
            verdict=None,
            confidence=0.0,
            source="fallback",
            rationale="model undecided/unreachable; caller applies safe default",
            neighbors=few_shot_examples,
        )

    # Teacher-cache: the LLM just answered a HARD case. Persist a confident
    # verdict as a weak correction so the head learns this region and the LLM
    # is not consulted here again — the mechanism that decays llm_call_rate.
    if (
        _teacher_cache_enabled()
        and store is not None
        and conf >= _TEACHER_MIN_CONF
        and hasattr(store, "learn_from_teacher")
    ):
        try:
            store.learn_from_teacher(
                relation=relation, text=text, verdict=role,
                confidence=conf, scope=scope, instruction=instruction,
            )
            _STATS["teacher_writes"] += 1
        except Exception:  # pragma: no cover - never break decide()
            pass

    return Decision(
        verdict=role,
        confidence=conf,
        source="llm",
        correction_id=None,
        rationale="classified by model",
        neighbors=few_shot_examples,
    )


def resolve_or(
    relation: str,
    text: str,
    candidates: list[str],
    *,
    lexical: str | None,
    instruction: str = "",
    context: str = "",
    scope: DecisionScope | None = None,
    relations: dict | None = None,
) -> tuple[str | None, Decision | None]:
    """Store-fronts-regex resolution: let a confident store correction override
    a lexical/regex verdict, with **zero** LLM cost and a guaranteed safe
    default.

    This is the seam that lets the hand-tuned lexical gates keep deciding by
    default while a PM's learned correction takes precedence where it confidently
    applies. The store only speaks on a confident, context-rich hit
    (``source="store"``); otherwise the caller's existing ``lexical`` verdict is
    returned unchanged. Because the LLM tier is disabled, this can only ever
    *remove* an error a PM already flagged — it never introduces a new
    model-driven judgment, and with no store wired it returns ``lexical``
    byte-for-byte.

    Args:
        lexical: the verdict the caller's existing regex/keyword gate produced
            (may be ``None`` if the gate was itself undecided). This is the
            fallback returned whenever the store has nothing confident to say.

    Returns:
        ``(verdict, decision)`` where ``verdict`` is the store's verdict on a
        confident hit else ``lexical``, and ``decision`` is the store
        :class:`Decision` when it decided (so the caller can cite
        ``correction_id`` for provenance) else ``None``.
    """
    d = decide(
        relation,
        text,
        candidates,
        instruction=instruction,
        context=context,
        scope=scope,
        relations=relations,
        llm=False,
    )
    if d.source == "store" and d.verdict is not None:
        return d.verdict, d
    return lexical, None


def _with_examples(instruction: str, examples: list[Any]) -> str:
    """Append retrieved corrections to the instruction as few-shot guidance.

    No-op in Phase 2 (``examples`` is always empty). Kept here so Phase 3 only
    has to populate the store's ``few_shot`` method — the prompt-assembly seam
    already exists and is exercised by the same code path.
    """
    if not examples:
        return instruction
    lines = ["", "Worked examples from prior corrections (follow these):"]
    for ex in examples[:5]:
        text = getattr(ex, "text", None) or (ex.get("text") if isinstance(ex, dict) else "")
        verdict = getattr(ex, "verdict", None) or (ex.get("verdict") if isinstance(ex, dict) else "")
        if text and verdict:
            lines.append(f"- TEXT: {str(text)[:200]} -> {verdict}")
    return instruction + "\n".join(lines)


__all__ = [
    "decide",
    "resolve_or",
    "Decision",
    "DecisionScope",
    "FeedbackStore",
    "set_store",
    "get_store",
    "get_decide_stats",
    "reset_decide_stats",
]
