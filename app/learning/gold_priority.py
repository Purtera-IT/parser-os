"""Which item is worth a PM's next click — ranked by what it teaches the heads.

The existing review queue answers *"what is most likely wrong?"*. That is a
triage question. Active learning answers a different one: *"what will teach the
model the most?"* — and the two disagree in ways that matter.

The clearest example is confidence. The queue scores novelty as ``1 - confidence``,
which is monotonic: it always prefers the least confident item. But an atom the
head scores at 0.02 is usually junk — OCR noise, a stray fragment — and labeling
it teaches almost nothing. The item that teaches most sits at the **decision
boundary**, where the head is genuinely torn. Uncertainty has to peak at the
threshold, not run away to zero.

The four signals, and why each earns its weight:

* **boundary** — classic uncertainty sampling. Peaks at the abstain threshold and
  decays in BOTH directions, so confident-and-right and confident-and-noise both
  rank low.
* **starvation** — how few examples of this label the training log already holds.
  This is the signal that makes the queue about *learning* rather than *doubt*:
  the 500th ``scope_item`` teaches nothing, the first ``submission_req`` fills an
  empty class. With 12 fine classes sitting empty, this is where the leverage is.
* **impact** — an atom that governs a packet reaches the SOW, so a wrong label
  there costs a deliverable, not just a metric.
* **reach** — an atom whose delexicalized shape recurs across the corpus
  generalizes when corrected; a one-off does not.

Scores are bounded [0,1] and every contribution is recorded in ``reasons`` so a
PM (or an audit) can see WHY something was asked. Pure and side-effect-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# Weights sum to 1.0 so `score` stays interpretable as "share of the ideal ask".
W_BOUNDARY = 0.40
W_STARVATION = 0.30
W_IMPACT = 0.20
W_REACH = 0.10

# Above this many labeled examples a class is considered well covered; the
# ledger's readiness gate wants ~600-1200 atoms for the keystone head, and
# per-class saturation sets in far earlier than that.
SATURATED_AT = 40

# A shape seen this many times is fully "reaching" — beyond it, more repetition
# does not make the correction any more generalizable.
REACH_FULL = 10


@dataclass(frozen=True)
class GoldPriority:
    """A ranked ask, with the reasoning kept attached."""

    score: float
    reasons: list[str] = field(default_factory=list)


def boundary_uncertainty(confidence: float, threshold: float) -> float:
    """1.0 exactly at the decision boundary, decaying to 0 in both directions.

    Deliberately NOT ``1 - confidence``. A head that scores something 0.02 is not
    uncertain, it is confidently dismissive — usually of noise. The informative
    item is the one it cannot separate.
    """
    if threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    c = min(1.0, max(0.0, float(confidence)))
    # Each side is normalized by ITS OWN span, so the curve reaches 0 at both
    # c=0 and c=1. Using one shared span lets the narrower side decay too
    # slowly — with tau=0.72 a 0.99-confidence atom still scored 0.61 and got
    # labeled "at the boundary", which is exactly backwards.
    span = threshold if c < threshold else (1.0 - threshold)
    if span <= 0.0:
        return 0.0
    return round(max(0.0, 1.0 - abs(c - threshold) / span), 6)


def class_starvation(count: int) -> float:
    """1.0 for a label with no examples, decaying as the class fills up."""
    n = max(0, int(count))
    if n >= SATURATED_AT:
        return 0.0
    return round(1.0 / math.sqrt(1.0 + n), 6)


def reach(cohort_size: int) -> float:
    """How much a correction here generalizes, from how often the shape recurs."""
    n = max(0, int(cohort_size))
    if n <= 1:
        return 0.0
    return round(min(1.0, math.log1p(n - 1) / math.log1p(REACH_FULL)), 6)


def gold_priority(
    *,
    confidence: float,
    label: str,
    relation: str = "atom_type",
    class_counts: Mapping[tuple[str, str], int] | Mapping[str, int] | None = None,
    threshold: float = 0.72,
    governs_packet: bool = False,
    cohort_size: int = 1,
    already_gold: bool = False,
) -> GoldPriority:
    """Expected teaching value of asking a PM about this item."""
    if already_gold:
        # A human already ruled on this. Asking again spends a click to learn
        # nothing and reads to the PM as the system not listening.
        return GoldPriority(0.0, ["already_gold"])

    counts = class_counts or {}
    key: Any = (relation, label)
    n = counts.get(key)
    if n is None:
        n = counts.get(label, 0) if not isinstance(next(iter(counts), None), tuple) else 0
    n = int(n or 0)

    b = boundary_uncertainty(confidence, threshold)
    s = class_starvation(n)
    i = 1.0 if governs_packet else 0.4
    r = reach(cohort_size)

    score = W_BOUNDARY * b + W_STARVATION * s + W_IMPACT * i + W_REACH * r

    reasons: list[str] = []
    if b >= 0.6:
        reasons.append("at_decision_boundary")
    if n == 0:
        reasons.append("unseen_class")
    elif s >= 0.3:
        reasons.append("rare_class")
    if governs_packet:
        reasons.append("governs_a_packet")
    if r >= 0.5:
        reasons.append("shape_recurs")
    if not reasons:
        reasons.append("low_teaching_value")

    return GoldPriority(round(min(1.0, max(0.0, score)), 6), reasons)


def class_counts_from_log(log: Any, relation: str | None = None) -> dict[tuple[str, str], int]:
    """Label histogram from the training log — the input that makes this about
    learning rather than doubt. Returns {} on any failure, which degrades the
    ranker to pure uncertainty rather than breaking it."""
    counts: dict[tuple[str, str], int] = {}
    try:
        rows: Iterable[Any] = log.rows(relation=relation) if relation else log.rows()
    except Exception:
        try:
            rows = log.rows()
        except Exception:
            return {}
    try:
        for r in rows:
            rel = getattr(r, "relation", None) or (r.get("relation") if isinstance(r, dict) else None)
            lab = getattr(r, "label", None) or (r.get("label") if isinstance(r, dict) else None)
            if not rel or not lab:
                continue
            counts[(str(rel), str(lab))] = counts.get((str(rel), str(lab)), 0) + 1
    except Exception:
        return counts
    return counts
