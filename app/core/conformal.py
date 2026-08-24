"""Abstention with a coverage guarantee, instead of a hand-tuned knob.

Every abstain gate in this system today is a threshold somebody picked:
``sim_floor = 0.55``, ``tau = 0.30`` -- both in the calibration registry with
receipts honestly saying the corpus was tiny. A threshold answers "how
confident does the score FEEL"; it promises nothing.

Split conformal prediction answers a different question with an actual
guarantee. Calibrate once on labels the model never trained on (the PM
hold-out -- the one label set in this system a model did not write), and the
guarantee is distribution-free:

    P(true label is in the prediction set) >= 1 - alpha

for any score function, however badly calibrated its raw confidences are. The
decision rule is then not "score above X" but:

    answer when the prediction set is a SINGLETON; abstain otherwise.

A singleton at 90% coverage means "one label suffices to keep the guarantee".
A multi-label set means the model genuinely cannot separate candidates at the
promised coverage -- which is precisely what abstention is supposed to mean,
and what a raw threshold only gestures at.

Finite-sample honesty, enforced rather than footnoted:

* with n calibration points the achievable coverage is quantised -- you
  cannot promise 90% from 12 points. ``fit`` computes the coverage actually
  guaranteed and stores THAT, never the request.
* below a floor the guarantee is vacuous (the quantile is the maximum score
  and every set contains every label). ``fit`` refuses, the same way
  router_eval refuses to print a number that cannot mean anything.
* calibration rows must be human-authored and disjoint from training by DEAL
  -- the caller enforces deal-disjointness (split_by_deal exists for this);
  this module enforces that a refused fit cannot be silently used.

Deliberately dependency-free (no scipy): the whole method is a quantile and a
comparison, and it has to run wherever the eval runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

#: Below this many calibration points, even the weakest useful guarantee
#: (~80%) is out of reach and every prediction set saturates. Refuse.
MIN_CALIBRATION = 5


@dataclass
class ConformalGate:
    """A fitted split-conformal gate over per-class scores."""

    #: Nonconformity threshold: scores BELOW ``1 - qhat`` are excluded.
    qhat: float = 1.0
    #: The coverage actually guaranteed by n and alpha -- not the request.
    guaranteed_coverage: float = 0.0
    requested_alpha: float = 0.1
    n_calibration: int = 0
    refusals: list[str] = field(default_factory=list)

    @property
    def fitted(self) -> bool:
        return not self.refusals

    def prediction_set(self, scores: Mapping[str, float]) -> set[str]:
        """Every label whose score clears the calibrated threshold."""
        if not self.fitted:
            # A refused gate abstains on everything rather than guessing --
            # fail closed, like every other gate in this codebase.
            return set(scores)
        return {label for label, s in scores.items() if s >= 1.0 - self.qhat}

    def decide(self, scores: Mapping[str, float]) -> tuple[str | None, set[str]]:
        """(answer or None, the full prediction set).

        Answer only when the set is a singleton. The set is returned either
        way because a two-label set is INFORMATION -- "it is wireless or
        cabling, not the other twenty-seven" -- and the disagreement queue can
        show a PM exactly that.
        """
        pset = self.prediction_set(scores)
        if len(pset) == 1:
            return next(iter(pset)), pset
        return None, pset

    def explain(self) -> str:
        if not self.fitted:
            return "conformal gate REFUSED: " + "; ".join(self.refusals)
        return (
            f"conformal gate: answer when a single label scores >= "
            f"{1.0 - self.qhat:.3f}; guaranteed coverage "
            f"{self.guaranteed_coverage * 100:.1f}% "
            f"(requested {100 * (1 - self.requested_alpha):.0f}%, "
            f"n={self.n_calibration})"
        )


def fit(
    calibration: Sequence[tuple[Mapping[str, float], str]],
    *,
    alpha: float = 0.1,
    min_calibration: int = MIN_CALIBRATION,
) -> ConformalGate:
    """Fit the gate on ``(per-class scores, true label)`` pairs.

    The pairs must come from labels the scorer never trained on, split from
    training by DEAL -- the same discipline as router_eval, for the same
    reason: calibrating on training data yields a quantile that flatters the
    model and a "guarantee" that holds only in-sample.
    """
    gate = ConformalGate(requested_alpha=alpha, n_calibration=len(calibration))

    if not 0.0 < alpha < 1.0:
        gate.refusals.append(f"alpha={alpha} is not a miscoverage rate")
        return gate
    n = len(calibration)
    if n < min_calibration:
        gate.refusals.append(
            f"only {n} calibration points (need {min_calibration}); the "
            "quantile would be the sample maximum and every prediction set "
            "would contain every label -- a guarantee that guarantees nothing"
        )
        return gate

    missing = sum(1 for scores, label in calibration if label not in scores)
    if missing:
        gate.refusals.append(
            f"{missing} calibration row(s) whose true label the scorer cannot "
            "even name; the label spaces disagree and no quantile repairs that"
        )
        return gate

    # Nonconformity: how far the truth was from a perfect score.
    nonconformity = sorted(1.0 - float(scores[label]) for scores, label in calibration)

    # The finite-sample quantile. rank/n is the coverage actually guaranteed;
    # if the required rank exceeds n, the requested coverage is unreachable at
    # this n -- refuse rather than quietly promising less.
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        max_coverage = n / (n + 1)
        gate.refusals.append(
            f"coverage {100 * (1 - alpha):.0f}% is unreachable with n={n} "
            f"(max guaranteeable {100 * max_coverage:.1f}%); collect more "
            "labels or request less"
        )
        return gate

    gate.qhat = nonconformity[rank - 1]
    gate.guaranteed_coverage = rank / (n + 1)
    return gate


def coverage_on(
    gate: ConformalGate,
    holdout: Sequence[tuple[Mapping[str, float], str]],
) -> dict[str, float]:
    """Empirical check of the guarantee on a second held-out set.

    The math says coverage >= 1 - alpha; this measures it, because a guarantee
    nobody verifies decays into a slogan. Also reports the operating numbers a
    threshold never gives you: how often the gate answers, and how often an
    answer is right.
    """
    if not holdout:
        return {"n": 0}
    covered = answered = answered_right = 0
    for scores, label in holdout:
        pset = gate.prediction_set(scores)
        if label in pset:
            covered += 1
        if len(pset) == 1:
            answered += 1
            if label in pset:
                answered_right += 1
    n = len(holdout)
    return {
        "n": n,
        "empirical_coverage": covered / n,
        "answer_rate": answered / n,
        "accuracy_when_answering": (answered_right / answered) if answered else 0.0,
    }
