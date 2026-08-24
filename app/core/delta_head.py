"""The delta head: learn WHEN the system is wrong, not what is right.

Every correction a PM taps is treated today as one more labelled row for the
task head. It is a richer object than that: a pair

    (what the system said, what the human said, the receipted input)

-- a preference pair with provenance, produced accidentally, in the course of
work the PM was doing anyway. This module trains on those pairs a small model
of a different target:

    P(a human would correct this | the input)

That is disagreement-with-reality as a first-class output. It needs far fewer
labels than any task head, because "was this wrong" is a binary over inputs
the task heads already embed -- and it is the learned version of the
disagreement queue: instead of asking a PM only where two routers argue, ask
where the error model says the shipped answer smells like the ones humans have
overturned before.

THE CENSORING PROBLEM, HANDLED RATHER THAN HIDDEN. The naive negative set is
"everything nobody corrected" -- and that set is mostly deals nobody LOOKED
at. Absence of a correction is only evidence of correctness when a human
plausibly reviewed the deal. So negatives are drawn exclusively from deals
with review evidence (any correction or tap activity on that deal): "a PM
worked this deal, touched other things, and left this one standing" is a real
negative; "nobody opened it" is nothing, and is excluded. This is the
difference between an error model and a popularity model of which deals get
opened.

The classifier is deliberately tiny -- logistic regression over injected
embeddings, plain numpy, trained by gradient descent in a few hundred
iterations. At tens-of-labels scale anything larger memorises; a linear probe
over a strong encoder is the correct capacity, and it trains on a laptop in
milliseconds. The embedder is injected exactly as ``ContrastiveTypeKNN``
injects ``embed_fn``, so the architecture is testable without a GPU artifact
present.

Guess-free, like everything else here: below a floor of labels, or with only
one class observed, ``fit`` refuses and the head abstains on everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

#: Fewer positives than this and the probe is fitting noise. The delta head
#: is the FIRST model tap data feeds precisely because this floor is low.
MIN_POSITIVES = 8
MIN_NEGATIVES = 8


@dataclass(frozen=True)
class DeltaExample:
    """One shipped answer and whether a human overturned it."""

    text: str                 # the receipted input the system decided on
    corrected: bool           # True: a human changed the answer
    deal_id: str = ""
    relation: str = ""        # which head shipped it (service_routing, atom_type, ...)


def build_examples(
    corrections: Sequence[Any],
    shipped: Sequence[Any],
) -> list[DeltaExample]:
    """Pair positives (corrected) with review-conditioned negatives.

    ``corrections``: rows a human overturned -- objects with ``scope_key`` /
    ``deal_id``, ``relation``, and ``exemplars`` (the feedback store's
    Correction shape).
    ``shipped``: rows the system emitted -- objects with ``deal_id``,
    ``relation``, ``raw_text`` (the training log's TrainingRow shape).

    A shipped row becomes a NEGATIVE only when its deal carries review
    evidence -- some correction exists on that deal -- and the row itself was
    not the thing corrected. Deals with no human activity contribute nothing:
    silence from a deal nobody opened is not agreement.
    """
    examples: list[DeltaExample] = []
    reviewed_deals: set[str] = set()
    corrected_texts: set[str] = set()

    for row in corrections or []:
        deal = str(getattr(row, "scope_key", "") or getattr(row, "deal_id", "") or "")
        if deal:
            reviewed_deals.add(deal)
        for exemplar in list(getattr(row, "exemplars", []) or []) or [getattr(row, "raw_text", "")]:
            text = str(exemplar or "").strip()
            if text:
                corrected_texts.add(text)
                examples.append(DeltaExample(
                    text=text, corrected=True, deal_id=deal,
                    relation=str(getattr(row, "relation", "") or ""),
                ))

    for row in shipped or []:
        deal = str(getattr(row, "deal_id", "") or "")
        text = str(getattr(row, "raw_text", "") or "").strip()
        if not text or deal not in reviewed_deals or text in corrected_texts:
            continue
        examples.append(DeltaExample(
            text=text, corrected=False, deal_id=deal,
            relation=str(getattr(row, "relation", "") or ""),
        ))
    return examples


@dataclass
class DeltaHead:
    """P(correction | input): a linear probe over injected embeddings."""

    embed_fn: Callable[[Sequence[str]], np.ndarray] = field(repr=False)
    weights: np.ndarray | None = field(default=None, repr=False)
    bias: float = 0.0
    refusals: list[str] = field(default_factory=list)
    n_positive: int = 0
    n_negative: int = 0

    @property
    def fitted(self) -> bool:
        return self.weights is not None and not self.refusals

    def fit(
        self,
        examples: Sequence[DeltaExample],
        *,
        l2: float = 1.0,
        iterations: int = 400,
        learning_rate: float = 0.5,
    ) -> "DeltaHead":
        positives = [e for e in examples if e.corrected]
        negatives = [e for e in examples if not e.corrected]
        self.n_positive, self.n_negative = len(positives), len(negatives)

        if len(positives) < MIN_POSITIVES:
            self.refusals.append(
                f"{len(positives)} corrected examples (need {MIN_POSITIVES}); "
                "a probe fit on fewer is noise wearing a sigmoid"
            )
        if len(negatives) < MIN_NEGATIVES:
            self.refusals.append(
                f"{len(negatives)} review-conditioned negatives (need "
                f"{MIN_NEGATIVES}); without them this learns 'everything is "
                "wrong', which is a slogan, not a model"
            )
        if self.refusals:
            return self

        texts = [e.text for e in positives] + [e.text for e in negatives]
        y = np.array([1.0] * len(positives) + [0.0] * len(negatives))
        X = np.asarray(self.embed_fn(texts), dtype=np.float64)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        # Class-balanced logistic regression by plain gradient descent. Tiny
        # on purpose: at this label scale, capacity IS the failure mode.
        pos_w = 0.5 / max(y.sum(), 1.0)
        neg_w = 0.5 / max((1 - y).sum(), 1.0)
        sample_w = np.where(y == 1.0, pos_w, neg_w)
        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(iterations):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = sample_w * (p - y)
            w -= learning_rate * (X.T @ g + l2 * w / len(y))
            b -= learning_rate * float(g.sum())
        self.weights, self.bias = w, b
        return self

    def score(self, texts: Sequence[str]) -> list[float | None]:
        """P(correction) per input; None everywhere when the head refused."""
        if not self.fitted:
            return [None] * len(texts)
        X = np.asarray(self.embed_fn(list(texts)), dtype=np.float64)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        z = X @ self.weights + self.bias
        return [float(v) for v in 1.0 / (1.0 + np.exp(-z))]

    def rank_for_review(
        self, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """(deal_id, text) pairs -> deals ordered by P(correction), highest first.

        This is the learned successor of the disagreement queue: spend the
        scarcest resource in the system -- PM attention -- where shipped
        answers most resemble the ones humans have overturned before. Empty
        when the head refused: an unfitted error model must not order anyone's
        work.
        """
        if not self.fitted or not candidates:
            return []
        scores = self.score([text for _deal, text in candidates])
        ranked = [
            (deal, s) for (deal, _), s in zip(candidates, scores) if s is not None
        ]
        return sorted(ranked, key=lambda pair: -pair[1])

    def explain(self) -> str:
        if not self.fitted:
            return "delta head REFUSED: " + "; ".join(self.refusals)
        return (
            f"delta head: fitted on {self.n_positive} corrections vs "
            f"{self.n_negative} review-conditioned negatives"
        )
