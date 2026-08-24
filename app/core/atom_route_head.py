"""Route a deal by classifying its ATOMS and aggregating the evidence.

The existing router embeds one blob -- a FILES line plus forty sampled
sentences -- and asks a kNN which pack the blob resembles. That throws away the
structure the pipeline already produced. A deal is not a paragraph; it is a bag
of atoms, each of which is separately about something.

Classifying atoms and aggregating gives four things the blob cannot:

* **Sample efficiency.** Seventy labelled deals are seventy training examples
  for a deal-level head, and thousands for an atom-level one. The deal label is
  weak supervision over its atoms -- the standard multiple-instance setup.
* **Evidence.** The route arrives with the atoms that caused it. A PM
  correcting "this is not wireless" can be shown the six atoms that said it
  was, and the correction lands on those, not on an opaque deal vector.
* **Stability under deal size.** Every atom votes, and the decision is a SHARE
  of the voting atoms, so adding documents cannot lurch the answer. There is no
  sampling step left to have a cliff in.
* **Room for a second workstream.** A deal that is genuinely AV *and* cabling
  has two masses of evidence. The blob has to pick one and lose the other.

WHAT IS AND IS NOT HERE. This module is the aggregator and the decision rule.
The per-atom classifier is injected, exactly as ``ContrastiveTypeKNN`` injects
``embed_fn`` -- so the architecture is testable, and reviewable, without a GPU
artifact present. Training the atom-level head is a separate job on a box that
has one.

The guess-free rule is applied at both levels: an atom the classifier cannot
place abstains and contributes nothing, and a deal whose evidence is thin or
split abstains too, leaving the LLM base in charge. Silence is a legitimate
answer here and is never scored as a route.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

#: Atom types that are BOM/pricing noise rather than scope. Kept identical to
#: ``service_router._NOISE_TYPES``: a TV install reads as 313 cabling material
#: rows against 52 scope atoms, and counting those rows routes the deal by its
#: bill of materials instead of its work.
NOISE_TYPES = frozenset({
    "pricing_assumption", "commercial_total", "rate_card", "line_item",
})

#: One atom may not carry a deal. Confidence is capped before it is summed, so
#: a single hyper-confident row cannot outvote a spread of ordinary evidence.
_WEIGHT_CAP = 1.0


@dataclass(frozen=True)
class AtomVote:
    """One atom's opinion, kept so the decision can be shown its reasons."""

    index: int
    label: str
    confidence: float
    text: str = ""


@dataclass
class RouteDecision:
    """A routing verdict, the evidence for it, and why it abstained if it did."""

    label: str | None = None
    confidence: float = 0.0
    secondary: str | None = None
    evidence: list[AtomVote] = field(default_factory=list)
    mass: dict[str, float] = field(default_factory=dict)
    voting_atoms: int = 0
    considered_atoms: int = 0
    abstain_reason: str = ""

    @property
    def abstained(self) -> bool:
        return self.label is None

    def explain(self) -> str:
        if self.abstained:
            return (
                f"abstained ({self.abstain_reason}); "
                f"{self.voting_atoms}/{self.considered_atoms} atoms voted"
            )
        shares = ", ".join(
            f"{k} {v * 100:.0f}%" for k, v in sorted(self.mass.items(), key=lambda kv: -kv[1])
        )
        lines = [
            f"{self.label} at {self.confidence:.2f}"
            + (f" (secondary: {self.secondary})" if self.secondary else ""),
            f"  {self.voting_atoms}/{self.considered_atoms} atoms voted -> {shares}",
        ]
        for vote in self.evidence[:5]:
            lines.append(f"  - [{vote.confidence:.2f}] {vote.text[:88]}")
        return "\n".join(lines)


#: A batch classifier: texts in, ``(label, confidence)`` or ``None`` per text.
#: Matches ``ContrastiveTypeKNN.classify_batch`` so a trained head drops in.
AtomClassifier = Callable[[Sequence[str]], Sequence[tuple[str, float] | None]]


@dataclass
class PerAtomRouter:
    """Aggregate per-atom votes into a deal route.

    ``min_voting_atoms``   how much evidence before a deal may be routed at all.
                           Two confident atoms out of four hundred is a
                           coincidence, not a workstream.
    ``min_margin``         how far the winner must lead the runner-up, as a
                           share of the voting mass. Below it the deal is
                           genuinely mixed and the honest answer is silence.
    ``secondary_floor``    a runner-up above this share is reported as a real
                           second workstream rather than discarded.
    """

    classify: AtomClassifier
    min_voting_atoms: int = 6
    min_margin: float = 0.15
    secondary_floor: float = 0.25
    max_evidence: int = 12

    def route_texts(self, texts: Sequence[str]) -> RouteDecision:
        texts = [t for t in (s.strip() for s in texts) if t]
        decision = RouteDecision(considered_atoms=len(texts))
        if not texts:
            decision.abstain_reason = "no atoms"
            return decision

        try:
            verdicts = list(self.classify(texts))
        except Exception:
            decision.abstain_reason = "classifier unavailable"
            return decision

        votes: list[AtomVote] = []
        for i, verdict in enumerate(verdicts):
            if not verdict:
                continue  # the atom abstained; it contributes nothing
            label, confidence = verdict
            label = str(label).strip()
            if not label:
                continue
            votes.append(AtomVote(i, label, float(confidence), texts[i] if i < len(texts) else ""))

        decision.voting_atoms = len(votes)
        if len(votes) < self.min_voting_atoms:
            decision.abstain_reason = (
                f"only {len(votes)} atoms voted, need {self.min_voting_atoms}"
            )
            return decision

        weights: dict[str, float] = defaultdict(float)
        for vote in votes:
            weights[vote.label] += min(max(vote.confidence, 0.0), _WEIGHT_CAP)
        total = sum(weights.values())
        if total <= 0:
            decision.abstain_reason = "no positive evidence mass"
            return decision

        decision.mass = {k: v / total for k, v in weights.items()}
        ranked = sorted(decision.mass.items(), key=lambda kv: -kv[1])
        top_label, top_share = ranked[0]
        runner_share = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_share - runner_share

        if margin < self.min_margin:
            decision.abstain_reason = (
                f"{top_label} leads {ranked[1][0] if len(ranked) > 1 else '-'} by only "
                f"{margin:.2f} (need {self.min_margin:.2f}) -- the deal is mixed"
            )
            return decision

        decision.label = top_label
        # Confidence is the winner's SHARE OF THE MARGIN, not its raw share: a
        # class holding 60% against a 35% runner-up is a far weaker call than
        # one holding 60% against three classes at 13%, and a single number
        # that cannot tell those apart is the kind that gets over-trusted.
        decision.confidence = round(min(1.0, margin / max(top_share, 1e-9)), 4)
        if len(ranked) > 1 and runner_share >= self.secondary_floor:
            decision.secondary = ranked[1][0]
        decision.evidence = sorted(
            (v for v in votes if v.label == top_label),
            key=lambda v: -v.confidence,
        )[: self.max_evidence]
        return decision

    def route_deal(self, atoms: Sequence[Any]) -> RouteDecision:
        """Convenience: filter BOM/pricing noise, then route the remaining bodies."""
        texts = [t for a in atoms if (t := _atom_text(a)) and _atom_type(a) not in NOISE_TYPES]
        if len(texts) < self.min_voting_atoms:
            # Thin scope: fall back to everything rather than abstain on a
            # filter, mirroring the guard in service_router._scope_summary.
            texts = [t for a in atoms if (t := _atom_text(a))]
        return self.route_texts(texts)


def _atom_type(atom: Any) -> str:
    value = getattr(atom, "atom_type", None)
    if value is None and isinstance(atom, dict):
        value = atom.get("atom_type")
    return value.value if hasattr(value, "value") else str(value or "")


def _atom_text(atom: Any) -> str:
    for attr in ("raw_text", "normalized_text", "text"):
        value = getattr(atom, attr, None)
        if value:
            return str(value).strip()
    if isinstance(atom, dict):
        for key in ("text", "raw_text", "normalized_text", "body"):
            if atom.get(key):
                return str(atom[key]).strip()
    return ""
