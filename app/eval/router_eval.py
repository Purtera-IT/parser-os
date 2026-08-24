"""Score the service router against labels the router did not write.

The existing number for this router was 91%, and it meant nothing: the head was
trained on DeepSeek's labels and scored against DeepSeek's labels, so it
measured how well it had imitated the teacher. A distilled head graded by its
own teacher can only ever look good.

This scores it against PM corrections instead. A PM tapping the *Workstream /
domain* chip because the brief opened on the wrong pack produces a label no
teacher generated, on a deal that actually shipped. It is the only
non-circular signal available, and the point of the whole exercise is that it
is a number which **can go down**.

Four rules are enforced here rather than left to whoever runs it:

1. **Human-authored only.** A correction with no ``created_by`` is not gold.
2. **Split by DEAL, never by row.** Corrections inside one deal share a
   customer vocabulary, the same documents and the same PM in the same hour.
   A row-level split puts near-duplicates on both sides and hands back a
   number as circular as the one it replaced.
3. **Abstention is not an answer.** A guess-free router that stands aside is
   neither right nor wrong. Coverage and accuracy-on-covered are reported
   separately; an abstention is never scored as a hit.
4. **Refuse to report a number that cannot mean anything.** Too few deals, one
   class only, or a deal on both sides of the split -> the report says why and
   withholds the figure rather than printing something quotable.

Usage::

    from app.eval.router_eval import gold_cases_from_store, evaluate_router
    cases = gold_cases_from_store(store)
    report = evaluate_router(my_router, cases)
    print(report.summary())
"""

from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

#: The decide() relation the router head governs -- see pm_feedback.HEAD_REGISTRY,
#: where the "router" chip maps to this relation at deal scope.
ROUTER_RELATION = "service_routing"

#: Below this many held-out deals the accuracy figure is noise, and printing it
#: invites exactly the false confidence this module exists to prevent.
MIN_HOLDOUT_DEALS = 12

#: Known non-human authors, kept only so the refusal message can name them.
#: They are NOT the test -- see below for why a deny-list is the wrong shape.
_MACHINE_AUTHORS = frozenset({
    "", "system", "auto", "teacher", "deepseek", "llm", "pipeline",
    "universal-audit", "mined-audit", "seed",
})


@dataclass(frozen=True)
class RouterCase:
    """One deal, its PM-given label, and the text the router sees."""

    deal_id: str
    gold_label: str
    text: str
    author: str = ""

    @property
    def is_human_authored(self) -> bool:
        """A POSITIVE test for a person, not the absence of a known machine.

        This was written the other way first -- "not in _MACHINE_AUTHORS" --
        and a test against the real store caught it immediately. The store's
        136 corrections are authored by ``universal-audit``, ``mined-audit``
        and ``seed``; none of those were in the deny-list, so every one of them
        would have counted as gold and the circular number would have come
        straight back under a new name.

        A deny-list fails open, and it fails open silently every time somebody
        adds a labeller. The gate has to fail CLOSED: an author this cannot
        recognise as a person is treated as a machine, because the cost of
        wrongly excluding a real PM label is one missing row, and the cost of
        wrongly including a machine label is an eval that lies.

        A person is identified the way this system already identifies them --
        by an email address, which is what the ``pm`` field on the correction
        chip carries.
        """
        author = self.author.strip().lower()
        if not author or author in _MACHINE_AUTHORS:
            return False
        return "@" in author


@dataclass
class RouterResult:
    case: RouterCase
    predicted: str | None  # None == abstained
    confidence: float = 0.0

    @property
    def abstained(self) -> bool:
        return self.predicted is None

    @property
    def correct(self) -> bool:
        return (not self.abstained) and self.predicted == self.case.gold_label


@dataclass
class RouterReport:
    results: list[RouterResult] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    holdout_deals: int = 0

    @property
    def covered(self) -> list[RouterResult]:
        """The cases the router actually answered."""
        return [r for r in self.results if not r.abstained]

    @property
    def coverage(self) -> float:
        """Share of deals the router was willing to answer at all."""
        return (len(self.covered) / len(self.results)) if self.results else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy ON WHAT IT ANSWERED. Meaningless without coverage beside it:
        a router that abstains on all but its three easiest deals scores 1.0."""
        cov = self.covered
        return (sum(1 for r in cov if r.correct) / len(cov)) if cov else 0.0

    @property
    def majority_baseline(self) -> float:
        """Always answering the most common label. The head has to beat this;
        a specialist that cannot is an expensive way to guess the mode."""
        if not self.results:
            return 0.0
        counts = Counter(r.case.gold_label for r in self.results)
        return counts.most_common(1)[0][1] / len(self.results)

    @property
    def reportable(self) -> bool:
        return not self.refusals

    def per_class(self) -> dict[str, tuple[int, int]]:
        """gold label -> (correct, total) over answered cases."""
        out: dict[str, list[int]] = {}
        for r in self.covered:
            slot = out.setdefault(r.case.gold_label, [0, 0])
            slot[1] += 1
            if r.correct:
                slot[0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def summary(self) -> str:
        if not self.reportable:
            lines = ["router eval: NO NUMBER REPORTED", ""]
            lines += [f"  - {why}" for why in self.refusals]
            lines += [
                "",
                "  A figure from this set would not mean anything, so none is given.",
            ]
            return "\n".join(lines)

        lines = [
            f"router eval: {self.accuracy * 100:.1f}% on what it answered "
            f"({sum(1 for r in self.covered if r.correct)}/{len(self.covered)})",
            f"  coverage        : {self.coverage * 100:.1f}%  "
            f"({len(self.covered)}/{len(self.results)} deals answered)",
            f"  majority baseline: {self.majority_baseline * 100:.1f}%  "
            + ("<-- BEATEN" if self.accuracy > self.majority_baseline else "<-- NOT BEATEN"),
            f"  held-out deals  : {self.holdout_deals}",
            "",
        ]
        for label, (ok, total) in sorted(self.per_class().items()):
            lines.append(f"  {label:<24} {ok}/{total}")
        wrong = [r for r in self.covered if not r.correct]
        if wrong:
            lines += ["", "  wrong:"]
            for r in wrong[:20]:
                lines.append(
                    f"    {r.case.deal_id[:24]:<24} gold {r.case.gold_label:<20} "
                    f"said {r.predicted} ({r.confidence:.2f})"
                )
        return "\n".join(lines)


def gold_cases_from_store(
    store: Any, *, relation: str = ROUTER_RELATION
) -> list[RouterCase]:
    """PM corrections for ``relation``, as evaluation cases.

    Reads the same corrections the feedback store already holds -- nothing new
    is collected. Only human-authored, deal-scoped rows qualify: the label has
    to come from a person, and it has to be attached to a deal so the split
    below can be honest.
    """
    try:
        rows = store.list_corrections(status="active")
    except Exception:
        return []

    cases: list[RouterCase] = []
    for row in rows or []:
        if str(getattr(row, "relation", "")) != relation:
            continue
        deal_id = str(getattr(row, "scope_key", "") or "").strip()
        verdict = str(getattr(row, "verdict", "") or "").strip()
        if not deal_id or not verdict:
            continue
        exemplars = list(getattr(row, "exemplars", []) or [])
        case = RouterCase(
            deal_id=deal_id,
            gold_label=verdict,
            text="\n".join(str(e) for e in exemplars),
            author=str(getattr(row, "created_by", "") or ""),
        )
        if case.is_human_authored:
            cases.append(case)
    return cases


def split_by_deal(
    cases: Iterable[RouterCase], *, holdout_share: float = 0.5, salt: str = "router-eval-v1"
) -> tuple[list[RouterCase], list[RouterCase]]:
    """Partition into ``(trainable, holdout)`` on the DEAL, deterministically.

    The bucket is a stable hash of the deal id, so a deal lands on the same
    side forever: the hold-out does not drift as corrections accumulate, and
    two runs a month apart are comparable. crc32 rather than ``hash()``, whose
    string hashing is randomised per process by PYTHONHASHSEED.

    Splitting on the deal is the whole point. Corrections within one deal are
    correlated -- same customer vocabulary, same documents, same PM in the same
    hour -- so a row-level split leaks near-duplicates across the boundary and
    reports a number that looks earned and is not.
    """
    trainable: list[RouterCase] = []
    holdout: list[RouterCase] = []
    cut = max(0.0, min(1.0, holdout_share))
    for case in cases:
        bucket = zlib.crc32(f"{salt}|{case.deal_id}".encode("utf-8")) % 10_000
        (holdout if bucket < cut * 10_000 else trainable).append(case)
    return trainable, holdout


def evaluate_router(
    router: Callable[[RouterCase], tuple[str, float] | None],
    cases: Sequence[RouterCase],
    *,
    trainable: Sequence[RouterCase] | None = None,
    min_holdout_deals: int = MIN_HOLDOUT_DEALS,
) -> RouterReport:
    """Run ``router`` over held-out cases and score it, or refuse to.

    ``router`` returns ``(label, confidence)`` or ``None`` to abstain -- the
    same contract ``ContrastiveTypeKNN.classify`` already uses, so a head can
    be passed in directly.

    Pass ``trainable`` to have the leak check run: any deal appearing on both
    sides voids the report rather than quietly inflating it.
    """
    report = RouterReport()
    deals = {c.deal_id for c in cases}
    report.holdout_deals = len(deals)

    if len(deals) < min_holdout_deals:
        report.refusals.append(
            f"only {len(deals)} held-out deals (need {min_holdout_deals}); "
            "an accuracy figure here is noise"
        )
    labels = {c.gold_label for c in cases}
    if len(labels) < 2:
        report.refusals.append(
            f"held-out set contains a single label ({labels or 'none'}); "
            "nothing is being discriminated"
        )
    if trainable is not None:
        overlap = deals & {c.deal_id for c in trainable}
        if overlap:
            report.refusals.append(
                f"{len(overlap)} deal(s) appear in BOTH the trainable and held-out "
                f"sets, e.g. {sorted(overlap)[:3]} -- the split leaked"
            )
    non_human = [c for c in cases if not c.is_human_authored]
    if non_human:
        report.refusals.append(
            f"{len(non_human)} case(s) are not human-authored; gold must come "
            "from a person or this is circular again"
        )

    for case in cases:
        try:
            verdict = router(case)
        except Exception:
            verdict = None
        if verdict is None:
            report.results.append(RouterResult(case, None, 0.0))
        else:
            label, confidence = verdict
            report.results.append(RouterResult(case, str(label), float(confidence)))
    return report


# ── promotion: what has to be true before a head is allowed to decide ──────


@dataclass
class PromotionVerdict:
    """Whether a head may take over from the router that currently ships."""

    promote: bool = False
    blockers: list[str] = field(default_factory=list)
    candidate: RouterReport | None = None
    incumbent: RouterReport | None = None
    #: Held-out deals where exactly one of the two got it right.
    paired_wins: int = 0
    paired_losses: int = 0
    #: Two-sided McNemar p over those disagreements. 1.0 == no evidence at all.
    p_value: float = 1.0

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else "DO NOT PROMOTE"
        lines = [f"promotion: {head}", ""]
        for why in self.blockers:
            lines.append(f"  - {why}")
        if self.candidate and self.candidate.reportable:
            lines += [
                "",
                f"  candidate : {self.candidate.accuracy * 100:.1f}% "
                f"@ {self.candidate.coverage * 100:.0f}% coverage",
            ]
            if self.incumbent and self.incumbent.reportable:
                lines.append(
                    f"  incumbent : {self.incumbent.accuracy * 100:.1f}% "
                    f"@ {self.incumbent.coverage * 100:.0f}% coverage"
                )
            lines.append(
                f"  paired    : candidate wins {self.paired_wins}, loses "
                f"{self.paired_losses} of the disagreements (p={self.p_value:.3f})"
            )
        return "\n".join(lines)


def _paired_counts(candidate: RouterReport, incumbent: RouterReport) -> tuple[int, int]:
    """(candidate-right-incumbent-wrong, candidate-wrong-incumbent-right).

    Keyed on the deal, and an abstention counts as not-right: a router that
    stayed silent did not get the deal right, and pairing it as a win would
    reward abstaining.
    """
    inc_by_deal = {r.case.deal_id: r for r in incumbent.results}
    wins = losses = 0
    for row in candidate.results:
        other = inc_by_deal.get(row.case.deal_id)
        if other is None:
            continue
        if row.correct and not other.correct:
            wins += 1
        elif other.correct and not row.correct:
            losses += 1
    return wins, losses


def _mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar (binomial sign test on the discordant pairs).

    Written out rather than pulled from scipy: this module has to be runnable
    wherever the eval runs, and the exact test on a handful of pairs is a few
    lines of arithmetic. Returns 1.0 when nothing disagrees -- no evidence,
    rather than a flattering default.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare_routers(
    candidate: Callable[[RouterCase], tuple[str, float] | None],
    incumbent: Callable[[RouterCase], tuple[str, float] | None],
    cases: Sequence[RouterCase],
    *,
    trainable: Sequence[RouterCase] | None = None,
    min_holdout_deals: int = MIN_HOLDOUT_DEALS,
    min_gain: float = 0.0,
    max_p_value: float = 0.05,
) -> PromotionVerdict:
    """Head-to-head on one hold-out, then decide whether to promote.

    The bar is the router that SHIPS TODAY, not a majority-class baseline.
    "Better than guessing" is not the question anyone is actually asking when
    an LLM base is already answering correctly; the question is whether the
    cheap local head is at least as good as the expensive remote one.

    Coverage is part of the bar, not a footnote. A head that abstains on
    everything hard scores beautifully on the rest, so a candidate that is more
    accurate while answering far less has not earned anything -- it has just
    moved the work back to the incumbent and taken credit for what was left.
    """
    verdict = PromotionVerdict()
    verdict.candidate = evaluate_router(
        candidate, cases, trainable=trainable, min_holdout_deals=min_holdout_deals
    )
    verdict.incumbent = evaluate_router(
        incumbent, cases, trainable=trainable, min_holdout_deals=min_holdout_deals
    )

    if not verdict.candidate.reportable:
        verdict.blockers.extend(verdict.candidate.refusals)
        return verdict

    cand, inc = verdict.candidate, verdict.incumbent

    # Paired comparison, because both routers answered the SAME deals.
    #
    # Comparing two accuracies as if they were independent throws that away and
    # needs far more deals than exist. At the hold-out sizes this will have for
    # a long time -- tens, not hundreds -- the standard error on a raw accuracy
    # is around ten points, so a candidate identical to the incumbent wins
    # roughly half of any weekly comparison by chance alone. A promote-on-win
    # rule run weekly would then promote, roll back and promote again, and read
    # as an unstable head rather than an unstable measurement.
    #
    # Only the deals where they DISAGREE carry information. McNemar asks
    # whether those disagreements favour the candidate more than a coin would.
    wins, losses = _paired_counts(cand, inc)
    verdict.paired_wins, verdict.paired_losses = wins, losses
    verdict.p_value = _mcnemar_p(wins, losses)
    if wins + losses == 0 and inc.reportable:
        verdict.blockers.append(
            "candidate and incumbent never disagree on the held-out deals; "
            "there is nothing here to promote on"
        )
    elif inc.reportable and verdict.p_value > max_p_value:
        verdict.blockers.append(
            f"candidate wins {wins} / loses {losses} of the disagreements "
            f"(p={verdict.p_value:.3f} > {max_p_value:.2f}) -- consistent with chance"
        )

    if cand.accuracy <= cand.majority_baseline:
        verdict.blockers.append(
            f"candidate {cand.accuracy * 100:.1f}% does not beat the majority "
            f"class ({cand.majority_baseline * 100:.1f}%)"
        )
    if inc.reportable and cand.accuracy < inc.accuracy + min_gain:
        verdict.blockers.append(
            f"candidate {cand.accuracy * 100:.1f}% does not beat the shipping "
            f"router {inc.accuracy * 100:.1f}%"
            + (f" by the required {min_gain * 100:.0f} points" if min_gain else "")
        )
    if inc.reportable and cand.coverage < inc.coverage * 0.9:
        verdict.blockers.append(
            f"candidate answers {cand.coverage * 100:.0f}% of deals against the "
            f"incumbent's {inc.coverage * 100:.0f}% -- accuracy bought by "
            "abstaining is not an improvement"
        )

    verdict.promote = not verdict.blockers
    if verdict.promote:
        verdict.blockers.append("all gates cleared")
    return verdict


def readiness(cases: Sequence[RouterCase], *, min_holdout_deals: int = MIN_HOLDOUT_DEALS) -> str:
    """How far the gold set is from being able to gate anything at all.

    Reported separately from accuracy because it is a different question, and
    the honest answer today is a distance rather than a score.
    """
    human = [c for c in cases if c.is_human_authored]
    deals = {c.deal_id for c in human}
    labels = {c.gold_label for c in human}
    _train, hold = split_by_deal(human)
    hold_deals = {c.deal_id for c in hold}
    lines = [
        "router gold readiness",
        f"  corrections           : {len(cases)}",
        f"  human-authored        : {len(human)}",
        f"  distinct deals        : {len(deals)}",
        f"  distinct labels       : {len(labels)}",
        f"  would land in hold-out: {len(hold_deals)}",
        "",
    ]
    if len(hold_deals) >= min_holdout_deals and len(labels) >= 2:
        lines.append("  -> enough to gate a promotion.")
    else:
        need = max(0, min_holdout_deals - len(hold_deals))
        lines.append(
            f"  -> NOT enough to gate anything: {need} more held-out deal(s) needed"
            + ("" if len(labels) >= 2 else ", and at least two distinct labels")
        )
    return "\n".join(lines)
