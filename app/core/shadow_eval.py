"""Shadow harness — does the student beat the teacher *yet*, per relation?

This is the gate the whole cutover hangs on. Before any LLM call is replaced,
we must prove — on names the student never saw — that the student matches the
teacher. This module measures exactly that, per relation, and emits a blunt
verdict: ``ready`` or ``not ready``.

The evaluation is **leave-one-deal-out**, for free, because every training row
already carries a deterministic deal-based ``split`` (see
:func:`app.core.training_log.assign_split`). We fit the student on the
``train`` rows and score it on the ``holdout`` rows. Since splits are by *deal*,
a holdout row's deal — and therefore its proper nouns — were never in the
student's memory. So a high holdout accuracy means the student learned the
**rule**, not the **names**. That is the only kind of "learning" we credit.

Metrics, per relation:

* **coverage** — fraction of holdout rows the student answered (did not
  abstain). Abstention is safe (falls through to the LLM), so low coverage is
  "not ready", never "wrong".
* **accuracy** — of the answered rows, fraction whose label matched the gold
  label. This is the precision of speaking.
* **gold accuracy** — accuracy restricted to PM-taught (gold) rows, reported
  separately because those are the labels that actually matter.

A relation is **ready to cut over** only when it clears *all* of: enough
holdout examples to trust the estimate, high accuracy on answered rows, and
enough coverage to be worth the latency win. Until then the LLM stays in the
loop for that relation. Nothing here mutates the compile path — it only reads
the log and reports. Cutover (#70/#71) consumes this verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.core.extractor_student import ExtractionStudent
from app.core.training_log import TEACHER_PM, TrainingLog

# Readiness bar. Deliberately strict: replacing the LLM is only worth it if the
# student is *right* on unseen names. Coverage can grow over time; accuracy is
# non-negotiable.
_MIN_HOLDOUT = 20      # too few examples → estimate is noise → not ready
_MIN_ACCURACY = 0.90   # of answered rows
_MIN_COVERAGE = 0.60   # must answer enough to matter
_MIN_GOLD_ACCURACY = 0.95  # PM-taught labels are the ones we cannot get wrong


@dataclass
class RelationReport:
    """Per-relation shadow result on the holdout split."""

    relation: str
    n_holdout: int = 0
    n_answered: int = 0
    n_correct: int = 0
    n_gold: int = 0
    n_gold_correct: int = 0
    confusions: list[tuple[str, str, str]] = field(default_factory=list)  # (text, gold, pred)

    @property
    def coverage(self) -> float:
        return self.n_answered / self.n_holdout if self.n_holdout else 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_answered if self.n_answered else 0.0

    @property
    def gold_accuracy(self) -> float:
        return self.n_gold_correct / self.n_gold if self.n_gold else 1.0

    def ready(
        self,
        *,
        min_holdout: int = _MIN_HOLDOUT,
        min_accuracy: float = _MIN_ACCURACY,
        min_coverage: float = _MIN_COVERAGE,
        min_gold_accuracy: float = _MIN_GOLD_ACCURACY,
    ) -> bool:
        """Blunt cutover verdict. All bars must clear simultaneously."""
        return (
            self.n_holdout >= min_holdout
            and self.coverage >= min_coverage
            and self.accuracy >= min_accuracy
            and self.gold_accuracy >= min_gold_accuracy
        )

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "relation": self.relation,
            "n_holdout": self.n_holdout,
            "n_answered": self.n_answered,
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "gold_accuracy": round(self.gold_accuracy, 4),
            "ready": self.ready(),
        }


def evaluate_relation(
    log: TrainingLog,
    relation: str,
    *,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    reachable_fn: Callable[[], bool] | None = None,
    k: int | None = None,
    threshold: float | None = None,
) -> RelationReport:
    """Fit a student on TRAIN rows, score it on HOLDOUT rows for one relation."""
    kwargs = {"fit_split": "train", "embed_fn": embed_fn, "reachable_fn": reachable_fn}
    if k is not None:
        kwargs["k"] = k
    if threshold is not None:
        kwargs["threshold"] = threshold
    student = ExtractionStudent(log, **kwargs)

    holdout = log.rows(relation=relation, split="holdout")
    report = RelationReport(relation=relation, n_holdout=len(holdout))
    for r in holdout:
        role_map = None
        if isinstance(r.provenance, dict):
            role_map = r.provenance.get("role_map")
        pred = student.classify(r.raw_text or r.masked_text, relation, role_map=role_map)
        is_gold = r.teacher == TEACHER_PM
        if is_gold:
            report.n_gold += 1
        if pred.abstained:
            continue
        report.n_answered += 1
        if pred.label == r.label:
            report.n_correct += 1
            if is_gold:
                report.n_gold_correct += 1
        else:
            if is_gold:
                # gold miss is the worst case — record it for inspection
                report.confusions.append((r.raw_text[:80], r.label, pred.label))
            elif len(report.confusions) < 25:
                report.confusions.append((r.raw_text[:80], r.label, pred.label))
    return report


def _head_feature(r) -> str:
    """The delexicalized feature a head trains/evaluates on — masked_text if
    present, else delexicalize raw_text (mirrors ExtractionStudent)."""
    from app.core.delexicalize import delexicalize
    mt = r.masked_text
    if not mt and r.raw_text:
        rm = (r.provenance or {}).get("role_map") if isinstance(r.provenance, dict) else None
        mt = delexicalize(r.raw_text, rm).masked
    return mt or ""


def evaluate_relation_head(
    log: TrainingLog,
    relation: str,
    *,
    embed_fn: Callable[[list[str]], np.ndarray],
    head: "object | None" = None,
    **head_kwargs,
) -> RelationReport:
    """Leave-one-deal-out score for the **trained NeuralHead** on one relation.

    Mirrors :func:`evaluate_relation` (same TRAIN/HOLDOUT split, same report
    shape) but the scorer is the contrastive head, not the kNN student — this
    is the eval-gate the model registry (#72) consumes to decide whether a
    freshly-trained head is safe to promote. One batched embedding pass per
    split keeps it kind to the embedding host.

    ``embed_fn`` must return an L2-normalized ``(N, D)`` matrix (the pipeline
    contract). A pre-fitted ``head`` may be passed to score an existing
    champion on the current holdout; otherwise a new head is fit on TRAIN rows
    (gold-weighted via each row's ``weight``).
    """
    from app.core.neural_head import NeuralHead

    train = log.rows(relation=relation, split="train")
    holdout = log.rows(relation=relation, split="holdout")
    report = RelationReport(relation=relation, n_holdout=len(holdout))

    feats_tr, ytr, wtr = [], [], []
    for r in train:
        f = _head_feature(r)
        if f and r.label:
            feats_tr.append(f); ytr.append(r.label); wtr.append(float(r.weight) or 1.0)
    feats_ho, yho, gold_ho = [], [], []
    for r in holdout:
        f = _head_feature(r)
        if f and r.label:
            feats_ho.append(f); yho.append(r.label)
            gold_ho.append(r.teacher == TEACHER_PM)

    # n_holdout reflects the rows we can actually score (have a feature+label).
    report.n_holdout = len(feats_ho)
    if not feats_ho:
        return report
    candidates = sorted(set(ytr))

    if head is None:
        if not feats_tr:
            return report  # nothing to learn from → abstains everywhere
        Xtr = np.asarray(embed_fn(feats_tr), dtype=np.float32)
        head = NeuralHead(**head_kwargs).fit(Xtr, ytr, sample_weight=np.asarray(wtr, dtype=np.float32))

    Xho = np.asarray(embed_fn(feats_ho), dtype=np.float32)
    for i, (gold_label, is_gold) in enumerate(zip(yho, gold_ho)):
        if is_gold:
            report.n_gold += 1
        dec = head.classify(Xho[i], candidates)
        if dec.verdict is None:
            continue
        report.n_answered += 1
        if dec.verdict == gold_label:
            report.n_correct += 1
            if is_gold:
                report.n_gold_correct += 1
        elif is_gold:
            report.confusions.append((feats_ho[i][:80], gold_label, dec.verdict))
        elif len(report.confusions) < 25:
            report.confusions.append((feats_ho[i][:80], gold_label, dec.verdict))
    return report


def evaluate_all(
    log: TrainingLog,
    *,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    reachable_fn: Callable[[], bool] | None = None,
    k: int | None = None,
    threshold: float | None = None,
) -> dict[str, RelationReport]:
    """Shadow-score every relation present in the log. Keyed by relation."""
    relations = sorted({
        r.relation for r in log.rows()  # cheap: ids+labels only used downstream
    })
    out: dict[str, RelationReport] = {}
    for rel in relations:
        out[rel] = evaluate_relation(
            log, rel, embed_fn=embed_fn, reachable_fn=reachable_fn,
            k=k, threshold=threshold,
        )
    return out


def ready_relations(reports: dict[str, RelationReport]) -> list[str]:
    """The relations whose LLM call is safe to replace, today."""
    return sorted(rel for rel, rep in reports.items() if rep.ready())


def summary(reports: dict[str, RelationReport]) -> dict[str, object]:
    """A glanceable readiness census for the PM / CI log."""
    rels = list(reports.values())
    ready = [r for r in rels if r.ready()]
    return {
        "relations": len(rels),
        "ready": [r.relation for r in ready],
        "not_ready": [r.relation for r in rels if not r.ready()],
        "detail": {r.relation: r.as_dict() for r in rels},
    }
