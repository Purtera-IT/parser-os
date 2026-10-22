"""Holdout-stratified evaluation of Division proposal (PUR-29).

Local labelled data only (default: the synthetic fixture). Stratified k-fold on
the normalised raw label; label clustering is re-fit INSIDE each training fold
so the holdout never informs which spellings merge.

Reports coverage (answered / total), accuracy on answered, macro-F1 over
answered, abstention rate on examples flagged ambiguous vs clean, and the
spelling clusters learned on the full set.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.division_proposal import DivisionProposer, load_examples, normalize_label  # noqa: E402


def stratified_folds(examples, k, seed):
    by = defaultdict(list)
    for i, e in enumerate(examples):
        by[normalize_label(e["division"])].append(i)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for idxs in by.values():
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def evaluate(examples, k=3, seed=0, **params):
    folds = stratified_folds(examples, k, seed)
    rows = []
    for f, test_idx in enumerate(folds):
        test = set(test_idx)
        model = DivisionProposer(**params).fit([e for i, e in enumerate(examples) if i not in test])
        for i in test_idx:
            e = examples[i]
            p = model.propose(e["work_order"])
            rows.append({"gold": model.canonical(e["division"]), "pred": p.division,
                         "ambiguous": bool(e.get("ambiguous")), "reason": p.reason, "fold": f})
    answered = [r for r in rows if r["pred"] is not None]
    correct = sum(r["pred"] == r["gold"] for r in answered)
    labels = sorted({r["gold"] for r in rows})
    f1s = {}
    for lab in labels:
        tp = sum(r["pred"] == lab and r["gold"] == lab for r in answered)
        fp = sum(r["pred"] == lab and r["gold"] != lab for r in answered)
        fn = sum(r["pred"] != lab and r["gold"] == lab for r in answered)
        f1s[lab] = (2 * tp / (2 * tp + fp + fn)) if tp else 0.0
    amb = [r for r in rows if r["ambiguous"]]
    clean = [r for r in rows if not r["ambiguous"]]
    rate = lambda rs: round(sum(r["pred"] is None for r in rs) / len(rs), 3) if rs else None  # noqa: E731
    full = DivisionProposer(**params).fit(examples)
    clusters = defaultdict(list)
    for norm, canon in full.label_map.items():
        clusters[canon].append(norm)
    return {
        "n": len(rows), "folds": k,
        "coverage": round(len(answered) / len(rows), 3),
        "accuracy_on_answered": round(correct / len(answered), 3) if answered else None,
        "macro_f1_on_answered": round(sum(f1s.values()) / len(f1s), 3) if f1s else None,
        "per_division_f1": {k_: round(v, 3) for k_, v in f1s.items()},
        "abstention_rate_ambiguous": rate(amb),
        "abstention_rate_clean": rate(clean),
        "accuracy_on_answered_clean": (
            round(sum(r["pred"] == r["gold"] for r in clean if r["pred"]) / max(1, sum(1 for r in clean if r["pred"])), 3)
        ),
        "abstain_reasons": dict(Counter(r["reason"].split(":")[0] for r in rows if r["pred"] is None)),
        "label_clusters": {c: sorted(m) for c, m in sorted(clusters.items())},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="tests/fixtures/division/synthetic_labelled.json")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(evaluate(load_examples(a.labels), a.folds, a.seed), indent=2))
