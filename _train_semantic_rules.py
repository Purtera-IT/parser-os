"""Train (calibrate) the SemanticRules' per-rule thresholds against their labelled
examples and write the eval-gated result to the threshold registry the rules read
at construction (models/semantic_rule_thresholds.json).

A rule fires iff its candidate's nearest POSITIVE prototype cosine clears
``threshold`` and beats the nearest NEGATIVE. The threshold is hand-set today; this
re-fits it from data:

  labels = the rule's own positives (should fire) + negatives (should NOT) +
           any accumulated, human-labelled decisions from the SOWSMITH_RULE_LOG
           feedback log (text + ground-truth fire/no-fire).

For each labelled example we score it LEAVE-ONE-OUT (exclude itself from the
prototype matrices so a positive isn't trivially its own nearest match), sweep the
threshold, and pick the value maximising leave-one-out F1. We ADOPT it only if it
beats the current threshold's F1 (eval gate) — same pattern as the grounded-extractor
model registry — so a retrain can never regress a rule.

Requires the SAME embedder production uses (qwen3-embedding); thresholds are
model-specific and won't transfer across embedders. If the embedder is unreachable
the run reports it and writes nothing (the rules keep their hand-set defaults).

Usage:
    python _train_semantic_rules.py            # train all registered rules
    python _train_semantic_rules.py --dry-run  # report, don't write the registry
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REGISTRY = Path(os.environ.get(
    "SOWSMITH_RULE_THRESHOLDS",
    str(Path(__file__).resolve().parent / "models" / "semantic_rule_thresholds.json"),
))


def _collect_rules() -> list:
    """Every SemanticRule the parser uses — the shared cross-cutting ones plus the
    per-format rules defined as rule-getter functions in the parsers."""
    import app.core.semantic_rules as S
    import app.parsers.orbitbrief_pdf as P

    rules = []
    seen = set()
    for mod in (S, P):
        for name in dir(mod):
            if not name.endswith("_rule"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                r = fn()
            except Exception:
                continue
            if isinstance(r, S.SemanticRule) and r.name not in seen and r.positives:
                seen.add(r.name)
                rules.append(r)
    return rules


def _feedback_examples(rule_name: str) -> list[tuple[str, int]]:
    """Human-labelled decisions for this rule from the feedback log, if present.
    A row is used only when it carries a ground-truth label ('label': 0/1) added
    by joining the logged decision to the reviewer's accept/reject."""
    path = os.environ.get("SOWSMITH_RULE_LOG")
    if not path or not Path(path).exists():
        return []
    out: list[tuple[str, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("rule") == rule_name and rec.get("label") in (0, 1) and rec.get("text"):
            out.append((rec["text"], int(rec["label"])))
    return out


def _f1(labels: list[int], preds: list[int]) -> float:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _loo_scores(rule, examples: list[tuple[str, int]]):
    """Leave-one-out (best_pos, best_neg) per example, against the rule's own
    prototype sets — embedding each text once with the production embedder."""
    from app.core.embedding_retrieval import embed_texts

    pos_texts = list(rule.positives)
    neg_texts = list(rule.negatives)
    ex_texts = [t for t, _ in examples]
    # one embed call for everything (cache-backed)
    all_texts = pos_texts + neg_texts + ex_texts
    V = embed_texts(all_texts)  # already L2-normalized
    if not V.any():
        return None  # embedder down -> zero matrix
    n_pos, n_neg = len(pos_texts), len(neg_texts)
    P = V[:n_pos]
    N = V[n_pos:n_pos + n_neg]
    scored = []
    pos_set = {t: i for i, t in enumerate(pos_texts)}
    neg_set = {t: i for i, t in enumerate(neg_texts)}
    for k, (t, lab) in enumerate(examples):
        q = V[n_pos + n_neg + k]
        Pk, Nk = P, N
        if t in pos_set:  # exclude itself from positives
            Pk = np.delete(P, pos_set[t], axis=0)
        if t in neg_set:
            Nk = np.delete(N, neg_set[t], axis=0)
        bp = float((Pk @ q).max()) if len(Pk) else -1.0
        bn = float((Nk @ q).max()) if len(Nk) else -1.0
        scored.append((bp, bn, lab))
    return scored


def _best_threshold(scored):
    """Sweep thresholds, return (threshold, f1) maximising leave-one-out F1."""
    labels = [lab for _, _, lab in scored]
    best_t, best_f1 = None, -1.0
    for i in range(25, 96):  # 0.25 .. 0.95
        t = i / 100.0
        preds = [1 if (bp >= t and bp > bn) else 0 for bp, bn, _ in scored]
        f1 = _f1(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main() -> int:
    dry = "--dry-run" in sys.argv
    # Fast guard: thresholds are model-specific, so train ONLY against the live
    # production embedder. If it's unreachable, bail before any (blocking) embed
    # call — the rules keep their hand-set defaults.
    try:
        from app.core.embedding_retrieval import embedding_endpoint_reachable
        reachable = bool(embedding_endpoint_reachable())
    except Exception:
        reachable = False
    if not reachable:
        print("Embedder UNREACHABLE — thresholds are model-specific, so training "
              "must run against the live qwen3 embedder. Bring it online and re-run.\n"
              "(No registry written; rules keep their hand-set defaults.)")
        return 1
    rules = _collect_rules()
    print(f"Collected {len(rules)} SemanticRules")
    registry: dict = {}
    if REGISTRY.exists():
        try:
            registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    trained = adopted = 0
    for rule in rules:
        examples = [(t, 1) for t in rule.positives] + [(t, 0) for t in rule.negatives]
        examples += _feedback_examples(rule.name)
        scored = _loo_scores(rule, examples)
        if scored is None:
            print(f"  {rule.name:26} SKIP — embedder unreachable (zero vectors)")
            continue
        trained += 1
        cur = rule.threshold
        cur_preds = [1 if (bp >= cur and bp > bn) else 0 for bp, bn, _ in scored]
        cur_f1 = _f1([lab for _, _, lab in scored], cur_preds)
        new_t, new_f1 = _best_threshold(scored)
        adopt = new_f1 >= cur_f1 + 1e-9 and new_t is not None
        flag = "ADOPT" if adopt else "keep "
        print(f"  {rule.name:26} cur thr={cur:.2f} f1={cur_f1:.3f}  ->  "
              f"best thr={new_t:.2f} f1={new_f1:.3f}  [{flag}]  (n={len(scored)})")
        if adopt:
            adopted += 1
            registry[rule.name] = {"threshold": new_t, "f1": round(new_f1, 4),
                                   "n": len(scored), "prev": cur}

    if not trained:
        print("\nNo rules trained — bring the qwen3 embedder online and re-run.")
        return 1
    print(f"\nTrained {trained} rules; {adopted} threshold(s) improved.")
    if dry:
        print("--dry-run: registry NOT written.")
        return 0
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Wrote {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
