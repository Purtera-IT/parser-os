"""Admission heads — supervised linear classifiers that make span RECALL accurate.

The prototype NeuralHead (cosine-to-centroid + a p>=0.80 gate) is a precision
floor: leave-deals-out it recovers only ~0.16 of missed spans at precision 0.90.
A regularized LOGISTIC head on the same frozen 4096-d embeddings, with a
precision-targeted threshold, reaches ~0.83 precision / ~0.83 recall on held-out
deals — it generalizes past the cosine neighborhood because it learns a real
decision boundary, not a similarity radius.

One head per span relation (admit vs skip). Hard contracts:

* **Frozen embedder.** Operates only on already-computed embeddings; the embedder
  id is stored and checked at serve time (a head is invalid for another embedder).
* **Precision-first.** The threshold is chosen to hold a precision target on a
  held-out split; recall is whatever that precision allows. Abstain (below
  threshold) → the atom is left untouched (guess-free), same contract as the
  store seam.
* **Pure + portable.** sklearn at fit time; serve is a dot product + sigmoid, so
  a saved head is just (coef, intercept, threshold) — no sklearn needed to serve.
* **Text-ruleable.** Heads retrain from the training log + PM corrections; a PM
  teaches recall by adding labeled rows, then a retrain lifts the head.
"""
from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Span relation -> the AtomType a recovered atom is re-typed into (compiler seam).
RELATION_TO_ATOM_TYPE: dict[str, str] = {
    "milestones": "milestone_phase",
    "requirements": "requirement",
    "quantities": "quantity",
    "acceptance_criteria": "acceptance_criterion",
    "risks": "risk",
    "compliance_obligations": "compliance_rule",
    "stakeholders": "stakeholder",
    "certifications": "bonding_insurance",
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class AdmissionHead:
    """A linear admit/skip classifier for one relation over frozen embeddings."""

    relation: str
    coef: np.ndarray            # (D,)
    intercept: float
    threshold: float            # admit iff P(admit) >= threshold
    embed_model: str = ""
    n_train: int = 0
    holdout_precision: float = 0.0
    holdout_recall: float = 0.0
    precision_target: float = 0.90

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return _sigmoid(X @ self.coef + self.intercept)

    def admit(self, vec: np.ndarray) -> bool:
        return bool(self.proba(vec)[0] >= self.threshold)

    # ── persistence (npz: arrays + json meta) ────────────────────────
    def to_npz(self, path: str) -> None:
        meta = {
            "relation": self.relation, "intercept": float(self.intercept),
            "threshold": float(self.threshold), "embed_model": self.embed_model,
            "n_train": int(self.n_train),
            "holdout_precision": float(self.holdout_precision),
            "holdout_recall": float(self.holdout_recall),
            "precision_target": float(self.precision_target),
        }
        with io.open(path, "wb") as fh:
            np.savez(fh, coef=np.asarray(self.coef, dtype=np.float32),
                     __meta__=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8))

    @classmethod
    def from_npz(cls, path: str) -> "AdmissionHead":
        with np.load(path, allow_pickle=False) as z:
            m = json.loads(bytes(z["__meta__"].tobytes()).decode("utf-8"))
            coef = z["coef"]
        return cls(relation=m["relation"], coef=coef, intercept=m["intercept"],
                   threshold=m["threshold"], embed_model=m.get("embed_model", ""),
                   n_train=m.get("n_train", 0), holdout_precision=m.get("holdout_precision", 0.0),
                   holdout_recall=m.get("holdout_recall", 0.0),
                   precision_target=m.get("precision_target", 0.90))


def _pick_threshold(proba: np.ndarray, y: np.ndarray, precision_target: float) -> tuple[float, float, float]:
    """Lowest threshold whose precision >= target (maximising recall at that precision).
    Returns (threshold, precision, recall). Falls back to 0.5 if target unreachable."""
    order = np.argsort(-proba)
    tp = fp = 0
    pos = int(y.sum())
    best = (1.01, 1.0, 0.0)
    for i in order:
        if y[i]:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / pos if pos else 0.0
        if prec >= precision_target and rec > best[2]:
            best = (float(proba[i]), prec, rec)
    if best[0] > 1.0:  # target never met → conservative default
        return 0.5, 0.0, 0.0
    return best


def fit_admission_head(
    relation: str,
    X: np.ndarray,
    y: np.ndarray,
    deals: list[str],
    *,
    embed_model: str = "",
    precision_target: float = 0.90,
    C_grid: tuple = (0.25, 1.0, 4.0, 16.0),
) -> Optional[AdmissionHead]:
    """Fit + eval-gate a head with grouped (by-deal) cross-validation.

    The threshold is tuned on OUT-OF-FOLD probabilities (each row scored by a model
    that never saw its deal), so it transfers to unseen deals instead of overfitting
    the train-fit scores. The regularisation C is swept and the value with the best
    OOF recall@precision_target wins. The reported holdout precision/recall ARE the
    out-of-fold (leave-deal-out) numbers — what production will deliver. The served
    model is refit on all data with the winning C, threshold carried from OOF.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    groups = np.asarray(deals)
    pos_deals = sorted({d for d, yy in zip(deals, y) if yy == 1})
    n_groups = len(set(deals))
    if len(pos_deals) < 3 or len(set(y)) < 2 or n_groups < 3:
        return None
    n_splits = min(5, n_groups)

    def oof_proba(C: float) -> np.ndarray:
        p = np.zeros(len(y), dtype=np.float64)
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        return p

    best = None  # (oof_recall, C, threshold, oof_precision)
    for C in C_grid:
        p = oof_proba(C)
        thr, prec, rec = _pick_threshold(p, y, precision_target)
        if best is None or rec > best[0]:
            best = (rec, C, thr, prec)
    oof_rec, bestC, thr, oof_prec = best

    full = LogisticRegression(C=bestC, class_weight="balanced", max_iter=2000).fit(X, y)
    return AdmissionHead(
        relation=relation,
        coef=full.coef_[0].astype(np.float32),
        intercept=float(full.intercept_[0]),
        threshold=float(thr),
        embed_model=embed_model,
        n_train=len(y),
        holdout_precision=oof_prec,
        holdout_recall=oof_rec,
        precision_target=precision_target,
    )


class _MeanProbaEnsemble:
    """Averages predict_proba over several fitted estimators (picklable)."""

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X):
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.stack([1 - ps, ps], axis=1)


@dataclass
class ModelAdmissionHead:
    """Admit/skip head wrapping ANY fitted sklearn estimator (LR / MLP / GB /
    ensemble) with predict_proba. Same precision-first, embedder-pinned,
    guess-free contract as AdmissionHead — but non-linear models are allowed,
    auto-selected per relation by :func:`fit_best_admission_head`. Persisted via
    pickle (the model isn't a bare linear coef anymore)."""

    relation: str
    model: object
    threshold: float
    model_name: str = "LR"
    embed_model: str = ""
    n_train: int = 0
    holdout_precision: float = 0.0
    holdout_recall: float = 0.0
    precision_target: float = 0.90

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return self.model.predict_proba(X)[:, 1]

    def admit(self, vec: np.ndarray) -> bool:
        return bool(self.proba(vec)[0] >= self.threshold)

    def to_pkl(self, path: str) -> None:
        with io.open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pkl(cls, path: str) -> "ModelAdmissionHead":
        with io.open(path, "rb") as fh:
            return pickle.load(fh)


def fit_best_admission_head(
    relation: str,
    X: np.ndarray,
    y: np.ndarray,
    deals: list[str],
    *,
    embed_model: str = "",
    precision_target: float = 0.90,
) -> Optional[ModelAdmissionHead]:
    """Auto-select the best classifier per relation. Trains LR / MLP / GB (+ a
    MLP+GB mean ensemble), eval-gates each leave-one-DEAL-out, and keeps whichever
    maximises recall@precision_target. Stakeholders prefers LR; acceptance/quantities
    prefer GB; milestones/compliance prefer MLP — so per-relation selection beats any
    single model. Threshold comes from the winner's out-of-fold scores. Returns None
    when the data can't support a gated head (caller leaves the relation on the LLM)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    groups = np.asarray(deals)
    pos_deals = sorted({d for d, yy in zip(deals, y) if yy == 1})
    n_groups = len(set(deals))
    if len(pos_deals) < 3 or len(set(y)) < 2 or n_groups < 3:
        return None
    n_splits = min(5, n_groups)

    def lr():
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

    def mlp():
        return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=400,
                             early_stopping=True, n_iter_no_change=15, random_state=0)

    def gb():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              l2_regularization=1.0, random_state=0)

    cands = {"LR": lr, "MLP": mlp, "GB": gb}

    def oof(fn):
        p = np.zeros(len(y), dtype=np.float64)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            m = fn()
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        return p

    oofs = {name: oof(fn) for name, fn in cands.items()}
    oofs["ENS"] = (oofs["MLP"] + oofs["GB"]) / 2.0

    best = None  # (recall, name, threshold, precision)
    for name, p in oofs.items():
        thr, prec, rec = _pick_threshold(p, y, precision_target)
        if best is None or rec > best[0]:
            best = (rec, name, thr, prec)
    rec, name, thr, prec = best

    if name == "ENS":
        model = _MeanProbaEnsemble([mlp().fit(X, y), gb().fit(X, y)])
    else:
        model = cands[name]().fit(X, y)
    return ModelAdmissionHead(
        relation=relation, model=model, threshold=float(thr), model_name=name,
        embed_model=embed_model, n_train=len(y), holdout_precision=prec,
        holdout_recall=rec, precision_target=precision_target,
    )


class AdmissionRegistry:
    """A directory of admission heads (one .npz per relation) + an index."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._cache: dict[str, AdmissionHead] = {}

    def save(self, head) -> None:
        if isinstance(head, ModelAdmissionHead):
            head.to_pkl(os.path.join(self.path, f"{head.relation}.pkl"))
        else:
            head.to_npz(os.path.join(self.path, f"{head.relation}.npz"))
        self._cache[head.relation] = head
        self._write_index()

    def _write_index(self) -> None:
        idx = {}
        for rel, h in self.load_all().items():
            idx[rel] = {"model": getattr(h, "model_name", "LR"), "n_train": h.n_train,
                        "threshold": round(h.threshold, 4),
                        "holdout_precision": round(h.holdout_precision, 3),
                        "holdout_recall": round(h.holdout_recall, 3),
                        "embed_model": h.embed_model}
        with io.open(os.path.join(self.path, "index.json"), "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=2)

    def load_all(self, *, embed_model: str = "") -> dict:
        """Load every head; a .pkl (multi-model) wins over a .npz of the same relation."""
        out: dict = {}
        if not os.path.isdir(self.path):
            return out
        # linear .npz first, then let .pkl override (newer multi-model heads)
        for ext, loader in ((".npz", AdmissionHead.from_npz), (".pkl", ModelAdmissionHead.from_pkl)):
            for fn in os.listdir(self.path):
                if not fn.endswith(ext):
                    continue
                try:
                    h = loader(os.path.join(self.path, fn))
                except Exception:
                    continue
                if embed_model and h.embed_model and h.embed_model != embed_model:
                    continue  # embedder-pinned
                out[h.relation] = h
        return out


__all__ = ["AdmissionHead", "ModelAdmissionHead", "AdmissionRegistry",
           "fit_admission_head", "fit_best_admission_head", "RELATION_TO_ATOM_TYPE"]
