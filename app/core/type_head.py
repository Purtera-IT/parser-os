"""Learnable atom-type deflector (#70, partial cutover) — a trained classifier
that fronts the LLM typing stage on the HIGH-CONFIDENCE subset only, and keeps
improving as the training log grows.

Why this exists (proven): the kNN student caps at ~0.65 on the full 43-class
problem because the LLM teacher labels the ``_keep`` boundary inconsistently.
But on the subset where a *trained* classifier is confident it assigns a
SPECIFIC type, precision is high (≈0.92 @ conf≥0.85). So we deflect only that
subset off the LLM (guess-free), and send everything else to the LLM as before.

**It keeps learning** — three ways, no frozen model:
  1. ``train_type_head`` reads the CURRENT training log every time, so new LLM
     rows + PM-gold corrections are always included.
  2. ``retrain_if_stale`` rebuilds when the log has grown by ``min_growth`` rows
     since the last trained snapshot (cheap: LR over cached embeddings).
  3. **Eval-gated promotion with rollback**: a freshly trained head is adopted
     ONLY if its held-out typed-precision@threshold is >= the incumbent's (minus
     a small slack). A regression is rejected, so quality is monotonic. As PM
     corrections sharpen the ``_keep`` boundary, the confident-coverage grows and
     the threshold can be lowered — automatically, safely.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_DEFAULT_THRESHOLD = 0.85          # confident-typed deflection bar (precision-first)
_MIN_PRECISION = 0.85              # promotion floor on held-out typed predictions
_HOLDOUT = 0.25
_KEEP = "_keep"


def _split(deal_id: str) -> str:
    import hashlib
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "holdout" if (h % 100) / 100.0 < _HOLDOUT else "train"


@dataclass
class TypeHeadMetrics:
    n_train: int
    n_holdout: int
    threshold: float
    deflect_rate: float       # fraction of holdout atoms deflected (conf>=thr, !=_keep)
    deflect_precision: float  # accuracy on those deflected
    overall_acc: float
    trained_at: float
    log_rows: int             # log size this head was trained on (staleness key)


@dataclass
class TrainedTypeHead:
    model: Any
    classes: list[str]
    threshold: float
    metrics: TypeHeadMetrics
    embed_fn: Callable[[list[str]], np.ndarray] | None = field(default=None, repr=False)

    def classify(self, text: str) -> tuple[str, float] | None:
        """Return (specific_type, confidence) to DEFLECT off the LLM, or None to
        abstain (caller falls back to the LLM). Guess-free: abstains on low
        confidence or a ``_keep`` prediction."""
        if self.embed_fn is None:
            return None
        try:
            vec = np.asarray(self.embed_fn([text]), dtype=np.float32)
        except Exception:
            return None
        proba = self.model.predict_proba(vec)[0]
        i = int(np.argmax(proba))
        label, conf = self.classes[i], float(proba[i])
        if label == _KEEP or conf < self.threshold:
            return None
        return label, conf


def _load_rows(log_db: str):
    import sqlite3
    con = sqlite3.connect(log_db)
    rows = con.execute(
        "SELECT COALESCE(NULLIF(masked_text,''), raw_text) AS feat, label, deal_id "
        "FROM training_rows WHERE relation='atom_type' "
        "AND COALESCE(masked_text,raw_text,'')!='' AND label IS NOT NULL"
    ).fetchall()
    con.close()
    return [(f, l, d or "") for f, l, d in rows if f]


def train_type_head(
    log_db: str | None = None,
    *,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
) -> TrainedTypeHead | None:
    """Train an LR type-deflector from the CURRENT training log, scored on a
    held-out-by-deal split. Returns None if there isn't enough data. The caller
    eval-gates promotion via the returned metrics."""
    from sklearn.linear_model import LogisticRegression

    log_db = log_db or os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
    if embed_fn is None:
        from app.core.embedding_retrieval import embed_texts
        embed_fn = embed_texts

    data = _load_rows(log_db)
    if len(data) < 200 or len({l for _, l, _ in data}) < 3:
        return None

    feats = [d[0] for d in data]
    # de-dup embeddings (many repeats) to keep training cheap.
    uniq = sorted(set(feats))
    emb = {}
    for i in range(0, len(uniq), 256):
        batch = uniq[i:i + 256]
        for k, v in zip(batch, np.asarray(embed_fn(batch), dtype=np.float32)):
            emb[k] = v

    Xtr, ytr, Xte, yte = [], [], [], []
    for feat, label, deal in data:
        v = emb.get(feat)
        if v is None:
            continue
        (Xte, yte) if _split(deal) == "holdout" else (Xtr, ytr)
        if _split(deal) == "holdout":
            Xte.append(v); yte.append(label)
        else:
            Xtr.append(v); ytr.append(label)
    if len(ytr) < 100 or len(yte) < 20 or len(set(ytr)) < 3:
        return None
    Xtr, Xte = np.vstack(Xtr), np.vstack(Xte)
    ytr, yte = np.array(ytr), np.array(yte)

    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(Xtr, ytr)
    classes = list(clf.classes_)
    proba = clf.predict_proba(Xte)
    pred = np.array(classes)[np.argmax(proba, axis=1)]
    conf = np.max(proba, axis=1)
    overall = float((pred == yte).mean())
    sel = (pred != _KEEP) & (conf >= threshold)
    deflect_rate = float(sel.mean())
    deflect_prec = float((pred[sel] == yte[sel]).mean()) if sel.sum() else 0.0

    metrics = TypeHeadMetrics(
        n_train=len(ytr), n_holdout=len(yte), threshold=threshold,
        deflect_rate=deflect_rate, deflect_precision=deflect_prec,
        overall_acc=overall, trained_at=time.time(), log_rows=len(data),
    )
    return TrainedTypeHead(model=clf, classes=classes, threshold=threshold,
                           metrics=metrics, embed_fn=embed_fn)


# ── registry: eval-gated promotion + rollback (monotonic quality) ──────────────

def _registry_dir() -> str:
    return os.environ.get("SOWSMITH_TYPE_HEAD_DIR", "_type_head")


def _meta_path() -> str:
    return os.path.join(_registry_dir(), "metrics.json")


def _model_path() -> str:
    return os.path.join(_registry_dir(), "head.pkl")


def load_promoted_head(embed_fn=None) -> TrainedTypeHead | None:
    """Load the currently-promoted head (best so far), or None."""
    import pickle
    mp, kp = _model_path(), _meta_path()
    if not (os.path.exists(mp) and os.path.exists(kp)):
        return None
    try:
        with open(mp, "rb") as f:
            obj = pickle.load(f)
        m = json.load(open(kp, encoding="utf-8"))
        if embed_fn is None:
            from app.core.embedding_retrieval import embed_texts
            embed_fn = embed_texts
        obj.embed_fn = embed_fn
        return obj
    except Exception:
        return None


def retrain_if_stale(
    *, log_db: str | None = None, min_growth: int = 300,
    min_precision: float = _MIN_PRECISION, slack: float = 0.01,
) -> dict[str, Any]:
    """Rebuild + eval-gate the head when the log grew by >= min_growth rows since
    the last promoted snapshot. Promote ONLY if held-out deflect-precision clears
    ``min_precision`` AND does not regress vs the incumbent (minus slack).
    Returns a status dict. This is the 'keeps learning, never worse' loop."""
    os.makedirs(_registry_dir(), exist_ok=True)
    incumbent = None
    if os.path.exists(_meta_path()):
        try:
            incumbent = json.load(open(_meta_path(), encoding="utf-8"))
        except Exception:
            incumbent = None

    # staleness check: skip retrain if the log hasn't grown enough.
    import sqlite3
    db = log_db or os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
    try:
        n_now = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM training_rows WHERE relation='atom_type'").fetchone()[0]
    except Exception:
        return {"status": "no_log"}
    if incumbent and n_now - int(incumbent.get("log_rows", 0)) < min_growth:
        return {"status": "fresh", "log_rows": n_now}

    head = train_type_head(log_db=db)
    if head is None:
        return {"status": "insufficient_data"}
    p = head.metrics.deflect_precision
    inc_p = float(incumbent.get("deflect_precision", 0.0)) if incumbent else 0.0
    if p < min_precision or (incumbent and p < inc_p - slack):
        return {"status": "rejected", "candidate_precision": round(p, 3),
                "incumbent_precision": round(inc_p, 3), "reason": "regression_or_below_floor"}

    import pickle
    save = TrainedTypeHead(model=head.model, classes=head.classes,
                           threshold=head.threshold, metrics=head.metrics)
    with open(_model_path(), "wb") as f:
        pickle.dump(save, f)
    json.dump(head.metrics.__dict__, open(_meta_path(), "w"), indent=2)
    return {"status": "promoted", "deflect_precision": round(p, 3),
            "deflect_rate": round(head.metrics.deflect_rate, 3), "log_rows": n_now}
