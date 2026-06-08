"""#71 — learnable span extractors that replace the doc-excerpt LLM enrich
extractors (the ~325s stage) with a per-ATOM scan.

The insight: the LLM extractors run on a 30K-char doc EXCERPT and lose recall to
truncation ("Pack 18 has 196 clauses; a single excerpt loses 80%+"). But the
parser already atomizes the WHOLE document — every clause/row/contact is an atom.
So we don't need to re-extract from an excerpt: we classify each EXISTING atom
"is this a <relation> item?" with a trained, recall-tuned, embedder-pinned binary
head. Full-document coverage, local, free, and recall can EXCEED the excerpt LLM.

Per relation:
  * SpanHead — binary "is this atom a <relation> span?" (LogisticRegression over
    frozen embeddings). Recall-tuned: pick the threshold at the highest recall
    whose precision still clears a floor. Held-out BY DEAL.
  * Norm — value normalization is mostly deterministic (amounts, emails, dates)
    and handled by the existing typed extractors / atom_type_sanity; this module
    owns the RECALL half (which atoms are items). The LLM is consulted only on
    the residual the head is unsure about (guess-free).

Same learnable contract as app.core.type_head: trains from the CURRENT log,
eval-gated promotion + rollback (recall must not regress), retrains as the log
grows. OFF until wired + flag-enabled.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_HOLDOUT = 0.25
# Relations worth replacing (recall-heavy doc-excerpt extractors).
DEFAULT_RELATIONS = ("requirements", "stakeholders", "commercial_line_items",
                     "site_clusters", "quantities", "milestones")
_PRECISION_FLOOR = 0.80   # don't accept a recall point below this precision


def _split(deal_id: str) -> str:
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "holdout" if (h % 100) / 100.0 < _HOLDOUT else "train"


@dataclass
class SpanHeadMetrics:
    relation: str
    n_pos: int
    n_neg: int
    n_holdout_pos: int
    threshold: float
    recall: float
    precision: float
    trained_at: float
    log_rows: int


@dataclass
class TrainedSpanHead:
    relation: str
    model: Any
    pos_index: int          # index of the positive class in model.classes_
    threshold: float
    metrics: SpanHeadMetrics
    embed_fn: Callable[[list[str]], np.ndarray] | None = None

    def is_item(self, text: str) -> tuple[bool, float]:
        """(is_a_<relation>_item, score). Recall-tuned threshold."""
        if self.embed_fn is None:
            return False, 0.0
        try:
            vec = np.asarray(self.embed_fn([text]), dtype=np.float32)
        except Exception:
            return False, 0.0
        p = float(self.model.predict_proba(vec)[0][self.pos_index])
        return (p >= self.threshold), p


def _load(log_db: str):
    import sqlite3
    con = sqlite3.connect(log_db)
    rows = con.execute(
        "SELECT relation, COALESCE(NULLIF(masked_text,''),raw_text) AS feat, deal_id "
        "FROM training_rows WHERE COALESCE(masked_text,raw_text,'')!=''"
    ).fetchall()
    con.close()
    return [(r, f, d or "") for r, f, d in rows if f]


def _pick_threshold(scores: np.ndarray, y: np.ndarray, floor: float) -> tuple[float, float, float]:
    """Highest-recall threshold whose precision >= floor (recall-first)."""
    best = (0.5, 0.0, 0.0)  # thr, recall, precision
    # Use the raw unique scores as candidate thresholds (NOT rounded — rounding
    # up can push the threshold above a score and exclude the point it should
    # include, yielding tp=0 and a false "no valid threshold").
    order = np.unique(scores)
    for thr in order:
        pred = scores >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec >= floor and rec > best[1]:
            best = (float(thr), rec, prec)
    return best


def train_span_head(
    relation: str,
    *,
    log_db: str | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    precision_floor: float = _PRECISION_FLOOR,
) -> TrainedSpanHead | None:
    """Train the binary span head for ``relation`` from the current log, scored
    held-out-by-deal. Positives = atoms the LLM extracted as this relation;
    negatives = atoms it did not (other relations / _keep). Returns None if too
    little data."""
    from sklearn.linear_model import LogisticRegression

    log_db = log_db or os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
    if embed_fn is None:
        from app.core.embedding_retrieval import embed_texts
        embed_fn = embed_texts

    data = _load(log_db)
    pos_feats = {f for r, f, _ in data if r == relation}
    if len(pos_feats) < 30:
        return None
    # label each unique feature: 1 if ever a positive for this relation, else 0.
    feat_deal: dict[str, str] = {}
    for _r, f, d in data:
        feat_deal.setdefault(f, d)
    labels = {f: (1 if f in pos_feats else 0) for f in feat_deal}
    # balance negatives ~3x positives to keep precision meaningful.
    pos = [f for f, y in labels.items() if y == 1]
    neg = [f for f, y in labels.items() if y == 0]
    if len(neg) > 3 * len(pos):
        import random
        random.seed(0)
        neg = random.sample(neg, 3 * len(pos))
    feats = pos + neg
    emb = {}
    for i in range(0, len(feats), 256):
        b = feats[i:i + 256]
        for k, v in zip(b, np.asarray(embed_fn(b), dtype=np.float32)):
            emb[k] = v

    Xtr, ytr, Xte, yte = [], [], [], []
    for f in feats:
        v = emb.get(f)
        if v is None:
            continue
        if _split(feat_deal[f]) == "holdout":
            Xte.append(v); yte.append(labels[f])
        else:
            Xtr.append(v); ytr.append(labels[f])
    if sum(ytr) < 15 or sum(yte) < 5 or len(set(ytr)) < 2:
        return None
    Xtr, Xte = np.vstack(Xtr), np.vstack(Xte)
    ytr, yte = np.array(ytr), np.array(yte)

    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pos_idx = list(clf.classes_).index(1)
    scores = clf.predict_proba(Xte)[:, pos_idx]
    thr, rec, prec = _pick_threshold(scores, yte, precision_floor)

    m = SpanHeadMetrics(
        relation=relation, n_pos=len(pos), n_neg=len(neg),
        n_holdout_pos=int((yte == 1).sum()), threshold=thr, recall=rec,
        precision=prec, trained_at=time.time(), log_rows=len(data),
    )
    return TrainedSpanHead(relation=relation, model=clf, pos_index=pos_idx,
                           threshold=thr, metrics=m, embed_fn=embed_fn)


# ── learnable registry: eval-gated promotion + rollback (recall-monotonic) ─────

def _reg_dir() -> str:
    return os.environ.get("SOWSMITH_SPAN_HEAD_DIR", "_span_heads")


def _paths(relation: str):
    d = _reg_dir()
    return os.path.join(d, f"{relation}.pkl"), os.path.join(d, f"{relation}.json")


def load_span_head(relation: str, embed_fn=None) -> TrainedSpanHead | None:
    import pickle
    mp, kp = _paths(relation)
    if not (os.path.exists(mp) and os.path.exists(kp)):
        return None
    try:
        with open(mp, "rb") as f:
            obj = pickle.load(f)
        if embed_fn is None:
            from app.core.embedding_retrieval import embed_texts
            embed_fn = embed_texts
        obj.embed_fn = embed_fn
        return obj
    except Exception:
        return None


def retrain_span_heads(
    relations=DEFAULT_RELATIONS, *, log_db: str | None = None,
    min_precision: float = _PRECISION_FLOOR, min_recall: float = 0.5, slack: float = 0.02,
) -> dict[str, Any]:
    """Train + eval-gate each relation's span head. Promote ONLY if precision
    clears the floor, recall clears min_recall, AND recall does not regress vs the
    incumbent (minus slack) -> recall is monotonic. Returns per-relation status."""
    import pickle
    os.makedirs(_reg_dir(), exist_ok=True)
    out = {}
    for rel in relations:
        head = train_span_head(rel, log_db=log_db, precision_floor=min_precision)
        if head is None:
            out[rel] = {"status": "insufficient_data"}
            continue
        _, kp = _paths(rel)
        inc = json.load(open(kp, encoding="utf-8")) if os.path.exists(kp) else None
        r = head.metrics.recall
        inc_r = float(inc.get("recall", 0.0)) if inc else 0.0
        if head.metrics.precision < min_precision or r < min_recall or (inc and r < inc_r - slack):
            out[rel] = {"status": "rejected", "recall": round(r, 3),
                        "precision": round(head.metrics.precision, 3), "incumbent_recall": round(inc_r, 3)}
            continue
        mp, kp = _paths(rel)
        save = TrainedSpanHead(relation=rel, model=head.model, pos_index=head.pos_index,
                               threshold=head.threshold, metrics=head.metrics)
        with open(mp, "wb") as f:
            pickle.dump(save, f)
        json.dump(head.metrics.__dict__, open(kp, "w"), indent=2)
        out[rel] = {"status": "promoted", "recall": round(r, 3),
                    "precision": round(head.metrics.precision, 3), "threshold": round(head.threshold, 3)}
    return out


class SpanExtractorSet:
    """Loaded promoted span heads. extract(atoms) -> {relation: [atom,...]} of the
    atoms each head identifies as a <relation> item (the recall half of #71).
    Value normalization is handled downstream (deterministic + existing
    extractors); the LLM backstops the residual the heads don't cover."""

    def __init__(self, relations=DEFAULT_RELATIONS, embed_fn=None):
        self.heads = {r: h for r in relations if (h := load_span_head(r, embed_fn))}

    def __bool__(self):
        return bool(self.heads)

    def extract(self, atoms) -> dict[str, list]:
        def _txt(a):
            return (getattr(a, "raw_text", "") or getattr(a, "normalized_text", "") or "")
        out: dict[str, list] = {r: [] for r in self.heads}
        for a in atoms:
            t = _txt(a)
            if not t:
                continue
            for rel, head in self.heads.items():
                hit, _score = head.is_item(t)
                if hit:
                    out[rel].append(a)
        return out
