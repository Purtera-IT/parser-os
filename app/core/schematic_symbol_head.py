"""Neural symbol-classifier head — the distillation target for the schematic
symbol detector.

The CV ``symbol_detector`` (template + text-tag) and the VLM
``vision_symbol_detector`` are the *teachers*. This head is the *student*: it
learns ``glyph crop -> legend class`` so that, after enough teacher-labeled
crops accumulate, symbol classification runs **local, free, and fast** with the
VLM only consulted when the head abstains (guess-free precedence — same MAX
pattern as the text admission heads).

Design mirrors :mod:`app.core.training_log` + :mod:`app.core.admission_head`:

* **Feature** = a deterministic fixed-size grayscale vector of the crop
  (default 64x64 -> 4096-dim, L2-normalized). No new model dependency, CPU-only,
  byte-reproducible. (A learned vision embedder can be swapped in later behind
  the same interface; the head code is feature-agnostic.)
* **Label** = the legend class (``entity_key``) the teacher assigned.
* **Honest eval** = split by ``deal_id`` hash so all crops from one drawing set
  land in the same split — leave-one-deal-out, no train/test leakage of a firm's
  drawing conventions.
* **Auto-select** = try LogisticRegression / MLP / GradientBoosting and keep the
  one with the best leave-one-deal-out macro-F1. Multiclass throughout.
* **Abstain** = if top-class probability < threshold, return ``None`` so the
  caller falls back to the VLM teacher. Precision over recall.
* **Durable** = SQLite training store rides along as a warm base
  (``SOWSMITH_SCHEMATIC_HEAD_DB``); ``:memory:`` is the test/default mode.

Nothing in the compile path changes until a trained head is explicitly loaded
and wired into the detector — this module is the foundation, not the cutover.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_SIDE = 64
FEATURE_DIM = FEATURE_SIDE * FEATURE_SIDE
_DEFAULT_ABSTAIN = 0.55
TEACHER_VLM = "vlm"      # silver
TEACHER_PM = "pm"        # gold
TEACHER_CV = "cv"        # the deterministic detector agreed
_WEIGHT = {TEACHER_PM: 5.0, TEACHER_VLM: 1.0, TEACHER_CV: 1.5}


# ── feature extraction ────────────────────────────────────────────────────────


def crop_feature(png_bytes: bytes, side: int = FEATURE_SIDE) -> np.ndarray:
    """Deterministic glyph feature: resize to side×side grayscale, contrast-
    normalize, flatten, L2-normalize. Symbol shape — not color/size — drives the
    vector, so the same glyph at different scales lands close in feature space."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(png_bytes)) as im:
        g = im.convert("L")
        # Translation+scale invariance: crop to the ink (non-white) bounding box
        # before resizing, so the same glyph anywhere/any-size maps to the same
        # vector. getbbox() works on the inverted image (ink -> nonzero).
        inv = ImageOps.invert(g)
        bbox = inv.getbbox()
        if bbox:
            g = g.crop(bbox)
        g = g.resize((side, side))
    arr = np.asarray(g, dtype=np.float32).reshape(-1)
    arr -= arr.mean()
    std = arr.std()
    if std > 1e-6:
        arr /= std
    n = np.linalg.norm(arr)
    if n > 1e-6:
        arr /= n
    return arr


def feature_sha(feat: np.ndarray) -> str:
    return hashlib.sha256(np.round(feat, 5).tobytes()).hexdigest()[:16]


# ── training store ────────────────────────────────────────────────────────────


_HOLDOUT_FRACTION = 0.25


def assign_split(deal_id: str) -> str:
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "test" if (h % 100) / 100.0 < _HOLDOUT_FRACTION else "train"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbol_rows (
  row_id TEXT PRIMARY KEY,
  deal_id TEXT, sheet TEXT, crop_sha TEXT,
  label TEXT, teacher TEXT, confidence REAL,
  weight REAL, split TEXT, feature TEXT, created REAL
)
"""


@dataclass
class SymbolRow:
    deal_id: str
    sheet: str | None
    crop_sha: str
    label: str
    teacher: str
    confidence: float
    feature: np.ndarray


class SchematicSymbolStore:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("SOWSMITH_SCHEMATIC_HEAD_DB", ":memory:")
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def log(self, rows: list[SymbolRow]) -> int:
        n = 0
        for r in rows:
            rid = hashlib.sha256(
                f"{r.deal_id}|{r.crop_sha}|{r.label}|{r.teacher}".encode()
            ).hexdigest()[:24]
            try:
                self.conn.execute(
                    "INSERT OR REPLACE INTO symbol_rows VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, r.deal_id, r.sheet, r.crop_sha, r.label, r.teacher,
                     float(r.confidence), _WEIGHT.get(r.teacher, 1.0),
                     assign_split(r.deal_id), json.dumps(r.feature.tolist()),
                     time.time()),
                )
                n += 1
            except Exception:
                pass
        self.conn.commit()
        return n

    def fetch(self, split: str | None = None):
        q = "SELECT label, teacher, weight, split, feature FROM symbol_rows"
        if split:
            q += " WHERE split=?"
            cur = self.conn.execute(q, (split,))
        else:
            cur = self.conn.execute(q)
        X, y, w = [], [], []
        for label, _teacher, weight, _split, feat in cur.fetchall():
            X.append(np.asarray(json.loads(feat), dtype=np.float32))
            y.append(label)
            w.append(weight)
        return (np.vstack(X) if X else np.zeros((0, FEATURE_DIM), np.float32),
                np.array(y), np.array(w, dtype=np.float32))

    def label_counts(self) -> dict[str, int]:
        cur = self.conn.execute("SELECT label, COUNT(*) FROM symbol_rows GROUP BY label")
        return {k: v for k, v in cur.fetchall()}


# ── head: auto-select + leave-one-deal-out eval-gate ──────────────────────────


@dataclass
class TrainedSymbolHead:
    model: Any
    classes: list[str]
    abstain: float
    val_macro_f1: float
    chosen: str

    def classify(self, feat: np.ndarray) -> tuple[str, float] | None:
        """Return (label, prob) or None to abstain (caller falls back to VLM)."""
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        p = float(proba[i])
        if p < self.abstain:
            return None
        return self.classes[i], p


def _candidates():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    return {
        "LR": lambda: LogisticRegression(max_iter=1000, C=2.0),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=400),
        "GB": lambda: GradientBoostingClassifier(),
    }


def train_symbol_head(store: SchematicSymbolStore,
                      abstain: float = _DEFAULT_ABSTAIN) -> TrainedSymbolHead | None:
    """Auto-select the best classifier by train/test (leave-one-deal-out) macro-F1.
    Returns None if there isn't enough labeled data yet (need >=2 classes and a
    non-empty test split)."""
    from sklearn.metrics import f1_score

    Xtr, ytr, wtr = store.fetch("train")
    Xte, yte, _ = store.fetch("test")
    if len(set(ytr)) < 2 or len(Xte) == 0:
        return None

    best = None
    for name, make in _candidates().items():
        try:
            clf = make()
            try:
                clf.fit(Xtr, ytr, sample_weight=wtr)
            except TypeError:
                clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            f1 = f1_score(yte, pred, average="macro", zero_division=0)
        except Exception:
            continue
        if best is None or f1 > best[1]:
            best = (name, f1, clf)
    if best is None:
        return None

    # Refit the winner on ALL data for the shipped head.
    Xall, yall, wall = store.fetch(None)
    name, f1, _ = best
    final = _candidates()[name]()
    try:
        final.fit(Xall, yall, sample_weight=wall)
    except TypeError:
        final.fit(Xall, yall)
    return TrainedSymbolHead(
        model=final, classes=list(final.classes_),
        abstain=abstain, val_macro_f1=float(f1), chosen=name,
    )
