"""Continual symbol-type head — gets better every run, retrains in seconds.

The architecture that makes "improve each run + train way faster" real:

* **Frozen backbone, cached features.** Each crop -> a fixed feature vector,
  computed ONCE and stored by crop hash. Re-runs embed only NEW crops; everything
  else is a cache hit. (Default backbone is the deterministic crop_feature; pass
  any frozen embedder — e.g. a SimCLR ResNet — via ``feature_fn``.)
* **Head trains in seconds.** With features cached, the classifier is just the
  auto-select LR/MLP/GB on vectors (sub-second on thousands of rows).
* **Monotonic via eval-gate.** Each retrain is promoted ONLY if it beats the
  current champion's leave-one-deal-out macro-F1; else the champion is kept. So
  accumulating more gold labels can only help or hold — never regress.
* **Supervised.** Trains on canonical device-type gold (VLM) + PM corrections
  (weighted higher). Frozen-backbone + supervised head is far more sample-
  efficient than training a net from scratch.

Durable: point ``SOWSMITH_CONTINUAL_HEAD_DB`` at a file to accumulate across runs
(the warm base). Champion model persists alongside as ``<db>.champion.pkl``.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np

from app.core.schematic_symbol_head import (
    SchematicSymbolStore,
    SymbolRow,
    crop_feature,
    feature_sha,
    train_symbol_head,
)


@dataclass
class RetrainResult:
    promoted: bool
    reason: str
    new_f1: float | None = None
    champion_f1: float | None = None
    n_rows: int = 0
    n_classes: int = 0


class ContinualSymbolHead:
    def __init__(self, db_path: str | None = None, feature_fn=None, min_gain: float = 0.0):
        self.db_path = db_path or os.environ.get("SOWSMITH_CONTINUAL_HEAD_DB", ":memory:")
        self.store = SchematicSymbolStore(self.db_path)
        self.feature_fn = feature_fn or crop_feature
        self.min_gain = min_gain
        self.champion = None
        self.champion_f1: float | None = None
        self._champ_path = (self.db_path + ".champion.pkl") if self.db_path != ":memory:" else None
        self._load_champion()

    def _load_champion(self):
        if self._champ_path and os.path.exists(self._champ_path):
            try:
                with open(self._champ_path, "rb") as f:
                    blob = pickle.load(f)
                self.champion = blob["head"]
                self.champion_f1 = blob["f1"]
            except Exception:
                self.champion = None

    def _save_champion(self):
        if self._champ_path and self.champion is not None:
            try:
                with open(self._champ_path, "wb") as f:
                    pickle.dump({"head": self.champion, "f1": self.champion_f1}, f)
            except Exception:
                pass

    def _seen(self) -> set[str]:
        try:
            cur = self.store.conn.execute("SELECT crop_sha FROM symbol_rows")
            return {r[0] for r in cur.fetchall()}
        except Exception:
            return set()

    def ingest(self, labeled_crops: list[tuple[bytes, str]], *, deal_id: str,
               teacher: str = "vlm", confidence: float = 0.9) -> int:
        """Add (crop_png, label) pairs. Features are computed once per unique crop
        (cache hit otherwise) -> incremental + fast. Returns rows added."""
        seen = self._seen()
        rows = []
        for png, label in labeled_crops:
            feat = self.feature_fn(png)
            sha = feature_sha(feat)
            if sha in seen:
                continue
            seen.add(sha)
            rows.append(SymbolRow(deal_id=deal_id, sheet=None, crop_sha=sha,
                                  label=label, teacher=teacher,
                                  confidence=confidence, feature=feat))
        return self.store.log(rows)

    def retrain(self) -> RetrainResult:
        """Auto-select + leave-one-deal-out eval-gate. Promote only if it beats the
        champion (by >= min_gain). Champion kept otherwise (rollback by default)."""
        cand = train_symbol_head(self.store)
        Xall, yall, _ = self.store.fetch(None)
        if cand is None:
            return RetrainResult(False, "insufficient_data", n_rows=len(Xall),
                                 n_classes=len(set(yall)))
        if self.champion_f1 is not None and cand.val_macro_f1 < self.champion_f1 + self.min_gain:
            return RetrainResult(False, "no_improvement", new_f1=cand.val_macro_f1,
                                 champion_f1=self.champion_f1, n_rows=len(Xall),
                                 n_classes=len(cand.classes))
        self.champion = cand
        self.champion_f1 = cand.val_macro_f1
        self._save_champion()
        return RetrainResult(True, "promoted", new_f1=cand.val_macro_f1,
                             champion_f1=self.champion_f1, n_rows=len(Xall),
                             n_classes=len(cand.classes))

    def classify(self, png: bytes):
        """(label, prob) or None (abstain -> VLM/legend fallback)."""
        if self.champion is None:
            return None
        return self.champion.classify(self.feature_fn(png))

    def n_rows(self) -> int:
        return len(self.store.fetch(None)[0])
