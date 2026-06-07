"""Unified schematic neural-head framework — the spine every per-module head
plugs into, so "make them all neural heads" is one pattern, not seven one-offs.

Each schematic sub-decision that is perceptual/fuzzy (symbol class, page kind,
discipline, room type, "do these two symbols connect") becomes a registered head
with the SAME guarantees, taught by the VLM teacher and distilled to local:

  feature_fn  : turn the raw thing (crop / page / symbol-pair) into a vector
  teacher     : the VLM/CV label that supervises it (silver) + PM corrections (gold)
  head        : auto-select LR/MLP/GB by leave-one-DEAL-out macro-F1 (honest eval)
  abstain     : low confidence -> None -> caller falls back to the VLM teacher
  store       : SQLite training rows, split by deal, shippable warm base

What is NOT a neural head (stays deterministic — neural would be worse):
  crop_sha256 / NMS / count cross-check / bbox math / source_replay / legend
  TABLE geometry / sheet_metadata regex. Provenance + exact parsing must be exact.

What stays per-document (NOT a global head): symbol grounding — handled by
:class:`app.core.schematic_symbol_head.LegendIndex` (the legend is the answer key
for its own set). The heads here are the GLOBAL-vocabulary decisions whose label
space is the same across every drawing set.

This module reuses the proven training store + auto-select trainer; it adds the
registry, teacher-capture hook, eval-gated promotion, and abstain contract that
make all heads uniform.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.schematic_symbol_head import (
    SchematicSymbolStore,
    SymbolRow,
    TrainedSymbolHead,
    train_symbol_head,
)

FeatureFn = Callable[[Any], np.ndarray]


@dataclass
class HeadSpec:
    """Declares one neural head. The framework gives it train/predict/abstain/
    teacher-capture for free."""

    name: str                       # "page_kind", "discipline", "room_type", ...
    feature_fn: FeatureFn           # raw input -> vector
    abstain: float = 0.55           # below top-prob -> abstain -> VLM fallback
    db_env: str | None = None       # env var pointing at this head's store DB
    global_vocab: bool = True       # False => belongs in a per-document index

    def store(self) -> SchematicSymbolStore:
        path = os.environ.get(self.db_env) if self.db_env else None
        return SchematicSymbolStore(path or ":memory:")


@dataclass
class RegisteredHead:
    spec: HeadSpec
    trained: TrainedSymbolHead | None = None

    def predict(self, raw: Any):
        """(label, prob) or None (abstain -> caller consults the VLM teacher)."""
        if self.trained is None:
            return None
        feat = self.spec.feature_fn(raw)
        return self.trained.classify(feat)


class HeadRegistry:
    """Holds head specs + their live trained models, with eval-gated promotion
    and rollback (a new model only replaces the live one if it clears the gate)."""

    def __init__(self, min_macro_f1: float = 0.70):
        self._heads: dict[str, RegisteredHead] = {}
        self._stores: dict[str, SchematicSymbolStore] = {}
        self.min_macro_f1 = min_macro_f1

    def register(self, spec: HeadSpec) -> None:
        self._heads[spec.name] = RegisteredHead(spec=spec)
        self._stores[spec.name] = spec.store()

    def names(self) -> list[str]:
        return sorted(self._heads)

    def capture(self, head: str, *, deal_id: str, raw: Any, label: str,
                teacher: str, sheet: str | None = None, confidence: float = 0.9) -> None:
        """Teacher-capture: log one labeled example for `head`. Silver from the
        VLM/CV, gold from PM corrections (weighted higher in the store)."""
        from app.core.schematic_symbol_head import feature_sha

        spec = self._heads[head].spec
        feat = spec.feature_fn(raw)
        self._stores[head].log([SymbolRow(
            deal_id=deal_id, sheet=sheet, crop_sha=feature_sha(feat),
            label=label, teacher=teacher, confidence=confidence, feature=feat,
        )])

    def train(self, head: str) -> dict[str, Any]:
        """Train (auto-select + leave-one-deal-out eval-gate) and PROMOTE only if
        the candidate clears ``min_macro_f1``. Returns a status dict; on failure
        the previously-live model is kept (rollback by default)."""
        reg = self._heads[head]
        candidate = train_symbol_head(self._stores[head], abstain=reg.spec.abstain)
        if candidate is None:
            return {"head": head, "promoted": False, "reason": "insufficient_data"}
        if candidate.val_macro_f1 < self.min_macro_f1:
            return {"head": head, "promoted": False, "reason": "below_gate",
                    "candidate_f1": candidate.val_macro_f1, "gate": self.min_macro_f1}
        reg.trained = candidate  # promote
        return {"head": head, "promoted": True, "chosen": candidate.chosen,
                "val_macro_f1": candidate.val_macro_f1,
                "n_classes": len(candidate.classes)}

    def predict(self, head: str, raw: Any):
        return self._heads[head].predict(raw)

    def store(self, head: str) -> SchematicSymbolStore:
        return self._stores[head]


# ── default head feature extractors ───────────────────────────────────────────
# A rendered-page (or crop) PNG -> vector reuses the symbol head's invariant
# extractor (works on any image). Page-level heads get a coarser thumbnail; richer
# layout/text features can drop in behind the same feature_fn later.


def page_feature(png_bytes: bytes) -> np.ndarray:
    from app.core.schematic_symbol_head import crop_feature
    return crop_feature(png_bytes)


def default_registry() -> HeadRegistry:
    """The heads that are GLOBAL-vocabulary (label set is the same across every
    drawing set) and therefore legitimately a trained classifier. Symbol grounding
    is intentionally absent — it is per-document (LegendIndex), not global."""
    reg = HeadRegistry()
    reg.register(HeadSpec("page_kind", page_feature,
                          db_env="SOWSMITH_PAGEKIND_HEAD_DB"))
    reg.register(HeadSpec("discipline", page_feature,
                          db_env="SOWSMITH_DISCIPLINE_HEAD_DB"))
    return reg
