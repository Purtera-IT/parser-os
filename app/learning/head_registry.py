"""Head registry — the versioned champion store for trained extractor heads.

The training log accumulates labeled rows; :mod:`app.core.neural_head` turns
them into a fitted head; :mod:`app.core.shadow_eval` scores a head on the
leave-one-deal-out holdout. What was missing — and what this module is — is the
*memory between retrains*: a durable, versioned store that

* keeps every head we ever fit per relation (an audit trail of the curve),
* records each version's holdout metrics + the data/embedder signature it was
  trained against, and
* tracks a single **champion** per relation — the head currently allowed to
  serve — with one-call **promote** and **rollback**.

This is the "breedable" part of the self-improving loop (#72): the retrainer
(:mod:`app.learning.retrain`) fits a candidate, the eval-gate decides if it
beats the incumbent on unseen deals, and only then does it call
:meth:`HeadRegistry.promote`. A regression is undone with
:meth:`HeadRegistry.rollback`. Nothing here ever trains or scores — it only
stores, indexes, and selects, so it has no embedding/network dependency.

Design mirrors the other durable stores (training_log, shadow_history):

* **On disk, versioned.** ``<root>/<relation>/<version>.npz`` is the head;
  ``<version>.json`` its metadata. ``<root>/index.json`` names the champion and
  the promotion history per relation.
* **Env-gated, default-off.** :func:`get_head_registry` returns ``None`` unless
  ``SOWSMITH_HEAD_REGISTRY_DIR`` is set, so production behavior is unchanged
  until we opt in.
* **Never raises into a caller's hot path.** Load failures degrade to ``None``
  (serve falls through to kNN/LLM) rather than breaking a compile.
* **Embedder-pinned.** A head is only valid for the embedding model it was
  trained on; that model id is stored in metadata and checked at serve time.
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.neural_head import NeuralHead


@dataclass
class HeadMeta:
    """Metadata for one stored head version (the audit record next to the npz)."""

    version: str
    relation: str
    created_at: float
    embed_model: str = ""
    data_signature: str = ""
    n_train: int = 0
    n_holdout: int = 0
    n_classes: int = 0
    coverage: float = 0.0
    accuracy: float = 0.0
    gold_accuracy: float = 0.0
    ready: bool = False
    trained: bool = False
    status: str = "candidate"   # "candidate" | "champion" | "retired"
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeadMeta":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}  # type: ignore[attr-defined]
        return cls(**known)


class HeadRegistry:
    """Versioned per-relation head store with a single promotable champion."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._index_path = os.path.join(self.root, "index.json")

    # ── index plumbing ───────────────────────────────────────────────
    def _load_index(self) -> dict[str, Any]:
        try:
            with io.open(self._index_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_index(self, index: dict[str, Any]) -> None:
        tmp = self._index_path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
        os.replace(tmp, self._index_path)  # atomic on same filesystem

    def _rel_entry(self, index: dict[str, Any], relation: str) -> dict[str, Any]:
        return index.setdefault(relation, {"champion": None, "history": []})

    def _rel_dir(self, relation: str) -> str:
        d = os.path.join(self.root, relation)
        os.makedirs(d, exist_ok=True)
        return d

    def _meta_path(self, relation: str, version: str) -> str:
        return os.path.join(self._rel_dir(relation), f"{version}.json")

    def _head_path(self, relation: str, version: str) -> str:
        return os.path.join(self._rel_dir(relation), f"{version}.npz")

    # ── write path ───────────────────────────────────────────────────
    def register(
        self,
        relation: str,
        head: NeuralHead,
        *,
        embed_model: str = "",
        data_signature: str = "",
        n_train: int = 0,
        n_holdout: int = 0,
        coverage: float = 0.0,
        accuracy: float = 0.0,
        gold_accuracy: float = 0.0,
        ready: bool = False,
        notes: str = "",
    ) -> HeadMeta:
        """Persist a freshly-fit head as a new *candidate* version. Does NOT
        promote it — that is the eval-gate's decision (see :meth:`promote`)."""
        version = "h_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        meta = HeadMeta(
            version=version,
            relation=relation,
            created_at=time.time(),
            embed_model=embed_model,
            data_signature=data_signature,
            n_train=int(n_train),
            n_holdout=int(n_holdout),
            n_classes=len(head.classes_),
            coverage=float(coverage),
            accuracy=float(accuracy),
            gold_accuracy=float(gold_accuracy),
            ready=bool(ready),
            trained=bool(head.trained),
            status="candidate",
            notes=notes,
        )
        head.save(self._head_path(relation, version))
        with io.open(self._meta_path(relation, version), "w", encoding="utf-8") as fh:
            json.dump(meta.as_dict(), fh, indent=2)
        index = self._load_index()
        entry = self._rel_entry(index, relation)
        entry["history"].append(version)
        self._save_index(index)
        return meta

    def promote(self, relation: str, version: str) -> None:
        """Make ``version`` the champion for ``relation``. The previous champion
        stays on disk and in history, so :meth:`rollback` can restore it."""
        index = self._load_index()
        entry = self._rel_entry(index, relation)
        if version not in entry["history"]:
            raise ValueError(f"unknown version {version} for relation {relation}")
        prev = entry.get("champion")
        entry["champion"] = version
        entry["previous_champion"] = prev
        self._save_index(index)
        # Reflect status in the per-version metadata for human audit.
        self._set_status(relation, version, "champion")
        if prev and prev != version:
            self._set_status(relation, prev, "retired")

    def rollback(self, relation: str) -> Optional[str]:
        """Restore the previous champion (undo the last promotion). Returns the
        version restored, or ``None`` if there is nothing to roll back to."""
        index = self._load_index()
        entry = self._rel_entry(index, relation)
        prev = entry.get("previous_champion")
        if not prev:
            return None
        current = entry.get("champion")
        entry["champion"] = prev
        entry["previous_champion"] = None
        self._save_index(index)
        self._set_status(relation, prev, "champion")
        if current and current != prev:
            self._set_status(relation, current, "retired")
        return prev

    def _set_status(self, relation: str, version: str, status: str) -> None:
        try:
            meta = self.meta(relation, version)
            if meta is None:
                return
            meta.status = status
            with io.open(self._meta_path(relation, version), "w", encoding="utf-8") as fh:
                json.dump(meta.as_dict(), fh, indent=2)
        except Exception:
            pass

    # ── read path ────────────────────────────────────────────────────
    def relations(self) -> list[str]:
        return sorted(self._load_index().keys())

    def champion_version(self, relation: str) -> Optional[str]:
        return self._load_index().get(relation, {}).get("champion")

    def meta(self, relation: str, version: str) -> Optional[HeadMeta]:
        try:
            with io.open(self._meta_path(relation, version), encoding="utf-8") as fh:
                return HeadMeta.from_dict(json.load(fh))
        except Exception:
            return None

    def champion_meta(self, relation: str) -> Optional[HeadMeta]:
        v = self.champion_version(relation)
        return self.meta(relation, v) if v else None

    def load_head(self, relation: str, version: str) -> Optional[NeuralHead]:
        try:
            return NeuralHead.load(self._head_path(relation, version))
        except Exception:
            return None

    def champion(self, relation: str) -> Optional[tuple[NeuralHead, HeadMeta]]:
        """The (head, meta) currently allowed to serve ``relation``, or None."""
        v = self.champion_version(relation)
        if not v:
            return None
        head = self.load_head(relation, v)
        meta = self.meta(relation, v)
        if head is None or meta is None:
            return None
        return head, meta

    def history(self, relation: str) -> list[HeadMeta]:
        out = []
        for v in self._load_index().get(relation, {}).get("history", []):
            m = self.meta(relation, v)
            if m is not None:
                out.append(m)
        return out

    def summary(self) -> dict[str, Any]:
        """A glanceable census: champion + metrics per relation."""
        out: dict[str, Any] = {}
        for rel in self.relations():
            cm = self.champion_meta(rel)
            out[rel] = {
                "champion": cm.version if cm else None,
                "accuracy": round(cm.accuracy, 4) if cm else None,
                "coverage": round(cm.coverage, 4) if cm else None,
                "ready": cm.ready if cm else False,
                "versions": len(self.history(rel)),
            }
        return out


# ── process-wide singleton + env-gated accessor ─────────────────────────────
_REGISTRY: HeadRegistry | None = None


def get_head_registry() -> HeadRegistry | None:
    """Return the process registry iff ``SOWSMITH_HEAD_REGISTRY_DIR`` is set.

    Default-off: no env var → no registry → serve falls through to kNN/LLM, so
    production behavior is unchanged until we opt in.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    root = os.environ.get("SOWSMITH_HEAD_REGISTRY_DIR")
    if not root:
        return None
    try:
        _REGISTRY = HeadRegistry(root)
    except Exception:
        _REGISTRY = None
    return _REGISTRY


def set_head_registry(registry: HeadRegistry | None) -> None:
    """Inject a registry (tests / explicit wiring)."""
    global _REGISTRY
    _REGISTRY = registry


__all__ = [
    "HeadRegistry",
    "HeadMeta",
    "get_head_registry",
    "set_head_registry",
]
