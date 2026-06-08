"""Runtime kNN over the supervised-contrastive type space — Layer 2 of the
cascade, the part the worker actually runs.

The GPU trainer (runpod_detector/train_contrastive_encoder_gpu.py) re-sorts the
embedding space around the type DECISION and ships two artifacts:
  - the fine-tuned encoder (sentence-transformers save dir), and
  - a labeled store (store.npz: emb, y, text) + knn_meta.json.
This module loads them and answers the decision by nearest-neighbor vote. That
buys two properties a classifier head cannot:
  * INSTANT-LEARNING — a PM correction is ``append(text, label)``; it influences
    the very next atom, no retrain.
  * GUESS-FREE — abstains below the operating threshold and on out-of-distribution
    atoms (top-1 neighbor too far), so the caller falls back to the LLM. Skip
    rather than emit a wrong label.

Classes depend on how the encoder was trained (knn_meta 'mode'):
  unified -> _keep + 7 facets ;  gate -> _keep/typed ;  facet -> 7 facets.

EMBEDDING IS PLUGGABLE. By default it lazy-loads the saved sentence-transformers
encoder (the bge build). For the qwen3-embedding-LoRA-via-vLLM path, pass an
``embed_fn`` that hits the vLLM endpoint — the kNN math here is identical, so the
runtime doesn't change when the encoder does.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_KEEP = "_keep"
_DEFAULT_K = 15
_DEFAULT_SIM_FLOOR = 0.55
_DEFAULT_TAU = 0.30        # vote-margin operating point if knn_meta omits one


def _registry_dir() -> str:
    return os.environ.get("SOWSMITH_CONTRASTIVE_TYPE_DIR", "_contrastive_type")


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


@dataclass
class ContrastiveTypeKNN:
    emb: np.ndarray                       # (N, D) L2-normalized store vectors
    y: np.ndarray                         # (N,) store labels
    k: int = _DEFAULT_K
    sim_floor: float = _DEFAULT_SIM_FLOOR
    tau: float = _DEFAULT_TAU
    prior_alpha: float = 0.5      # class-prior debias in vote (0=raw kNN, 1=balanced)
    mode: str = "unified"
    text: np.ndarray | None = None        # (N,) store texts (for inspection/dedup)
    embed_fn: Callable[[list[str]], np.ndarray] | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── inference ──────────────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if self.embed_fn is None:
            return None
        try:
            return _norm(np.asarray(self.embed_fn(texts), dtype=np.float32))
        except Exception:
            return None

    def _vote(self, q: np.ndarray) -> tuple[str, float]:
        """q: (D,) normalized. Returns (label, confidence). confidence = distance-
        weighted vote margin, hard-gated to 0 when the nearest neighbor is beyond
        the OOD similarity floor."""
        sims = self.emb @ q                       # (N,) cosine
        kk = min(self.k, sims.shape[0])
        nb = np.argpartition(-sims, kk - 1)[:kk]
        top1 = float(sims[nb].max())
        votes: dict[str, float] = {}
        for j in nb:
            votes[self.y[j]] = votes.get(self.y[j], 0.0) + max(float(sims[j]), 0.0)
        if self.prior_alpha:
            import collections as _c
            counts = _c.Counter(self.y.tolist())
            votes = {c: v / (counts.get(c, 1) ** self.prior_alpha) for c, v in votes.items()}
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        total = sum(votes.values()) + 1e-9
        win = ranked[0][0]
        margin = (ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)) / total
        conf = margin if top1 >= self.sim_floor else 0.0
        return win, float(conf)

    def classify(self, text: str) -> tuple[str, float] | None:
        """Return (label, confidence), or None to abstain (caller -> LLM).
        Guess-free: abstains below the operating tau or on OOD."""
        qs = self._embed([text])
        if qs is None:
            return None
        label, conf = self._vote(qs[0])
        return (label, conf) if conf >= self.tau else None

    def classify_batch(self, texts: list[str]) -> list[tuple[str, float] | None]:
        qs = self._embed(texts)
        if qs is None:
            return [None] * len(texts)
        out: list[tuple[str, float] | None] = []
        for i in range(len(texts)):
            label, conf = self._vote(qs[i])
            out.append((label, conf) if conf >= self.tau else None)
        return out

    # ── instant-learning ────────────────────────────────────────────────────────
    def append(self, text: str, label: str, *, persist: bool = True) -> bool:
        """Add a (text, label) exemplar to the store so it influences the NEXT
        atom — no retrain. Used by the PM-correction loop. Thread-safe."""
        v = self._embed([text])
        if v is None:
            return False
        with self._lock:
            self.emb = np.vstack([self.emb, v])
            self.y = np.append(self.y, label)
            if self.text is not None:
                self.text = np.append(self.text, text)
            if persist:
                try:
                    self._save()
                except Exception:
                    return True  # in-memory append still took effect
        return True

    def _save(self) -> None:
        path = os.path.join(_registry_dir(), "store.npz")
        os.makedirs(_registry_dir(), exist_ok=True)
        kw = {"emb": self.emb, "y": self.y}
        if self.text is not None:
            kw["text"] = self.text
        np.savez_compressed(path, **kw)


def _encoder_embed_fn():
    """Default embed_fn: lazy-load the saved sentence-transformers encoder once."""
    enc_dir = os.path.join(_registry_dir(), "best")
    if not os.path.isdir(enc_dir):
        enc_dir = _registry_dir()
    holder: dict[str, Any] = {}

    def _fn(texts: list[str]) -> np.ndarray:
        model = holder.get("m")
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(enc_dir)
            holder["m"] = model
        return model.encode(texts, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)

    return _fn


def load_promoted(embed_fn: Callable[[list[str]], np.ndarray] | None = None) -> ContrastiveTypeKNN | None:
    """Load the promoted contrastive store + meta from the registry, or None.
    Pass ``embed_fn`` to override the embedder (e.g. a vLLM-backed qwen3-LoRA
    endpoint); default loads the saved sentence-transformers encoder."""
    reg = _registry_dir()
    store_p = os.path.join(reg, "store.npz")
    if not os.path.exists(store_p):
        return None
    try:
        z = np.load(store_p, allow_pickle=True)
        emb = _norm(np.asarray(z["emb"], dtype=np.float32))
        y = np.asarray(z["y"])
        text = np.asarray(z["text"]) if "text" in z.files else None
    except Exception:
        return None

    meta: dict[str, Any] = {}
    for cand in (os.path.join(reg, "best", "knn_meta.json"), os.path.join(reg, "knn_meta.json")):
        if os.path.exists(cand):
            try:
                meta = json.load(open(cand, encoding="utf-8"))
                break
            except Exception:
                pass

    if embed_fn is None:
        embed_fn = _encoder_embed_fn()
    tau = meta.get("operating_tau")
    return ContrastiveTypeKNN(
        emb=emb, y=y, text=text, embed_fn=embed_fn,
        k=int(meta.get("k", _DEFAULT_K)),
        sim_floor=float(meta.get("sim_floor", _DEFAULT_SIM_FLOOR)),
        tau=float(tau) if tau is not None else _DEFAULT_TAU,
        mode=str(meta.get("mode", "unified")),
    )
