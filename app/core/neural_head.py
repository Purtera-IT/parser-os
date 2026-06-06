"""The neural decision head — a learned metric over frozen embeddings.

The feedback store used to score a query against each correction with **raw
cosine similarity** and a flat ``0.82`` threshold. That has three weaknesses:

1. *Generic geometry.* Cosine in the pretrained embedding space measures
   generic semantic similarity, not *your* decision boundary. "PurTera's
   letterhead" and "the customer's HQ address" sit close in raw space because
   they are both corporate US addresses — exactly the pair we must separate.
2. *Uncalibrated.* ``0.82`` is a magic number. A score of 0.83 vs 0.81 carries
   no probabilistic meaning, so the confident/uncertain line is guesswork.
3. *No novelty signal.* A query unlike anything ever corrected still gets a
   cosine score; the store can't tell "I am sure" from "I have never seen
   anything like this."

This module fixes all three **without fine-tuning the embedder** (so compiles
stay reproducible — the frozen embedder is pinned in the compile signature):

* **#1 Learned projection (metric learning).** A small linear map ``W`` (D→d)
  trained with a prototypical-softmax objective pulls same-verdict exemplars
  together and pushes different verdicts apart. kNN/prototype scoring then
  happens in a space *shaped to the PM's corrections*. At cold start (too few
  labels) ``W`` is identity — byte-identical to today's raw-cosine behavior —
  and it sharpens automatically as corrections accrue.
* **#3 Calibration.** A fitted temperature turns prototype distances into
  honest probabilities, so "confident" is ``P >= p_hi`` (a real probability),
  not a cosine magic number.
* **#4 OOD / uncertainty.** Every query gets a novelty score (similarity to the
  nearest in-distribution prototype) and a margin (top-1 vs top-2 probability).
  Far-from-everything or near-the-boundary → ``route_llm`` (a *genuine* hard
  decision worth an LLM call); confidently-typical → decide or skip the LLM.

Hard contracts (mirror :mod:`app.core.feedback_store`):

* **Pure numpy, no torch.** Trains in milliseconds on the tens-to-hundreds of
  exemplars a store holds. No new heavy dependency, no GPU.
* **Deterministic.** Seeded init + full-batch gradient descent → identical
  weights for identical inputs, so a compile is reproducible.
* **Offline.** Operates only on already-computed embedding matrices. The single
  network call (embedding) is owned by the caller; this module never touches
  the network.
* **Never raises into the hot path.** ``classify`` degrades to a safe abstain
  on any internal error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── routing defaults (probabilities, not cosine magic numbers) ────────
# A decision is CONFIDENT (store decides, no LLM) only when the calibrated
# probability clears p_hi AND the top-1/top-2 margin is decisive AND the query
# is in-distribution. Anything uncertain or novel becomes ROUTE_LLM — the
# genuine hard decisions we actually want the model spending time on.
_P_HI = 0.80          # min calibrated prob to decide without the LLM
_MARGIN = 0.25        # min (p_top1 - p_top2) to call it decisive
_OOD_SIM = 0.55       # min cosine (projected) to the nearest prototype to be
                      # considered in-distribution at all
_MIN_PER_CLASS = 3    # exemplars/class needed before the projection is learned
_PROJ_DIM = 128       # projected dimensionality
_EPOCHS = 250
_LR = 0.5
_L2 = 1e-3
_SEED = 1729


@dataclass
class HeadDecision:
    """One classification outcome from the head.

    Attributes:
        verdict: best-scoring class label, or ``None`` when the head abstains.
        confidence: calibrated probability of ``verdict`` (0..1).
        route_llm: True → uncertain or out-of-distribution; the caller SHOULD
            ask the LLM (this is a genuine hard decision). False → the head is
            confident, so the LLM must NOT be called for this item.
        ood: True → the query is unlike anything in the training set.
        margin: p_top1 - p_top2 (decisiveness).
        nearest_sim: cosine to the nearest prototype in the projected space.
        probs: full {label: prob} map (audit / few-shot trace).
        trained: True iff a learned projection was used (vs cold-start identity).
    """

    verdict: str | None
    confidence: float = 0.0
    route_llm: bool = True
    ood: bool = False
    margin: float = 0.0
    nearest_sim: float = 0.0
    probs: dict[str, float] = field(default_factory=dict)
    trained: bool = False


def _l2norm(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize an (N, D) matrix; zero rows stay zero."""
    if m.ndim == 1:
        n = float(np.linalg.norm(m))
        return m / n if n > 1e-9 else m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms > 1e-9, norms, 1.0)


class NeuralHead:
    """A learned, calibrated, OOD-aware classifier over frozen embeddings.

    Fit once per (relation, training-data signature); cached by the store.
    """

    def __init__(
        self,
        *,
        proj_dim: int = _PROJ_DIM,
        p_hi: float = _P_HI,
        margin: float = _MARGIN,
        ood_sim: float = _OOD_SIM,
        min_per_class: int = _MIN_PER_CLASS,
        seed: int = _SEED,
    ) -> None:
        self.proj_dim = proj_dim
        self.p_hi = p_hi
        self.margin = margin
        self.ood_sim = ood_sim
        self.min_per_class = min_per_class
        self.seed = seed

        self.classes_: list[str] = []
        self._W: np.ndarray | None = None        # (D, d) projection; None = identity
        self._protos: np.ndarray | None = None    # (K, d) normalized class prototypes
        self._temp: float = 1.0                    # calibration temperature
        self._radius: float = self.ood_sim         # learned in-distribution floor
        self.trained: bool = False                 # learned projection engaged?
        self.n_train: int = 0

    # ── geometry ─────────────────────────────────────────────────────
    def _project(self, X: np.ndarray) -> np.ndarray:
        """Map raw embeddings into the learned (or identity) metric space,
        then L2-normalize so dot product = cosine."""
        Z = X if self._W is None else X @ self._W
        return _l2norm(Z)

    # ── training ─────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: list[str],
        sample_weight: np.ndarray | list[float] | None = None,
    ) -> "NeuralHead":
        """Fit the head from L2-normalized embeddings ``X`` (N, D) and labels
        ``y`` (length N). Safe on tiny/degenerate data — falls back to an
        identity projection (today's raw-cosine behavior) rather than failing.

        ``sample_weight`` (length N) lets gold (PM) rows dominate silver (LLM)
        rows so a single human correction is not drowned by abundant teacher
        labels that agree with the old mistake. Weights are normalized to mean
        1 internally; ``None`` means uniform (every row weight 1). They shape
        the prototypes, the metric-learning gradient, and the calibration
        temperature — i.e. every place the data votes.
        """
        try:
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != len(y):
                return self
            w = self._norm_weights(sample_weight, len(y))
            self.n_train = X.shape[0]
            self.classes_ = sorted(set(y))
            counts = {c: y.count(c) for c in self.classes_}

            # Learn a projection only with enough signal: >=2 classes and
            # >=min_per_class exemplars each. Otherwise identity (cold start).
            learnable = (
                len(self.classes_) >= 2
                and min(counts.values()) >= self.min_per_class
            )
            if learnable:
                self._W = self._learn_projection(X, y, w)
                self.trained = True
            else:
                self._W = None
                self.trained = False

            Z = self._project(X)
            self._protos = self._class_prototypes(Z, y, w)     # (K, d)
            self._temp = self._fit_temperature(Z, y, w)        # calibration
            self._radius = self._fit_radius(Z, y)              # OOD floor
            return self
        except Exception:  # pragma: no cover - never break the store
            self._W = None
            self._protos = None
            self.trained = False
            return self

    @staticmethod
    def _norm_weights(sample_weight, n: int) -> np.ndarray:
        """Coerce optional weights to a length-n float vector with mean 1.

        Mean-1 normalization keeps the effective learning rate / gradient scale
        identical to the uniform case, so adding weights never silently changes
        step sizes — it only changes the *relative* pull of each row.
        """
        if sample_weight is None:
            return np.ones(n, dtype=np.float32)
        w = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if w.shape[0] != n:
            return np.ones(n, dtype=np.float32)
        w = np.clip(w, 1e-6, None)
        mean = float(w.mean()) or 1.0
        return (w / mean).astype(np.float32)

    def _class_prototypes(self, Z: np.ndarray, y: list[str],
                          w: np.ndarray | None = None) -> np.ndarray:
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        protos = []
        for c in self.classes_:
            idx = [i for i, yi in enumerate(y) if yi == c]
            wi = w[idx].reshape(-1, 1)
            # Weighted class mean: gold exemplars pull the prototype harder.
            wsum = float(wi.sum()) or 1.0
            protos.append(_l2norm((Z[idx] * wi).sum(axis=0) / wsum))
        return np.vstack(protos).astype(np.float32)

    def _learn_projection(self, X: np.ndarray, y: list[str],
                          w: np.ndarray | None = None) -> np.ndarray:
        """Prototypical-softmax metric learning.

        Minimize the weighted NLL of the true class under logits =
        (Zi . proto_k)/T over a linear projection ``W``. Prototypes are
        recomputed each epoch from the current projection but treated as
        constants in the gradient (stop-gradient) — the standard, numerically
        stable prototypical-network trick. Full-batch GD with momentum + L2;
        seeded for determinism. ``w`` weights each row's loss (mean-1), so the
        learned metric bends toward separating the cases PMs actually corrected.
        """
        rng = np.random.default_rng(self.seed)
        D = X.shape[1]
        d = min(self.proj_dim, D)
        # Small random init scaled by 1/sqrt(D) keeps activations ~unit.
        W = rng.standard_normal((D, d)).astype(np.float32) / np.sqrt(D)
        y_idx = np.array([self.classes_.index(c) for c in y])
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        wcol = w.reshape(-1, 1)
        K = len(self.classes_)
        T = 0.1  # sharp during training; calibration temperature fit separately
        vel = np.zeros_like(W)
        mom = 0.9
        for _ in range(_EPOCHS):
            Z = _l2norm(X @ W)                                  # (N, d)
            protos = []
            for k in range(K):
                mask = y_idx == k
                wk = w[mask].reshape(-1, 1)
                wks = float(wk.sum()) or 1.0
                protos.append(_l2norm((Z[mask] * wk).sum(axis=0) / wks))
            P = np.vstack(protos)                               # (K, d)
            logits = (Z @ P.T) / T                              # (N, K)
            logits -= logits.max(axis=1, keepdims=True)
            ex = np.exp(logits)
            probs = ex / ex.sum(axis=1, keepdims=True)          # (N, K)
            # dL/dlogit = p - onehot ; dlogit/dZ = P/T  (stop-grad through P)
            g = probs.copy()
            g[np.arange(len(y_idx)), y_idx] -= 1.0              # (N, K)
            g = g * wcol                                        # weight each row
            dZ = (g @ P) / T                                    # (N, d)
            gradW = X.T @ dZ / X.shape[0] + _L2 * W             # (D, d)
            vel = mom * vel - _LR * gradW
            W = W + vel
        return W.astype(np.float32)

    def _fit_temperature(self, Z: np.ndarray, y: list[str],
                         w: np.ndarray | None = None) -> float:
        """Pick the temperature that minimizes the weighted NLL of the true
        class under the prototype-softmax — turns distances into honest
        probabilities, with gold rows counting more toward calibration."""
        if self._protos is None:
            return 1.0
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        y_idx = np.array([self.classes_.index(c) for c in y])
        sims = Z @ self._protos.T                               # (N, K)
        wsum = float(w.sum()) or 1.0
        best_t, best_nll = 1.0, float("inf")
        for t in (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0):
            logits = sims / t
            logits -= logits.max(axis=1, keepdims=True)
            ex = np.exp(logits)
            probs = ex / ex.sum(axis=1, keepdims=True)
            p_true = probs[np.arange(len(y_idx)), y_idx]
            nll = float((-np.log(np.clip(p_true, 1e-9, 1.0)) * w).sum() / wsum)
            if nll < best_nll:
                best_nll, best_t = nll, t
        return best_t

    def _fit_radius(self, Z: np.ndarray, y: list[str]) -> float:
        """In-distribution floor: the 5th-percentile of each training point's
        cosine to its OWN class prototype. A query whose nearest-prototype
        cosine falls below this is genuinely novel (OOD)."""
        if self._protos is None:
            return self.ood_sim
        y_idx = np.array([self.classes_.index(c) for c in y])
        own = (Z * self._protos[y_idx]).sum(axis=1)             # (N,)
        floor = float(np.percentile(own, 5)) if own.size else self.ood_sim
        # Never trust a floor looser than the global OOD guard.
        return max(self.ood_sim, min(floor, 0.99))

    # ── inference ────────────────────────────────────────────────────
    def classify(self, query_vec: np.ndarray, candidates: list[str]) -> HeadDecision:
        """Classify one L2-normalized query embedding, restricted to the
        caller's ``candidates`` (the closed verdict set for this decision).

        Routing:
          * confident (P>=p_hi, margin ok, in-distribution) → decide, no LLM.
          * uncertain (margin small) or OOD (novel) → route_llm=True.
          * always restricted to candidates; a best class outside the candidate
            set means abstain + route_llm (we don't force an off-menu verdict).
        """
        try:
            # The head only speaks when it has a real decision boundary: >=2
            # learned classes to contrast. A single-class correction (e.g. the
            # PurTera self-address rule) has no contrast here — the store keeps
            # deciding it on its own calibrated-cosine path, behavior unchanged.
            if self._protos is None or len(self.classes_) < 2 or not candidates:
                return HeadDecision(verdict=None, route_llm=True, trained=self.trained)
            q = self._project(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
            sims = self._protos @ q                              # (K,)

            # Calibrated probabilities over ALL trained classes.
            logits = sims / max(self._temp, 1e-6)
            logits -= logits.max()
            ex = np.exp(logits)
            probs_all = ex / ex.sum()
            prob_map = {c: float(probs_all[i]) for i, c in enumerate(self.classes_)}

            # Novelty: nearest prototype across all classes.
            nearest_sim = float(sims.max())
            ood = nearest_sim < self._radius

            # Restrict to candidates the caller will accept.
            cand_in = [c for c in candidates if c in prob_map]
            if not cand_in:
                return HeadDecision(
                    verdict=None, route_llm=True, ood=ood,
                    nearest_sim=nearest_sim, probs=prob_map, trained=self.trained,
                )
            ranked = sorted(cand_in, key=lambda c: prob_map[c], reverse=True)
            top = ranked[0]
            p1 = prob_map[top]
            p2 = prob_map[ranked[1]] if len(ranked) > 1 else 0.0
            margin = p1 - p2

            # The best learned class overall might NOT be a candidate (e.g. the
            # query looks like a third role we've learned). If so, the candidate
            # decision is unreliable → hand to the LLM.
            best_overall = self.classes_[int(np.argmax(probs_all))]
            best_is_candidate = best_overall in candidates

            confident = (
                not ood
                and best_is_candidate
                and p1 >= self.p_hi
                and margin >= self.margin
            )
            return HeadDecision(
                verdict=top if confident else None,
                confidence=p1,
                route_llm=not confident,
                ood=ood,
                margin=margin,
                nearest_sim=nearest_sim,
                probs=prob_map,
                trained=self.trained,
            )
        except Exception:  # pragma: no cover - never break the store
            return HeadDecision(verdict=None, route_llm=True, trained=self.trained)

    # ── persistence (for the model registry, #72) ────────────────────
    # A fitted head is just a few small numpy arrays + scalars. Persisting it
    # lets a registry keep a versioned champion per relation and load it at
    # serve time without re-fitting. Vectors live in the *projected* metric
    # space the head learned; we never persist the raw embedder (it stays
    # pinned in the compile signature), so a saved head is only valid for the
    # embedding model it was trained on — recorded in the registry metadata.
    def to_state(self) -> dict[str, Any]:
        """Serialize the fitted head to a plain dict (numpy arrays + scalars)."""
        return {
            "version": 1,
            "proj_dim": int(self.proj_dim),
            "p_hi": float(self.p_hi),
            "margin": float(self.margin),
            "ood_sim": float(self.ood_sim),
            "min_per_class": int(self.min_per_class),
            "seed": int(self.seed),
            "classes_": list(self.classes_),
            "W": None if self._W is None else np.asarray(self._W, dtype=np.float32),
            "protos": None if self._protos is None else np.asarray(self._protos, dtype=np.float32),
            "temp": float(self._temp),
            "radius": float(self._radius),
            "trained": bool(self.trained),
            "n_train": int(self.n_train),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "NeuralHead":
        """Rebuild a fitted head from :meth:`to_state` output."""
        head = cls(
            proj_dim=int(state.get("proj_dim", _PROJ_DIM)),
            p_hi=float(state.get("p_hi", _P_HI)),
            margin=float(state.get("margin", _MARGIN)),
            ood_sim=float(state.get("ood_sim", _OOD_SIM)),
            min_per_class=int(state.get("min_per_class", _MIN_PER_CLASS)),
            seed=int(state.get("seed", _SEED)),
        )
        head.classes_ = list(state.get("classes_", []))
        W = state.get("W")
        head._W = None if W is None else np.asarray(W, dtype=np.float32)
        protos = state.get("protos")
        head._protos = None if protos is None else np.asarray(protos, dtype=np.float32)
        head._temp = float(state.get("temp", 1.0))
        head._radius = float(state.get("radius", head.ood_sim))
        head.trained = bool(state.get("trained", False))
        head.n_train = int(state.get("n_train", 0))
        return head

    def save(self, path: str) -> None:
        """Persist the fitted head to a single ``.npz`` file (arrays + json meta)."""
        import io as _io
        import json as _json
        state = self.to_state()
        meta = {k: v for k, v in state.items() if k not in ("W", "protos")}
        arrays: dict[str, np.ndarray] = {"__meta__": np.frombuffer(
            _json.dumps(meta).encode("utf-8"), dtype=np.uint8)}
        if state["W"] is not None:
            arrays["W"] = state["W"]
        if state["protos"] is not None:
            arrays["protos"] = state["protos"]
        with _io.open(path, "wb") as fh:
            np.savez(fh, **arrays)

    @classmethod
    def load(cls, path: str) -> "NeuralHead":
        """Load a head saved by :meth:`save`."""
        import json as _json
        with np.load(path, allow_pickle=False) as z:
            meta = _json.loads(bytes(z["__meta__"].tobytes()).decode("utf-8"))
            meta["W"] = z["W"] if "W" in z.files else None
            meta["protos"] = z["protos"] if "protos" in z.files else None
        return cls.from_state(meta)


__all__ = ["NeuralHead", "HeadDecision"]
