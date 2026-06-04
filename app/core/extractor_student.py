"""Extractor student — the trained-head replacement for the LLM, day one.

This is the *student* in the teacher→student distillation (see
:mod:`app.core.training_log`). The LLM is the teacher: slow, occasionally
hallucinating, but able to read novel prose. The student is fast, local, and —
critically — **cannot hallucinate**, because it is *discriminative over
retrieved evidence*: it only ever returns a label by voting over labeled
examples that already exist in the training log. There is no generation step,
so there is nothing to invent.

Two-speed learning, one object:

* **kNN memory (this module).** The moment a row lands in the training log it
  is usable. A PM correction is live on the very next compile — no retrain.
  This is the instant-memory speed.
* **Trained head (later, #70/#71).** A small contrastive head learns the
  *generalizing* boundary from the same rows. Slower to update, better at
  unseen phrasings. It will front this kNN path when it provably wins on the
  held-out split — never before.

**Generalization, not memorization** (the binding constraint). The student
scores the **delexicalized** text (:mod:`app.core.delexicalize`): proper nouns
are already masked to role placeholders before embedding, so the vote is over
the *rule shape*, never the identity. Swap "PurTera" for "Acme" and the masked
text — and therefore the prediction — is identical. The student literally
cannot learn "the thing called PurTera"; it can only learn "the thing in the
self-org role."

**Guess-free.** Below the confidence threshold the student :meth:`abstains
<Prediction.abstained>`. Abstention is not failure — it is the student
correctly saying "defer to the teacher / keep + flag." Precedence is unchanged:
store/student → LLM → UNDECIDED. The student never emits a wrong label to fill
a gap.

No network is required for the *logic*: inject ``embed_fn`` for deterministic
tests, exactly as :class:`app.core.feedback_store.FeedbackStore` does. When the
real embedding endpoint is unreachable the student abstains everywhere (safe
no-op), so wiring it into the compile path can never break a compile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.delexicalize import delexicalize
from app.core.training_log import TrainingLog, TrainingRow

# Default kNN settings. The threshold is deliberately high: the student should
# only speak when it is genuinely confident, because the alternative (abstain →
# LLM) is correct, just slower. Recall is bought back over time as the log
# grows, never by lowering the bar and guessing.
_DEFAULT_K = 9
_DEFAULT_THRESHOLD = 0.78


@dataclass
class Neighbor:
    """One retrieved training row that contributed to a vote (audit trail)."""

    label: str
    similarity: float
    teacher: str
    weight: float
    masked_text: str


@dataclass
class Prediction:
    """The student's answer for one (text, relation) query.

    ``abstained`` is the load-bearing field: when True the caller MUST fall
    through to the teacher (LLM) or keep+flag. ``label`` is meaningless then.
    """

    relation: str
    label: str = ""
    confidence: float = 0.0
    abstained: bool = True
    reason: str = ""
    neighbors: list[Neighbor] = field(default_factory=list)


class ExtractionStudent:
    """kNN-over-the-training-log span classifier. The day-one trained head.

    Args:
        log: the :class:`TrainingLog` to learn from.
        embed_fn: ``list[str] -> (N, D)`` L2-normalized matrix. Defaults to the
            pipeline embedder. Injected in tests for determinism.
        reachable_fn: ``() -> bool`` endpoint probe; when False the student
            abstains everywhere (safe no-op). Defaults to the pipeline probe.
        fit_split: which split to learn from. ``"train"`` for honest
            leave-one-deal-out eval (so holdout deals are never in memory);
            ``None`` (default) uses every row, which is what production wants.
        k: neighbours per vote. threshold: minimum confidence to not abstain.
    """

    def __init__(
        self,
        log: TrainingLog,
        *,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        reachable_fn: Callable[[], bool] | None = None,
        fit_split: str | None = None,
        k: int = _DEFAULT_K,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._log = log
        self._embed_fn = embed_fn
        self._reachable_fn = reachable_fn
        self._fit_split = fit_split
        self._k = max(1, int(k))
        self._threshold = float(threshold)
        # Per-relation fitted memory: (labels, weights, matrix). Lazily built,
        # cached until invalidated. Vectors are recomputed from masked_text and
        # never persisted — swapping the embed model can't strand stale memory.
        self._memory: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

    # ── embedding plumbing (mirrors FeedbackStore) ──────────────────────
    def _reachable(self) -> bool:
        if self._reachable_fn is not None:
            try:
                return bool(self._reachable_fn())
            except Exception:
                return False
        try:
            from app.core.embedding_retrieval import embedding_endpoint_reachable
            return embedding_endpoint_reachable()
        except Exception:
            return False

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return None
        try:
            if self._embed_fn is not None:
                mat = self._embed_fn(texts)
            else:
                from app.core.embedding_retrieval import embed_texts
                mat = embed_texts(texts)
            arr = np.asarray(mat, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != len(texts):
                return None
            return arr
        except Exception:
            return None

    # ── fit (lazy, cached) ──────────────────────────────────────────────
    def invalidate(self, relation: str | None = None) -> None:
        """Drop cached memory so the next query re-reads the log."""
        if relation is None:
            self._memory.clear()
        else:
            self._memory.pop(relation, None)

    def _memory_for(self, relation: str) -> tuple[list[str], np.ndarray, np.ndarray] | None:
        cached = self._memory.get(relation)
        if cached is not None:
            return cached
        rows: list[TrainingRow] = self._log.rows(
            relation=relation, split=self._fit_split
        )
        # Use masked_text (generalization). Fall back to delexicalizing raw_text
        # if an old row predates the autofill. Skip empty-feature rows.
        feats: list[str] = []
        labels: list[str] = []
        weights: list[float] = []
        for r in rows:
            mt = r.masked_text
            if not mt and r.raw_text:
                mt = delexicalize(
                    r.raw_text,
                    (r.provenance or {}).get("role_map") if isinstance(r.provenance, dict) else None,
                ).masked
            if not mt or not r.label:
                continue
            feats.append(mt)
            labels.append(r.label)
            weights.append(float(r.weight) or 1.0)
        if not feats:
            return None
        mat = self._embed(feats)
        if mat is None:
            return None
        built = (labels, np.asarray(weights, dtype=np.float32), mat)
        self._memory[relation] = built
        return built

    # ── predict ─────────────────────────────────────────────────────────
    def classify(
        self,
        text: str,
        relation: str,
        *,
        role_map: dict[str, str] | None = None,
        candidates: list[str] | None = None,
    ) -> Prediction:
        """Vote a label for ``text`` under ``relation``, or abstain.

        ``role_map`` is applied before embedding so the query is masked the
        same way the training rows were. ``candidates`` optionally restricts the
        allowed labels (the decide() contract: never return a verdict the caller
        can't use) — neighbours with other labels are ignored.
        """
        if not text or not text.strip():
            return Prediction(relation=relation, reason="empty_text")
        if not self._reachable():
            return Prediction(relation=relation, reason="embedder_unreachable")

        mem = self._memory_for(relation)
        if mem is None:
            return Prediction(relation=relation, reason="no_training_rows")
        labels, weights, mat = mem

        masked = delexicalize(text, role_map).masked
        qv = self._embed([masked])
        if qv is None:
            return Prediction(relation=relation, reason="embed_failed")
        sims = mat @ qv[0]  # both L2-normalized → cosine

        # Top-k by similarity, then weighted vote per label.
        order = np.argsort(-sims)
        allowed = set(candidates) if candidates else None
        votes: dict[str, float] = {}
        sim_by_label: dict[str, float] = {}
        used: list[Neighbor] = []
        take = 0
        for idx in order:
            if take >= self._k:
                break
            lbl = labels[idx]
            if allowed is not None and lbl not in allowed:
                continue
            sim = float(sims[idx])
            if sim <= 0.0:
                continue
            w = float(weights[idx]) * sim  # teacher-weighted, similarity-scaled
            votes[lbl] = votes.get(lbl, 0.0) + w
            sim_by_label[lbl] = max(sim_by_label.get(lbl, 0.0), sim)
            used.append(Neighbor(label=lbl, similarity=sim,
                                 teacher="", weight=float(weights[idx]),
                                 masked_text=""))
            take += 1

        if not votes:
            return Prediction(relation=relation, reason="no_neighbors",
                              neighbors=used)

        best_label = max(votes, key=lambda k: votes[k])
        total = sum(votes.values()) or 1.0
        share = votes[best_label] / total          # margin over rival labels
        peak_sim = sim_by_label[best_label]          # closeness of best evidence
        # Confidence is the product: the student must be BOTH close to known
        # evidence AND have that evidence agree. Either being weak → abstain.
        confidence = float(share * peak_sim)

        if confidence < self._threshold:
            return Prediction(relation=relation, label=best_label,
                              confidence=confidence, abstained=True,
                              reason="below_threshold", neighbors=used)
        return Prediction(relation=relation, label=best_label,
                          confidence=confidence, abstained=False,
                          reason="knn_vote", neighbors=used)
