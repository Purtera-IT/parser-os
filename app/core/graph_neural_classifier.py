"""Graph neural edge classifier — a learned relation head over a diffused
deal graph.

`graph_builder.py` decides edges (supports / contradicts / excludes /
requires / located_in / same_as / …) with regex + keyword rules. Those rules
are brittle and customer-specific. This module learns the *same* decision from
geometry instead, so the graph layer becomes neural without ever blindly
dropping or inventing an edge.

Architecture (mirrors :mod:`app.core.neural_head` — frozen features, learned
head, cold-start safe):

  1. **Frozen graph diffusion.** Atoms are nodes; an edge in the *message-
     passing* graph exists between atoms that share an entity key (the same
     inverted index ``graph_builder`` already builds). We propagate the frozen
     atom embeddings over the degree-normalized adjacency for a few hops
     (a GCN with the identity weight — a deterministic "graph echo", NO trained
     propagation weights). Each node ends up carrying a context-aware summary
     of its neighborhood. This is stable on the tens-to-hundreds of atoms a
     deal holds and needs no training.
  2. **Learned relation head.** For a candidate atom pair ``(i, j)`` we build a
     pair feature ``[h_i, h_j, h_i*h_j, |h_i-h_j|]`` and classify it with a
     :class:`~app.core.neural_head.NeuralHead` over the relation labels
     (including the explicit ``no_edge`` class, so the head learns when NOT to
     connect two atoms). The head brings calibration, an OOD/novelty signal
     and the abstain-and-route behaviour for free.

Hard contracts (identical to neural_head / feedback_store):
  * Pure numpy, no torch. Deterministic. Offline (operates on already-computed
    embeddings; never touches the network). Never raises into the hot path —
    every public method degrades to "abstain" on internal error, so the caller
    keeps the deterministic ``graph_builder`` edge as the fallback.

Cold start: with too few labels the head uses an identity projection and
abstains on inference, so a fresh deploy behaves exactly like today's pipeline
(deterministic edges) and sharpens automatically as corrections accrue — the
verify-gate retirement path, never a hard cutover.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.neural_head import NeuralHead, _l2norm

NO_EDGE = "no_edge"

_DIFFUSION_HOPS = 2
_DIFFUSION_ALPHA = 0.5   # self-retention per hop (teleport); 1.0 = no mixing
_SEED = 1729


@dataclass
class EdgePrediction:
    """One relation outcome for an atom pair from the neural graph head.

    Attributes:
        edge_type: predicted relation label, or ``None`` when abstaining.
        confidence: calibrated probability of ``edge_type`` (0..1).
        route_fallback: True → the head is untrained / uncertain / OOD; the
            caller MUST fall back to the deterministic graph_builder decision
            for this pair (never a silent edge gain or loss).
        is_edge: convenience — ``edge_type not in (None, NO_EDGE)``.
        ood: query pair is unlike anything trained on.
        probs: full {label: prob} map for audit.
        trained: True iff a learned projection was engaged.
    """

    edge_type: str | None
    confidence: float = 0.0
    route_fallback: bool = True
    is_edge: bool = False
    ood: bool = False
    probs: dict[str, float] = field(default_factory=dict)
    trained: bool = False


def _atom_entity_keys(atom) -> list[str]:
    keys = getattr(atom, "entity_keys", None)
    if not keys:
        return []
    return [str(k) for k in keys if k]


def build_adjacency(atoms: list) -> np.ndarray:
    """Symmetric degree-normalized adjacency over the entity-key graph.

    Two atoms are linked (for message passing) when they share at least one
    entity key. Returns an (N, N) row-normalized matrix with self-loops and a
    teleport weight ``_DIFFUSION_ALPHA`` on the diagonal. Empty/degenerate
    inputs return an identity-like matrix (no mixing), so diffusion is a no-op.
    """
    n = len(atoms)
    A = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return A
    # inverted index: entity key -> atom indices
    by_key: dict[str, list[int]] = {}
    for i, a in enumerate(atoms):
        for k in _atom_entity_keys(a):
            by_key.setdefault(k, []).append(i)
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                A[idxs[a_i], idxs[b_i]] = 1.0
                A[idxs[b_i], idxs[a_i]] = 1.0
    # degree-normalize the off-diagonal mass, then add teleport self-loop:
    #   row = alpha * e_i + (1-alpha) * normalized_neighbors
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        deg = float(A[i].sum())
        out[i, i] = 1.0
        if deg > 0:
            out[i] = (1.0 - _DIFFUSION_ALPHA) * (A[i] / deg)
            out[i, i] += _DIFFUSION_ALPHA
    return out


def diffuse(node_vecs: np.ndarray, A: np.ndarray, hops: int = _DIFFUSION_HOPS) -> np.ndarray:
    """Propagate frozen node embeddings over ``A`` for ``hops`` and L2-normalize.

    No trained weights — a deterministic GCN-with-identity. Each hop blends a
    node with its entity-key neighborhood, so two atoms that disagree on a
    shared device/site/part will end up with features that reflect that
    contradiction context.
    """
    if node_vecs.ndim != 2 or node_vecs.shape[0] == 0:
        return node_vecs
    H = node_vecs.astype(np.float32)
    for _ in range(max(0, hops)):
        H = A @ H
        H = _l2norm(H)
    return H


def pair_feature(hi: np.ndarray, hj: np.ndarray) -> np.ndarray:
    """Direction-aware pair feature: concat of both endpoints, their Hadamard
    product (agreement) and absolute difference (disagreement)."""
    return np.concatenate([hi, hj, hi * hj, np.abs(hi - hj)]).astype(np.float32)


class GraphNeuralClassifier:
    """Learned, calibrated, cold-start-safe edge-relation classifier.

    Lifecycle: ``fit_graph`` once per deal (diffuses node features and trains
    the relation head from labeled pairs), then ``predict(i, j, candidates)``
    per candidate pair.
    """

    def __init__(self, *, hops: int = _DIFFUSION_HOPS, seed: int = _SEED) -> None:
        self.hops = hops
        self.head = NeuralHead(seed=seed)
        self._H: np.ndarray | None = None
        self.n_nodes = 0
        self.trained = False

    def fit_graph(
        self,
        node_vecs: np.ndarray,
        atoms: list,
        labeled_pairs: list[tuple[int, int, str]],
    ) -> "GraphNeuralClassifier":
        """Diffuse ``node_vecs`` over the entity-key graph of ``atoms`` and fit
        the relation head from ``labeled_pairs`` = ``(i, j, label)`` triples.

        ``label`` is an edge_type string or :data:`NO_EDGE`. Safe on tiny /
        degenerate data: the head stays in its cold-start abstain mode.
        """
        try:
            node_vecs = np.asarray(node_vecs, dtype=np.float32)
            self.n_nodes = node_vecs.shape[0] if node_vecs.ndim == 2 else 0
            if self.n_nodes == 0:
                return self
            A = build_adjacency(atoms)
            self._H = diffuse(node_vecs, A, self.hops)
            if labeled_pairs:
                feats, labels = [], []
                for i, j, lbl in labeled_pairs:
                    if 0 <= i < self.n_nodes and 0 <= j < self.n_nodes:
                        feats.append(pair_feature(self._H[i], self._H[j]))
                        labels.append(str(lbl))
                if feats:
                    self.head.fit(np.vstack(feats), labels)
                    self.trained = self.head.trained
            return self
        except Exception:  # pragma: no cover - never break graph build
            self._H = None
            self.trained = False
            return self

    def predict(self, i: int, j: int, candidates: list[str]) -> EdgePrediction:
        """Predict the relation for the atom pair (i, j) restricted to
        ``candidates`` (the edge types the caller is willing to draw, which
        should include :data:`NO_EDGE`). Abstains → ``route_fallback=True``.
        """
        try:
            if self._H is None or not (0 <= i < self.n_nodes) or not (0 <= j < self.n_nodes):
                return EdgePrediction(edge_type=None, route_fallback=True, trained=self.trained)
            feat = pair_feature(self._H[i], self._H[j]).reshape(1, -1)
            # NeuralHead.classify normalizes internally via its projection; feed
            # the raw pair feature as the "query embedding".
            hd = self.head.classify(feat[0], candidates)
            if hd.verdict is None or hd.route_llm:
                return EdgePrediction(
                    edge_type=None,
                    confidence=hd.confidence,
                    route_fallback=True,
                    ood=hd.ood,
                    probs=hd.probs,
                    trained=hd.trained,
                )
            return EdgePrediction(
                edge_type=hd.verdict,
                confidence=hd.confidence,
                route_fallback=False,
                is_edge=hd.verdict not in (None, NO_EDGE),
                ood=hd.ood,
                probs=hd.probs,
                trained=hd.trained,
            )
        except Exception:  # pragma: no cover - never break graph build
            return EdgePrediction(edge_type=None, route_fallback=True, trained=self.trained)


def neural_edge_gate(
    atoms: list,
    edges: list,
    *,
    embed_fn,
    high_precision_families: set[str],
    ambiguous_families: set[str],
    min_train: int = 6,
    drop_confidence: float = 0.80,
):
    """Self-supervised spurious-edge gate over a single deal's graph.

    The deterministic ``graph_builder`` over-links: noisy entity keys and the
    capped pair generator emit low-signal ``semantic_link`` /
    ``cross_artifact_co_mention`` edges. This pass learns the deal's OWN
    decision boundary and drops only the ambiguous edges it is confident are
    spurious — it NEVER touches a high-precision edge and NEVER adds one.

    Training signal (no external labels — works on a cold deploy):
      * positives: the high-precision edges (exact part-number/quantity
        contradictions, value_support) labelled by their real ``edge_type``;
      * negatives (``no_edge``): atom pairs that share no entity key at all.

    With too little signal (``< min_train`` labels or a cold head) the head
    abstains, so every edge is kept — identical to today's output. Returns
    ``(kept_edges, dropped_count)``.

    ``embed_fn(list[str]) -> (N, D) ndarray`` is injected so this stays offline
    and unit-testable; production passes ``embedding_retrieval.embed_texts``.
    """
    try:
        if not edges or len(atoms) < 3:
            return edges, 0
        ordered = list(atoms)
        idx_of = {getattr(a, "id", i): i for i, a in enumerate(ordered)}
        texts = [str(getattr(a, "normalized_text", "") or getattr(a, "raw_text", "")) for a in ordered]
        node_vecs = embed_fn(texts)
        if node_vecs is None or getattr(node_vecs, "shape", (0,))[0] != len(ordered):
            return edges, 0

        def _fam(e) -> str:
            md = getattr(e, "metadata", {}) or {}
            return str(md.get("edge_family", ""))

        def _et(e) -> str:
            et = getattr(e, "edge_type", None)
            return getattr(et, "value", str(et))

        labeled: list[tuple[int, int, str]] = []
        for e in edges:
            if _fam(e) in high_precision_families:
                i = idx_of.get(getattr(e, "from_atom_id", None))
                j = idx_of.get(getattr(e, "to_atom_id", None))
                if i is not None and j is not None:
                    labeled.append((i, j, _et(e)))

        # negative (no_edge) sampling: pairs with disjoint entity keys
        keysets = [set(_atom_entity_keys(a)) for a in ordered]
        rng = np.random.default_rng(_SEED)
        n = len(ordered)
        tries, want = 0, max(len(labeled), min_train)
        while len([p for p in labeled if p[2] == NO_EDGE]) < want and tries < want * 20:
            tries += 1
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i != j and not (keysets[i] & keysets[j]):
                labeled.append((i, j, NO_EDGE))

        distinct = {lbl for _, _, lbl in labeled}
        if len(labeled) < min_train or len(distinct) < 2:
            return edges, 0

        gnc = GraphNeuralClassifier()
        gnc.fit_graph(node_vecs, ordered, labeled)
        if not gnc.trained:
            return edges, 0

        cands = sorted(distinct)
        kept, dropped = [], 0
        for e in edges:
            if _fam(e) not in ambiguous_families:
                kept.append(e)
                continue
            i = idx_of.get(getattr(e, "from_atom_id", None))
            j = idx_of.get(getattr(e, "to_atom_id", None))
            if i is None or j is None:
                kept.append(e)
                continue
            pred = gnc.predict(i, j, cands)
            if (not pred.route_fallback and pred.edge_type == NO_EDGE
                    and pred.confidence >= drop_confidence):
                dropped += 1
                continue
            kept.append(e)
        return kept, dropped
    except Exception:  # pragma: no cover - never break graph build
        return edges, 0


__all__ = [
    "GraphNeuralClassifier",
    "EdgePrediction",
    "build_adjacency",
    "diffuse",
    "pair_feature",
    "neural_edge_gate",
    "NO_EDGE",
]
