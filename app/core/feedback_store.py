"""The feedback store — where a PM's correction lives forever.

A correction is the durable memory of a judgment a human fixed: "PurTera is our
company, never list its address as a job site." It is NOT a keyword rule
(``if "PurTera" in text``) — those never generalize to the next deal's phrasing.
It is an **embedding prototype** plus a **relation verdict**, so it fires on any
text semantically close to what the PM corrected, including paraphrases, a
renamed entity, or PurTera's next office.

Why this gives learning-system power without 10,000 examples: the store rides a
*pre-trained* embedding model (qwen3-embedding:8b, 4096-dim, already in the
pipeline). Similarity is semantic, not lexical, so one exemplar generalizes.
~30-60 recurring patterns cover the deal types; they accrue over the first
20-30 deals. No fine-tuning, no LoRA on the critical path.

Design contract (mirrors :mod:`app.core.semantic_role`):

* **Offline-safe.** If the embedding endpoint is unreachable, ``resolve``
  returns ``None`` — never a guess. ``decide()`` then falls to the LLM/fallback.
* **Never raises.** Any internal error yields ``None`` (undecided).
* **Scoped.** Corrections live at ``global`` | ``pack:<domain>`` | ``deal:<id>``;
  resolution is narrowest-first so a deal override beats a global rule for that
  deal only — the cross-domain expansion story.
* **Inspectable / reversible.** Every correction is a SQLite row a PM can read,
  disable, or supersede; every decision it drives cites its ``id``.
* **Deterministic in tests.** The embedder and reachability probe are injected,
  so tests exercise resolution without the network.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.decide import Decision, DecisionScope
from app.core.neural_head import NeuralHead

# Scope tiers, narrowest first. A deal correction overrides a pack correction,
# which overrides global.
SCOPE_DEAL = "deal"
SCOPE_PACK = "pack"
SCOPE_GLOBAL = "global"

_DEFAULT_THRESHOLD = 0.82  # cosine; per-correction tunable (Phase 5 calibrates)


@dataclass
class Correction:
    """One learned judgment. See module docstring."""

    id: str
    relation: str  # the decide() relation this governs, e.g. "physical_site"
    verdict: str  # the verdict to return on a hit; MUST be a caller candidate
    scope: str = SCOPE_GLOBAL  # "global" | "pack" | "deal"
    scope_key: str = ""  # pack id or deal id when scope != global
    exemplars: list[str] = field(default_factory=list)
    threshold: float = _DEFAULT_THRESHOLD
    relations: dict = field(default_factory=dict)  # structured grounding
    instruction: str = ""
    complaint_id: str | None = None
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "active"  # "active" | "disabled" | "superseded"
    supersedes: str | None = None
    confidence_floor: float = 0.0
    hit_count: int = 0
    last_fired: float | None = None
    wrongful_override_count: int = 0

    # ── SQLite (de)serialization ──────────────────────────────────────
    def to_row(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["exemplars"] = json.dumps(self.exemplars)
        d["relations"] = json.dumps(self.relations)
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Correction":
        d = dict(row)
        d["exemplars"] = json.loads(d.get("exemplars") or "[]")
        d["relations"] = json.loads(d.get("relations") or "{}")
        return cls(**d)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    relation TEXT NOT NULL,
    verdict TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    scope_key TEXT NOT NULL DEFAULT '',
    exemplars TEXT NOT NULL DEFAULT '[]',
    threshold REAL NOT NULL DEFAULT 0.82,
    relations TEXT NOT NULL DEFAULT '{}',
    instruction TEXT NOT NULL DEFAULT '',
    complaint_id TEXT,
    created_by TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    supersedes TEXT,
    confidence_floor REAL NOT NULL DEFAULT 0,
    hit_count INTEGER NOT NULL DEFAULT 0,
    last_fired REAL,
    wrongful_override_count INTEGER NOT NULL DEFAULT 0
);
"""

_COLUMNS = [
    "id", "relation", "verdict", "scope", "scope_key", "exemplars",
    "threshold", "relations", "instruction", "complaint_id", "created_by",
    "created_at", "updated_at", "status", "supersedes", "confidence_floor",
    "hit_count", "last_fired", "wrongful_override_count",
]


class FeedbackStore:
    """Embedding-backed correction store implementing the decide() contract.

    Args:
        db_path: SQLite file, or ``":memory:"`` for an ephemeral store.
        embed_fn: ``list[str] -> (N, D)`` L2-normalized matrix. Defaults to the
            pipeline's :func:`app.core.embedding_retrieval.embed_texts`. Injected
            in tests for determinism.
        reachable_fn: ``() -> bool`` embedding-endpoint probe. Defaults to the
            pipeline's. When it returns ``False`` the store is a safe no-op.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        reachable_fn: Callable[[], bool] | None = None,
        rerank_fn: Callable[[str, list[str]], list[float] | None] | None = None,
    ) -> None:
        # check_same_thread=False: the store is a process-wide singleton (see
        # decide.set_store) consulted from the compiler's worker threads AND
        # from FastAPI's request threadpool (the PM feedback endpoints). Without
        # this, a resolve()/add() from a different thread than the one that
        # opened the connection raises ProgrammingError. Behavior is otherwise
        # byte-identical to a thread-bound connection for single-threaded use.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._embed_fn = embed_fn
        self._reachable_fn = reachable_fn
        # Lazily-built, in-memory prototype cache: id -> normalized vector.
        # Embeddings are recomputed from exemplars (never persisted) so swapping
        # the embed model can't leave stale vectors behind.
        self._proto: dict[str, np.ndarray] = {}
        self._proto_dirty = True
        # Per-(relation, scope, data) neural-head cache. A head is the learned,
        # calibrated, OOD-aware scorer that fronts the raw-cosine path whenever
        # a relation has a real decision boundary (>=2 verdicts with enough
        # exemplars). Keyed by a content signature so a write rebuilds it; an
        # individual-exemplar embedding cache backs the fits.
        self._heads: dict[str, NeuralHead] = {}
        self._ex_vec: dict[str, np.ndarray] = {}  # exemplar text -> vector
        self._enable_head = True
        # NATURAL-LANGUAGE LEARNING: a correction's free-text ``instruction`` is
        # the PM's expert advice ("an airport concourse is the site itself, not a
        # sub-area"). By default it only primes the LLM; with this on it ALSO
        # becomes a labeled anchor for the correction's verdict in the SAME
        # learned metric space — so a PM moves the decision boundary by *writing a
        # sentence*, not by hand-balancing example strings. One anchor per
        # correction → a gentle, calibrated nudge (not the recall-killing flood a
        # pile of exemplars causes). Off → byte-identical to exemplar-only fits.
        self._enable_nl = os.getenv(
            "SOWSMITH_NEURAL_NL_LEARN", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        # Per-correction matrix of its individual (normalized) exemplar vectors,
        # built alongside the mean prototype. Backs the optional max-similarity
        # scorer: instead of cosine-to-mean (which blurs a correction whose
        # exemplars are heterogeneous — e.g. site codes + names + addresses), the
        # query scores against its single NEAREST exemplar (ColBERT-style late
        # interaction, lite). Off by default → byte-identical to the mean path.
        self._proto_ex: dict[str, np.ndarray] = {}
        self._enable_maxsim = os.getenv(
            "SOWSMITH_NEURAL_MAXSIM", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        # Cross-encoder reranker (retrieve-then-rerank). The bi-encoder above is
        # stage-1 recall; an optional cross-encoder is stage-2 precision. Off by
        # default → byte-identical to the bi-encoder-only path. ``rerank_fn`` is
        # an injectable scorer (tests); otherwise the env-selected backend in
        # app.core.reranker is used when SOWSMITH_NEURAL_RERANK is set.
        self._rerank_fn = rerank_fn
        self._enable_rerank = os.getenv(
            "SOWSMITH_NEURAL_RERANK", ""
        ).strip().lower() in ("1", "true", "yes", "on")

    # ── embedding plumbing (lazy import keeps this module import-light) ──
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        from app.core.embedding_retrieval import embed_texts
        return embed_texts(texts)

    def _reachable(self) -> bool:
        if self._reachable_fn is not None:
            return self._reachable_fn()
        try:
            from app.core.embedding_retrieval import embedding_endpoint_reachable
            return embedding_endpoint_reachable()
        except Exception:
            return False

    # ── CRUD ─────────────────────────────────────────────────────────
    def add(self, c: Correction) -> None:
        row = c.to_row()
        placeholders = ",".join("?" for _ in _COLUMNS)
        cols = ",".join(_COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO corrections ({cols}) VALUES ({placeholders})",
            [row[k] for k in _COLUMNS],
        )
        self._conn.commit()
        self._proto_dirty = True
        self._heads.clear()

    def all_corrections(self, *, active_only: bool = True) -> list[Correction]:
        q = "SELECT * FROM corrections"
        if active_only:
            q += " WHERE status = 'active'"
        return [Correction.from_row(r) for r in self._conn.execute(q)]

    def get(self, correction_id: str) -> Correction | None:
        r = self._conn.execute(
            "SELECT * FROM corrections WHERE id = ?", (correction_id,)
        ).fetchone()
        return Correction.from_row(r) if r else None

    def list_corrections(self, *, status: str | None = None) -> list[Correction]:
        """Return stored corrections, newest last. ``status`` filters to one
        lifecycle state (e.g. ``"active"``); ``None`` returns all. Read-only —
        used by the PM feedback surface to show which learned rules exist."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM corrections WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM corrections ORDER BY created_at"
            ).fetchall()
        return [Correction.from_row(r) for r in rows]

    def set_status(self, correction_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE corrections SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), correction_id),
        )
        self._conn.commit()
        self._proto_dirty = True
        self._heads.clear()

    def _record_hit(self, correction_id: str) -> None:
        self._conn.execute(
            "UPDATE corrections SET hit_count = hit_count + 1, last_fired = ? WHERE id = ?",
            (time.time(), correction_id),
        )
        self._conn.commit()

    # ── prototype cache ──────────────────────────────────────────────
    def _ensure_protos(self, corrections: list[Correction]) -> None:
        """Embed each correction's exemplars into a single normalized prototype
        (mean of exemplar vectors). Cached until a write dirties it."""
        if not self._proto_dirty and all(c.id in self._proto for c in corrections):
            return
        self._proto = {}
        self._proto_ex = {}
        # Batch-embed all exemplars in one call, then group back per correction.
        flat: list[str] = []
        spans: list[tuple[str, int, int]] = []
        for c in corrections:
            if not c.exemplars:
                continue
            start = len(flat)
            flat.extend(c.exemplars)
            spans.append((c.id, start, len(flat)))
        if not flat:
            self._proto_dirty = False
            return
        mat = self._embed(flat)  # (sum_exemplars, D), normalized
        for cid, s, e in spans:
            block = mat[s:e]
            # Keep the per-exemplar matrix for the optional max-sim scorer.
            self._proto_ex[cid] = block.astype(np.float32)
            proto = block.mean(axis=0)
            norm = float(np.linalg.norm(proto))
            if norm > 1e-9:
                proto = proto / norm
            self._proto[cid] = proto.astype(np.float32)
        self._proto_dirty = False

    def _correction_score(self, cid: str, qv: np.ndarray) -> float:
        """Similarity of query ``qv`` to correction ``cid``.

        Default: cosine to the mean prototype. With ``SOWSMITH_NEURAL_MAXSIM``
        on: the MAX cosine over the correction's individual exemplars (late
        interaction, lite). Max-sim refuses to let a heterogeneous correction
        dilute itself — a query that matches ONE exemplar strongly fires even if
        the other exemplars drag the mean down — while staying identical to the
        mean path for single-exemplar corrections."""
        if self._enable_maxsim:
            block = self._proto_ex.get(cid)
            if block is not None and block.shape[0] > 0:
                return float(np.max(block @ qv))
        proto = self._proto.get(cid)
        if proto is None:
            return -1.0
        return float(np.dot(proto, qv))

    # ── cross-encoder reranker (retrieve-then-rerank, stage-2 precision) ──
    def _best_exemplar_text(self, c: "Correction", qv: np.ndarray) -> str | None:
        """The single exemplar of correction ``c`` nearest the query — the text
        the cross-encoder should score the query against. (Reranking a mean
        prototype is meaningless; cross-encoders consume real text pairs.)"""
        if not c.exemplars:
            return None
        block = self._proto_ex.get(c.id)
        if block is None or block.shape[0] == 0 or block.shape[0] != len(c.exemplars):
            return c.exemplars[0]
        return c.exemplars[int(np.argmax(block @ qv))]

    def _rerank_active(self) -> bool:
        return self._rerank_fn is not None or self._enable_rerank

    def _rerank(self, query: str, docs: list[str]) -> list[float] | None:
        """Cross-encoder relevance of each doc to ``query`` in [0,1], aligned to
        input order. ``None`` → reranker unreachable (caller falls back to the
        bi-encoder; fail-open, never a guess)."""
        if self._rerank_fn is not None:
            try:
                out = self._rerank_fn(query, docs)
            except Exception:
                return None
            if out is None or len(out) != len(docs):
                return None
            try:
                return [float(x) for x in out]
            except (TypeError, ValueError):
                return None
        try:
            from app.core.reranker import rerank as _rr
            return _rr(query, docs)
        except Exception:
            return None

    def _rerank_threshold(self) -> float:
        try:
            from app.core.reranker import threshold
            return threshold()
        except Exception:
            return 0.5

    def _rerank_topk(self) -> int:
        try:
            from app.core.reranker import top_k
            return top_k()
        except Exception:
            return 20

    def _resolve_tier_reranked(
        self, tier_corrs: list["Correction"], text: str, qv: np.ndarray
    ) -> tuple[bool, "Decision | None"]:
        """Stage-2 rerank within one scope tier. Returns ``(ran, decision)``:

          * ``(True, Decision)``  — cross-encoder fired on its top pick;
          * ``(True, None)``      — cross-encoder ran but vetoed every candidate
                                    (definitive tier abstain — do NOT fall back);
          * ``(False, None)``     — reranker unreachable (caller fails open to
                                    the bi-encoder path for this tier).

        The bi-encoder casts a WIDE net (top-k by cosine/max-sim, NOT gated by
        the per-correction threshold) so the cross-encoder can rescue a
        paraphrased ghost the bi-encoder under-scored — that recall rescue, plus
        the cross-encoder's own threshold, is the ceiling-lift.
        """
        scored: list[tuple[float, "Correction"]] = []
        for c in tier_corrs:
            if c.id not in self._proto and c.id not in self._proto_ex:
                continue
            scored.append((self._correction_score(c.id, qv), c))
        if not scored:
            return (True, None)
        scored.sort(key=lambda t: -t[0])
        topk = scored[: self._rerank_topk()]
        docs = [self._best_exemplar_text(c, qv) or "" for _, c in topk]
        rr = self._rerank(text, docs)
        if rr is None or len(rr) != len(topk):
            return (False, None)  # unreachable → fail open to bi-encoder
        rt = self._rerank_threshold()
        best_i = max(range(len(topk)), key=lambda i: rr[i])
        if rr[best_i] < rt:
            return (True, None)  # cross-encoder vetoed the whole tier
        bi_score, c = topk[best_i]
        self._record_hit(c.id)
        return (True, Decision(
            verdict=c.verdict,
            confidence=float(rr[best_i]),
            source="store",
            correction_id=c.id,
            rationale=(
                f"reranked correction {c.id} "
                f"(cross-encoder {rr[best_i]:.3f}; bi-encoder {bi_score:.3f})"
            ),
        ))

    # ── neural head (learned metric, calibrated, OOD-aware) ──────────
    def _embed_exemplars(self, texts: list[str]) -> np.ndarray:
        """Embed individual exemplar strings, memoized per text. Unlike
        ``_ensure_protos`` (which collapses each correction to a mean), the head
        needs every exemplar as its own labeled training point."""
        missing = [t for t in texts if t not in self._ex_vec]
        if missing:
            mat = self._embed(missing)
            for t, v in zip(missing, mat):
                self._ex_vec[t] = v.astype(np.float32)
        return np.vstack([self._ex_vec[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def _relation_head(
        self, relation: str, allowed: set[str], scope: DecisionScope
    ) -> NeuralHead | None:
        """Fit (or reuse) the neural head for this relation, over the
        corrections visible at ``scope`` (global + matching pack + matching
        deal). Returns a *trained* head only when there is a real decision
        boundary — >=2 distinct verdicts, each with enough exemplars — else
        ``None`` so resolve() uses the legacy per-correction cosine path
        (which still handles single-class rules like PurTera unchanged).
        """
        if not self._enable_head:
            return None
        # Scope-visible corrections for this relation (verdict need NOT be in
        # `allowed` here — the head learns the full boundary, e.g. job_site vs
        # vendor, even if the caller's candidate set is a subset; classify()
        # restricts to candidates at decision time).
        visible: list[Correction] = []
        for c in self.all_corrections(active_only=True):
            if c.relation != relation or not c.exemplars:
                continue
            if c.scope == SCOPE_GLOBAL:
                visible.append(c)
            elif c.scope == SCOPE_PACK and c.scope_key == scope.pack:
                visible.append(c)
            elif c.scope == SCOPE_DEAL and c.scope_key == scope.deal_id:
                visible.append(c)
        if len({c.verdict for c in visible}) < 2:
            return None  # no boundary → single-class cosine path handles it

        pairs: list[tuple[str, str]] = []
        for c in visible:
            for ex in c.exemplars:
                pairs.append((ex, c.verdict))
            # NATURAL-LANGUAGE LEARNING: fold the PM's prose advice in as a
            # labeled anchor for this correction's verdict. One point per
            # correction keeps it a calibrated nudge, not an exemplar flood.
            if self._enable_nl:
                instr = (c.instruction or "").strip()
                if instr:
                    pairs.append((instr, c.verdict))
        sig = f"{relation}|{scope.deal_id}|{scope.pack}|nl={int(self._enable_nl)}|" + "␟".join(
            sorted(f"{v}␞{t}" for t, v in pairs)
        )
        cached = self._heads.get(sig)
        if cached is not None:
            return cached if cached.trained else None
        X = self._embed_exemplars([t for t, _ in pairs])
        if X.shape[0] != len(pairs) or X.shape[0] == 0:
            return None
        head = NeuralHead().fit(X, [v for _, v in pairs])
        self._heads[sig] = head
        return head if head.trained else None

    def _nearest_correction_id(
        self, corrs: list[Correction], verdict: str, qv: np.ndarray
    ) -> str | None:
        """Best raw-cosine correction of ``verdict`` for provenance citation."""
        self._ensure_protos(corrs)
        best: tuple[float, str] | None = None
        for c in corrs:
            if c.verdict != verdict:
                continue
            proto = self._proto.get(c.id)
            if proto is None:
                continue
            s = float(np.dot(proto, qv))
            if best is None or s > best[0]:
                best = (s, c.id)
        return best[1] if best else None

    # ── decide() contract: resolve + few_shot ────────────────────────
    def resolve(
        self,
        *,
        relation: str,
        text: str,
        candidates: list[str],
        context: str,
        scope: DecisionScope,
        instruction: str,
        relations: dict | None,
    ) -> Decision | None:
        """Return a Decision on a confident, in-candidate-set hit; else None.

        Narrowest scope wins: deal corrections are searched first, then pack,
        then global. A correction only fires if (a) its relation matches, (b)
        its verdict is one of the caller's ``candidates``, and (c) the query's
        cosine similarity to its prototype meets the correction's threshold.
        """
        try:
            if not text or not candidates:
                return None
            if not self._reachable():
                return None
            allowed = set(candidates)
            corrs = [
                c for c in self.all_corrections(active_only=True)
                if c.relation == relation and c.verdict in allowed
            ]
            if not corrs:
                return None
            self._ensure_protos(corrs)

            q = self._embed([text])
            if q.shape[0] == 0:
                return None
            qv = q[0]
            if float(np.linalg.norm(qv)) < 1e-9:  # failed embed → undecided
                return None

            # 0) NEURAL HEAD — when this relation has a real decision boundary
            #    (>=2 verdicts), a learned, calibrated, OOD-aware scorer decides
            #    the confident cases (positive AND negative) and abstains on the
            #    uncertain/novel ones. Abstain → fall through to the cosine path,
            #    then (in decide()) to the LLM. This is what keeps the LLM for
            #    genuinely hard decisions only.
            head = self._relation_head(relation, allowed, scope)
            if head is not None:
                hd = head.classify(qv, candidates)
                if hd.verdict is not None and not hd.route_llm:
                    # Cite the nearest correction of the winning verdict for
                    # human-traceable provenance.
                    cid = self._nearest_correction_id(corrs, hd.verdict, qv)
                    self._record_hit(cid) if cid else None
                    return Decision(
                        verdict=hd.verdict,
                        confidence=hd.confidence,
                        source="store",
                        correction_id=cid,
                        rationale=(
                            f"neural head (calibrated p={hd.confidence:.3f}, "
                            f"margin={hd.margin:.2f}, in-distribution)"
                        ),
                    )
                # Head abstained: uncertain or OOD → let cosine/LLM handle it.

            use_rerank = self._rerank_active()

            # Narrowest scope first.
            for tier, key in (
                (SCOPE_DEAL, scope.deal_id),
                (SCOPE_PACK, scope.pack),
                (SCOPE_GLOBAL, ""),
            ):
                tier_corrs = [
                    c for c in corrs
                    if c.scope == tier and (tier == SCOPE_GLOBAL or c.scope_key == key)
                ]
                if not tier_corrs:
                    continue

                # Stage-2 cross-encoder rerank (when enabled). Authoritative
                # within a tier: a fire returns; a veto skips to the next scope
                # tier; only an UNREACHABLE reranker falls open to the
                # bi-encoder path below.
                if use_rerank:
                    ran, decision = self._resolve_tier_reranked(tier_corrs, text, qv)
                    if decision is not None:
                        return decision
                    if ran:
                        continue

                best: tuple[float, Correction] | None = None
                for c in tier_corrs:
                    if c.id not in self._proto and c.id not in self._proto_ex:
                        continue
                    score = self._correction_score(c.id, qv)
                    if score >= c.threshold and (best is None or score > best[0]):
                        best = (score, c)
                if best is not None:
                    score, c = best
                    self._record_hit(c.id)
                    metric = "max-sim" if self._enable_maxsim else "cosine"
                    return Decision(
                        verdict=c.verdict,
                        confidence=score,
                        source="store",
                        correction_id=c.id,
                        rationale=f"matched correction {c.id} ({metric} {score:.3f})",
                    )
            return None
        except Exception:  # pragma: no cover - never break decide()
            return None

    def learn_from_teacher(
        self,
        *,
        relation: str,
        text: str,
        verdict: str,
        confidence: float,
        scope: DecisionScope,
        instruction: str = "",
        max_per_class: int = 200,
    ) -> str | None:
        """Persist a confident LLM verdict as a *weak* correction so the neural
        head learns this region and the LLM is not consulted here again.

        Distinct from a human correction:
          * ``created_by="teacher"`` and ``complaint_id="teacher:<relation>"``
            so it is auditable and prunable, never confused with a PM's call;
          * content-addressed id → re-seeing the same text is idempotent (no
            duplicate rows, no unbounded growth on repeats);
          * scoped to the deal when known (a teacher label generalizes within a
            deal first); global only when no deal context.

        Returns the correction id, or None when not written. Never raises.
        """
        try:
            t = (text or "").strip()
            if not t or not verdict or not relation:
                return None
            import hashlib

            digest = hashlib.sha256(
                f"{relation}\x1f{verdict}\x1f{t.lower()}".encode("utf-8")
            ).hexdigest()[:16]
            cid = f"teacher_{digest}"
            if self.get(cid) is not None:
                return cid  # idempotent: already learned this exact case

            if scope.deal_id:
                tier, key = SCOPE_DEAL, scope.deal_id
            elif scope.pack:
                tier, key = SCOPE_PACK, scope.pack
            else:
                tier, key = SCOPE_GLOBAL, ""

            # Capacity guard per (relation, verdict, scope): keep the store from
            # ballooning if a deal repeats near-identical lines forever.
            same = [
                c for c in self.all_corrections(active_only=True)
                if c.relation == relation and c.verdict == verdict
                and c.created_by == "teacher" and c.scope == tier and c.scope_key == key
            ]
            if len(same) >= max_per_class:
                return None

            self.add(Correction(
                id=cid, relation=relation, verdict=verdict,
                scope=tier, scope_key=key, exemplars=[t],
                threshold=_DEFAULT_THRESHOLD, instruction=instruction,
                created_by="teacher", complaint_id=f"teacher:{relation}",
                confidence_floor=float(confidence),
            ))
            return cid
        except Exception:  # pragma: no cover - never break decide()
            return None

    def evaluation_twin(
        self, *, extra: list[Correction] | None = None
    ) -> "FeedbackStore":
        """Return an ephemeral in-memory copy of this store (same embedder /
        reachability), optionally with ``extra`` corrections added.

        Used by the eval harness to test a *candidate* correction against a
        hold-out without ever writing it to the real store — so a correction
        that causes collateral damage is rejected before it can fire in
        production.
        """
        twin = FeedbackStore(
            ":memory:", embed_fn=self._embed_fn, reachable_fn=self._reachable_fn
        )
        for c in self.all_corrections(active_only=False):
            twin.add(c)
        for c in extra or []:
            twin.add(c)
        return twin

    def few_shot(
        self,
        *,
        relation: str,
        text: str,
        scope: DecisionScope,
        k: int = 3,
    ) -> list[dict]:
        """Return up to ``k`` nearest corrections for this relation as
        ``{"text", "verdict"}`` few-shot examples to prime the LLM when the
        store has no *confident* hit. Safe no-op when offline."""
        try:
            if not text or not self._reachable():
                return []
            corrs = [
                c for c in self.all_corrections(active_only=True)
                if c.relation == relation and c.exemplars
            ]
            if not corrs:
                return []
            self._ensure_protos(corrs)
            q = self._embed([text])
            if q.shape[0] == 0:
                return []
            qv = q[0]
            scored: list[tuple[float, Correction]] = []
            for c in corrs:
                proto = self._proto.get(c.id)
                if proto is not None:
                    scored.append((float(np.dot(proto, qv)), c))
            scored.sort(key=lambda t: t[0], reverse=True)
            return [
                {"text": c.exemplars[0], "verdict": c.verdict}
                for _, c in scored[:k]
            ]
        except Exception:  # pragma: no cover
            return []


# ── default seed corrections ─────────────────────────────────────────
#
# The one correction every deal needs from day zero: PurTera is the user's own
# company, used as the service provider in ALL deals, so its office / letterhead
# / billing address must NEVER be minted as a job site. This is a GLOBAL-scoped
# correction (it holds across every deal and pack), and it is the canonical
# proof that the store generalizes from a single human judgment: one exemplar,
# semantic match, fires forever with zero LLM cost.
#
# It is seeded by deterministic id, so re-seeding REPLACES rather than
# duplicates it (INSERT OR REPLACE on the primary key) — safe to call on every
# store init.
_GLOBAL_PURTERA_ID = "global_purtera_self_address"


def seed_default_corrections(store: "FeedbackStore") -> int:
    """Install the built-in global corrections into ``store``. Idempotent.

    Returns the number of corrections seeded:
      * the PurTera self-address rule (one global judgment), and
      * the universal site ghost-rejection gate — the learned, deal-agnostic
        replacement for the bulk of the hand-curated ``_OBVIOUS_NON_SITES``
        denylist (a 3-way site-role head plus one tight binary kNN per ghost
        concept). All exemplars are plain-language ROLE descriptions, never a
        deal's specific site/city/vendor name.

    All entries are seeded by deterministic id, so re-seeding REPLACES rather
    than duplicates (INSERT OR REPLACE on the primary key) — safe on every init.
    """
    store.add(
        Correction(
            id=_GLOBAL_PURTERA_ID,
            relation="physical_site",
            verdict="vendor_or_billing_address",
            scope=SCOPE_GLOBAL,
            exemplars=[
                "PurTera LLC, 11720 Amber Park Dr, Alpharetta GA 30009",
                "PurTera, Alpharetta GA — service provider letterhead / billing address",
            ],
            instruction=(
                "Classify the role of this address within the deal. PurTera is "
                "the service provider's own company; its office, letterhead, or "
                "billing address is never a customer job site."
            ),
            created_by="seed",
            complaint_id="seed:purtera_self_address",
        )
    )
    seeded = 1

    # Universal site ghost-rejection gate (role head + concept gates). Imported
    # lazily so a problem in the seed module can never break store construction.
    try:
        from app.core.site_role_seed import site_role_gate_corrections

        for corr in site_role_gate_corrections():
            store.add(corr)
            seeded += 1
    except Exception:  # pragma: no cover - seed module must never break init
        pass

    return seeded


__all__ = [
    "Correction",
    "FeedbackStore",
    "SCOPE_DEAL",
    "SCOPE_PACK",
    "SCOPE_GLOBAL",
    "seed_default_corrections",
]
