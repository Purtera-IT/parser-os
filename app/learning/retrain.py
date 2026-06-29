"""Retrainer — the eval-gated promotion loop that makes the head self-improving.

This is the engine that turns *accumulating corrections* into a *better served
model*, safely. Each run, per relation:

1. **Fit a candidate** head on the current TRAIN rows (gold-weighted, so PM
   corrections dominate abundant LLM silver — see
   :meth:`app.core.neural_head.NeuralHead.fit`).
2. **Eval-gate** it leave-one-deal-out on the HOLDOUT split
   (:func:`app.core.shadow_eval.evaluate_relation_head`) — accuracy/coverage on
   deals whose names it never trained on.
3. **Compare to the incumbent champion**, re-scored on the *same* current
   holdout, and promote the candidate **only** if it clears the absolute
   readiness bar AND does not regress the champion. Otherwise the champion
   stays (rollback-by-default — a bad candidate can never reach serve).
4. **Record** the serving metrics into the shadow-history curve and register
   every candidate in the head registry for audit.

The contract that keeps this honest:

* **Never promote a regression.** A candidate that is worse than the champion on
  unseen deals — especially on gold rows — is held, never served.
* **Guess-free preserved.** Promotion only changes *which fast scorer* answers;
  abstention still falls through to the LLM. A promoted head that abstains is
  not "wrong", just deferring.
* **Embedder-pinned.** The head is valid only for the embedding model it was fit
  on; the model id is stored and checked at serve time.
* **Idempotent.** If the data hasn't changed since the champion was trained
  (same signature), retrain is skipped — no churn, no needless re-promotion.

CLI (the "do it for me" entrypoint)::

    SOWSMITH_TRAINING_LOG_DB=_training_cloud.db \\
    SOWSMITH_HEAD_REGISTRY_DIR=_head_registry \\
    SOWSMITH_SHADOW_HISTORY_DB=_shadow_history.db \\
    OLLAMA_HOST=http://... python -X utf8 -m app.learning.retrain

Run it after each batch of compiles (or on a schedule); it advances the
champions only when the data earns it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from app.core.neural_head import NeuralHead
from app.core.shadow_eval import RelationReport, _head_feature, evaluate_relation_head
from app.core.training_log import TrainingLog
from app.learning.head_registry import HeadMeta, HeadRegistry

# How much worse than the champion (on unseen deals) a candidate may be on any
# axis and still be considered "not a regression". Tight on purpose.
_REGRESSION_TOL = 0.01
# A candidate must beat the champion's accuracy by at least this to *replace* a
# still-ready champion (avoids churn from eval noise). A stale (no-longer-ready)
# champion is replaced by any ready candidate regardless.
_MIN_IMPROVEMENT = 0.0


@dataclass
class RetrainResult:
    """Outcome of one relation's retrain attempt."""

    relation: str
    action: str                       # promote_first | promote_better | promote_stale_champion | hold_not_ready | hold_champion_better | skip_no_new_data | skip_no_data
    candidate_version: Optional[str] = None
    promoted: bool = False
    candidate: Optional[dict] = None  # candidate holdout metrics
    champion: Optional[dict] = None   # incumbent holdout metrics (re-scored)
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "relation": self.relation,
            "action": self.action,
            "candidate_version": self.candidate_version,
            "promoted": self.promoted,
            "candidate": self.candidate,
            "champion": self.champion,
            "detail": self.detail,
        }


def data_signature(log: TrainingLog, relation: str) -> str:
    """A stable fingerprint of a relation's training data (membership + labels +
    weights). Changes whenever a row is added or a label/weight is corrected, so
    the retrainer can skip relations with no new signal."""
    rows = log.rows(relation=relation)
    h = hashlib.sha256()
    h.update(f"n={len(rows)}".encode("utf-8"))
    for r in sorted(rows, key=lambda r: r.id):
        h.update(f"|{r.id}:{r.label}:{r.weight}:{r.split}".encode("utf-8"))
    return h.hexdigest()[:16]


def fit_candidate(
    log: TrainingLog,
    relation: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    **head_kwargs,
) -> Optional[tuple[NeuralHead, int]]:
    """Fit a head on TRAIN rows for one relation. Returns (head, n_train) or
    None when there is nothing to learn from."""
    train = log.rows(relation=relation, split="train")
    feats, y, w = [], [], []
    for r in train:
        f = _head_feature(r)
        if f and r.label:
            feats.append(f)
            y.append(r.label)
            w.append(float(r.weight) or 1.0)
    if not feats or len(set(y)) < 1:
        return None
    X = np.asarray(embed_fn(feats), dtype=np.float32)
    head = NeuralHead(**head_kwargs).fit(X, y, sample_weight=np.asarray(w, dtype=np.float32))
    return head, len(feats)


def _is_regression(cand: RelationReport, champ: RelationReport, tol: float) -> bool:
    """True if the candidate is meaningfully worse than the champion on unseen
    deals on any axis we care about (gold accuracy weighted most)."""
    return (
        cand.accuracy < champ.accuracy - tol
        or cand.coverage < champ.coverage - tol
        or cand.gold_accuracy < champ.gold_accuracy - tol
    )


def retrain_relation(
    log: TrainingLog,
    relation: str,
    registry: HeadRegistry,
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    embed_model: str = "",
    regression_tol: float = _REGRESSION_TOL,
    min_improvement: float = _MIN_IMPROVEMENT,
    **head_kwargs,
) -> RetrainResult:
    """Fit, eval-gate, and conditionally promote one relation's head."""
    sig = data_signature(log, relation)
    champ_meta = registry.champion_meta(relation)

    # No new data since the current champion → nothing to do.
    if champ_meta and champ_meta.data_signature == sig:
        return RetrainResult(
            relation=relation, action="skip_no_new_data",
            candidate_version=champ_meta.version, promoted=False,
            champion=_meta_metrics(champ_meta),
            detail="data unchanged since champion was trained",
        )

    fit = fit_candidate(log, relation, embed_fn, **head_kwargs)
    if fit is None:
        return RetrainResult(relation=relation, action="skip_no_data",
                             detail="no trainable rows")
    cand_head, n_train = fit

    cand_rep = evaluate_relation_head(log, relation, embed_fn=embed_fn, head=cand_head)
    cand_meta = registry.register(
        relation, cand_head,
        embed_model=embed_model, data_signature=sig, n_train=n_train,
        n_holdout=cand_rep.n_holdout, coverage=cand_rep.coverage,
        accuracy=cand_rep.accuracy, gold_accuracy=cand_rep.gold_accuracy,
        ready=cand_rep.ready(),
    )

    # Candidate must clear the absolute bar before it can ever serve.
    if not cand_rep.ready():
        return RetrainResult(
            relation=relation, action="hold_not_ready",
            candidate_version=cand_meta.version, promoted=False,
            candidate=cand_rep.as_dict(),
            champion=_meta_metrics(champ_meta) if champ_meta else None,
            detail="candidate does not clear readiness bar; champion (if any) unchanged",
        )

    # First-ever ready head for this relation → promote.
    if champ_meta is None:
        registry.promote(relation, cand_meta.version)
        return RetrainResult(
            relation=relation, action="promote_first",
            candidate_version=cand_meta.version, promoted=True,
            candidate=cand_rep.as_dict(),
            detail="first ready head promoted to champion",
        )

    # Re-score the incumbent champion on the SAME current holdout for a fair
    # head-to-head on the deals that exist now.
    champ_obj = registry.champion(relation)
    if champ_obj is None:
        registry.promote(relation, cand_meta.version)
        return RetrainResult(
            relation=relation, action="promote_first",
            candidate_version=cand_meta.version, promoted=True,
            candidate=cand_rep.as_dict(),
            detail="champion artifact unreadable; promoted candidate",
        )
    champ_head, _ = champ_obj
    champ_rep = evaluate_relation_head(log, relation, embed_fn=embed_fn, head=champ_head)

    if _is_regression(cand_rep, champ_rep, regression_tol):
        return RetrainResult(
            relation=relation, action="hold_champion_better",
            candidate_version=cand_meta.version, promoted=False,
            candidate=cand_rep.as_dict(), champion=champ_rep.as_dict(),
            detail="candidate regresses champion on unseen deals; champion kept",
        )

    champ_ready_now = champ_rep.ready()
    beats = cand_rep.accuracy >= champ_rep.accuracy + min_improvement

    if not champ_ready_now:
        registry.promote(relation, cand_meta.version)
        return RetrainResult(
            relation=relation, action="promote_stale_champion",
            candidate_version=cand_meta.version, promoted=True,
            candidate=cand_rep.as_dict(), champion=champ_rep.as_dict(),
            detail="champion no longer clears the bar on current data; candidate promoted",
        )
    if beats:
        registry.promote(relation, cand_meta.version)
        return RetrainResult(
            relation=relation, action="promote_better",
            candidate_version=cand_meta.version, promoted=True,
            candidate=cand_rep.as_dict(), champion=champ_rep.as_dict(),
            detail="candidate beats champion without regression; promoted",
        )
    return RetrainResult(
        relation=relation, action="hold_champion_better",
        candidate_version=cand_meta.version, promoted=False,
        candidate=cand_rep.as_dict(), champion=champ_rep.as_dict(),
        detail="candidate not better than a still-ready champion; champion kept",
    )


def _meta_metrics(meta: Optional[HeadMeta]) -> Optional[dict]:
    if meta is None:
        return None
    return {
        "coverage": round(meta.coverage, 4),
        "accuracy": round(meta.accuracy, 4),
        "gold_accuracy": round(meta.gold_accuracy, 4),
        "ready": meta.ready,
    }


def retrain_all(
    log: TrainingLog,
    registry: HeadRegistry,
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    relations: Optional[list[str]] = None,
    embed_model: str = "",
    record_history: bool = True,
    **head_kwargs,
) -> list[RetrainResult]:
    """Retrain + eval-gate every relation present in the log (or a given subset).
    Records the serving metrics into the shadow-history curve."""
    # Safety guard: NEVER train on a dead embedder. embed_texts returns
    # zero-vectors (not an error) when the qwen3-Mac/Ollama host is offline, so
    # an unguarded scheduled run during an outage would fit + eval-gate heads on
    # garbage and could promote a degenerate champion. Probe first; abort the
    # whole run cleanly (no-op) if the embedder is unreachable.
    try:
        _probe = np.asarray(embed_fn(["__embed_probe__"]))
        if _probe.size == 0 or float(np.linalg.norm(_probe.reshape(-1))) == 0.0:
            print("[retrain] embedder unreachable (zero-vector probe) — aborting; no training this run")
            return []
    except Exception as _e:  # pragma: no cover - probe failure must abort, not train
        print(f"[retrain] embedder probe failed ({_e}) — aborting; no training this run")
        return []
    # Pull any PM gold rows the SERVICE mirrored to blob into this log first, so
    # the retrain learns from corrections written on the other container (the
    # feedback endpoint runs on the service; retrain runs here on the worker).
    # Gated + best-effort: no-op unless SOWSMITH_FEEDBACK_BLOB is on.
    try:
        from app.core import feedback_blob as _fb
        _n = _fb.sync_training_rows_into_log(log)
        if _n:
            print(f"[retrain] imported {_n} PM gold rows from blob")
    except Exception:
        pass
    if relations is None:
        relations = sorted({r.relation for r in log.rows()})
    results: list[RetrainResult] = []
    serving_reports: dict[str, RelationReport] = {}
    for rel in relations:
        res = retrain_relation(
            log, rel, registry, embed_fn,
            embed_model=embed_model, **head_kwargs,
        )
        results.append(res)
        # Snapshot the metrics of whatever now serves this relation.
        champ = registry.champion(rel)
        if champ is not None:
            head, _ = champ
            serving_reports[rel] = evaluate_relation_head(
                log, rel, embed_fn=embed_fn, head=head)
    if record_history and serving_reports:
        try:
            from app.core import shadow_history
            shadow_history.record(serving_reports, log=log, label="retrain")
        except Exception:
            pass
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────
def _embed_model_id() -> str:
    # The head is only valid for the embedder it was fit on; capture the SAME id
    # the pipeline resolves (OLLAMA_EMBED_MODEL or the pinned default) so the
    # registry can enforce embedder-pinning at serve time.
    try:
        from app.core import embedding_retrieval as er
        v = er._embed_model()
        if isinstance(v, str) and v:
            return v
    except Exception:
        pass
    # Fallbacks: module-level constants, then explicit override.
    for name in ("EMBED_MODEL", "EMBEDDING_MODEL", "MODEL", "_DEFAULT_MODEL"):
        try:
            from app.core import embedding_retrieval as er
            v = getattr(er, name, None)
            if isinstance(v, str) and v:
                return v
        except Exception:
            pass
    return os.environ.get("SOWSMITH_EMBED_MODEL", "unknown")


def main() -> None:
    from app.core.embedding_retrieval import embed_texts
    from app.core.training_log import get_training_log

    log = get_training_log()
    if log is None:
        raise SystemExit("set SOWSMITH_TRAINING_LOG_DB to the training-log DB first")
    registry_dir = os.environ.get("SOWSMITH_HEAD_REGISTRY_DIR")
    if not registry_dir:
        raise SystemExit("set SOWSMITH_HEAD_REGISTRY_DIR to the registry path first")
    registry = HeadRegistry(registry_dir)

    _ = embed_texts(["probe"])  # fail loud if the embedding host is down
    model = _embed_model_id()

    print(f"retrain: log_rows={log.count()} registry={registry_dir} embed_model={model}\n")
    results = retrain_all(log, registry, embed_texts, embed_model=model)

    hdr = f"{'relation':22s} {'action':24s} {'cand acc/cov':>14s} {'champ acc/cov':>14s} {'promoted':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        c = r.candidate or {}
        ch = r.champion or {}
        cstr = f"{c.get('accuracy',0):.2f}/{c.get('coverage',0):.2f}" if c else "   -   "
        hstr = f"{ch.get('accuracy',0):.2f}/{ch.get('coverage',0):.2f}" if ch else "   -   "
        print(f"{r.relation:22s} {r.action:24s} {cstr:>14s} {hstr:>14s} {('YES' if r.promoted else '-'):>9s}")

    print("\nchampions now serving:")
    for rel, info in registry.summary().items():
        if info["champion"]:
            print(f"  {rel:22s} acc={info['accuracy']} cov={info['coverage']} "
                  f"ready={info['ready']} ({info['versions']} versions)")
    promoted = [r.relation for r in results if r.promoted]
    print(f"\npromoted this run: {promoted or '— none —'}")


if __name__ == "__main__":
    main()
