"""Dedicated nightly retrain entry — run as a scheduled Azure Container Apps Job,
SEPARATE from the warm worker so it never holds a compile slot (the in-worker
post-compile retrain is OFF in dev for exactly that reason: it embeds via the Mac
and can hang the next reparse).

Safe by construction:
  * ABORTS if the embedder is unreachable — never trains on zero-vectors (when
    the qwen3-Mac/Ollama host is offline, embed_texts returns zeros, not an
    error, which would otherwise fit + promote degenerate heads);
  * imports the PM gold rows the SERVICE mirrored to blob, so PM corrections
    actually become training data;
  * runs the serving deflector retrains (type + span heads), whose artifacts the
    worker already round-trips via write_back_ml -> fetch_ml, so a promotion here
    reaches live serving on the worker's next start;
  * runs the eval-gated registry retrain (consumes the same gold; promotes a
    champion only when it beats the incumbent on held-out);
  * persists the grown log + retrained heads back to blob.

Run: ``python -m app.learning.nightly_retrain``
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

# PM-gold holdout labels required before a calibrator may be promoted. Silver
# labels are derived from the system's own review_flags / contradictions, so
# validating against them is circular — a calibrator that merely rediscovers
# that rule scores near-perfectly. Only human corrections are real evidence
# about whether an output was right. Override with CALIBRATOR_MIN_GOLD.
MIN_GOLD_HOLDOUT = int(os.getenv("CALIBRATOR_MIN_GOLD", "20"))


def _embedder_live() -> bool:
    """True only if the embedder returns a real (non-zero) vector. Guards every
    downstream retrain so an offline Mac makes the whole run a safe no-op."""
    try:
        import numpy as np

        from app.core.embedding_retrieval import embed_texts

        probe = np.asarray(embed_texts(["__nightly_embed_probe__"]))
        return probe.size > 0 and float(np.linalg.norm(probe.reshape(-1))) > 0.0
    except Exception as e:  # pragma: no cover - any probe failure => skip, never train
        print(f"[nightly] embedder probe failed: {e}")
        return False


def _fit_calibrator() -> None:
    """Best-effort eval-gated calibrator fit. Loads recent CompileResults from
    blob (the worker persists deals/<id>/parser-os/latest/result.json), labels
    them via PM corrections (gold) + silver bootstrap, fits on a train split,
    and PROMOTES _calibrator/calibrator.joblib ONLY if its Brier beats the raw
    heuristic on a held-out split (rollback-by-default). No-op until enough
    data + result.json exist — the deterministic review gate serves meanwhile."""
    import hashlib
    import json as _json
    import tempfile
    from pathlib import Path

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        print("[nightly] calibrator: no storage conn — skip")
        return
    try:
        from azure.storage.blob import ContainerClient

        from app.core.schemas import CompileResult
        from app.core.training_log import TEACHER_PM, get_training_log
        from app.learning import calibration as C
        from app.learning.features import build_atom_feature_row
    except Exception as e:
        print(f"[nightly] calibrator deps missing: {e}")
        return

    # 1. Recent compile results (the feature source).
    try:
        cc = ContainerClient.from_connection_string(
            conn, os.environ.get("AZURE_STORAGE_BLOB_CONTAINER", "orbitbrief-artifacts"))
        names = [b.name for b in cc.list_blobs(name_starts_with="deals/")
                 if b.name.endswith("/parser-os/latest/result.json")][:50]
        results = []
        for nm in names:
            try:
                results.append(CompileResult.model_validate_json(cc.download_blob(nm).readall()))
            except Exception:
                continue
    except Exception as e:
        print(f"[nightly] calibrator: result.json load failed: {e}")
        return
    if not results:
        print("[nightly] calibrator: no result.json in blob yet — skip (deterministic gate serves)")
        return

    # 2. PM-corrected atom ids (the gold correctness signal).
    pm_ids: set[str] = set()
    try:
        log = get_training_log()
        if log is not None:
            for r in log.rows():
                if getattr(r, "teacher", "") == TEACHER_PM and getattr(r, "complaint_id", None):
                    pm_ids.add(r.complaint_id)
    except Exception:
        pass

    labels = C.build_calibration_labels(results, pm_corrected_atom_ids=pm_ids)
    print(f"[nightly] calibrator labels: {len(labels['atom_labels'])} atom / {len(labels['reviews'])} packet (pm_gold={len(pm_ids)})")
    if len(labels["reviews"]) < 8 or len({r["correct_packet"] for r in labels["reviews"]}) < 2:
        print("[nightly] calibrator: need >=8 packet labels with both classes — skip")
        return

    def _holdout(_id: str) -> bool:
        return int(hashlib.md5(_id.encode()).hexdigest(), 16) % 5 == 0  # ~20%

    train = {
        "atom_labels": [r for r in labels["atom_labels"] if not _holdout(r["atom_id"])],
        "reviews": [r for r in labels["reviews"] if not _holdout(r["packet_id"])],
    }
    ho_atoms = [r for r in labels["atom_labels"] if _holdout(r["atom_id"])]

    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "labels.json"
        lp.write_text(_json.dumps(train))
        cand = Path(td) / "cand.joblib"
        try:
            C.train_calibrator(lp, results, cand)
        except Exception as e:
            print(f"[nightly] calibrator fit skipped: {e}")
            return
        payload = C.load_calibrator(cand)
        am = payload.get("atom_model")
        if am is None or not ho_atoms:
            print("[nightly] calibrator: no atom model / holdout — not promoting")
            return
        by_id = {a.id: a for r in results for a in (getattr(r, "atoms", []) or [])}
        feats, ys, heur = [], [], []
        # Kept index-aligned with feats/ys/heur so the gold subset can be sliced
        # out below; `ho_atoms` itself is not, because atoms missing from the
        # results are skipped.
        ho_rows = []
        for r in ho_atoms:
            a = by_id.get(r["atom_id"])
            if a is None:
                continue
            feats.append(build_atom_feature_row(a))
            ys.append(r["label"])
            ho_rows.append(r)
            hc = getattr(a, "calibrated_confidence", None)
            heur.append(hc if hc is not None else (getattr(a, "confidence", 0.5) or 0.5))
        if len(ys) < 5 or len(set(ys)) < 2:
            print("[nightly] calibrator: holdout too thin — not promoting")
            return

        # Promotion MUST be decided on PM gold, not on silver.
        #
        # Silver labels come from review_flags / contradictions / high-confidence,
        # i.e. from the system's own prior opinion — so a calibrator can score
        # near-perfectly by rediscovering the rule that generated them. The
        # previous gate could not catch that, because the holdout was labelled by
        # the SAME rule: circularity inflated candidate and baseline alike, so a
        # degenerate fit promoted most easily rather than least. Only a human
        # correction is real evidence about whether an output was right.
        gold_idx = [i for i, r in enumerate(ho_rows) if r["atom_id"] in pm_ids]
        if len(gold_idx) < MIN_GOLD_HOLDOUT:
            print(
                f"[nightly] calibrator: only {len(gold_idx)} PM-gold holdout labels "
                f"(need {MIN_GOLD_HOLDOUT}) — NOT promoting; deterministic gate serves. "
                "Silver-only validation is circular and cannot justify a promotion."
            )
            return
        probs_all = [float(p[1]) for p in am.predict_proba(feats)]
        probs = [probs_all[i] for i in gold_idx]
        ys = [ys[i] for i in gold_idx]
        heur = [heur[i] for i in gold_idx]
        if len(set(ys)) < 2:
            print("[nightly] calibrator: PM-gold holdout is single-class — not promoting")
            return
        b_cal, b_heur = C.brier_score(probs, ys), C.brier_score(heur, ys)
        print(
            f"[nightly] calibrator Brier={b_cal:.4f} vs heuristic={b_heur:.4f} "
            f"(PM-gold holdout n={len(ys)})"
        )
        if b_cal < b_heur:
            import shutil
            dst = Path(os.environ.get("ML_ARTIFACT_DIR", "/tmp/ml")) / "_calibrator"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(cand, dst / "calibrator.joblib")
            print(f"[nightly] PROMOTED calibrator -> {dst / 'calibrator.joblib'}")
        else:
            print("[nightly] calibrator did NOT beat heuristic — not promoted (rollback-by-default)")


def main() -> int:
    if not _embedder_live():
        print("[nightly] embedder unreachable — skipping retrain (no training on zero-vectors)")
        return 0

    # Import PM gold rows (mirrored to blob by the service feedback endpoint) into
    # the training log so BOTH the deflector retrains and the eval-gated retrain
    # learn from PM corrections.
    try:
        from app.core import feedback_blob
        from app.core.training_log import get_training_log

        log = get_training_log()
        if log is not None:
            n = feedback_blob.sync_training_rows_into_log(log)
            print(f"[nightly] imported {n} PM gold rows into the training log")
            # Silver from every compile. The workers mirror each batch to its own
            # blob because /tmp does not survive a recycle; merging them here is
            # what finally gives the eval gate a holdout to score against.
            # Idempotent (row id is the PRIMARY KEY, add_many upserts).
            try:
                from app.core import training_row_blob

                m = training_row_blob.sync_into_log(log)
                print(f"[nightly] imported {m} compile training rows from blob")
            except Exception as exc:
                print(f"[nightly] compile-row import skipped: {exc}")
        else:
            print("[nightly] no training log (SOWSMITH_TRAINING_LOG_DB unset) — skipping gold import")
    except Exception as e:
        print(f"[nightly] gold import skipped: {e}")

    # Serving deflector retrains — _type_head / _span_heads round-trip to the
    # worker via write_back_ml -> fetch_ml, so promotions reach live serving.
    for mod_name, fn_name in (
        ("app.core.type_head", "retrain_if_stale"),
        ("app.core.span_extractor", "retrain_span_heads"),
    ):
        try:
            getattr(importlib.import_module(mod_name), fn_name)()
            print(f"[nightly] {fn_name} ok")
        except Exception as e:
            print(f"[nightly] {fn_name} skipped: {e}")

    # Eval-gated registry retrain (guarded internally; consumes the same gold).
    try:
        from app.learning.retrain import main as retrain_main

        retrain_main()
    except SystemExit as e:
        print(f"[nightly] eval-gated retrain: {e}")
    except Exception as e:
        print(f"[nightly] eval-gated retrain skipped: {e}")

    # Eval-gated calibrator fit (best-effort; promotes _calibrator only if it
    # beats the heuristic on a holdout). No-op until result.json + labels exist.
    try:
        _fit_calibrator()
    except Exception as e:
        print(f"[nightly] calibrator fit skipped: {e}")

    # Retention. This run just registered another candidate per relation, and
    # nothing ever removed the old ones — the registry reached 14,165 files /
    # ~5GB, most of the ~9GB ml-artifacts container, which the workers fetch onto
    # 4-8Gi of ephemeral disk. That filled /tmp and killed every compile with
    # OSError: [Errno 28]. Prune BEFORE the write-back so the deletions mirror to
    # blob in the same pass. Champion and previous_champion are always kept, so
    # serving and rollback are unaffected.
    try:
        from app.learning.head_registry import get_head_registry

        _reg = get_head_registry()
        if _reg is not None:
            keep = int(os.getenv("HEAD_REGISTRY_KEEP", "5"))
            dropped = _reg.prune(keep_per_relation=keep)
            print(f"[nightly] registry prune: dropped {len(dropped)} old versions (keep={keep}/relation)")
        else:
            print("[nightly] registry prune skipped: no registry configured")
    except Exception as e:
        print(f"[nightly] registry prune skipped: {e}")

    # Persist the grown log + retrained heads back to blob for the worker to load.
    # Timeout is generous on purpose. At 300s this silently killed the write-back
    # EVERY night for weeks: the upload was non-incremental over a ~5GB / 14k-file
    # append-only registry, so it could never finish, and every eval-gated
    # promotion was thrown away before it reached blob. The upload is incremental
    # now (seconds), and the job's own replicaTimeout (3600s) is the real ceiling.
    wb = "/write_back_ml.py"
    if os.path.exists(wb):
        try:
            done = subprocess.run(
                [sys.executable, wb],
                timeout=int(os.getenv("NIGHTLY_WRITEBACK_TIMEOUT_S", "1800")),
                check=False,
            )
            if done.returncode == 0:
                print("[nightly] persisted heads + log to blob")
            else:
                # Loud: a silent write-back failure means the whole run is lost.
                print(f"[nightly] WRITE-BACK FAILED rc={done.returncode} — this run was NOT persisted")
        except Exception as e:
            print(f"[nightly] WRITE-BACK FAILED: {e} — this run was NOT persisted")
    else:
        print("[nightly] /write_back_ml.py not present — skipping persist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
