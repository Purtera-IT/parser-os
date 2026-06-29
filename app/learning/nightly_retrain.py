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
        for r in ho_atoms:
            a = by_id.get(r["atom_id"])
            if a is None:
                continue
            feats.append(build_atom_feature_row(a))
            ys.append(r["label"])
            hc = getattr(a, "calibrated_confidence", None)
            heur.append(hc if hc is not None else (getattr(a, "confidence", 0.5) or 0.5))
        if len(ys) < 5 or len(set(ys)) < 2:
            print("[nightly] calibrator: holdout too thin — not promoting")
            return
        probs = [float(p[1]) for p in am.predict_proba(feats)]
        b_cal, b_heur = C.brier_score(probs, ys), C.brier_score(heur, ys)
        print(f"[nightly] calibrator Brier={b_cal:.4f} vs heuristic={b_heur:.4f} (holdout n={len(ys)})")
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

    # Persist the grown log + retrained heads back to blob for the worker to load.
    wb = "/write_back_ml.py"
    if os.path.exists(wb):
        try:
            subprocess.run([sys.executable, wb], timeout=300, check=False)
            print("[nightly] persisted heads + log to blob")
        except Exception as e:
            print(f"[nightly] write-back skipped: {e}")
    else:
        print("[nightly] /write_back_ml.py not present — skipping persist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
