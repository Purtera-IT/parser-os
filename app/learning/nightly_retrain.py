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
