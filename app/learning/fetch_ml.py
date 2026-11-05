"""What the ML artifact dir actually holds, as a receipt the compile can report.

``ml_capabilities._artifacts_loaded()`` has always called
``app.learning.fetch_ml.artifact_self_check()`` inside a bare ``try/except`` —
and this module did not exist, so the import failed every time and the manifest
recorded ``{}``. The envelope therefore could not say whether the heads had
anything to load, which is how a worker that fetched no artifacts produced a
normal-looking envelope with no ``c3_brief`` in it and nobody noticed.

The worker's ``fetch_ml.py`` now writes ``_fetch_ml_receipt.json`` at boot and
refuses to start when a configured artifact is missing. This reads that receipt
and re-checks the paths at COMPILE time (a boot-time pass does not prove the
files survived a /tmp scrub or a disk-full eviction).

Guess-free: with no receipt the state is ``"unknown"``, never ``"ok"``.
"""
from __future__ import annotations

import json
import os
from typing import Any

#: Paths under the ML dir the runtime CREATES rather than fetches; absent is normal.
_RUNTIME_CREATED = ("cache", "_shadow_history", "_feedback", "_rule_decisions")


def _ml_dir() -> str:
    return os.environ.get("SOWSMITH_ML_ARTIFACT_DIR") or os.environ.get("ML_ARTIFACT_DIR") or "/tmp/ml"


def configured_artifacts() -> dict[str, str]:
    """Artifact paths this process is configured to load out of the ML dir.

    Read from the environment, so a head added to the container config is covered
    the day it is added. Mirrors the worker's ``_required_paths``; the head
    registry is excluded because fetch_ml deliberately skips it unless
    ``FETCH_ML_INCLUDE_REGISTRY=1``, and runtime-created caches are excluded
    because they are written, not fetched.
    """
    out: dict[str, str] = {}
    try:
        base = os.path.realpath(_ml_dir())
    except OSError:
        return out
    for key, val in os.environ.items():
        if not key.startswith("SOWSMITH_") or not str(val).strip():
            continue
        if not (key.endswith("_DIR") or key.endswith("_DB") or key.endswith("_PATH")):
            continue
        p = str(val).strip()
        try:
            rp = os.path.realpath(p)
        except OSError:
            continue
        if rp == base or not rp.startswith(base + os.sep):
            continue
        low = os.path.basename(p).lower() + "|" + p.lower()
        if any(m in low for m in _RUNTIME_CREATED) or "_head_registry" in low:
            continue
        out[key] = p
    return out


def artifact_self_check() -> dict[str, Any]:
    """Did the heads have anything to load, and does it still exist?

    Returns, for ``compile_capabilities.artifacts_loaded``:
      ``artifacts``        "ok" | "missing" | "unknown"
      ``artifacts_missing``  configured-but-absent env keys (only when missing)
      ``artifacts_present``  how many configured artifacts are on disk
      ``fetch_ml``         "ok" | "failed" | "absent"  (what the boot receipt said)
      ``fetch_ml_error``   the boot failure, when there was one
    """
    out: dict[str, Any] = {}
    receipt_path = os.path.join(_ml_dir(), "_fetch_ml_receipt.json")
    receipt: dict[str, Any] = {}
    try:
        with open(receipt_path) as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            receipt = loaded
    except (OSError, ValueError):
        receipt = {}

    if not receipt:
        out["fetch_ml"] = "absent"
    elif receipt.get("ok") is True:
        out["fetch_ml"] = "ok"
    else:
        out["fetch_ml"] = "failed"
        err = receipt.get("error")
        if err:
            out["fetch_ml_error"] = str(err)[:300]
    if isinstance(receipt.get("downloaded"), int):
        out["fetch_ml_files"] = receipt["downloaded"]

    configured = configured_artifacts()
    if not configured:
        # Nothing is configured to load from the ML dir: there is no claim to check.
        out["artifacts"] = "unknown" if out["fetch_ml"] == "absent" else out["fetch_ml"]
        return out
    missing = sorted(k for k, p in configured.items() if not os.path.exists(p))
    out["artifacts_present"] = len(configured) - len(missing)
    out["artifacts_configured"] = len(configured)
    if missing:
        out["artifacts"] = "missing"
        out["artifacts_missing"] = missing
    else:
        out["artifacts"] = "ok"
    return out
