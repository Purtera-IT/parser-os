"""Optional blob-mirror for the :class:`~app.core.feedback_store.FeedbackStore`
so PM corrections written by the SERVICE container reach the WORKER container
(which runs ``decide()`` during compile) and survive container recycles.

Why this exists
---------------
The ``FeedbackStore`` is per-process SQLite. In dev the service writes
corrections (PM chip → ``/feedback/correction``) into its own ``/tmp`` DB while
the worker reads a *separate* ``/tmp`` DB during compile — so a fix returns 200
but never reaches the model and is lost on recycle. A shared SQLite over an
Azure Files (SMB) mount corrupts (byte-range locking), so instead each
correction is mirrored to blob as a small JSON object and the worker loads new
ones at the start of every compile.

Contract
--------
* **Gated**: no-op unless ``SOWSMITH_FEEDBACK_BLOB`` is truthy.
* **Offline-safe**: any failure (missing dep, no conn string, network) is
  swallowed — mirroring must NEVER break a compile or a correction.
* **Idempotent**: one blob per correction id; ``store.add`` upserts by id and
  the loader skips ids already present, so a re-run is a no-op.

Layout: ``<container>/_feedback/corrections/<id>.json`` (one object/correction).
Container defaults to ``orbitbrief-artifacts`` (already readable+writable by
both containers); override with ``SOWSMITH_FEEDBACK_BLOB_CONTAINER``.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.core.feedback_store import Correction, FeedbackStore

_PREFIX = "_feedback/corrections/"
_TRUTHY = {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return os.getenv("SOWSMITH_FEEDBACK_BLOB", "").strip().lower() in _TRUTHY


def _container_client():
    """A blob ContainerClient, or ``None`` when disabled/unconfigured/offline."""
    if not _enabled():
        return None
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        return None
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception:
        return None
    container = os.environ.get(
        "SOWSMITH_FEEDBACK_BLOB_CONTAINER", "orbitbrief-artifacts"
    ).strip() or "orbitbrief-artifacts"
    try:
        svc = BlobServiceClient.from_connection_string(conn)
        return svc.get_container_client(container)
    except Exception:
        return None


def upload_correction(corr: "Correction") -> bool:
    """Mirror one correction to blob (overwrite). Best-effort → returns success."""
    cc = _container_client()
    if cc is None:
        return False
    try:
        data = json.dumps(dataclasses.asdict(corr)).encode("utf-8")
        cc.upload_blob(name=f"{_PREFIX}{corr.id}.json", data=data, overwrite=True)
        return True
    except Exception:
        return False


def sync_into_store(store: "FeedbackStore") -> int:
    """Bring ``store`` up to date with the blob-mirrored corrections. Returns the
    number added or refreshed. Best-effort; one list call, plus a download per
    correction that is new or whose blob copy is newer than the local one.

    New ids alone were not enough. A PM correction that repeats a judgment is
    MERGED into the existing correction (same id, more exemplars), so the id was
    already present and the update was skipped: on dev the worker kept the
    11-exemplar `task` correction it loaded after a roll while blob held 12, and
    000036's request -- the twelfth, taught from its Deal Kit -- scored 0.28
    against a copy that did not contain it. A merged or retired correction now
    reaches every process that already holds an older copy.
    """
    cc = _container_client()
    if cc is None:
        return 0
    try:
        from app.core.feedback_store import Correction
    except Exception:
        return 0
    try:
        local = {c.id: float(getattr(c, "updated_at", 0.0) or 0.0)
                 for c in store.all_corrections(active_only=False)}
    except Exception:
        local = {}
    changed = 0
    try:
        for b in cc.list_blobs(name_starts_with=_PREFIX):
            cid = b.name[len(_PREFIX):]
            if cid.endswith(".json"):
                cid = cid[: -len(".json")]
            if cid in local:
                modified = getattr(b, "last_modified", None)
                try:
                    blob_ts = modified.timestamp() if modified is not None else None
                except Exception:
                    blob_ts = None
                # Upload happens after updated_at is stamped, so an unchanged
                # correction's blob is never meaningfully newer than its row.
                if blob_ts is not None and blob_ts <= local[cid] + 1.0:
                    continue
            try:
                raw = cc.download_blob(b.name).readall()
                incoming = Correction(**json.loads(raw))
            except Exception:
                continue
            if cid in local and float(getattr(incoming, "updated_at", 0.0) or 0.0) <= local[cid]:
                continue
            try:
                store.add(incoming)
                changed += 1
            except Exception:
                continue
    except Exception:
        return changed
    return changed


# ── Training rows (gold) ─────────────────────────────────────────────────
# PM corrections also write a gold TrainingRow for the nightly retrain. The
# retrain runs on the WORKER reading its TrainingLog, while corrections are
# written on the SERVICE — so mirror the rows to blob too and let the worker's
# retrain import them before training.
_TRAIN_PREFIX = "_feedback/training_rows/"


def upload_training_rows(correction_id: str, rows) -> bool:
    """Mirror the gold TrainingRows for one correction to blob. Best-effort."""
    cc = _container_client()
    if cc is None or not rows:
        return False
    try:
        payload = [dataclasses.asdict(r) for r in rows]
        data = json.dumps(payload).encode("utf-8")
        cc.upload_blob(name=f"{_TRAIN_PREFIX}{correction_id}.json", data=data, overwrite=True)
        return True
    except Exception:
        return False


def sync_training_rows_into_log(log) -> int:
    """Import blob-mirrored PM TrainingRows into ``log`` (the worker's
    TrainingLog) so the nightly retrain learns from service-written corrections.
    add_many is INSERT OR REPLACE, so re-imports are idempotent. Best-effort."""
    cc = _container_client()
    if cc is None:
        return 0
    try:
        from app.core.training_log import TrainingRow
    except Exception:
        return 0
    added = 0
    try:
        for b in cc.list_blobs(name_starts_with=_TRAIN_PREFIX):
            try:
                raw = cc.download_blob(b.name).readall()
                rows = [TrainingRow(**d) for d in json.loads(raw)]
                added += log.add_many(rows)
            except Exception:
                continue
    except Exception:
        return added
    return added
