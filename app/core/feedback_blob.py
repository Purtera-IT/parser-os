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
    """Load any blob-mirrored corrections NOT already in ``store``. Returns the
    number newly added. Best-effort; one cheap list call + a download per *new*
    correction only."""
    cc = _container_client()
    if cc is None:
        return 0
    try:
        from app.core.feedback_store import Correction
    except Exception:
        return 0
    try:
        existing = {c.id for c in store.all_corrections(active_only=False)}
    except Exception:
        existing = set()
    added = 0
    try:
        for b in cc.list_blobs(name_starts_with=_PREFIX):
            cid = b.name[len(_PREFIX):]
            if cid.endswith(".json"):
                cid = cid[: -len(".json")]
            if cid in existing:
                continue
            try:
                raw = cc.download_blob(b.name).readall()
                store.add(Correction(**json.loads(raw)))
                added += 1
            except Exception:
                continue
    except Exception:
        return added
    return added
