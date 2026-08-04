"""Blob mirror for training rows, so what a compile learns survives the container.

The worker logs training rows to ``SOWSMITH_TRAINING_LOG_DB``, which lives under
``/tmp/ml`` — ephemeral. ``entrypoint.sh`` calls ``fetch_ml.py`` to DOWNLOAD that
log at startup but nothing ever uploaded it back, so every row every compile
produced died with the container. The blob log stayed at 3 rows (only the PM
corrections the nightly imports), which is why the eval gate reported
``hold_not_ready 0.00/0.00`` for every relation: not because candidates lost, but
because there was no holdout to score them against, so nothing could ever be
promoted.

Why per-batch blobs instead of uploading the DB
-----------------------------------------------
Several workers compile concurrently against the same blob path. Uploading the
whole SQLite file is last-write-wins — one worker silently erases another's
rows. Each batch instead gets its own immutable blob and the nightly merges them
all, which is the same shape as the correction mirror in
:mod:`app.core.feedback_blob` and is safe under any amount of concurrency.

Re-merging is idempotent: ``TrainingRow.id`` is the table's PRIMARY KEY and
``add_many`` uses INSERT OR REPLACE, so importing the same blob every night is a
no-op rather than a duplicate.

Contract mirrors :mod:`app.core.feedback_blob`:

* **Gated**: no-op unless ``SOWSMITH_FEEDBACK_BLOB`` is truthy.
* **Offline-safe**: every failure is swallowed — mirroring must NEVER break a
  compile. Losing a row is bad; failing a customer's compile is worse.

Layout: ``<container>/_training_rows/<batch_id>.jsonl`` (one JSON row per line).
"""
from __future__ import annotations

import dataclasses
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.core.training_log import TrainingLog, TrainingRow

_PREFIX = "_training_rows/"
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


def rows_to_jsonl(rows: Iterable["TrainingRow"]) -> str:
    """Serialize rows one-per-line. ``provenance`` stays a dict (not the
    JSON-string form ``to_row`` produces) so the reload is a clean round-trip."""
    out: list[str] = []
    for r in rows:
        try:
            out.append(json.dumps(dataclasses.asdict(r), ensure_ascii=False))
        except Exception:
            continue
    return "\n".join(out)


def jsonl_to_rows(text: str) -> list["TrainingRow"]:
    """Parse rows back, skipping malformed lines.

    Tolerant on purpose: one bad line must not cost the whole batch.
    """
    from app.core.training_log import TrainingRow

    fields = {f.name for f in dataclasses.fields(TrainingRow)}
    rows: list[TrainingRow] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        # Drop unknown keys so an older worker's blob cannot crash a newer
        # nightly on a field that has since been renamed.
        kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in fields}
        if not kwargs.get("relation") or not kwargs.get("label"):
            continue
        try:
            rows.append(TrainingRow(**kwargs))
        except Exception:
            continue
    return rows


def upload_rows(rows: list["TrainingRow"]) -> bool:
    """Mirror one batch to its own blob. Best-effort → returns success."""
    if not rows:
        return False
    cc = _container_client()
    if cc is None:
        return False
    try:
        payload = rows_to_jsonl(rows)
        if not payload:
            return False
        # Unique per batch: concurrent workers can never overwrite each other.
        name = f"{_PREFIX}{uuid.uuid4().hex}.jsonl"
        cc.upload_blob(name=name, data=payload.encode("utf-8"), overwrite=True)
        return True
    except Exception:
        return False


def sync_into_log(log: "TrainingLog" | None = None) -> int:
    """Merge every mirrored batch into the training log. Returns rows imported.

    Safe to run repeatedly — ids are the primary key and add_many upserts.
    """
    cc = _container_client()
    if cc is None:
        return 0
    if log is None:
        try:
            from app.core.training_log import get_training_log

            log = get_training_log()
        except Exception:
            return 0
    if log is None:
        return 0
    imported = 0
    try:
        blobs = list(cc.list_blobs(name_starts_with=_PREFIX))
    except Exception:
        return 0
    for b in blobs:
        try:
            raw = cc.download_blob(b.name).readall()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            rows = jsonl_to_rows(raw)
            if rows:
                imported += log.add_many(rows)
        except Exception:
            # One unreadable batch must not stop the rest.
            continue
    return imported
