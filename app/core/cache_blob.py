"""Blob mirror for the content-addressed caches, so a read survives the container.

The OCR cache and the embedding cache both live under ``/tmp/ml``, which
``fetch_ml.py`` DOWNLOADS at startup — and nothing ever uploaded back. So the
caches only ever held whatever was last put in blob by hand, every cold start
re-read every picture through Document Intelligence, and every compile
re-embedded lines it had embedded a hundred times before.

That is not only cost. A cache that is empty on Monday and warm on Tuesday
gives the same deal two different readings, which is the class of bug the whole
determinism pass existed to remove. Fixing it inside the parser and leaving it
at the container boundary finishes nothing.

Why per-batch blobs instead of uploading the sqlite file
--------------------------------------------------------
Up to three workers compile concurrently against one blob path. Uploading a
whole SQLite file is last-write-wins: one worker silently erases another's
entries. Each batch instead gets its own immutable blob and the merge folds
them all in, which is the same shape as :mod:`app.core.training_row_blob` and
:mod:`app.core.feedback_blob`, and is safe under any amount of concurrency.

Re-merging is idempotent: both caches are CONTENT-ADDRESSED, so the same key
always carries the same value and a re-import is a no-op rather than a
conflict. That is the property that makes this safe in a way a mutable cache
would not be.

Contract, mirroring the other two:

* **Gated**: no-op unless ``SOWSMITH_FEEDBACK_BLOB`` is truthy.
* **Offline-safe**: every failure is swallowed. Losing a cached read is bad;
  failing a customer's compile over one is worse.

Layout: ``<container>/_cache_rows/<kind>/<batch_id>.jsonl``.
"""
from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any, Iterable

_PREFIX = "_cache_rows/"
_TRUTHY = {"1", "true", "yes", "on"}

#: The two caches this mirrors. ``kind`` is the blob sub-prefix and the table.
OCR = "ocr"
EMBED = "embed"


def _enabled() -> bool:
    return os.getenv("SOWSMITH_FEEDBACK_BLOB", "").strip().lower() in _TRUTHY


def _container_client():
    """A blob ContainerClient, or None when disabled/unconfigured/offline."""
    if not _enabled():
        return None
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        return None
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception:
        return None
    container = (
        os.environ.get("SOWSMITH_FEEDBACK_BLOB_CONTAINER", "orbitbrief-artifacts").strip()
        or "orbitbrief-artifacts"
    )
    try:
        return BlobServiceClient.from_connection_string(conn).get_container_client(container)
    except Exception:
        return None


def rows_to_jsonl(rows: Iterable[dict]) -> str:
    """One JSON object per line. Bytes are base64'd so the line stays text."""
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("key"):
            continue
        rec = dict(row)
        blob = rec.get("vec")
        if isinstance(blob, (bytes, bytearray)):
            rec["vec"] = base64.b64encode(bytes(blob)).decode("ascii")
        try:
            out.append(json.dumps(rec, ensure_ascii=False))
        except Exception:
            continue
    return "\n".join(out)


def jsonl_to_rows(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict) or not rec.get("key"):
            continue
        if isinstance(rec.get("vec"), str):
            try:
                rec["vec"] = base64.b64decode(rec["vec"])
            except Exception:
                continue
        rows.append(rec)
    return rows


def upload_rows(kind: str, rows: list[dict]) -> bool:
    """Mirror one batch of cache entries to its own blob. Best-effort."""
    if not rows or kind not in (OCR, EMBED):
        return False
    cc = _container_client()
    if cc is None:
        return False
    try:
        payload = rows_to_jsonl(rows)
        if not payload:
            return False
        # Unique per batch: concurrent workers can never overwrite each other.
        name = f"{_PREFIX}{kind}/{uuid.uuid4().hex}.jsonl"
        cc.upload_blob(name=name, data=payload.encode("utf-8"), overwrite=True)
        return True
    except Exception:
        return False


def _iter_batches(cc, kind: str):
    for b in cc.list_blobs(name_starts_with=f"{_PREFIX}{kind}/"):
        if not b.name.endswith(".jsonl"):
            continue
        try:
            yield jsonl_to_rows(cc.download_blob(b.name).readall().decode("utf-8", "replace"))
        except Exception:
            continue


def sync_ocr_into_cache() -> int:
    """Merge every mirrored OCR batch into the local cache. Returns rows imported.

    Safe to run repeatedly: the key is ``sha256(version || image bytes)``, so a
    re-import writes the same text for the same picture.
    """
    cc = _container_client()
    if cc is None:
        return 0
    try:
        from app.core.ocr_cache import get_cache
    except Exception:
        return 0
    cache = get_cache()
    if cache is None:
        return 0
    n = 0
    for rows in _iter_batches(cc, OCR):
        for row in rows:
            text = str(row.get("text") or "")
            if not text.strip():
                # A miss is a statement about the day, not the picture, and it
                # is never stored -- so it is never restored either.
                continue
            try:
                cache.put_raw(
                    key=str(row["key"]),
                    text=text,
                    backend=str(row.get("backend") or ""),
                    confidence=float(row.get("confidence") or 0.0),
                    extra=row.get("extra"),
                )
                n += 1
            except Exception:
                continue
    return n


def export_ocr_rows(limit: int = 5000) -> list[dict]:
    """The local OCR cache as mirrorable rows."""
    try:
        from app.core.ocr_cache import get_cache
    except Exception:
        return []
    cache = get_cache()
    if cache is None:
        return []
    try:
        return cache.export_rows(limit=limit)
    except Exception:
        return []


def mirror_ocr(limit: int = 5000) -> bool:
    """Upload what this container has read. Called at the end of a compile."""
    rows = export_ocr_rows(limit=limit)
    return upload_rows(OCR, rows) if rows else False


# ---- embeddings -------------------------------------------------------------
#
# The larger of the two by volume: OCR sees a handful of pictures per deal,
# the embedder sees every candidate line of every rule. A warmed cache reaches
# gigabytes, so this mirrors only what the PROCESS computed rather than the
# whole file -- re-uploading a warm cache would cost more than the embeddings
# it saves.


def sync_embed_into_cache() -> int:
    """Merge every mirrored embedding batch into the local cache.

    Safe to run repeatedly: the key is sha256(model || text), so a restored
    vector is the one the endpoint would have returned. The model is IN the
    key, so swapping embed models can never hand back a stale vector of the
    wrong dimensionality -- it simply misses and re-embeds.
    """
    cc = _container_client()
    if cc is None:
        return 0
    try:
        from app.core.embedding_cache import get_cache
    except Exception:
        return 0
    cache = get_cache()
    if cache is None:
        return 0
    n = 0
    for rows in _iter_batches(cc, EMBED):
        try:
            n += cache.put_raw_rows(rows)
        except Exception:
            continue
    return n


def export_embed_rows(limit: int = 4000) -> list[dict]:
    """The vectors this process computed."""
    try:
        from app.core.embedding_cache import get_cache
    except Exception:
        return []
    cache = get_cache()
    if cache is None:
        return []
    try:
        return cache.export_fresh_rows(limit=limit)
    except Exception:
        return []


def mirror_embed(limit: int = 4000) -> bool:
    """Upload what this compile embedded. Called at the end of a compile."""
    rows = export_embed_rows(limit=limit)
    return upload_rows(EMBED, rows) if rows else False
