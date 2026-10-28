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
VISION = "vision"


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
    if not rows or kind not in (OCR, EMBED, VISION):
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


#: Batches this process already merged, by blob name -> etag. Every compile
#: used to download and re-merge EVERY mirrored batch: 1.7 GB of embedding
#: rows and 76 MB of OCR rows on dev (2026-10-15), ~100 s before a compile's
#: first stage, growing with every compile. A batch is now merged once per
#: local cache; a changed batch (new etag) is merged again.
_MERGED: dict[str, str] = {}


def _seen_path() -> str:
    """Beside the local caches, so a wiped cache (a new container's /tmp)
    wipes the record with it and everything is merged again."""
    db = os.environ.get("SOWSMITH_EMBED_CACHE_DB", "").strip() or "/tmp/ml/_embed_cache.db"
    return os.path.join(os.path.dirname(db) or ".", "_cache_rows_merged.json")


def _load_seen() -> dict[str, str]:
    if not _MERGED:
        try:
            with open(_seen_path(), encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _MERGED.update({str(k): str(v) for k, v in data.items()})
        except Exception:
            pass
    return _MERGED


def _save_seen() -> None:
    try:
        path = _seen_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_MERGED, f)
        os.replace(tmp, path)
    except Exception:
        pass


#: Batches downloaded at once while merging. The merge applies them in listing
#: order regardless (a later row for a key wins, exactly as one after another);
#: only the network round trips overlap. 1,170 OCR batches one at a time were
#: most of a cold container's ~3 minute merge (dev, 2026-10-28).
_FETCH_AHEAD = max(1, int(os.environ.get("SOWSMITH_CACHE_SYNC_FETCH", "16") or 16))


def _iter_batches(cc, kind: str, raw: bool = False):
    """Each mirrored batch not merged yet, in listing order: its rows (or, with
    ``raw``, its JSONL text). Downloads run up to ``_FETCH_AHEAD`` ahead."""
    import collections
    import concurrent.futures

    seen = _load_seen()
    merged_any = False
    todo: list[tuple[str, str]] = []
    for b in cc.list_blobs(name_starts_with=f"{_PREFIX}{kind}/"):
        if not b.name.endswith(".jsonl"):
            continue
        tag = str(getattr(b, "etag", "") or getattr(b, "last_modified", "") or "")
        if tag and seen.get(b.name) == tag:
            continue
        todo.append((b.name, tag))

    def _fetch(name: str):
        try:
            return cc.download_blob(name).readall().decode("utf-8", "replace")
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=_FETCH_AHEAD,
                                               thread_name_prefix=f"cache-{kind}") as pool:
        window: collections.deque = collections.deque()
        it = iter(todo)
        for item in it:
            window.append((item, pool.submit(_fetch, item[0])))
            if len(window) >= _FETCH_AHEAD:
                break
        while window:
            (name, tag), fut = window.popleft()
            nxt = next(it, None)
            if nxt is not None:
                window.append((nxt, pool.submit(_fetch, nxt[0])))
            text = fut.result()
            if text is None:
                continue
            yield text if raw else jsonl_to_rows(text)
            # Marked only after the caller merged the rows (the generator resumes
            # here), so a batch that failed part way is fetched again next time.
            if tag:
                seen[name] = tag
                merged_any = True
    if merged_any:
        _save_seen()


#: A mirrored row's key, read without parsing the row (rows_to_jsonl writes it first).
_KEY_AT_START = __import__("re").compile(r'^\{"key": "([^"\\]+)"')


def sync_ocr_into_cache() -> int:
    """Merge every mirrored OCR batch into the local cache. Returns rows written.

    Safe to run repeatedly: the key is ``sha256(version || image bytes)``, so a
    re-import writes the same text for the same picture. Each batch is written
    in one transaction, in order, and a row identical to the one this merge
    last wrote for its key is skipped: writing it again changes nothing. The
    mirror held each picture's row ~1,000 times over (whole-cache re-exports),
    and writing every copy with its own commit was most of the merge.
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
    last: dict[str, str] = {}   # key -> the exact line this merge last wrote for it
    for text in _iter_batches(cc, OCR, raw=True):
        rows: list[dict] = []
        wrote: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _KEY_AT_START.match(line)
            if m and m.group(1) not in wrote and last.get(m.group(1)) == line:
                continue
            for row in jsonl_to_rows(line):
                if not str(row.get("text") or "").strip():
                    # A miss is a statement about the day, not the picture, and it
                    # is never stored -- so it is never restored either.
                    continue
                wrote[str(row["key"])] = line
                rows.append(row)
        try:
            n += cache.put_raw_many(rows)
        except Exception:
            continue
        # Only what reached the cache can be skipped later.
        last.update(wrote)
    return n


def export_ocr_rows(limit: int = 5000) -> list[dict]:
    """The reads this process paid for, as mirrorable rows (never the restored
    ones: re-uploading the whole cache every compile is what grew the mirror)."""
    try:
        from app.core.ocr_cache import get_cache
    except Exception:
        return []
    cache = get_cache()
    if cache is None:
        return []
    try:
        return cache.export_fresh_rows(limit=limit)
    except Exception:
        return []


def mirror_ocr(limit: int = 5000) -> bool:
    """Upload what this container has read since its last upload. Called at the end of a compile."""
    rows = export_ocr_rows(limit=limit)
    ok = upload_rows(OCR, rows) if rows else False
    if ok:
        _mark_mirrored("ocr_cache", rows)
    return ok


def _mark_mirrored(module: str, rows: list[dict]) -> None:
    """Uploaded rows stop being fresh, so the next compile on this container
    sends only what it adds (a warm worker re-sent every earlier compile's rows)."""
    try:
        import importlib

        cache = importlib.import_module(f"app.core.{module}").get_cache()
        if cache is not None:
            cache.mark_mirrored([str(r.get("key")) for r in rows if r.get("key")])
    except Exception:
        pass


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


_EMBED_SYNC: dict[str, Any] = {"thread": None}
_OCR_SYNC: dict[str, Any] = {"thread": None}
#: Set whenever no OCR merge is running: an OCR lookup waits on it, so a
#: picture is never re-read through the billed service because its cached
#: text had not been merged yet, and a deal with no pictures never waits.
OCR_MERGED = __import__("threading").Event()
OCR_MERGED.set()


def sync_ocr_into_cache_background() -> bool:
    """``sync_ocr_into_cache`` on a background thread; ``OcrCache.get`` waits
    for it (``wait_for_ocr_merge``). Returns True when a merge was started."""
    import threading

    t = _OCR_SYNC.get("thread")
    if t is not None and t.is_alive():
        return False
    OCR_MERGED.clear()

    def _run():
        try:
            sync_ocr_into_cache()
        except Exception:
            pass
        finally:
            OCR_MERGED.set()

    t = threading.Thread(target=_run, daemon=True, name="ocr-cache-sync")
    _OCR_SYNC["thread"] = t
    t.start()
    return True


def wait_for_ocr_merge(timeout: float = 120.0) -> bool:
    """Block until no OCR merge is running (at most ``timeout`` seconds)."""
    return OCR_MERGED.wait(timeout)


def sync_embed_into_cache_background() -> bool:
    """``sync_embed_into_cache`` on a background thread, so a compile never
    waits for it: a vector not merged yet is embedded again (the same vector
    from the same model), and one merged in time is a cache hit. After a
    container starts, the first merge is the whole mirror (1.7 GB on dev);
    after that only new batches. Returns True when a sync was started; False
    when one is still running."""
    import threading

    t = _EMBED_SYNC.get("thread")
    if t is not None and t.is_alive():
        return False

    def _run():
        try:
            sync_embed_into_cache()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True, name="embed-cache-sync")
    _EMBED_SYNC["thread"] = t
    t.start()
    return True


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
    ok = upload_rows(EMBED, rows) if rows else False
    if ok:
        _mark_mirrored("embedding_cache", rows)
    return ok


# ---- vision replies ----------------------------------------------------------
#
# The vision LLM's description of a picture, keyed by sha256(model, prompt,
# mime, image). Unset SOWSMITH_VISION_CACHE_DB meant no cache at all, and the
# worker scales from zero, so every compile asked again about every picture:
# 42.8 s of a warm 010043 compile (2026-10-16). Same immutable-batch mirror as
# OCR; only replies this process paid for are uploaded.


def sync_vision_into_cache() -> int:
    cc = _container_client()
    if cc is None:
        return 0
    try:
        from app.core import llm_client
    except Exception:
        return 0
    conn = llm_client._vision_cache_conn()
    if conn is None:
        return 0
    n = 0
    for rows in _iter_batches(cc, VISION):
        good = [(str(r["key"]), str(r.get("model") or ""), str(r.get("reply") or ""))
                for r in rows if r.get("key") and str(r.get("reply") or "").strip()]
        if not good:
            continue
        try:
            with llm_client._VCACHE_LOCK:
                conn.executemany("INSERT OR IGNORE INTO vision_cache (k, model, reply) VALUES (?,?,?)", good)
                conn.commit()
            n += len(good)
        except Exception:
            continue
    return n


def sync_vision_into_cache_background() -> bool:
    """``sync_vision_into_cache`` on a thread; a vision lookup waits for it."""
    import threading

    try:
        from app.core import llm_client
    except Exception:
        return False
    if llm_client._vision_cache_conn() is None or not _enabled():
        return False
    llm_client.VCACHE_MERGED.clear()

    def _run():
        try:
            sync_vision_into_cache()
        except Exception:
            pass
        finally:
            llm_client.VCACHE_MERGED.set()

    threading.Thread(target=_run, daemon=True, name="vision-cache-sync").start()
    return True


def mirror_vision() -> bool:
    """Upload the vision replies this process stored. Called at compile end."""
    try:
        from app.core import llm_client
    except Exception:
        return False
    rows = list(llm_client.VCACHE_NEW)
    if not rows:
        return False
    ok = upload_rows(VISION, rows)
    if ok:
        del llm_client.VCACHE_NEW[:len(rows)]
    return ok
