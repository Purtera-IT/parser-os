"""Persistent content-addressed embedding cache.

Every embedding the pipeline computes is keyed by ``sha256(model || text)`` and
persisted to a small sqlite file. A re-compile of the same (or overlapping)
corpus then skips the remote embedder entirely — turning the dominant
cold/repeat-compile cost (serial HTTP round-trips to a remote Ollama over
Tailscale) into a local disk read.

Design goals:
  * Universal — sits under ``embedding_retrieval.embed_texts`` so EVERY
    consumer (entity enrichment, typed classifier, feedback store, neural
    head, GNN) shares one cache.
  * Content-addressed + model-scoped — swapping the embed model can never
    return a stale vector of the wrong dimensionality.
  * Shippable — the cache file is a plain artifact. Warming it across the
    training deals and shipping it means a "cold" deal that reuses common
    phrasing is already warm. Point ``SOWSMITH_EMBED_CACHE_DB`` at the
    shipped base to reuse it.
  * Fail-open — any sqlite error degrades to "no cache" (returns vectors
    uncached); it must never break a compile.

Env:
  SOWSMITH_EMBED_CACHE_DB       path to the sqlite file (default:
                                ~/.parseros/embed_cache.db)
  SOWSMITH_EMBED_CACHE_DISABLE  set to disable the cache entirely
"""
from __future__ import annotations

import array
import hashlib
import os
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key  TEXT PRIMARY KEY,
    dim  INTEGER NOT NULL,
    vec  BLOB NOT NULL
)
"""


def _default_path() -> Path:
    return Path(os.path.expanduser("~")) / ".parseros" / "embed_cache.db"


def _key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(text.encode("utf-8", "ignore"))
    return h.hexdigest()


class EmbeddingCache:
    """Thread-safe sqlite-backed float32 vector cache."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        #: Keys this PROCESS computed, so the blob mirror uploads new work
        #: instead of re-sending the whole file. A warmed cache reaches
        #: gigabytes; re-uploading it every compile would cost more than the
        #: embeddings it saves. Restores deliberately do not land here --
        #: re-uploading what was just downloaded is the same waste wearing a
        #: different hat.
        self._fresh: set[str] = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: embed_texts may touch the cache from worker
        # threads; the lock below serializes all access.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get_many(self, model: str, texts: list[str]) -> list[list[float] | None]:
        """Return cached vectors aligned to ``texts`` (None where absent)."""
        keys = [_key(model, t) for t in texts]
        found: dict[str, list[float]] = {}
        with self._lock:
            # chunk the IN() query to stay under sqlite's variable limit
            uniq = list(dict.fromkeys(keys))
            for start in range(0, len(uniq), 500):
                chunk = uniq[start:start + 500]
                ph = ",".join("?" * len(chunk))
                cur = self._conn.execute(
                    f"SELECT key, dim, vec FROM embeddings WHERE key IN ({ph})",
                    chunk,
                )
                for k, dim, blob in cur.fetchall():
                    a = array.array("f")
                    a.frombytes(blob)
                    if len(a) == dim:
                        found[k] = list(a)
        return [found.get(k) for k in keys]

    def put_many(self, model: str, items: list[tuple[str, list[float]]]) -> None:
        """Persist (text, vector) pairs. Idempotent (content-addressed)."""
        if not items:
            return
        rows = []
        for text, vec in items:
            if not vec:
                continue
            a = array.array("f", vec)
            rows.append((_key(model, text), len(vec), a.tobytes()))
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()
            self._fresh.update(r[0] for r in rows)

    def export_fresh_rows(self, limit: int = 4000) -> list[dict]:
        """Vectors this process computed, as rows the blob mirror can carry.

        Only what is NEW. `fetch_ml.py` downloads this file at boot and
        nothing ever uploaded it back, so the cache held whatever was last
        put there by hand and every compile re-embedded lines it had embedded
        a hundred times before -- paying Azure twice and, when the endpoint
        blinked, reading the deal differently.
        """
        with self._lock:
            keys = list(self._fresh)[:int(limit)]
            if not keys:
                return []
            out: list[dict] = []
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                ph = ",".join("?" * len(chunk))
                cur = self._conn.execute(
                    f"SELECT key, dim, vec FROM embeddings WHERE key IN ({ph})", chunk
                )
                for k, dim, blob in cur.fetchall():
                    out.append({"key": k, "dim": int(dim), "vec": bytes(blob)})
            return out

    def put_raw_rows(self, rows: list[dict]) -> int:
        """Restore mirrored vectors by KEY. Returns how many landed.

        The key is sha256(model || text), so a restored vector is the same
        vector the endpoint would return -- which is what makes re-merging a
        no-op rather than a conflict. Restored keys are NOT marked fresh.
        """
        good = []
        for row in rows:
            key = str(row.get("key") or "")
            vec = row.get("vec")
            dim = row.get("dim")
            if not key or not isinstance(vec, (bytes, bytearray)) or not dim:
                continue
            good.append((key, int(dim), bytes(vec)))
        if not good:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
                good,
            )
            self._conn.commit()
        return len(good)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


_CACHE: EmbeddingCache | None = None
_CACHE_INIT = False
_INIT_LOCK = threading.Lock()


def get_cache() -> EmbeddingCache | None:
    """Lazily open the shared cache. Returns None when disabled or on error
    (caller then embeds uncached). The opened path is remembered so changing
    SOWSMITH_EMBED_CACHE_DB at runtime (tests) requires reset_cache()."""
    global _CACHE, _CACHE_INIT
    if _CACHE_INIT:
        return _CACHE
    with _INIT_LOCK:
        if _CACHE_INIT:
            return _CACHE
        _CACHE_INIT = True
        if os.environ.get("SOWSMITH_EMBED_CACHE_DISABLE"):
            _CACHE = None
            return None
        raw = os.environ.get("SOWSMITH_EMBED_CACHE_DB")
        path = Path(raw) if raw else _default_path()
        try:
            _CACHE = EmbeddingCache(path)
        except Exception:
            _CACHE = None
        return _CACHE


def reset_cache() -> None:
    """Drop the cached singleton (re-reads env on next get_cache). For tests."""
    global _CACHE, _CACHE_INIT
    with _INIT_LOCK:
        if _CACHE is not None:
            _CACHE.close()
        _CACHE = None
        _CACHE_INIT = False
