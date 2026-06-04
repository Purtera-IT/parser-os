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
