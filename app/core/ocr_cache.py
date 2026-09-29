"""Persistent content-addressed OCR cache.

The text OCR returns for a picture BECOMES the atom's text, and atom text is
three quarters of ``label_key``. So a page that reads back slightly differently
produces a different key and a gold label silently stops attaching. That makes
OCR one of the two remaining sources of "the same deal parsed twice does not
agree" -- the other was the embedder, fixed by reading its cache before asking
whether its endpoint was up.

``_ocr_chain`` already memoises on ``sha256(image_bytes)``, but in a process
dict capped at 512 entries::

    _OCR_CACHE: dict[str, dict[str, Any]] = {}

The worker scales to zero with a 300s cooldown, so that dict dies with the
container and every cold start re-reads every picture from scratch -- a fresh
roll of the dice against a service whose version can move, or against a vision
LLM, which is not reproducible even at ``temperature: 0``.

Persisting it makes the answer a property of the IMAGE rather than of the day:
the same bytes return the same text across containers, rolls and months. Same
design as ``embedding_cache``, and for the same reason.

Two deliberate choices:

  version in the key   ``sha256(version || bytes)``. Bump ``OCR_CACHE_VERSION``
                       to re-read the corpus on purpose. An upgrade is then an
                       explicit decision with a diff, instead of a drift nobody
                       ordered.

  never cache a miss   An empty result means "no backend fired", which is
                       usually an outage, a missing binary or a throttle --
                       not a fact about the picture. Storing it would freeze a
                       bad afternoon into every future compile. Same rule
                       ``embed_texts`` follows for failed embeds.

Env:
  SOWSMITH_OCR_CACHE_DB       path to the sqlite file (default:
                              ~/.parseros/ocr_cache.db)
  SOWSMITH_OCR_CACHE_DISABLE  set to disable the cache entirely
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

#: Bump to invalidate every cached read -- after changing the backend order,
#: upgrading a vision model, or changing what the chain returns.
OCR_CACHE_VERSION = "v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr (
    key        TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    backend    TEXT NOT NULL,
    confidence REAL NOT NULL,
    extra      TEXT
)
"""


def _default_path() -> Path:
    return Path(os.path.expanduser("~")) / ".parseros" / "ocr_cache.db"


def _key(image_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(OCR_CACHE_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(image_bytes)
    return h.hexdigest()


class OcrCache:
    """Thread-safe sqlite-backed OCR result cache."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the parse pool reads this from worker
        # threads; the lock below serialises all access.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, image_bytes: bytes) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT text, backend, confidence, extra FROM ocr WHERE key = ?",
                (_key(image_bytes),),
            )
            row = cur.fetchone()
        if row is None:
            return None
        text, backend, confidence, extra = row
        out: dict[str, Any] = {
            "text": text,
            "backend": backend,
            "confidence": float(confidence),
            "notes": [],
        }
        if extra:
            try:
                out.update(json.loads(extra))
            except Exception:
                pass
        return out

    def put(self, image_bytes: bytes, result: dict[str, Any]) -> None:
        """Persist a result. Idempotent; a miss is never stored."""
        text = (result.get("text") or "").strip()
        if not text:
            # "Nothing fired" is a statement about the day, not the picture.
            return
        # `notes` are this call's diagnostics and belong to the caller, not to
        # the image; everything else the chain chose to return is kept.
        extra = {k: v for k, v in result.items()
                 if k not in {"text", "backend", "confidence", "notes"}}
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ocr (key, text, backend, confidence, extra) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _key(image_bytes),
                    result.get("text") or "",
                    str(result.get("backend") or ""),
                    float(result.get("confidence") or 0.0),
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
            self._conn.commit()

    def export_rows(self, limit: int = 5000) -> list[dict[str, Any]]:
        """This container's reads, as rows the blob mirror can carry.

        `fetch_ml.py` downloads this file at boot and nothing ever uploaded it
        back, so every cold start re-read every picture through a billed
        service and got a fresh chance to read it differently. See
        `app.core.cache_blob`.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, text, backend, confidence, extra FROM ocr LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        return [
            {"key": k, "text": t, "backend": b, "confidence": c, "extra": e}
            for (k, t, b, c, e) in rows
        ]

    def put_raw(self, *, key: str, text: str, backend: str = "",
                confidence: float = 0.0, extra: Any = None) -> None:
        """Insert a row by its KEY, for restoring a mirrored batch.

        `put` hashes the image bytes; a restore has the key and not the bytes,
        which is the whole point of a content-addressed cache -- the key is
        enough. An empty read is still never stored.
        """
        if not key or not (text or "").strip():
            return
        if extra is not None and not isinstance(extra, str):
            try:
                extra = json.dumps(extra, ensure_ascii=False)
            except Exception:
                extra = None
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ocr (key, text, backend, confidence, extra) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(key), text, str(backend or ""), float(confidence or 0.0), extra),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM ocr").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


_CACHE: OcrCache | None = None
_CACHE_INIT = False
_INIT_LOCK = threading.Lock()


def get_cache() -> OcrCache | None:
    """Lazily open the shared cache.

    Returns None when disabled or on any error, in which case the caller OCRs
    exactly as it does today -- a cache must never be the reason a parse fails.
    """
    global _CACHE, _CACHE_INIT
    if _CACHE_INIT:
        return _CACHE
    with _INIT_LOCK:
        if _CACHE_INIT:
            return _CACHE
        _CACHE_INIT = True
        if os.environ.get("SOWSMITH_OCR_CACHE_DISABLE"):
            _CACHE = None
            return None
        raw = os.environ.get("SOWSMITH_OCR_CACHE_DB")
        path = Path(raw) if raw else _default_path()
        try:
            _CACHE = OcrCache(path)
        except Exception:
            _CACHE = None
        return _CACHE


def reset_cache() -> None:
    """Drop the open handle so the next call re-reads the env (tests)."""
    global _CACHE, _CACHE_INIT
    with _INIT_LOCK:
        if _CACHE is not None:
            _CACHE.close()
        _CACHE = None
        _CACHE_INIT = False
