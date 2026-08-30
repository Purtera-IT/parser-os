"""Precomputed lifecycle labels, looked up by content hash.

Classifying a document means reading it with a model, which is neither free nor
deterministic and has no place in a compile. So the corpus was classified once,
offline, and the answers are stored here keyed by ``content_sha256`` -- the same
bytes always get the same label, and a compile pays a dict lookup.

Keys are the first 32 hex characters of the sha256. That is 128 bits of content
address; collisions are not a practical concern and the shorter key keeps the
file readable in review.

A document not in the dataset returns ``None``. It is not guessed at: unknown
documents quarantine, exactly like an unrecognised type.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA = Path(__file__).parent / "data" / "lifecycle_by_sha.json"
KEY_LEN = 32


@lru_cache(maxsize=1)
def _table() -> dict[str, dict[str, Any]]:
    try:
        with _DATA.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def lookup(sha256: str | None) -> dict[str, Any] | None:
    """Lifecycle record for these bytes, or None when we have never seen them."""
    if not sha256:
        return None
    return _table().get(str(sha256).strip().lower()[:KEY_LEN])


def coverage() -> int:
    """How many documents the dataset can label. Useful in tests and gates."""
    return len(_table())
