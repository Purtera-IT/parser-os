"""The super-heads registry as data: which heads exist, base or Purtera layer,
whether each is ready to decide, what it replaces and its fallback.
Regenerate with ``python -m ml.c3.heads_registry`` after a retrain or a
labeling round; parser stages ask here before trusting a head."""
from __future__ import annotations

import functools
import json
from pathlib import Path

#: Statuses whose answers a stage may act on (others run in shadow at most).
TRUSTED = ("live", "ready")


@functools.lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    path = Path(__file__).with_name("heads.json")
    try:
        heads = json.loads(path.read_text(encoding="utf-8"))["heads"]
    except Exception:
        return {}
    return {h["key"]: h for h in heads}


def head(key: str) -> dict | None:
    return registry().get(key)


def trusted(key: str) -> bool:
    """True when the head beat guessing on the unseen holdout (or is live)."""
    h = head(key)
    return bool(h) and h["status"] in TRUSTED


def layer(key: str) -> str | None:
    h = head(key)
    return h["layer"] if h else None


def by_layer(name: str) -> list[dict]:
    return [h for h in registry().values() if h["layer"] == name]
