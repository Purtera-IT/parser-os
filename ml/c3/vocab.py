"""Canonical entity keys for the C3 entity loss, from ``app/core/label_vocab.json``.

Lines naming the same entity pull together (``losses.entity_loss``), so
``org:x`` and ``party:x`` on two lines must be one key. Only the registry's
one-to-one renames apply; an off-list key is kept as written, never dropped.

Mirrors ``app.core.label_vocab.canonical_key`` (tests/test_label_vocab.py
checks the two agree) but reads the JSON by path, so this package never
imports ``app``.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

VOCAB_PATH = Path(__file__).resolve().parents[2] / "app" / "core" / "label_vocab.json"


@lru_cache(maxsize=1)
def _tags() -> tuple[dict[str, str], tuple[tuple[re.Pattern[str], str], ...], dict[str, str]]:
    tags = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))["entity_tags"]
    pats = tuple((re.compile(p["match"]), p["to"]) for p in tags.get("pattern_aliases") or [])
    return dict(tags["key_aliases"]), pats, dict(tags["prefix_aliases"])


def canonical_key(key: str) -> str:
    keys, pats, prefixes = _tags()
    k = str(key).strip()
    if k in keys:
        return keys[k]
    for rx, to in pats:
        if rx.match(k):
            return rx.sub(to, k)
    prefix, sep, rest = k.partition(":")
    if sep and prefix in prefixes:
        return canonical_key(f"{prefixes[prefix]}:{rest}")
    return k


def canonical_keys(keys: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for k in keys:
        c = canonical_key(str(k))
        if c and c not in out:
            out.append(c)
    return out
