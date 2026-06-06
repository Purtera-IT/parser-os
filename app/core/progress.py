"""Lightweight, dependency-optional progress bars for long compile stages.

The compiler's telemetry only logs at stage *end*, so a slow stage is a black
box while it runs. ``track()`` wraps the hot loops (dedup, the entity-extract
and canonicalize LLM pools) in a tqdm bar so an operator can see exactly where
a multi-minute stage is and how fast it's moving.

Design constraints:
* **Never a hard dependency.** If tqdm isn't installed we silently fall back to
  a plain iterator — the pipeline must run identically without it.
* **Off-switchable.** ``SOWSMITH_PROGRESS=0`` disables all bars (e.g. for clean
  CI logs or when stderr is captured to a file).
* **No spam.** ``min_total`` suppresses a bar for tiny iterables so per-artifact
  loops over a handful of atoms don't each draw (and tear down) a bar.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Iterator, TypeVar

_T = TypeVar("_T")

try:  # tqdm is optional — fall back to a no-op iterator if absent.
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is missing
    _tqdm = None


def _enabled() -> bool:
    return os.environ.get("SOWSMITH_PROGRESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def track(
    iterable: Iterable[_T],
    *,
    desc: str,
    total: int | None = None,
    min_total: int = 1,
) -> Iterator[_T]:
    """Yield from ``iterable``, drawing a tqdm bar when it's worth it.

    Falls back to a plain iterator when tqdm is unavailable, progress is
    disabled, or ``total`` is below ``min_total``. The returned object is always
    a normal iterator, so a wrapped ``concurrent.futures.as_completed`` generator
    still propagates its ``TimeoutError`` exactly as before.
    """
    if _tqdm is None or not _enabled():
        return iter(iterable)
    if total is not None and total < min_total:
        return iter(iterable)
    # disable=None → tqdm auto-disables when stderr is NOT a real terminal
    # (e.g. piped to Tee-Object / redirected to a file). In a pipe the in-place
    # \r bar can't redraw, so every update just appends — and several parallel
    # bars stomp on each other and on the JSON stage banners. Auto-disabling
    # keeps a piped/tee'd log to clean one-line-per-event (stage banners +
    # heartbeats), while a real interactive terminal still gets live bars.
    # ascii=True draws the bar with plain '#' so it never mojibakes on a
    # non-UTF-8 console code page.
    return iter(
        _tqdm(
            iterable,
            desc=desc,
            total=total,
            file=sys.stderr,
            disable=None,
            ascii=True,
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            smoothing=0.1,
        )
    )
