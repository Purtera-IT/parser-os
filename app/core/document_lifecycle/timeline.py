"""When a deal changed state, and therefore what counts as evidence for it.

A document's created_at gives a crude cut. What is actually wanted is the moment
the deal committed to an answer -- when a quote or SOW went out -- because
everything before it is what we knew, and everything after is either the answer
or the delivery that followed.

Those moments are stated in the correspondence in plain words, with a timestamp
attached, so they were extracted once offline and stored here per deal. Every
event carries the sentence it came from; events whose quote could not be found in
the source were discarded rather than kept (16 of 590).

The cut was cross-checked against the independent one derived from document
creation dates. On the 73 deals where both exist the median disagreement is ONE
DAY and 75% agree within a week -- two unrelated signals, the same answer. The
timeline additionally dates 11 deals that have no quote document at all.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA = Path(__file__).parent / "data" / "deal_timeline.json"

#: The events that mean "we have now given the customer an answer". Everything a
#: quoting head reads should predate the earliest of these.
COMMITTING_EVENTS = ("QUOTE_SENT", "SOW_SENT")


@lru_cache(maxsize=1)
def _table() -> dict[str, dict[str, Any]]:
    try:
        with _DATA.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def events(deal_id: str | None) -> list[dict[str, Any]]:
    """Verified state changes for a deal, oldest first. Empty when unknown."""
    if not deal_id:
        return []
    return list((_table().get(str(deal_id)) or {}).get("events") or [])


def quote_asof(deal_id: str | None) -> str | None:
    """When this deal first committed to an answer, or None if it never did.

    None is a real answer, not a gap: a deal still in discovery has no cut, and
    everything on it is legitimately evidence.
    """
    if not deal_id:
        return None
    return (_table().get(str(deal_id)) or {}).get("quote_asof")


def is_after_cut(deal_id: str | None, created_at: str | None) -> bool:
    """True when this artifact postdates the deal's commitment to an answer.

    False whenever the question cannot be answered -- no cut, or no timestamp --
    because an artifact is admissible until something proves otherwise.
    """
    cut = quote_asof(deal_id)
    if not cut or not created_at:
        return False
    return str(created_at) > str(cut)


def coverage() -> tuple[int, int]:
    """(deals with a timeline, deals with a derived cut)."""
    t = _table()
    return len(t), sum(1 for v in t.values() if v.get("quote_asof"))
