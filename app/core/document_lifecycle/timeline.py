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
from datetime import datetime, timezone
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
    """Verified state changes for a deal, oldest first. Empty when unknown.

    Copies, for the same reason ``dataset.lookup`` copies: the table is cached
    for the life of the process, and an event handed out by reference is an
    event a caller can edit for every later deal that reads it.
    """
    if not deal_id:
        return []
    stored = (_table().get(str(deal_id)) or {}).get("events") or []
    return [dict(e) if isinstance(e, dict) else e for e in stored]


def quote_asof(deal_id: str | None) -> str | None:
    """When this deal first committed to an answer, or None if it never did.

    None is a real answer, not a gap: a deal still in discovery has no cut, and
    everything on it is legitimately evidence.
    """
    if not deal_id:
        return None
    return (_table().get(str(deal_id)) or {}).get("quote_asof")


def parse_ts(value: str | None) -> datetime | None:
    """An ISO timestamp as a naive UTC datetime, or None if it is not one.

    The two sides of the comparison do not agree on format and never will: the
    timeline was extracted from HubSpot, which writes naive UTC with microseconds
    ("2026-05-22T19:35:16.654000"), while a document's delivery time comes off
    the mirrored message as "2026-08-18T18:26:11Z". Compared as STRINGS those two
    shapes are not ordered by time -- "...:11Z" sorts after "...:11" for the same
    instant, and an offset like "+00:00" sorts before every digit. So both sides
    are parsed and normalised to naive UTC before anything is compared.
    """
    if not value:
        return None
    raw = str(value).strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def delivered_at(lifecycle: dict[str, Any] | None) -> str | None:
    """When this document reached us: the EARLIEST message that carried it.

    Earliest, not latest, because a document that was sent again after the quote
    went out did not thereby become post-quote material -- we already had it. The
    later send is a re-send, and dating the document by it would silently drop
    real evidence.

    None when the document has no delivering message, which is 253 of the 1,114
    classified documents: those are dateless, and a dateless document is never
    ruled out by the cut.
    """
    if not isinstance(lifecycle, dict):
        return None
    stamps = [
        d.get("ts") for d in (lifecycle.get("delivered") or [])
        if isinstance(d, dict) and d.get("ts")
    ]
    parsed = [(parse_ts(s), s) for s in stamps]
    usable = [(dt, s) for dt, s in parsed if dt is not None]
    if not usable:
        return None
    return min(usable)[1]


def is_after_cut(deal_id: str | None, created_at: str | None) -> bool:
    """True when this artifact postdates the deal's commitment to an answer.

    False whenever the question cannot be answered -- no cut, no timestamp, or a
    timestamp that will not parse -- because an artifact is admissible until
    something proves otherwise. Ruling evidence OUT on a guess is the expensive
    direction of this error.
    """
    cut = parse_ts(quote_asof(deal_id))
    when = parse_ts(created_at)
    if cut is None or when is None:
        return False
    return when > cut


def coverage() -> tuple[int, int]:
    """(deals with a timeline, deals with a derived cut)."""
    t = _table()
    return len(t), sum(1 for v in t.values() if v.get("quote_asof"))
