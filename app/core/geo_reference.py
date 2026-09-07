"""US place reference — look a location up instead of guessing at its spelling.

Every heuristic this replaces was a word list. "Is ``St. Louis`` a city or a
street?" was answered by asking whether the token ``st`` appeared, which gets
``Broad St`` right and ``St. Louis`` wrong, and the fix for that is another
list, and the fix for *that* is another list. There is no vocabulary that
converges, because the question is not about vocabulary: it is about whether a
real place exists with that name.

So we carry the places. ``app/data/us_postal_places.tsv.gz`` is derived from
the GeoNames postal export (CC BY 4.0), 40,979 postal codes over 29,542
distinct city/state pairs, 247KB on disk. It answers three things the parsers
previously guessed at:

  * **check** — is ``(city, state)`` a real place? (``Suite 400`` is not)
  * **fill** — a ZIP determines its city and state outright
  * **complete** — a city that exists in exactly one state names that state

Offline, deterministic, no network and no API key: a compile must not depend
on a third party being up, and a geocoder that returns a confident pin for a
street with no locality is worse than an honest blank.

Degrades open. If the data file is missing the lookups return "unknown"
rather than "no", so a missing asset weakens enrichment instead of rejecting
every real address.
"""

from __future__ import annotations

import gzip
import re
import threading
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data" / "us_postal_places.tsv.gz"

#: Place-name prefixes that are conventionally abbreviated. These do NOT decide
#: whether a string is a city — the gazetteer does. They only generate the
#: spelling variants under which a REAL place is indexed, so that "St. Louis"
#: finds the row GeoNames spells "Saint Louis".
_PREFIX_VARIANTS: tuple[tuple[str, str], ...] = (
    ("saint", "st"),
    ("mount", "mt"),
    ("fort", "ft"),
)

_lock = threading.Lock()
_zip_index: dict[str, tuple[str, str]] | None = None
_place_index: dict[str, frozenset[str]] | None = None
_loaded = False


def _normalize(name: str) -> str:
    """Casefold, drop punctuation, collapse whitespace."""
    text = re.sub(r"[.,'’]", "", str(name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _keys_for(name: str) -> list[str]:
    """Every spelling a real place should be findable under."""
    base = _normalize(name)
    if not base:
        return []
    keys = [base]
    head, _, rest = base.partition(" ")
    if rest:
        for full, abbrev in _PREFIX_VARIANTS:
            if head == full:
                keys.append(f"{abbrev} {rest}")
            elif head == abbrev:
                keys.append(f"{full} {rest}")
    return keys


def _load() -> None:
    global _zip_index, _place_index, _loaded
    with _lock:
        if _loaded:
            return
        _loaded = True
        zips: dict[str, tuple[str, str]] = {}
        places: dict[str, set[str]] = {}
        try:
            with gzip.open(_DATA, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    postal, city, state = parts[0], parts[1], parts[2]
                    # First row for a ZIP wins; GeoNames lists the primary
                    # place first and alternates after it.
                    zips.setdefault(postal, (city, state))
                    for key in _keys_for(city):
                        places.setdefault(key, set()).add(state)
        except FileNotFoundError:
            pass
        except Exception:  # a corrupt asset must not fail a compile
            zips, places = {}, {}
        _zip_index = zips
        _place_index = {k: frozenset(v) for k, v in places.items()}


def available() -> bool:
    """True when the reference data actually loaded."""
    _load()
    return bool(_zip_index)


def city_state_for_zip(postal_code: str | None) -> tuple[str, str] | None:
    """The city and state a 5-digit ZIP belongs to, or None."""
    _load()
    if not _zip_index:
        return None
    digits = re.sub(r"\D", "", str(postal_code or ""))
    if len(digits) < 5:
        return None
    return _zip_index.get(digits[:5])


def states_for_city(city: str | None) -> frozenset[str]:
    """Every state containing a place with this name."""
    _load()
    if not _place_index:
        return frozenset()
    for key in _keys_for(city or ""):
        found = _place_index.get(key)
        if found:
            return found
    return frozenset()


def is_known_place(city: str | None, state: str | None = None) -> bool | None:
    """Is this a real US place? None when the reference data is unavailable.

    ``None`` is not ``False``. A caller must not reject an address because we
    could not check it.
    """
    _load()
    if not _place_index:
        return None
    states = states_for_city(city)
    if not states:
        return False
    if not state:
        return True
    return str(state).strip().upper() in states


def resolve(
    *,
    city: str | None = None,
    state: str | None = None,
    postal_code: str | None = None,
) -> tuple[str | None, str | None]:
    """Fill in whichever of city/state the others determine. Never overwrites.

    A ZIP is authoritative for both. A city that exists in exactly one state
    names that state; a city in several does not, and abstains.
    """
    city_out = (str(city).strip() or None) if city else None
    state_out = (str(state).strip().upper() or None) if state else None

    by_zip = city_state_for_zip(postal_code)
    if by_zip:
        city_out = city_out or by_zip[0]
        state_out = state_out or by_zip[1]
        return city_out, state_out

    if city_out and not state_out:
        states = states_for_city(city_out)
        if len(states) == 1:
            state_out = next(iter(states))
    return city_out, state_out
