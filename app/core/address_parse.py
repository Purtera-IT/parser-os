"""Universal US address parsing for parser-os site emission.

Right-anchors ``City, ST ZIP`` so street tokens (``Park Blvd``) are never
absorbed into the city field. Used by site roster extraction, geo fallback,
and physical_site atom normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

US_STATES: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
})

_STREET_SUFFIXES = frozenset({
    "st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road",
    "dr", "drive", "ln", "lane", "ct", "court", "pl", "place", "way",
    "pkwy", "parkway", "hwy", "highway", "cir", "circle", "trl", "trail",
})

_HOUSE_NUMBER_RE = re.compile(r"^\d{1,6}\b")

# Embedded or line-ending ``City, ST ZIP`` (not greedy-left across the whole line).
_CITY_STATE_ZIP_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})\s*,?\s+"
    r"([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b"
)


@dataclass(frozen=True)
class ParsedAddress:
    street_address: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None

    def has_location(self) -> bool:
        return bool(
            (self.city and self.state)
            or (self.street_address and _HOUSE_NUMBER_RE.search(self.street_address))
        )


def _clean(part: str | None) -> str | None:
    s = re.sub(r"\s+", " ", (part or "").strip())
    return s or None


def _city_looks_valid(city: str) -> bool:
    """Reject cities that are street fragments or house-number runs."""
    city = _clean(city) or ""
    if not city:
        return False
    if re.match(r"^\d", city):
        return False
    tokens = [t.rstrip(".,") for t in city.split()]
    if not tokens:
        return False
    for t in tokens:
        bare = t.lower().rstrip(".")
        if bare in _STREET_SUFFIXES:
            return False
    return True


def _split_prefix_into_street_and_city(prefix: str) -> tuple[str | None, str | None]:
    """Take the rightmost 1–4 tokens of ``prefix`` as city when valid."""
    prefix = _clean(prefix) or ""
    if not prefix:
        return None, None
    if "," in prefix:
        left, right = prefix.rsplit(",", 1)
        city = _clean(right)
        street = _clean(left)
        if city and _city_looks_valid(city):
            return street, city
    words = prefix.split()
    for n in range(min(4, len(words)), 0, -1):
        city = " ".join(words[-n:])
        street = " ".join(words[:-n]).strip(" ,")
        if not _city_looks_valid(city):
            continue
        if n == 1 and street and not _HOUSE_NUMBER_RE.search(street):
            continue
        return _clean(street), city
    return prefix, None


def _parse_trailing_city_state_no_zip(street: str) -> ParsedAddress | None:
    """Parse ``3030 E 1st Ave, Denver CO`` when ZIP is absent."""
    raw = _clean(street) or ""
    m = re.search(
        r"^(.+?),\s*([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,2})\s+([A-Z]{2})\s*$",
        raw,
    )
    if not m:
        return None
    street_part = _clean(m.group(1))
    city = _clean(m.group(2))
    state = m.group(3).upper()
    if not street_part or not city or state not in US_STATES or not _city_looks_valid(city):
        return None
    return ParsedAddress(street_address=street_part, city=city, state=state)


def _parsed_from_city_state_zip_match(
    raw: str, m: re.Match[str]
) -> ParsedAddress | None:
    city_guess = _clean(m.group(1))
    state = m.group(2).upper()
    zipc = m.group(3)
    if state not in US_STATES or not city_guess:
        return None

    prefix = raw[: m.start()].strip().rstrip(" ,")

    if _city_looks_valid(city_guess):
        return ParsedAddress(
            street_address=_clean(prefix) or None,
            city=city_guess,
            state=state,
            zip=zipc,
        )

    combined = f"{prefix} {city_guess}".strip() if prefix else city_guess
    street, city = _split_prefix_into_street_and_city(combined)
    if not city or not _city_looks_valid(city):
        return None
    return ParsedAddress(
        street_address=street or None,
        city=city,
        state=state,
        zip=zipc,
    )


def parse_us_address_line(text: str) -> ParsedAddress:
    """Parse a US address from a line or prose fragment."""
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ParsedAddress()

    best: ParsedAddress | None = None
    for m in _CITY_STATE_ZIP_RE.finditer(raw):
        candidate = _parsed_from_city_state_zip_match(raw, m)
        if candidate and candidate.city and candidate.state:
            best = candidate

    if best:
        return best

    return ParsedAddress(street_address=raw)


def parse_city_state_field(city_state: str | None) -> tuple[str | None, str | None]:
    """Split ``Highland Park, MI`` or ``Seattle / WA`` into city + state."""
    s = _clean(city_state)
    if not s:
        return None, None
    if len(s) == 2 and s.upper() in US_STATES:
        return None, s.upper()
    m = re.match(r"^(.+?)\s*[,/]\s*([A-Z]{2})\s*$", s, re.IGNORECASE)
    if m:
        city = _clean(m.group(1))
        state = m.group(2).upper()
        if city and _city_looks_valid(city) and state in US_STATES:
            return city, state
    if _city_looks_valid(s):
        return s, None
    return None, None


def enrich_location_fields(
    *,
    street_address: str | None = None,
    city: str | None = None,
    state: str | None = None,
    zip_code: str | None = None,
    city_state: str | None = None,
    facility_name: str | None = None,
) -> dict[str, str | None]:
    """Merge separate columns + combined address lines into canonical fields."""
    street = _clean(street_address)
    out_city = _clean(city)
    out_state = _clean(state)
    out_zip = _clean(zip_code)

    if city_state and (not out_city or not out_state):
        cs_city, cs_state = parse_city_state_field(city_state)
        out_city = out_city or cs_city
        out_state = out_state or cs_state

    if street and (not out_city or not out_state or not out_zip):
        parsed = parse_us_address_line(street)
        if parsed.city and parsed.state:
            if not out_city:
                out_city = parsed.city
            if not out_state:
                out_state = parsed.state
            if not out_zip and parsed.zip:
                out_zip = parsed.zip
            if parsed.street_address:
                street = parsed.street_address
        elif not out_city or not out_state:
            no_zip = _parse_trailing_city_state_no_zip(street)
            if no_zip and no_zip.city and no_zip.state:
                if not out_city:
                    out_city = no_zip.city
                if not out_state:
                    out_state = no_zip.state
                if no_zip.street_address:
                    street = no_zip.street_address

    if not street and facility_name and not out_city:
        parsed = parse_us_address_line(facility_name)
        if parsed.city and parsed.state:
            out_city, out_state = parsed.city, parsed.state
            out_zip = out_zip or parsed.zip
            if parsed.street_address:
                street = parsed.street_address

    return {
        "street_address": street,
        "city": out_city,
        "state": out_state.upper() if out_state and len(out_state) == 2 else out_state,
        "zip": out_zip,
    }


_STREET_DEDUP_NOISE = frozenset({
    "location", "again", "site", "at", "note", "near", "in", "the",
})


def _street_for_dedup(street: str) -> str:
    s = street.lower().strip()
    if not s:
        return ""
    tokens = [t for t in s.split() if t not in _STREET_DEDUP_NOISE]
    return " ".join(tokens)


def normalized_address_key(fields: dict[str, Any]) -> str:
    """Dedup key for site atoms (city+state+zip when available, else full address)."""
    city = str(fields.get("city") or "").lower().strip()
    state = str(fields.get("state") or "").upper().strip()
    zipc = str(fields.get("zip") or "").strip()
    if city and state and zipc:
        street = _street_for_dedup(
            str(fields.get("street_address") or fields.get("address") or "")
        )
        if street:
            return f"{street}|{city}|{state}|{zipc}"
        return f"{city}|{state}|{zipc}"
    parts = [
        str(fields.get("street_address") or fields.get("address") or "").lower(),
        city,
        state,
        zipc,
        str(fields.get("facility_name") or fields.get("name") or "").lower(),
    ]
    return "|".join(p.strip() for p in parts if p.strip())


def find_us_addresses_in_text(text: str) -> list[ParsedAddress]:
    """Return every ``City, ST ZIP`` anchor found in prose."""
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return []
    out: list[ParsedAddress] = []
    seen: set[tuple[str, str, str]] = set()
    for m in _CITY_STATE_ZIP_RE.finditer(raw):
        parsed = _parsed_from_city_state_zip_match(raw, m)
        if not parsed or not parsed.city or not parsed.state:
            continue
        key = (parsed.city.lower(), parsed.state, parsed.zip or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)
    return out


__all__ = [
    "ParsedAddress",
    "US_STATES",
    "enrich_location_fields",
    "find_us_addresses_in_text",
    "normalized_address_key",
    "parse_city_state_field",
    "parse_us_address_line",
]
