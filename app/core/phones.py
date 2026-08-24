"""Phone-number recognition, in one place, using libphonenumber.

Four modules each carried their own phone regex and they disagreed with each
other. Measured on 7 real numbers and 9 things that merely look numeric:

                                 recall   false positives
    atom_substance_gate           7/7          3/9     <- read "invoice
                                                          2026-05-14 total
                                                          18,500" as a phone
    markdown_parser               6/7          1/9     <- and missed +44
    phonenumbers                  7/7          0/9

The regexes cannot do better, because "is this a phone number" is not a
question about shape. ``4500123456`` has the shape; it is a purchase order.
``phonenumbers`` is Google's libphonenumber, which validates against the
actual numbering plan of each country -- area codes that exist, central
office codes that exist, lengths that are legal -- and that is the only way
to tell the two apart.

``entity_extraction`` had rebuilt a piece of this by hand: strip a leading
US country code, require ten digits, reject area or central-office codes
starting 0 or 1. That is a partial reimplementation of the NANP rules, and
it is exactly what ``is_valid_number`` already knows for every country.

Entity keys stay ``phone:<10 digits>`` so nothing downstream has to change:
``national_digits`` reproduces the old canonical form.

Soft dependency, as with ``rapidfuzz`` in ``entity_resolution``: without the
library the old permissive pattern still runs, so recognition degrades to
what it was rather than failing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

try:  # pragma: no cover - exercised by whichever environment lacks it
    import phonenumbers as _pn
    from phonenumbers import PhoneNumberMatcher as _Matcher
except Exception:  # pragma: no cover
    _pn = None  # type: ignore[assignment]
    _Matcher = None  # type: ignore[assignment]

#: Default region for numbers written without a country code. Every deal in
#: this system is US commercial construction; an explicit +44 is still parsed
#: correctly because the matcher reads the prefix.
DEFAULT_REGION = "US"

#: The pre-library pattern, kept as the fallback so a missing dependency is a
#: quality regression and never an exception.
_FALLBACK_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"
)


@dataclass(frozen=True)
class PhoneMatch:
    """One recognised number and where it sits in the source text."""

    start: int
    end: int
    raw: str
    #: Canonical national significant number, digits only. For US numbers this
    #: is the ten-digit form the ``phone:`` entity key has always used.
    national_digits: str
    #: E.164, e.g. ``+18004234512``. Empty when only the fallback ran.
    e164: str


def _fallback_matches(text: str) -> Iterator[PhoneMatch]:
    for m in _FALLBACK_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            continue
        # The NANP rules the old entity_extraction code applied by hand.
        if digits[0] in {"0", "1"} or digits[3] in {"0", "1"}:
            continue
        yield PhoneMatch(m.start(), m.end(), m.group(0), digits, "")


def find_phones(text: str, region: str = DEFAULT_REGION) -> list[PhoneMatch]:
    """Every valid phone number in ``text``, with source offsets.

    Offsets are into ``text``, so a caller can redact or build a locator from
    the match rather than searching for the string again.
    """
    if not text:
        return []
    if _pn is None or _Matcher is None:  # pragma: no cover - fallback path
        return list(_fallback_matches(text))
    out: list[PhoneMatch] = []
    try:
        for m in _Matcher(text, region):
            if not _pn.is_valid_number(m.number):
                continue
            out.append(
                PhoneMatch(
                    start=m.start,
                    end=m.end,
                    raw=m.raw_string,
                    national_digits=str(m.number.national_number),
                    e164=_pn.format_number(m.number, _pn.PhoneNumberFormat.E164),
                )
            )
    except Exception:  # pragma: no cover - never fail a parse over a phone
        return list(_fallback_matches(text))
    return out


def has_phone(text: str, region: str = DEFAULT_REGION) -> bool:
    """True when ``text`` contains at least one valid phone number."""
    if not text:
        return False
    if _pn is None or _Matcher is None:  # pragma: no cover - fallback path
        return any(True for _ in _fallback_matches(text))
    try:
        for m in _Matcher(text, region):
            if _pn.is_valid_number(m.number):
                return True
    except Exception:  # pragma: no cover
        return any(True for _ in _fallback_matches(text))
    return False


def sub_phones(
    text: str,
    replacement: str,
    on_sub: Callable[[str, str], None] | None = None,
    region: str = DEFAULT_REGION,
) -> str:
    """Replace every phone number with ``replacement``.

    Used for delexicalisation, where the point is to remove the literal while
    keeping the sentence shape. ``on_sub`` receives ``(original, replacement)``
    for each substitution so callers can keep a reversal ledger.

    Replacements run right-to-left so earlier offsets stay valid.
    """
    matches = find_phones(text, region)
    if not matches:
        return text
    out = text
    for m in sorted(matches, key=lambda x: x.start, reverse=True):
        if on_sub is not None:
            on_sub(out[m.start:m.end], replacement)
        out = out[:m.start] + replacement + out[m.end:]
    return out
