"""Detect Purtera / internal authors for elevated source trust.

Domain-based (not name-based): any ``@purtera-it.com`` / ``@purtera.com`` /
``@optbotai.com`` author is treated as internal knowledge — high-value for
HubSpot notes and emails authored by our people.
"""

from __future__ import annotations

import re
from typing import Any

INTERNAL_EMAIL_DOMAINS: tuple[str, ...] = (
    "purtera-it.com",
    "purtera.com",
    "optbotai.com",
)

# Confidence floor / boost applied to atoms from internal-authored notes.
INTERNAL_AUTHOR_CONFIDENCE_FLOOR = 0.9
INTERNAL_AUTHOR_CONFIDENCE_BOOST = 0.08

_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@([a-z0-9.\-]+\.[a-z]{2,})", re.I)


def extract_email_domain(author_or_email: str | None) -> str | None:
    raw = (author_or_email or "").strip().lower()
    if not raw:
        return None
    m = _EMAIL_RE.search(raw)
    if not m:
        return None
    return m.group(1).lower()


def is_internal_author(
    author: str | None = None,
    *,
    author_email: str | None = None,
) -> bool:
    """True when author email (preferred) or author string is on an internal domain."""
    for candidate in (author_email, author):
        domain = extract_email_domain(candidate)
        if not domain:
            continue
        if any(domain == d or domain.endswith(f".{d}") for d in INTERNAL_EMAIL_DOMAINS):
            return True
    return False


def classify_author_affiliation(
    author: str | None = None,
    *,
    author_email: str | None = None,
) -> str:
    """Return ``internal`` | ``external`` | ``unknown``."""
    if is_internal_author(author, author_email=author_email):
        return "internal"
    email = (author_email or "").strip() or (author or "").strip()
    if extract_email_domain(email):
        return "external"
    return "unknown"


def apply_internal_author_elevation(
    *,
    confidence: float,
    review_flags: list[str] | None = None,
    value: dict[str, Any] | None = None,
) -> tuple[float, list[str], dict[str, Any]]:
    """Stamp affiliation + boost confidence for internal-authored atoms."""
    flags = list(review_flags or [])
    val = dict(value or {})
    val["author_affiliation"] = "internal"
    if "internal_author" not in flags:
        flags.append("internal_author")
    if "trusted_internal_source" not in flags:
        flags.append("trusted_internal_source")
    if confidence < INTERNAL_AUTHOR_CONFIDENCE_FLOOR:
        boosted = min(0.98, INTERNAL_AUTHOR_CONFIDENCE_FLOOR + INTERNAL_AUTHOR_CONFIDENCE_BOOST)
    else:
        boosted = min(0.98, confidence + INTERNAL_AUTHOR_CONFIDENCE_BOOST)
    return boosted, flags, val


__all__ = [
    "INTERNAL_AUTHOR_CONFIDENCE_BOOST",
    "INTERNAL_AUTHOR_CONFIDENCE_FLOOR",
    "INTERNAL_EMAIL_DOMAINS",
    "apply_internal_author_elevation",
    "classify_author_affiliation",
    "extract_email_domain",
    "is_internal_author",
]
