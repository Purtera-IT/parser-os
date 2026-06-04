"""Delexicalization (entity masking) — the generalization enforcer.

The trained extractor heads must learn the *general rule*, never a specific
name. "PurTera is our company, not a site" has to become "a party's own
self-referenced HQ address is not a deal site" — name-agnostic — or the model
is useless on the next deal and merely memorizes identities.

We enforce that here, *before* any text becomes a training feature: specific
identity-bearing surface forms are replaced with **role placeholders**
(``<SELF_ORG>``, ``<CUSTOMER>``, ``<VENDOR>``, ``<SITE>``, ``<PERSON>``) and
typed-literal placeholders (``<ADDR>``, ``<EMAIL>``, ``<MONEY>``, ``<DATE>``,
``<QTY>``, ``<PARTNO>``, ``<URL>``). The model never sees the literal string,
so it cannot key on it.

Design contract:
* **Deterministic.** Same (text, role_map) → same masked output.
* **Reversible / inspectable.** :func:`delexicalize` returns the applied
  substitutions so a PM (or an audit) can see exactly what was masked.
* **Conservative.** When a role is unknown we mask the *shape* (a literal
  address/email/money/date) but never invent a role. Generic capitalized runs
  are left alone — over-masking destroys signal as surely as under-masking.
* **Counterfactual-safe.** The whole point: swapping the proper nouns in the
  raw text must not change the masked text (see ``tests/test_delexicalize.py``
  ``test_name_swap_invariant``). That invariance is the training-time guarantee
  that the model learns the role, not the identity.

This module is pure string work — no network, no model. Role assignment is fed
in by the caller (it already knows customer/vendor/site/self from entity
extraction); this module only applies a role_map plus shape-based literal
masking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Role placeholder vocabulary. Stable tokens — the heads train on these, so
# renaming one is a breaking change to every trained checkpoint.
ROLE_SELF_ORG = "<SELF_ORG>"      # the proposing party / "our company"
ROLE_CUSTOMER = "<CUSTOMER>"      # the buyer / awarding entity
ROLE_VENDOR = "<VENDOR>"          # a named manufacturer / subcontractor
ROLE_SITE = "<SITE>"             # a physical job site / building
ROLE_PERSON = "<PERSON>"          # a named individual / stakeholder
ROLE_ORG = "<ORG>"               # a named organization of unknown role

# Typed-literal placeholders (shape-masked, role-agnostic).
LIT_ADDR = "<ADDR>"
LIT_EMAIL = "<EMAIL>"
LIT_PHONE = "<PHONE>"
LIT_MONEY = "<MONEY>"
LIT_DATE = "<DATE>"
LIT_QTY = "<QTY>"
LIT_PARTNO = "<PARTNO>"
LIT_URL = "<URL>"
LIT_ZIP = "<ZIP>"

ALL_PLACEHOLDERS = frozenset(
    {
        ROLE_SELF_ORG, ROLE_CUSTOMER, ROLE_VENDOR, ROLE_SITE, ROLE_PERSON,
        ROLE_ORG, LIT_ADDR, LIT_EMAIL, LIT_PHONE, LIT_MONEY, LIT_DATE,
        LIT_QTY, LIT_PARTNO, LIT_URL, LIT_ZIP,
    }
)


@dataclass
class DelexResult:
    """The masked text plus a reversible record of every substitution made."""

    masked: str
    # surface form → placeholder, in the order applied. Lets a PM/audit see
    # exactly which identity was hidden behind which role, and reverse it.
    substitutions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def role_map(self) -> dict[str, str]:
        """Flatten substitutions to {surface: placeholder} (last wins)."""
        return {surface: ph for surface, ph in self.substitutions}


# ── shape-based literal patterns (applied after role masking) ───────────────
# Order matters: more specific shapes first so e.g. an email isn't half-eaten
# by the URL rule. Each is conservative — anchored enough to avoid mauling
# ordinary prose.

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s<>()]+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)
_MONEY_RE = re.compile(r"(?<![\w$])\$\s?\d[\d,]*(?:\.\d{1,2})?(?:\s?[KMB])?\b", re.IGNORECASE)
# Street address: number + capitalized run + street suffix. Mirrors the
# extractor's conservative shape so masking and detection agree.
_ADDR_RE = re.compile(
    r"\b\d+(?:-\d+)?\s+[A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Za-z0-9'.\-]+){0,4}\s+"
    r"(?:Street|St|Ave|Avenue|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|"
    r"Court|Ct|Place|Pl|Highway|Hwy|Parkway|Pkwy|Trail|Trl|Circle|Cir|"
    r"Terrace|Loop|Run|Plaza|Square|Sq|Walk|Park)\.?\b"
)
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
# Dates: ISO, US slash, and "Month DD, YYYY".
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
# Part numbers / SKUs — letter+digit hyphenated identifiers.
_PARTNO_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{1,9}(?:[-/][A-Z0-9]{1,12}){1,6}|[A-Z]{2,5}\d{2,6}[A-Z]{0,3})\b"
)
# Bare quantities: "Qty: 136", "Quantity = 500", or "<number> <noun>" is left
# to the caller's richer extractor; here we only mask explicit qty fields so we
# don't blunt counts the head should learn from.
_QTY_FIELD_RE = re.compile(
    r"\b(?:qty|quantity|quantities|count)\s*[:=]?\s*\d[\d,]*\b", re.IGNORECASE
)


def _mask_role_surfaces(text: str, role_map: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """Replace each known surface form with its role placeholder.

    Longest surfaces first so "Acme Security Corp" is masked before a
    substring "Acme" would be. Word-boundary, case-insensitive, but only
    when the placeholder is a recognized role token.
    """
    subs: list[tuple[str, str]] = []
    # Normalize/validate the role map; ignore entries whose placeholder isn't
    # a known role token (defensive — a bad caller can't inject junk tokens).
    items = [
        (surface, ph)
        for surface, ph in role_map.items()
        if surface and surface.strip() and ph in ALL_PLACEHOLDERS
    ]
    # Longest surface first.
    for surface, ph in sorted(items, key=lambda kv: len(kv[0]), reverse=True):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(ph, text)
            subs.append((surface, ph))
    return text, subs


def _mask_literals(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Shape-mask identity-bearing literals. Order: specific → general."""
    subs: list[tuple[str, str]] = []

    def _apply(rx: re.Pattern[str], ph: str, t: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            subs.append((m.group(0), ph))
            return ph
        return rx.sub(_repl, t)

    text = _apply(_EMAIL_RE, LIT_EMAIL, text)
    text = _apply(_URL_RE, LIT_URL, text)
    text = _apply(_ADDR_RE, LIT_ADDR, text)
    text = _apply(_MONEY_RE, LIT_MONEY, text)
    text = _apply(_DATE_RE, LIT_DATE, text)
    text = _apply(_PHONE_RE, LIT_PHONE, text)
    text = _apply(_QTY_FIELD_RE, LIT_QTY, text)
    text = _apply(_PARTNO_RE, LIT_PARTNO, text)
    text = _apply(_ZIP_RE, LIT_ZIP, text)
    return text, subs


def delexicalize(
    text: str,
    role_map: dict[str, str] | None = None,
    *,
    mask_literals: bool = True,
) -> DelexResult:
    """Mask identities → role/shape placeholders so a head learns the rule.

    Parameters
    ----------
    text:
        The raw atom/span text.
    role_map:
        ``{surface_form: placeholder}`` from the caller's entity knowledge,
        e.g. ``{"PurTera": "<SELF_ORG>", "Yonah County": "<CUSTOMER>"}``.
        Placeholders MUST be members of :data:`ALL_PLACEHOLDERS`; unknown
        tokens are ignored (defensive).
    mask_literals:
        When True (default) also shape-mask addresses/emails/money/dates/etc.
        Set False to mask only the named roles (rarely needed).

    Returns
    -------
    DelexResult
        ``.masked`` is the training feature; ``.substitutions`` is the
        reversible audit trail.
    """
    if not text:
        return DelexResult(masked="", substitutions=[])
    subs: list[tuple[str, str]] = []
    out = text
    if role_map:
        out, role_subs = _mask_role_surfaces(out, role_map)
        subs.extend(role_subs)
    if mask_literals:
        out, lit_subs = _mask_literals(out)
        subs.extend(lit_subs)
    return DelexResult(masked=out, substitutions=subs)
