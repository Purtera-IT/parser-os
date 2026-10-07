"""What a commercial-term LINE says, read off the line itself.

Two different questions were sharing the ``commercial`` head:

* ``commercial`` (app.core.commercial_terms) is the SHAPE OF THE KIT a request
  got -- billing type, PM/PC hours, travel days. A finished Deal Kit teaches it
  on the request (a task line), and nothing in the deal's documents states it.
* The Terms tab of the labeling page shows a payment term, a contract term, a
  change-order rule or a commercial total and asks what THAT LINE says. Its
  answer is in the line. Stored under ``commercial_terms`` it was either
  unparseable (and silently dropped by ``parse_commercial_verdict``) or, worse,
  parseable and taught as a kit shape on text that is not a request.

This module is the second question's own grammar, relation
``commercial_term_read``, judged by the ``term`` head
(pm_feedback.HEAD_REGISTRY). The verdict is ``;``-separated ``key=value``:

    kind=payment_term;net_days=30;timing=arrears;tax=excluded
    kind=contract_term;term_months=36;renewal=auto;notice_days=60
    kind=pricing_assumption;min_hours=2;increment_min=15;after_hours_multiplier=1.5
    kind=change_order_rule;notice_days=5;approval=written
    kind=commercial_total
    kind=not_a_term

``kind`` is required. Every other key belongs to one kind; a key the kind does
not take, or one nobody defined, is dropped -- the reading keeps what it can
carry rather than refusing the whole answer. ``kind`` alone is a valid answer.

No money: there is no price, rate or amount key. What a company charges is the
company layer's business; this grammar is the universal reading of a line.
"""

from __future__ import annotations

from typing import Any

RELATION = "commercial_term_read"

KINDS = (
    "payment_term",
    "contract_term",
    "change_order_rule",
    "pricing_assumption",
    "commercial_total",
    "not_a_term",
)

#: Closed-set keys: the values each may take.
_ENUMS: dict[str, tuple[str, ...]] = {
    "renewal": ("auto", "manual", "none"),
    "timing": ("arrears", "advance", "on_completion", "milestone"),
    "tax": ("included", "excluded", "at_invoice"),
    "approval": ("written", "verbal"),
}

#: Number keys and whether zero is allowed (False: must be > 0).
_NUMBERS: dict[str, bool] = {
    "term_months": False,
    "notice_days": True,
    "net_days": True,
    "min_hours": True,
    "increment_min": True,
    "after_hours_multiplier": True,
    "rate_lock_months": True,
}

#: Which keys each kind takes, in the order an encoded verdict lists them.
KIND_KEYS: dict[str, tuple[str, ...]] = {
    "contract_term": ("term_months", "renewal", "notice_days"),
    "payment_term": ("net_days", "timing", "tax"),
    "pricing_assumption": ("min_hours", "increment_min", "after_hours_multiplier", "rate_lock_months"),
    "change_order_rule": ("notice_days", "approval"),
    "commercial_total": (),
    "not_a_term": (),
}


def _clean(key: str, val: Any) -> Any:
    """``val`` as ``key`` takes it, or None when it is not a value of ``key``."""
    if key in _ENUMS:
        v = str(val if val is not None else "").strip().lower()
        return v if v in _ENUMS[key] else None
    if key in _NUMBERS:
        if isinstance(val, bool) or val is None:
            return None
        try:
            f = float(str(val).strip())
        except (TypeError, ValueError):
            return None
        if f != f or f in (float("inf"), float("-inf")):
            return None
        if f < 0 or (f == 0 and not _NUMBERS[key]):
            return None
        return int(f) if f.is_integer() else f
    return None


def parse_term_verdict(verdict: Any) -> dict[str, Any] | None:
    """The reading a verdict carries, or None when it names no known kind.

    ``{"kind": ..., <key>: <value>, ...}`` -- only the keys the kind takes,
    each with a valid value. A repeated key keeps its last valid value.
    """
    pairs: list[tuple[str, str]] = []
    for part in str(verdict or "").split(";"):
        key, sep, val = part.partition("=")
        if sep:
            pairs.append((key.strip().lower(), val.strip()))
    kinds = [v.lower() for k, v in pairs if k == "kind"]
    kind = kinds[-1] if kinds else ""
    if kind not in KIND_KEYS:
        return None
    out: dict[str, Any] = {"kind": kind}
    allowed = KIND_KEYS[kind]
    for key, val in pairs:
        if key not in allowed:
            continue
        cleaned = _clean(key, val)
        if cleaned is not None:
            out[key] = cleaned
    return out


def encode_term_verdict(kind: str, **values: Any) -> str:
    """One canonical verdict string: ``kind`` first, then the kind's keys in a
    fixed order. Keys the kind does not take, and invalid values, are left out.
    Raises ValueError for an unknown kind -- there is nothing to encode."""
    k = str(kind or "").strip().lower()
    if k not in KIND_KEYS:
        raise ValueError(f"unknown term kind {kind!r}; one of {', '.join(KINDS)}")
    parts = [f"kind={k}"]
    for key in KIND_KEYS[k]:
        cleaned = _clean(key, values.get(key))
        if cleaned is not None:
            parts.append(f"{key}={cleaned:g}" if isinstance(cleaned, float) else f"{key}={cleaned}")
    return ";".join(parts)


def canonical_term_verdict(verdict: Any) -> str | None:
    """``verdict`` re-encoded canonically, or None when it does not parse."""
    parsed = parse_term_verdict(verdict)
    if parsed is None:
        return None
    kind = parsed.pop("kind")
    return encode_term_verdict(kind, **parsed)


__all__ = [
    "RELATION", "KINDS", "KIND_KEYS",
    "parse_term_verdict", "encode_term_verdict", "canonical_term_verdict",
]
