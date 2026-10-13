"""How many sites does the work order price — and if we can't say, why not?

PUR-10 (deal 010095, Lantronix): the work order came back right in every way
but one. Two devices, no onsite hands, full coverage of the priced work -- and
``site_count: null``. The kit then priced one site because hours x sites falls
back to 1. A null that silently becomes 1 is indistinguishable from "the
documents say one site", which is the problem.

The work order's ``site_count`` was, until this module, ONLY what the LLM
extractor returned. Three code paths made that null even when the documents
contain a site:

1. The extractor prompt says "do not invent a count the documents do not
   state". A single-ship-to quote never says "1 site", so a well-behaved model
   returns null. Nothing then looked at the address the quote does carry.
2. The work_order stage runs BEFORE ``site_geo_fallback``. A deal whose only
   locational anchor is a ``City, ST ZIP`` in a quote header gets its
   ``physical_site`` atom minted later in the compile, but the minted work
   lines had already frozen ``site_count: None``.
3. The extractor only reads the first N characters of each kept document, so a
   ship-to block past the truncation is invisible to it.

This module resolves the count deterministically from evidence already in the
compile, in order of strength, and ALWAYS returns a reason. When the documents
genuinely name no site it stays ``None`` with ``status: "unknown"`` -- never 0
and never a silent 1.
"""
from __future__ import annotations

import re
from typing import Any

#: Ordered from strongest to weakest.
STATUSES = (
    "declared_field",   # a structured "Qty of sites" / "# of sites" cell
    "work_order_llm",   # the extractor stated a number
    "stated_in_text",   # "10 sites", "two locations"
    "resolved_sites",   # distinct physical_site atoms
    "single_address",   # exactly one distinct non-vendor US address
    "unknown",          # nothing names a site -> None, with the reason
)


def _text(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None)
    if raw is None and isinstance(atom, dict):
        raw = atom.get("raw_text")
    return re.sub(r"\s+", " ", str(raw or "")).strip()


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    if v is None and isinstance(atom, dict):
        v = atom.get("value")
    return v if isinstance(v, dict) else {}


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def _flags(atom: Any) -> list[str]:
    return [str(f) for f in (getattr(atom, "review_flags", None) or [])]


def _evidence(atom: Any, text: str | None = None) -> dict[str, Any]:
    return {
        "atom_id": getattr(atom, "id", None),
        "artifact_id": getattr(atom, "artifact_id", None),
        "text": (text if text is not None else _text(atom))[:240],
    }


def _positive_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and float(v).is_integer() and 1 <= v <= 10_000:
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return _positive_int(int(v.strip()))
    return None


def _declared_field(atoms: list[Any]) -> tuple[int, Any] | None:
    for a in atoms:
        val = _value(a)
        fields = val.get("fields") if isinstance(val.get("fields"), dict) else {}
        n = _positive_int(fields.get("site_count"))
        if n is None and val.get("kind") == "deal_header" and val.get("field") == "site_count":
            n = _positive_int(val.get("value"))
        if n is not None:
            return n, a
    return None


def _is_work_order_minted(atom: Any) -> bool:
    return _value(atom).get("backfill_reason") == "work_order"


def _site_key(atom: Any) -> str:
    for k in getattr(atom, "entity_keys", None) or []:
        if isinstance(k, str) and k.startswith("site:"):
            return k
    val = _value(atom)
    try:
        from app.core.address_parse import normalized_address_key

        key = normalized_address_key(val)
        if key:
            return key
    except Exception:  # pragma: no cover
        pass
    return re.sub(r"[^a-z0-9]+", " ", _text(atom).lower()).strip()


def _resolved_sites(atoms: list[Any]) -> list[Any]:
    by_key: dict[str, Any] = {}
    for a in atoms:
        if _atom_type(a) != "physical_site":
            continue
        if any("vendor" in f for f in _flags(a)):
            continue
        key = _site_key(a)
        if key:
            by_key.setdefault(key, a)
    return list(by_key.values())


def _addresses(atoms: list[Any]) -> dict[tuple[str, str, str], tuple[Any, str]]:
    from app.core.address_parse import US_STATES, find_us_addresses_in_text
    from app.core.vendor_site_ban import is_purtera_vendor_address

    out: dict[tuple[str, str, str], tuple[Any, str]] = {}
    for a in atoms:
        if _is_work_order_minted(a):
            continue
        text = _text(a)
        if not text:
            continue
        for p in find_us_addresses_in_text(text):
            if not p.city or not p.state or p.state not in US_STATES:
                continue
            if is_purtera_vendor_address(text=text):
                continue
            key = (p.city.lower(), p.state, (p.zip or "")[:5])
            out.setdefault(key, (a, text))
    # The same city/state with and without a ZIP is one place.
    zipped = {(c, s) for c, s, z in out if z}
    return {k: v for k, v in out.items() if k[2] or (k[0], k[1]) not in zipped}


def resolve_site_count(
    atoms: list[Any],
    *,
    llm_site_count: Any = None,
    no_onsite_hands: bool = False,
) -> dict[str, Any]:
    """Best-supported site count with its provenance, or None with the reason.

    Returns ``{"site_count", "status", "reason", "evidence"}``. ``status`` is one
    of :data:`STATUSES`. ``site_count`` is never 0: "we could not find a site"
    and "there are zero sites" are different claims and only the first is ever
    made here.
    """
    atoms = [a for a in (atoms or []) if not _is_work_order_minted(a)]

    declared = _declared_field(atoms)
    if declared:
        n, a = declared
        return {
            "site_count": n,
            "status": "declared_field",
            "reason": f"a structured site-count field declares {n}",
            "evidence": [_evidence(a)],
        }

    llm_n = _positive_int(llm_site_count)
    if llm_n is not None:
        return {
            "site_count": llm_n,
            "status": "work_order_llm",
            "reason": f"the work order extractor read {llm_n} site(s) from the documents",
            "evidence": [],
        }

    from app.core.site_count_reconcile import _COUNT_NEAR_SITE, _as_int

    counts: dict[int, list[Any]] = {}
    for a in atoms:
        for m in _COUNT_NEAR_SITE.finditer(_text(a)):
            n = _as_int(m.group(1))
            if n is not None:
                counts.setdefault(n, []).append(a)
    if counts:
        best = sorted(counts.items(), key=lambda kv: (-len(kv[1]), -kv[0]))[0]
        return {
            "site_count": best[0],
            "status": "stated_in_text",
            "reason": f"documents state {best[0]} site(s) ({len(best[1])} mention(s))",
            "evidence": [_evidence(best[1][0])],
        }

    sites = _resolved_sites(atoms)
    if sites:
        return {
            "site_count": len(sites),
            "status": "resolved_sites",
            "reason": f"{len(sites)} distinct physical site(s) resolved from the documents",
            "evidence": [_evidence(s) for s in sites[:10]],
        }

    addrs = _addresses(atoms)
    if len(addrs) == 1:
        (a, text), = addrs.values()
        return {
            "site_count": 1,
            "status": "single_address",
            "reason": (
                "no site count is stated, but the documents carry exactly one "
                "distinct address; read as one site (needs PM confirmation that "
                "it is the work location, not a bill-to)"
            ),
            "evidence": [_evidence(a, text)],
        }

    if len(addrs) > 1:
        reason = (
            f"documents carry {len(addrs)} distinct addresses but no stated site "
            "count and no resolved site; cannot tell work locations from "
            "bill-to/corporate addresses"
        )
        evidence = [_evidence(a, t) for a, t in list(addrs.values())[:10]]
    else:
        reason = "documents name no site: no stated count, no resolved site, no address"
        if no_onsite_hands:
            reason += " (work is remote / no onsite hands, so none may exist)"
        evidence = []
    return {"site_count": None, "status": "unknown", "reason": reason, "evidence": evidence}


def backfill_work_order_site_count(atoms: list[Any]) -> int:
    """Fill ``site_count`` on minted work lines that froze it as None.

    Runs after site_geo_fallback, when physical_site atoms minted from a
    ``City, ST ZIP`` header finally exist. Returns the number of lines changed.
    Every minted line gets ``site_count_status`` / ``site_count_reason`` so a
    remaining None is explained.
    """
    minted = [a for a in atoms or [] if _is_work_order_minted(a)]
    if not minted:
        return 0
    pending = [a for a in minted if _value(a).get("site_count") is None]
    if not pending:
        return 0
    res = resolve_site_count(
        atoms, no_onsite_hands=any(_value(a).get("no_onsite_hands") for a in pending)
    )
    changed = 0
    for a in pending:
        val = dict(_value(a))
        val["site_count"] = res["site_count"]
        val["site_count_status"] = res["status"]
        val["site_count_reason"] = res["reason"]
        val["site_count_evidence"] = res["evidence"]
        a.value = val
        if res["site_count"] is not None:
            changed += 1
    return changed


__all__ = ["STATUSES", "backfill_work_order_site_count", "resolve_site_count"]
