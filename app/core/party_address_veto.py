"""An address on a signature page belongs to a party, not to the project.

Live 010300 (2026-09-03): the only physical_site the deal resolved was
"200 N. Milwaukee Ave., Vernon Hills, IL 60061" -- CDW's mailing address from
the signature page of a PSOW (page 6, beside six ``signatory`` atoms). The
customer HQ in the Customer-Designated Locations table was not it.

Shape only: a page that carries signatory atoms is a signature page; a
physical_site read from that page of the same document is re-typed to
``deal_metadata`` (kind ``party_address``) and loses its ``site:`` keys, so
no downstream head counts it as a place where work happens. The address is
kept and says what it is.
"""
from __future__ import annotations

from typing import Any


def _type_str(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def _page_of(atom: Any) -> Any:
    refs = getattr(atom, "source_refs", None) or []
    loc = getattr(refs[0], "locator", None) if refs else None
    return loc.get("page") if isinstance(loc, dict) else None


def veto_party_page_sites(atoms: list[Any]) -> int:
    """Retype physical_site atoms that sit on a signature page. Returns count."""
    signature_pages: set[tuple[str, Any]] = set()
    for a in atoms:
        if _type_str(a) == "signatory":
            page = _page_of(a)
            if page is not None:
                signature_pages.add((str(getattr(a, "artifact_id", "")), page))
    if not signature_pages:
        return 0
    n = 0
    for a in atoms:
        if _type_str(a) != "physical_site":
            continue
        key = (str(getattr(a, "artifact_id", "")), _page_of(a))
        if key not in signature_pages:
            continue
        try:
            from app.core.schemas import AtomType
            a.atom_type = AtomType.deal_metadata
        except Exception:
            a.atom_type = "deal_metadata"
        v = getattr(a, "value", None)
        if isinstance(v, dict):
            v["site_kind_before_veto"] = v.get("kind")
            v["kind"] = "party_address"
            v["why"] = "address on a signature page; a party's mailing address, not a work site"
        try:
            a.entity_keys = [k for k in (getattr(a, "entity_keys", None) or []) if not str(k).startswith("site:")]
        except Exception:
            pass
        flags = list(getattr(a, "review_flags", None) or [])
        if "party_address" not in flags:
            flags.append("party_address")
        try:
            a.review_flags = flags
        except Exception:
            pass
        n += 1
    return n


__all__ = ["veto_party_page_sites"]
