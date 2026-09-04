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


def _norm(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _street_of(atom: Any) -> str:
    """The street line of a site ("200 N. Milwaukee Ave"), normalised; at
    least a number and a word so a bare city can never match."""
    import re

    v = getattr(atom, "value", None)
    cand = ""
    if isinstance(v, dict):
        cand = str(v.get("street_address") or v.get("address") or "")
    if not cand:
        cand = str(getattr(atom, "raw_text", None) or "")
    cand = cand.split(",")[0]
    s = _norm(cand)
    return s if re.search(r"\d", s) and len(s.split()) >= 2 else ""


def veto_party_page_sites(atoms: list[Any]) -> int:
    """Retype physical_site atoms that sit on a signature page. Returns count."""
    signature_pages: set[tuple[str, Any]] = set()
    # The signature block's own words, per document: a site whose street sits
    # inside them is the party's mailing address even when its page index was
    # stamped by another extractor (table rows count pages from 0, paragraphs
    # from 1 — live 010300 round 23: "Vernon Hills Office, 200 N. Milwaukee
    # Ave" came back as the deal's second site).
    signature_text: dict[str, str] = {}
    for a in atoms:
        v = getattr(a, "value", None)
        kind = str((v or {}).get("kind") or "") if isinstance(v, dict) else ""
        if _type_str(a) == "signatory" or kind in ("signature_block", "signature_chrome"):
            art = str(getattr(a, "artifact_id", ""))
            page = _page_of(a)
            if page is not None:
                signature_pages.add((art, page))
            signature_text[art] = signature_text.get(art, "") + " " + _norm(str(getattr(a, "raw_text", None) or getattr(a, "text", None) or ""))
    if not signature_pages and not signature_text:
        return 0
    n = 0
    for a in atoms:
        art = str(getattr(a, "artifact_id", ""))
        key = (art, _page_of(a))
        on_signature_page = key in signature_pages
        in_signature_text = False
        if not on_signature_page and _type_str(a) == "physical_site":
            street = _street_of(a)
            in_signature_text = bool(street) and street in signature_text.get(art, "")
        if not (on_signature_page or in_signature_text):
            continue
        # Nothing on a signature page is a site fact. Strip site keys from
        # every atom there, so no later stage can promote the signatures
        # preamble into a site because it sits beside a mailing address
        # (live 010300 round 6: "In acknowledgement that the parties below…"
        # came back as physical_site site:vernon_hills).
        if _type_str(a) != "physical_site":
            keys = list(getattr(a, "entity_keys", None) or [])
            kept = [k for k in keys if not str(k).startswith("site:")]
            if len(kept) != len(keys):
                try:
                    a.entity_keys = kept
                    n += 1
                except Exception:
                    pass
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
