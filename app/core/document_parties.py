"""Who a contract-shaped document is BETWEEN, and whether we are a party to it.

Live 010300 (2026-09-03): both PSOWs in the deal are NewBold's contracts with
CDW for the customer's sites -- the incumbent's terms, which the customer asked
us to work within and update. The envelope dated them and read every clause,
but nothing said "these are NewBold's terms, not Purtera's"; a reader had to
know it.

The signal is in the document itself, by shape:
  * a header field whose LABEL is a contract role followed by "Name" or a bare
    role definition ("Provider Name: NewBold LLC", "Provider: NewBold",
    "Seller: CDW", "Customer: D4C")
  * the signature block's parties (one signer record per party)
We are whoever the internal email domains say we are (``purtera-it.com`` ->
"purtera"); a party whose name carries one of those tokens is us. A document
that names a provider / seller / contractor / vendor party and names none of us
carries a THIRD PARTY's terms: every atom from it is stamped with the owner of
those terms, so a consumer can tell "their price" from "our price".
"""
from __future__ import annotations

import re
from typing import Any

from app.core.internal_author import INTERNAL_EMAIL_DOMAINS

#: Contract roles: a closed grammatical class of party labels, not a vocabulary
#: of customers or vendors.
_ROLES = ("provider", "seller", "buyer", "customer", "client", "vendor", "contractor",
          "subcontractor", "supplier", "partner", "reseller", "affiliate")
_ROLE_LABEL_RE = re.compile(
    r"(?:^|\|)\s*(?:col_\d+:\s*)?(?:[A-Za-z ]{0,20}\s)?(?P<role>" + "|".join(_ROLES) + r")(?:\s+name)?\s*:\s*(?P<val>[^|]*)",
    re.I,
)
_PROVIDER_LIKE = {"provider", "seller", "vendor", "contractor", "subcontractor", "supplier", "reseller"}
_CUSTOMER_LIKE = {"customer", "client", "buyer"}


def our_org_tokens() -> set[str]:
    toks: set[str] = set()
    for d in INTERNAL_EMAIL_DOMAINS:
        head = str(d).split(".")[0]
        for part in re.split(r"[-_]", head):
            if len(part) >= 4:
                toks.add(part.lower())
    return toks


def _is_us(name: str) -> bool:
    low = re.sub(r"[^a-z]", "", str(name or "").lower())
    return any(tok in low for tok in our_org_tokens())


def _clean_party(val: str) -> str:
    v = re.sub(r"\s+", " ", str(val or "")).strip(" :;,.")
    # "D4C Site Assessment and Implementation Program – Phase 1: DENTISTRY FOR CHILDREN"
    # -> the value after the LAST colon is the cell's own value
    if ":" in v:
        v = v.rsplit(":", 1)[1].strip()
    # "NewBold c." / "D4C a. The Customer-designated…" -> first clause
    v = re.split(r"\s+[a-z][.)]\s|\s+\(|\.\s", v)[0].strip()
    return v[:80]


def parties_for_document(atoms: list[dict[str, Any]]) -> dict[str, Any]:
    """Roles -> party names read from one document's atoms."""
    roles: dict[str, str] = {}
    signers: list[str] = []
    for a in atoms:
        t = str(a.get("text") or "")
        s = a.get("structured") if isinstance(a.get("structured"), dict) else {}
        if a.get("atom_type") == "signatory" and isinstance(s.get("signers"), list):
            for rec in s["signers"]:
                p = str(rec.get("party") or "").strip()
                if p and p not in signers:
                    signers.append(p)
        if a.get("atom_type") not in ("deal_metadata", "stakeholder", "scope_item", "contract_term"):
            continue
        for m in _ROLE_LABEL_RE.finditer(t):
            role = m.group("role").lower()
            raw_val = m.group("val")
            if not raw_val.strip():
                # "col_0: Provider Name: | <column heading>: NewBold LLC …" --
                # the value is the cell AFTER the label cell; take the rest of
                # the row and let _clean_party keep what follows the last colon.
                raw_val = t[m.end():].lstrip(" |")
            val = _clean_party(raw_val)
            if len(val) < 2 or len(val.split()) > 8:
                continue
            roles.setdefault(role, val)
    return {"roles": roles, "signers": signers}


def annotate_document_parties(documents: list[dict[str, Any]], envelope: dict[str, Any]) -> int:
    """Stamp ``parties``, ``our_role`` and ``terms_owner`` on documents and
    ``decision_provenance.terms_owner`` on their atoms. Returns documents stamped."""
    atoms = envelope.get("atoms") or []
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for a in atoms:
        by_doc.setdefault(str(a.get("artifact_id")), []).append(a)
    n = 0
    third: list[dict[str, Any]] = []
    for d in documents or []:
        aid = str(d.get("artifact_id"))
        found = parties_for_document(by_doc.get(aid, []))
        roles, signers = found["roles"], found["signers"]
        if not roles and not signers:
            continue
        names = list(roles.values()) + signers
        ours = [nm for nm in names if _is_us(nm)]
        our_role = next((r for r, nm in roles.items() if _is_us(nm)), ("signatory" if ours else None))
        provider = next((roles[r] for r in ("provider", "contractor", "subcontractor", "vendor", "supplier", "seller", "reseller") if r in roles), None)
        customer = next((roles[r] for r in ("customer", "client", "buyer") if r in roles), None)
        third_party = bool(provider) and not ours
        d["parties"] = {"roles": roles, "signers": signers}
        d["our_role"] = our_role
        d["terms_owner"] = provider if third_party else (our_role and "us")
        d["third_party_terms"] = third_party
        if third_party:
            why = f"names {provider} as the provider and none of our organisations as a party; these are {provider}'s terms, not ours"
            d["terms_why"] = why
            third.append({"artifact_id": aid, "filename": d.get("filename"), "provider": provider, "customer": customer,
                          "signers": signers, "dated": d.get("authored_at"), "why": why})
            for a in by_doc.get(aid, []):
                prov = dict(a.get("decision_provenance") or {})
                prov["terms_owner"] = provider
                prov["applies_to"] = "third_party_terms"
                a["decision_provenance"] = prov
        n += 1
    summary = envelope.get("summary")
    if isinstance(summary, dict):
        summary["third_party_terms"] = third
    return n


__all__ = ["annotate_document_parties", "parties_for_document", "our_org_tokens"]
