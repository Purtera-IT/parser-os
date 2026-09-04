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
    # "Provider Name:", "ProviderName:" (OCR drops the space), "Provider:"
    r"(?:^|\|)\s*(?:col_\d+:\s*)?(?:[A-Za-z ]{0,20}\s)?(?P<role>" + "|".join(_ROLES) + r")(?:\s*name)?\s*:\s*(?P<val>[^|]*)",
    re.I,
)
# The defined-term shape every contract opens with: 'CDW Technologies LLC
# ("Buyer") and NewBold LLC ("Provider", "Seller" and "we")'. The party is the
# capitalised run just before the bracket; the role is the quoted term inside
# it. Live 010300 scanned PSOW: the header table's labels did not survive OCR,
# but this sentence did, and it names all three parties.
_DEFINED_TERM_RE = re.compile(
    r"(?P<val>[A-Z][A-Za-z0-9&.'\-]*(?:[ ,]+[A-Z][A-Za-z0-9&.'\-]*){0,6})\s*\(\s*(?:the\s+)?[“\"']\s*"
    r"(?P<role>(?i:" + "|".join(_ROLES) + r"))\s*[”\"']",
)
_CONTACT_TAIL_RE = re.compile(r"\s*(?:\+?\d[\d\s().-]{6,}\d|\S+@\S+|\+\d+)\s*$")


def _defined_term_roles(text: str) -> dict[str, str]:
    """Roles -> party names from '<Party> ("Role"' definitions in running text.
    Lines are unwrapped first: the party and its bracket often straddle a
    line break ('... and NewBold' / 'LLC ("Provider", ...')."""
    flat = " ".join(str(text or "").split())
    roles: dict[str, str] = {}
    for m in _DEFINED_TERM_RE.finditer(flat):
        role = m.group("role").lower()
        val = m.group("val").strip(" ,")
        # The run may start earlier than the name ("the undersigned, CDW ..."):
        # keep the part after the last comma-separated lowercase lead-in.
        if not val or not val[:1].isupper():
            continue
        if len(val) < 2 or len(val.split()) > 8:
            continue
        roles.setdefault(role, val)
    return roles


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
    # A header cell often runs into the next column ("DENTISTRY FOR CHILDREN
    # +1 (847) 9689740", "NewBold LLC FKA NewBold Corporation carlpai@cdw.com"):
    # a party name never ends in a phone number or an address.
    for _ in range(2):
        v2 = _CONTACT_TAIL_RE.sub("", v).strip(" ,;")
        if v2 == v:
            break
        v = v2
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
        # Any atom type: an OCR'd header line lands wherever the typer put it.
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
        for role, val in _defined_term_roles(t).items():
            roles.setdefault(role, val)
    return {"roles": roles, "signers": signers}


def parties_from_page_text(text: str) -> dict[str, str]:
    """Roles -> party names read line by line off a document's first page,
    for a scanned header whose cells never became atoms
    ("|ProviderName:|NewBold LLC FKA NewBold Corporation")."""
    roles: dict[str, str] = {}
    for line in str(text or "").splitlines()[:40]:
        for m in _ROLE_LABEL_RE.finditer(line):
            role = m.group("role").lower()
            raw_val = m.group("val") or line[m.end():].lstrip(" |")
            val = _clean_party(raw_val)
            if 2 <= len(val) and len(val.split()) <= 8:
                roles.setdefault(role, val)
    for role, val in _defined_term_roles(" ".join(str(text or "").splitlines()[:60])).items():
        roles.setdefault(role, val)
    return roles


def _edit_distance_small(a: str, b: str, *, limit: int = 2) -> bool:
    """True when two strings are within ``limit`` single-character edits."""
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
        if min(prev) > limit:
            return False
    return prev[-1] <= limit


def _reconcile_party_spellings(per_doc_roles: dict[str, dict[str, str]], all_text: str) -> None:
    """Rewrite near-identical party names to the spelling that occurs most
    often across the deal's atom texts. Mutates ``per_doc_roles`` in place."""
    names = sorted({v for roles in per_doc_roles.values() for v in roles.values() if v})
    if len(names) < 2:
        return
    canon: dict[str, str] = {}
    for nm in names:
        if nm in canon:
            continue
        cluster = [nm] + [o for o in names if o != nm and o not in canon and len(o) >= 6 and _edit_distance_small(nm, o)]
        if len(cluster) == 1:
            canon[nm] = nm
            continue
        best = max(cluster, key=lambda s: (all_text.count(s.lower()), -len(s)))
        for member in cluster:
            canon[member] = best
    for roles in per_doc_roles.values():
        for role, val in list(roles.items()):
            if val in canon and canon[val] != val:
                roles[role] = canon[val]


def annotate_document_parties(
    documents: list[dict[str, Any]],
    envelope: dict[str, Any],
    page_text_by_doc: dict[str, str] | None = None,
) -> int:
    """Stamp ``parties``, ``our_role`` and ``terms_owner`` on documents and
    ``decision_provenance.terms_owner`` on their atoms. Returns documents stamped."""
    atoms = envelope.get("atoms") or []
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for a in atoms:
        by_doc.setdefault(str(a.get("artifact_id")), []).append(a)
    n = 0
    third: list[dict[str, Any]] = []
    # First pass: every document's roles. Second pass: one spelling per party
    # across the deal -- an OCR'd copy reads "COW Technologies LLC" where the
    # text layer reads "CDW Technologies LLC" (live 010300, and the executive
    # summary printed the OCR spelling). For names within a letter or two of
    # each other, the spelling that occurs most across every atom's text wins.
    all_text = "\n".join(str(a.get("text") or "") for a in atoms).lower()
    per_doc_roles: dict[str, dict[str, str]] = {}
    per_doc_signers: dict[str, list[str]] = {}
    for d in documents or []:
        aid = str(d.get("artifact_id"))
        found = parties_for_document(by_doc.get(aid, []))
        roles, signers = found["roles"], found["signers"]
        if not roles and page_text_by_doc and page_text_by_doc.get(aid):
            roles = parties_from_page_text(page_text_by_doc[aid])
        per_doc_roles[aid] = roles
        per_doc_signers[aid] = signers
    _reconcile_party_spellings(per_doc_roles, all_text)
    for d in documents or []:
        aid = str(d.get("artifact_id"))
        roles, signers = per_doc_roles.get(aid, {}), per_doc_signers.get(aid, [])
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
