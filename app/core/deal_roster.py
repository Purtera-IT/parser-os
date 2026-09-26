"""Who is on this deal: one small table, not a card per signature.

A deal's people arrive twice. The message headers give an address, a domain and
a direction -- authoritative, always present, and enough to say which side
somebody is on. The signature block at the foot of a mail gives a NAME and a
JOB TITLE, which no header ever carries.

Neither is a statement anybody made. Nobody says their own footer, and five
"this person exists" cards reach nothing: on 010288 the stakeholder atoms are
referenced by zero packets and the workload matrix they nominally feed is all
zeros. So the people belong in a table, and the footers become chrome.

The catch is the order. Drop the signatures first and the deal loses the only
record of who anybody IS -- before this, an atom's own speaker said
``{"email": "t@purtera-it.com", "name": "T"}``, because the name was derived
from the address. The roster has to exist before the footers can go.

KEYED ON EMAIL, SO IT ACCUMULATES. One deal's roster is a convenience. The
same table across every deal is a record of who a person usually is -- which
side they sit on, what they are usually called, what they do -- and that is a
real signal for the heads that have to decide what a sentence is: a question
from a reseller's account executive is a different thing from the same words
from the customer's facilities manager.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

VERSION = "deal_roster_v1"

#: "Trent Torrence | Executive Vice President of Sales | t@x.com | 404.771.3490"
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
#: A job title, as signatures write them. Deliberately a shape, not a list:
#: any vocabulary here would be a list of the titles we happened to have seen.
_TITLE_HINT = re.compile(
    r"\b(officer|president|director|manager|executive|engineer|architect|lead|head|"
    r"specialist|coordinator|consultant|analyst|administrator|supervisor|"
    r"principal|partner|owner|founder|chief|vp|svp|evp|cto|ceo|coo|cfo|"
    r"account|sales|operations|solutions|technical|field|service)\b", re.I)
_SEP_RE = re.compile(r"\s*[|•·]\s*|\s{3,}")
#: A person's name: two to four capitalised words, no digits, no @.
_NAME_RE = re.compile(r"^[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){1,3}$")


def _parts(text: str) -> list[str]:
    return [p.strip() for p in _SEP_RE.split(str(text or "")) if p.strip()]


def _side_of(domain: str, ours: set[str]) -> str:
    d = (domain or "").lower()
    if not d:
        return "unknown"
    return "ours" if any(o and o in d for o in ours) else "theirs"


def read_signature(text: str) -> dict[str, str]:
    """name / role / phone / email from one signature block, or {}.

    A signature is pipe-separated in the shape the parsers emit. Anything that
    is not clearly a name, a title, an address or a number is left out rather
    than guessed at: a half-read title on a person record is worse than none,
    because it looks like knowledge.
    """
    out: dict[str, str] = {}
    chunks = _parts(text)
    if not chunks:
        return out
    for chunk in chunks:
        if "@" in chunk:
            m = _EMAIL_RE.search(chunk)
            if m and "email" not in out:
                out["email"] = m.group(0).lower()
            continue
        if _PHONE_RE.fullmatch(chunk.strip()):
            out.setdefault("phone", chunk.strip())
            continue
        if "name" not in out and _NAME_RE.match(chunk) and not _TITLE_HINT.search(chunk):
            out["name"] = chunk
            continue
        if _TITLE_HINT.search(chunk):
            # "Senior Client Executive, Commercial Majors" is ONE title. It
            # was being split on the comma into two people.
            out["role"] = out.get("role") or chunk
    return out


def build_deal_roster(*, atoms: list[Any], documents: list[dict[str, Any]],
                      our_domains: set[str] | None = None) -> dict[str, Any]:
    """The people on this deal, one row each, keyed on email."""
    ours = {d.lower() for d in (our_domains or set()) if d}
    if not ours:
        try:
            from app.core.document_parties import our_org_tokens

            ours = {t.lower() for t in (our_org_tokens() or set())}
        except Exception:
            ours = set()

    people: dict[str, dict[str, Any]] = {}

    def slot(email: str) -> dict[str, Any]:
        e = (email or "").strip().lower()
        return people.setdefault(e, {
            "email": e, "name": "", "role": "", "phone": "",
            "domain": e.split("@")[-1] if "@" in e else "",
            "side": "unknown", "messages_sent": 0, "seen_in": [],
        })

    # 1. Headers. Authoritative for the address and the side, and the only
    #    evidence that somebody actually wrote something rather than being
    #    quoted in somebody else's footer.
    for doc in documents or []:
        email = str(doc.get("sender_email") or "").strip().lower()
        if not _EMAIL_RE.fullmatch(email or ""):
            continue
        row = slot(email)
        row["messages_sent"] += 1
        row["domain"] = row["domain"] or email.split("@")[-1]
        row["side"] = _side_of(row["domain"], ours)
        name = str(doc.get("filename") or "")
        if name and name not in row["seen_in"]:
            row["seen_in"].append(name)

    # 2. Signatures. The only source of a name or a job title.
    for atom in atoms or []:
        try:
            kind = getattr(atom, "atom_type", None)
            kind = kind.value if hasattr(kind, "value") else str(kind or "")
            if kind != "stakeholder":
                continue
            got = read_signature(getattr(atom, "raw_text", "") or "")
            if not got.get("email"):
                continue
            row = slot(got["email"])
            row["domain"] = row["domain"] or got["email"].split("@")[-1]
            if row["side"] == "unknown":
                row["side"] = _side_of(row["domain"], ours)
            for field in ("name", "role", "phone"):
                if got.get(field) and not row[field]:
                    row[field] = got[field]
        except Exception:
            continue

    rows = sorted(people.values(),
                  key=lambda r: (-r["messages_sent"], r["side"], r["email"]))
    return {
        "version": VERSION,
        "people": rows,
        "by_side": dict(Counter(r["side"] for r in rows)),
        # Somebody quoted in a footer who never sent anything is on the deal
        # without being in the conversation -- worth seeing, never worth
        # emailing without asking first.
        "named_but_silent": [r["email"] for r in rows if not r["messages_sent"]],
    }


__all__ = ["build_deal_roster", "read_signature", "VERSION"]
