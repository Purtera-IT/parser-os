"""Who said it, to whom, and for which company.

A PM reading "we provide the parts that connect the PC to the relay" has to
know who "we" is. The parser knew the sender's address and nothing else, so
every atom read as if it came from nowhere: our own account exec and the
reseller's rep looked identical on the card.

A party is an email address resolved to a person, a company (its domain) and
a SIDE: ``ours`` for our own staff, ``theirs`` for everyone else. The side is
the one thing we can always be sure of -- our own mail domain is not a guess.
What that company IS to this deal (the customer, the reseller, the installer,
a manufacturer) is a judgement, so it ships as ``role_guess`` from the domain
and the PM confirms it.
"""
from __future__ import annotations

import re
from typing import Any

#: Our own mail domains. Everything else is the other side of the table.
OUR_DOMAINS = frozenset({"purtera-it.com", "purtera.com", "purteraits.com"})

_ADDR_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_NAME_RE = re.compile(r'^\s*"?(?P<name>[^"<@]+?)"?\s*<')

#: Roles a company can play. The guess is a starting point on the card, never
#: a fact: "cdw.com" is a reseller on this deal and a manufacturer's partner
#: on the next one.
ROLE_RESELLER = "reseller"
ROLE_INTERNAL = "internal"
ROLE_UNKNOWN = "unknown"

#: Distributors and resellers we meet often enough to name. Everything else
#: is unknown until a human says otherwise.
_KNOWN_RESELLERS = frozenset({"cdw.com", "shi.com", "connection.com", "insight.com", "zones.com", "shidirect.com"})


def address_of(raw: str) -> str:
    m = _ADDR_RE.search(str(raw or ""))
    return m.group(0).lower() if m else ""


def display_name_of(raw: str) -> str:
    m = _NAME_RE.match(str(raw or ""))
    if m:
        name = " ".join(m.group("name").split())
        if name and "@" not in name:
            return name
    addr = address_of(raw)
    return addr.split("@")[0].replace(".", " ").title() if addr else ""


def domain_of(raw: str) -> str:
    addr = address_of(raw)
    return addr.split("@")[1] if "@" in addr else ""


def party(raw: str) -> dict[str, Any] | None:
    """One person, as much as we honestly know about them."""
    addr = address_of(raw)
    if not addr:
        return None
    domain = domain_of(addr)
    ours = domain in OUR_DOMAINS
    return {
        "email": addr,
        "name": display_name_of(raw),
        "company": domain,
        "side": "ours" if ours else "theirs",
        "role_guess": ROLE_INTERNAL if ours else (ROLE_RESELLER if domain in _KNOWN_RESELLERS else ROLE_UNKNOWN),
    }


def parties_for_message(thread_block: dict[str, Any] | None) -> dict[str, Any]:
    """``{said_by, said_to}`` for one message, from its thread block."""
    tb = thread_block if isinstance(thread_block, dict) else {}
    said_by = party(str(tb.get("sender") or ""))
    said_to = []
    for raw in list(tb.get("to") or []) + list(tb.get("cc") or []):
        p = party(str(raw))
        if p and all(p["email"] != x["email"] for x in said_to):
            said_to.append(p)
    out: dict[str, Any] = {}
    if said_by:
        out["said_by"] = said_by
    if said_to:
        out["said_to"] = said_to[:8]
        # "we told the customer" vs "we said it among ourselves" -- the
        # difference between a commitment and a thought.
        out["internal_only"] = bool(said_by and said_by["side"] == "ours") and all(p["side"] == "ours" for p in said_to)
    return out


def stamp_parties(atoms: list[Any]) -> int:
    """Put who-said-it-to-whom on every email atom. Returns the number stamped."""
    stamped = 0
    for atom in atoms:
        v = getattr(atom, "value", None)
        if not isinstance(v, dict):
            continue
        tb = v.get("email_thread")
        if not isinstance(tb, dict) or v.get("said_by"):
            continue
        info = parties_for_message(tb)
        if not info:
            continue
        v.update(info)
        atom.value = v
        stamped += 1
    return stamped


__all__ = ["party", "parties_for_message", "stamp_parties", "address_of", "domain_of", "OUR_DOMAINS"]
