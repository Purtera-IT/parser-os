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
    # A machine (echosign@, dse@docusign.net, noreply@) said nothing and is
    # nobody's party; see app.core.automated_senders.
    from app.core.automated_senders import is_automated_sender

    if is_automated_sender(raw):
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
    # The message the line was written in, not the file it was found in. A
    # reply carries older messages quoted beneath it; their lines were said by
    # THAT message's author, to people the reply's header does not name. Live
    # 010087: Stephanie's quoted 7/7 email was credited to Trent, who only
    # replied above it.
    msg = tb.get("message") if isinstance(tb.get("message"), dict) else {}
    quoted = bool(msg.get("quoted")) or int(msg.get("index") or 0) > 0
    if quoted:
        said_by = party(str(msg.get("author") or ""))
        recipients: list = []
    else:
        said_by = party(str(tb.get("sender") or msg.get("author") or ""))
        recipients = list(tb.get("to") or []) + list(tb.get("cc") or [])
    said_to = []
    for raw in recipients:
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


#: The From: row of an email pasted into a note: "From: Stephanie Hechsel
#: <s@amtivo.com>", or the one-line "From: X | Sent: Y" form.
_PASTED_FROM_RE = re.compile(r"^[\s>*_]*from\s*:[\s*_]*(?P<who>[^|]+?)\s*(?:\|.*)?$", re.I)
#: A Gmail-style attribution: "On Tue, Jul 7, 2026 at 3:36 PM Stephanie <s@x.com> wrote:".
_PASTED_WROTE_RE = re.compile(r"^[\s>]*on\s.+?\d.*?(?P<who>[A-Z][^<>]*?<[^<>@\s]+@[^<>\s]+>)\s*wrote:\s*$", re.I)


def stamp_note_parties(atoms: list[Any]) -> int:
    """Who said each line of a HubSpot note.

    A note is written by its HubSpot author, except where the author pasted an
    email into it: from that email's "From:" row on, the lines are the
    pasted message's sender's. Live 010087: note 112490900231 is Stephanie's
    equipment-list email pasted by Trent, and its lines had no ``said_by``.
    Same note only; lines without a line number are left alone. Returns the
    number stamped. Atoms that already name a speaker keep it.
    """
    def _loc(a: Any) -> dict:
        refs = getattr(a, "source_refs", None) or []
        loc = getattr(refs[0], "locator", None) if refs else None
        return loc if isinstance(loc, dict) else {}

    def _line(a: Any) -> int | None:
        loc = _loc(a)
        ln = loc.get("line_start") if loc.get("line_start") is not None else loc.get("line")
        try:
            return int(ln) if ln is not None else None
        except (TypeError, ValueError):
            return None

    by_doc: dict[str, list[Any]] = {}
    for a in atoms or []:
        by_doc.setdefault(str(getattr(a, "artifact_id", "") or ""), []).append(a)
    stamped = 0
    for group in by_doc.values():
        meta = next((a for a in group if isinstance(getattr(a, "value", None), dict)
                     and a.value.get("kind") == "hubspot_note_meta"), None)
        if meta is None:
            continue
        mv = meta.value
        author = str(mv.get("author_email") or "")
        if author and mv.get("author"):
            author = f"{mv.get('author')} <{author}>"
        turns: list[tuple[int, str]] = []
        for a in group:
            ln = _line(a)
            if ln is None or a is meta:
                continue
            text = str(getattr(a, "raw_text", "") or "").strip()
            m = _PASTED_FROM_RE.match(text) or _PASTED_WROTE_RE.match(text)
            if m:
                turns.append((ln, m.group("who").strip()))
        turns.sort()
        for a in group:
            v = getattr(a, "value", None)
            if a is meta or not isinstance(v, dict) or v.get("said_by") or "email_thread" in v:
                continue
            ln = _line(a)
            if ln is None:
                continue
            who = author
            for at, sender in turns:
                if at <= ln:
                    who = sender
            p = party(who)
            if not p:
                continue
            v["said_by"] = p
            if who != author:
                v["pasted_email_from"] = who
            a.value = v
            stamped += 1
    return stamped


__all__ = ["party", "parties_for_message", "stamp_parties", "stamp_note_parties", "address_of", "domain_of", "OUR_DOMAINS"]
