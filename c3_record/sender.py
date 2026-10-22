"""Who sent each line, and where the deal stood when it arrived.

One set of rules for the training export (``ml/c3/export.py``, from the
labeling page's export) and the parser's compile (``app/core/c3_type_base.py``,
from the compile's documents), so the base reads the same fields however the
deal was produced. Pure Python, no dependencies.

Sender, per line, first that answers:

1. the line's own speaker (``said_by``: the page's ``saidBy``, the atom's
   ``value.said_by``);
2. the message it was written in (an email line's quoted message author,
   ``"Name <address>"``);
3. a meeting line's speaker, resolved by name against the deal's people;
4. its document's sender: an email's From, the note's author, and for a file
   the email or note that carried it (``doc_origin``: ``sender_email`` and
   ``direction``; the carrier's, for an attachment);
5. else blank (an upload nobody is known to have sent).

``speaker_side`` is ``ours`` (our own domains), ``partner`` (a reseller /
distributor, by the domains the labeling page names), ``vendor`` (a
manufacturer we buy from: no domain list yet, so only a speaker already
marked ``vendor``), else ``theirs`` (the customer). ``speaker_role`` keeps the
trained vocabulary: ``internal``, ``reseller``, ``vendor``, ``unknown``.
``speaker_org`` is the sender's organisation kind: ``internal`` / ``external``.

``doc_direction`` (per document): ``outbound``, ``partner_outbound``,
``inbound``, ``partner_inbound``, ``internal``, or ''.

``stage_at`` (per line): where the deal stood when the line arrived, from OUR
own milestones, the same cut the training views use (runE provenance.py):
``scoping`` until our first priced paper (a SOW draft, quote, Deal Kit, calc,
rate card, proposal, ...) or our first priced outbound message, then
``quoted``, then ``contracting`` from the first signed SOW or purchase order
at or after that. A document counts as ours only when nothing shows it came
in from the customer or a partner (an inbound direction or carrier, an
outside sender) and its role does not name another party as its author or
call it a blank template: a customer's own SOW draft or a partner's PO is
not our milestone (010043: a partner PO made every pre-quote line
"contracting"). The CRM stage history is not used: the training deals carry
none, and a field read one way in training and another in serving is worse
than none.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable

OUR_DOMAINS = frozenset({"purtera-it.com", "purtera.com", "purteraits.com"})
#: Platform-infra artifact-provenance.js PARTNER_DOMAINS (the page's partner_inbound).
PARTNER_DOMAINS = frozenset({
    "cdw.com", "cdwg.com", "shi.com", "shidirect.com", "insight.com", "connection.com",
    "govconnection.com", "presidio.com", "wwt.com", "carahsoft.com", "softchoice.com",
    "eplus.com", "zones.com", "ingrammicro.com", "tdsynnex.com",
})
VENDOR_DOMAINS: frozenset[str] = frozenset()

SIDES_FULL = ("ours", "theirs", "partner", "vendor")
ROLES_FULL = ("internal", "reseller", "vendor", "unknown")
ORG_KINDS = ("internal", "external")
DIRECTIONS = ("outbound", "partner_outbound", "inbound", "partner_inbound", "internal")
STAGES = ("scoping", "quoted", "contracting")

QUOTE_KINDS = frozenset({"sow_draft", "quote", "sow", "deal_kit", "estimating_calc", "cost_breakdown",
                         "rate_card", "pricing_workbook", "change_order", "loe", "proposal", "estimate"})
WON_KINDS = frozenset({"sow_signed", "signed_sow", "purchase_order"})
#: A price in a line (runE provenance.PRICE): our priced outbound message is the quote.
PRICE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,.]*\s*(?:/\s?(?:hr|hour|site|tech|day)|per\s+(?:hour|hr|site|"
                   r"tech|technician|day|visit|location|drop|ap|camera|unit))\b|\b(?:usd|eur)\s?\d", re.I)
_PROVIDER = re.compile(r"\b(?:supplied|provided|prepared|issued)\s+by\s+([A-Za-z][\w&.-]*)", re.I)
_MESSAGE_KINDS = frozenset({"email", "note", "meeting", "crm_note"})

_ADDR = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def address(raw: Any) -> str:
    m = _ADDR.search(str(raw or ""))
    return m.group(0).lower() if m else ""


def _on(domain: str, domains: Iterable[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in domains)


def _name(raw: Any) -> str:
    """'Patrick Kelly <p@x.com>' -> 'patrick kelly'; 'Kelly, Patrick' -> 'patrick kelly'."""
    s = str(raw or "")
    s = s.split("<", 1)[0] if "<" in s else ("" if "@" in s else s)
    s = s.strip().strip('"').strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first} {last}"
    return " ".join(re.findall(r"[a-z]+", s.lower()))


def party(email: Any) -> tuple[str, str, str]:
    """(side, role, org) for an address, else ('', '', '')."""
    addr = address(email)
    if not addr:
        return "", "", ""
    dom = addr.split("@", 1)[1]
    if _on(dom, OUR_DOMAINS):
        return "ours", "internal", "internal"
    if _on(dom, PARTNER_DOMAINS):
        return "partner", "reseller", "external"
    if _on(dom, VENDOR_DOMAINS):
        return "vendor", "vendor", "external"
    return "theirs", "unknown", "external"


def doc_direction(direction: Any, sender_email: Any = None) -> str:
    """The document's direction in one vocabulary, a partner's mail told apart."""
    d = str(direction or "").strip().lower()
    d = {"purtera_outbound": "outbound", "customer_inbound": "inbound"}.get(d, d)
    if d not in DIRECTIONS:
        return ""
    if d == "inbound" and party(sender_email)[0] == "partner":
        return "partner_inbound"
    return d


def _by_direction(direction: str) -> tuple[str, str, str]:
    if direction in ("outbound", "partner_outbound", "internal"):
        return "ours", "internal", "internal"
    if direction == "partner_inbound":
        return "partner", "reseller", "external"
    if direction == "inbound":
        return "theirs", "unknown", "external"
    return "", "", ""


class People:
    """The deal's people by name, from every name/address pair it holds."""

    def __init__(self) -> None:
        self.full: dict[str, str] = {}
        self.first: dict[str, set[str]] = {}

    def add(self, name: Any, email: Any = None) -> None:
        addr = address(email) or address(name)
        n = _name(name)
        if not addr or not n:
            return
        self.full.setdefault(n, addr)
        self.first.setdefault(n.split()[0], set()).add(addr)

    def find(self, name: Any) -> str:
        n = _name(name)
        if not n:
            return ""
        if n in self.full:
            return self.full[n]
        # "Craig Radloff Desk Mini": a device named after its owner.
        for full, addr in sorted(self.full.items(), key=lambda kv: -len(kv[0])):
            if len(full.split()) >= 2 and (n + " ").startswith(full + " "):
                return addr
        toks = n.split()
        if len(toks) == 1 and len(self.first.get(toks[0], ())) == 1:
            return next(iter(self.first[toks[0]]))
        return ""


def sender(*, said_by: Any = None, message_author: Any = None, speaker_name: Any = None,
           origin: Any = None, people: People | None = None) -> dict[str, str]:
    """``{speaker_side, speaker_role, speaker_org}`` for one line (rules above)."""
    def out(t: tuple[str, str, str]) -> dict[str, str]:
        return {"speaker_side": t[0], "speaker_role": t[1], "speaker_org": t[2]}

    by = said_by if isinstance(said_by, dict) else {}
    side = str(by.get("side") or "").strip().lower()
    if side:
        p = party(by.get("email"))
        role = str(by.get("role") or by.get("role_guess") or "").strip().lower()
        if side == "ours":
            return out(("ours", "internal", "internal"))
        if side in ("partner", "vendor"):
            return out((side, "reseller" if side == "partner" else "vendor", "external"))
        if p[0] in ("partner", "vendor") or role == "reseller":
            return out(("partner", "reseller", "external"))
        return out(("theirs", role if role in ROLES_FULL else "unknown", "external"))
    if message_author:
        p = party(message_author)
        if not p[0] and people is not None:
            p = party(people.find(message_author))
        if p[0]:
            return out(p)
    if speaker_name:
        p = party(people.find(speaker_name)) if people is not None else ("", "", "")
        # A speaker on a call nobody in the deal's mail is: the other side.
        return out(p if p[0] else ("theirs", "unknown", ""))
    o = origin if isinstance(origin, dict) else {}
    p = party(o.get("sender_email"))
    if p[0]:
        return out(p)
    via = str(o.get("via") or "").lower()
    if via and via != "upload":
        return out(_by_direction(doc_direction(o.get("direction"), o.get("sender_email"))))
    return out(("", "", ""))


def epoch(t: Any) -> float | None:
    if t in (None, ""):
        return None
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    for parse in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            dt = parse(s)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def is_ours(doc: dict[str, Any]) -> bool:
    """Our own paper: no sign it came in from outside, and its role (the
    parser's ``docRole``, when known) names no other author and no blank
    template."""
    o = doc.get("origin") if isinstance(doc.get("origin"), dict) else {}
    direction = doc_direction(o.get("direction"), o.get("sender_email"))
    if direction in ("inbound", "partner_inbound"):
        return False
    if party(o.get("sender_email"))[0] in ("theirs", "partner", "vendor"):
        return False
    role = str(doc.get("role") or "").lower()
    named = [m.lower() for m in _PROVIDER.findall(role)]
    if named and not any(n.startswith(("pur", "sales")) for n in named) and "purtera" not in role:
        return False
    return "template" not in role or "purtera" in role


def milestones(documents: Iterable[dict[str, Any]],
               rows: Iterable[dict[str, Any]] | None = None) -> list[tuple[float, str]]:
    """(arrival, stage) at each of our milestones: our first priced paper or
    priced outbound message (``quoted``), then the first signed SOW or
    purchase order at or after it (``contracting``). ``documents``:
    ``{"kind", "arrival", "origin", "role"?}``; ``rows`` (optional): lines
    with ``text``, ``doc_id``, ``entered_at``, ``speaker_side`` and
    ``doc_direction`` (an outbound line of ours with a price is the quote)."""
    docs = [d for d in documents if isinstance(d, dict)]
    quoted: float | None = None
    for d in docs:
        kind = str(d.get("kind") or "").strip().lower()
        t = epoch(d.get("arrival"))
        if t is not None and kind in QUOTE_KINDS and is_ours(d) and (quoted is None or t < quoted):
            quoted = t
    kinds = {str(d.get("doc_id") or ""): str(d.get("kind") or "").strip().lower() for d in docs}
    for r in rows or ():
        if (r.get("speaker_side") == "ours" and r.get("doc_direction") == "outbound"
                and (kinds.get(str(r.get("doc_id") or "")) in _MESSAGE_KINDS
                     or str(r.get("doc_kind") or "") in _MESSAGE_KINDS)
                and PRICE.search(str(r.get("text") or ""))):
            t = epoch(r.get("entered_at"))
            if t is not None and (quoted is None or t < quoted):
                quoted = t
    if quoted is None:
        return []
    won = [t for d in docs if str(d.get("kind") or "").strip().lower() in WON_KINDS
           for t in [epoch(d.get("arrival"))] if t is not None and t >= quoted]
    return [(quoted, "quoted")] + ([(min(won), "contracting")] if won else [])


def stage_at(entered_at: Any, marks: list[tuple[float, str]]) -> str:
    """The furthest milestone reached by ``entered_at`` (inclusive), else
    ``scoping``; '' when the line has no time."""
    t = epoch(entered_at)
    if t is None:
        return ""
    best = 0
    for when, stage in marks:
        if when <= t:
            best = max(best, STAGES.index(stage))
    return STAGES[best]


#: What a row may carry for ``annotate``; taken off the row once read.
SENDER_INPUTS = ("said_by", "said_to", "message_author", "speaker_name")


def annotate(rows: list[dict[str, Any]], documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every row ``speaker_side``, ``speaker_role``, ``speaker_org``,
    ``doc_direction`` and ``stage_at``, in place.

    A row: ``doc_id``, ``entered_at`` and, when known, ``said_by`` / ``said_to``
    (``{name, email, side, role}``), ``message_author`` and ``speaker_name``.
    A document: ``doc_id``, ``kind``, ``arrival`` and ``origin``
    (``{direction, sender_email, sender_name, via}``: for a file, the message
    or note that carried it)."""
    docs = [d for d in documents if isinstance(d, dict)]
    origin = {str(d.get("doc_id") or ""): (d.get("origin") if isinstance(d.get("origin"), dict) else {})
              for d in docs}
    people = People()
    for o in origin.values():
        people.add(o.get("sender_name"), o.get("sender_email"))
    for r in rows:
        for p in [r.get("said_by"), *(r.get("said_to") or [])]:
            if isinstance(p, dict):
                people.add(p.get("name"), p.get("email"))
        people.add(r.get("message_author"))
    for r in rows:
        o = origin.get(str(r.get("doc_id") or ""), {})
        r.update(sender(said_by=r.get("said_by"), message_author=r.get("message_author"),
                        speaker_name=r.get("speaker_name"), origin=o, people=people))
        r["doc_direction"] = doc_direction(o.get("direction"), o.get("sender_email"))
    marks = milestones(docs, rows)
    for r in rows:
        r["stage_at"] = stage_at(r.get("entered_at"), marks)
        for k in SENDER_INPUTS:
            r.pop(k, None)
    return rows


__all__ = ["OUR_DOMAINS", "PARTNER_DOMAINS", "SIDES_FULL", "ROLES_FULL", "ORG_KINDS", "DIRECTIONS",
           "STAGES", "SENDER_INPUTS", "People", "party", "doc_direction", "sender", "epoch", "is_ours",
           "milestones", "stage_at", "annotate", "PRICE"]
