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

``stage_at`` (per line): where the deal stood when the line arrived, from our
own milestones among the deal's documents, the line's own document included:
``scoping`` until the first SOW draft or quote arrives, then ``quoted``, then
``contracting`` from the first signed SOW or purchase order. The CRM stage
history is not used: the training deals carry none, and a field read one way
in training and another in serving is worse than none.
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

QUOTE_KINDS = frozenset({"sow_draft", "quote", "sow"})
WON_KINDS = frozenset({"sow_signed", "purchase_order"})

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


def milestones(documents: Iterable[dict[str, Any]]) -> list[tuple[float, str]]:
    """(arrival, stage) at each of our milestones: the first quote or SOW
    draft (``quoted``), the first signed SOW or purchase order
    (``contracting``). ``documents``: ``{"kind", "arrival"}``."""
    first: dict[str, float] = {}
    for d in documents:
        kind = str(d.get("kind") or "").strip().lower()
        t = epoch(d.get("arrival"))
        if t is None:
            continue
        stage = "quoted" if kind in QUOTE_KINDS else "contracting" if kind in WON_KINDS else ""
        if stage and (stage not in first or t < first[stage]):
            first[stage] = t
    return sorted((t, s) for s, t in first.items())


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
    marks = milestones(docs)
    for r in rows:
        o = origin.get(str(r.get("doc_id") or ""), {})
        r.update(sender(said_by=r.get("said_by"), message_author=r.get("message_author"),
                        speaker_name=r.get("speaker_name"), origin=o, people=people))
        r["doc_direction"] = doc_direction(o.get("direction"), o.get("sender_email"))
        r["stage_at"] = stage_at(r.get("entered_at"), marks)
        for k in SENDER_INPUTS:
            r.pop(k, None)
    return rows


__all__ = ["OUR_DOMAINS", "PARTNER_DOMAINS", "SIDES_FULL", "ROLES_FULL", "ORG_KINDS", "DIRECTIONS",
           "STAGES", "SENDER_INPUTS", "People", "party", "doc_direction", "sender", "epoch", "milestones",
           "stage_at", "annotate"]
