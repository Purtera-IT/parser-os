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

Rule 4 reads the document's AUTHOR where that is not its carrier (``document_authors``): a CRM record
(a HubSpot meeting/note/call export, a "HubSpot user" AI recap) is ours and internal; a file whose role
names its author is that author's (our SOW draft a partner forwarded back stays ours, a customer's RFQ we
uploaded is theirs), and so are its later revisions; a file we uploaded that the other side's message
names, and the files uploaded in the same batch, keep that sender.

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
#: A document that is an event (a message, note or call), never a file.
EVENT_DOC_KINDS = _MESSAGE_KINDS
#: Our quote going out in words, with no price in the line (000061, 2026-05-04: "Please see the attached
#: survey quote and install ROM."): an outbound line of ours that sends a quote, ROM, estimate, proposal,
#: pricing, SOW or LOE is the quote, as a priced one is. Asking for a quote ("request for quote", RFQ) is not.
QUOTE_SENT = re.compile(r"\b(?:attached|attaching|enclosed|here\s+is|here'?s|please\s+(?:see|find))\b[^.!?\n]{0,60}?"
                        r"\b(?:quotes?|quotation|rom|estimates?|proposal|pricing|sow|statement\s+of\s+work|loe)\b", re.I)
_ASKS_FOR_QUOTE = re.compile(r"\brequest(?:ing)?\s+(?:for\s+)?(?:a\s+)?(?:quot|proposal|pricing)|\bquote\s+request|\brf[qpi]\b",
                             re.I)


def sends_quote(text: Any) -> bool:
    """A line of ours that carries a price or sends our quote (``PRICE`` / ``QUOTE_SENT``)."""
    t = str(text or "")
    return bool(PRICE.search(t) or (QUOTE_SENT.search(t) and not _ASKS_FOR_QUOTE.search(t)))

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


#: Who a document's role sentence names as its author: "... provided by PurTera-IT to SHI" (ours),
#: "Request for quotation ..., issued by Medtronic" (the customer's).
_AUTHORED_BY = re.compile(r"\b(?:supplied|provided|prepared|issued|authored|written|created)\s+by\s+(?:the\s+)?"
                          r"([A-Za-z][\w&.-]*)", re.I)
#: A partner by the name its role sentence uses (the first label of PARTNER_DOMAINS: "CDW", "SHI").
PARTNER_NAMES = frozenset(d.split(".", 1)[0] for d in PARTNER_DOMAINS)
_NOT_A_PARTY = frozenset({"enterprise", "account", "technical", "project", "field", "team", "engineering",
                          "network", "operations", "site", "it", "our", "their", "its", "this", "that", "each",
                          "all", "one", "senior", "lead", "manager", "director", "staff", "internal", "external"})
_ROLE_WORDS = {"customer": "theirs", "client": "theirs", "partner": "partner", "reseller": "partner",
               "distributor": "partner"}


def role_author(role: Any) -> str:
    """The side whose paper a document is by its role sentence: ``ours``, ``partner``, ``theirs``, or ''
    when the role names no author (or names one this rule cannot place: "the provider")."""
    m = _AUTHORED_BY.search(str(role or ""))
    if not m:
        return ""
    raw = m.group(1)
    who = raw.lower().strip(".-")
    if who in _NOT_A_PARTY or (not raw[:1].isupper() and who not in _ROLE_WORDS):
        return ""                          # a job title or a description, not a party ("by technical team")
    if who.startswith(("pur", "sales")):
        return "ours"
    if who in PARTNER_NAMES:
        return "partner"
    if who in _ROLE_WORDS:
        return _ROLE_WORDS[who]
    if who in ("provider", "subcontractor", "vendor", "supplier", "us", "a", "an"):
        return ""
    return "theirs"


def is_ours(doc: dict[str, Any], author: dict[str, Any] | None = None) -> bool:
    """Our own paper: ``document_authors`` placed it with us (a SOW draft of ours a partner forwarded
    back stays ours), or there is no sign it came in from outside and its role (the parser's ``docRole``,
    when known) names no other author and no blank template."""
    if author and author.get("side") == "ours" and author.get("why") in ("role_author", "lineage"):
        return True
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
               rows: Iterable[dict[str, Any]] | None = None,
               authors: dict[str, dict[str, Any]] | None = None) -> list[tuple[float, str]]:
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
        if t is not None and kind in QUOTE_KINDS and is_ours(d, (authors or {}).get(str(d.get("doc_id") or ""))) \
                and (quoted is None or t < quoted):
            quoted = t
    kinds = {str(d.get("doc_id") or ""): str(d.get("kind") or "").strip().lower() for d in docs}
    for r in rows or ():
        if (r.get("speaker_side") == "ours" and r.get("doc_direction") == "outbound"
                and (kinds.get(str(r.get("doc_id") or "")) in _MESSAGE_KINDS
                     or str(r.get("doc_kind") or "") in _MESSAGE_KINDS)
                and sends_quote(r.get("text"))):
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

#: The CRM's own records: a HubSpot meeting, note or call export, and a note the CRM itself wrote
#: ("HubSpot user": the AI call recap). Ours, internal, whatever direction the upload was stamped with.
_CRM_RECORD = re.compile(r"-hs-(?:meeting|note|call)-", re.I)
_CRM_SYSTEM = re.compile(r"^\s*hubspot(?:\s+user)?\s*$", re.I)
#: Words a filename stem is matched on (runE provenance._stem_words): letters only, four or more.
_STEM_STOP = frozenset({"the", "and", "for", "with", "from", "this", "that", "copy", "final", "draft", "updated",
                        "update", "new", "docx", "xlsx", "pdf", "csv", "purtera", "file", "2026", "2025", "version"})
_LINEAGE_STOP = frozenset({"v", "rev", "version", "copy", "of", "final", "draft", "updated", "am", "pm", "jan",
                           "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
                           "docx", "doc", "xlsx", "xls", "pdf", "csv", "redline", "clean", "signed"})
_CALL_NAME = re.compile(r"\b(?:sync|call|meeting|transcript|recap|check-?in|catch-?up|debrief)\b", re.I)
#: Files uploaded together (one note, one email's attachments) arrive within this many seconds.
_BATCH_S = 120.0
_OUR_PAPER = QUOTE_KINDS | WON_KINDS


def _stem(filename: Any) -> str:
    return re.sub(r"\.[A-Za-z0-9]{2,4}$", "", str(filename or "").replace("\\", "/").rsplit("/", 1)[-1])


def _stem_words(filename: Any) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", _stem(filename))} - _STEM_STOP


def _lineage(filename: Any) -> str:
    """One draft's name across its revisions: "PurTeraIT_BitGo_..._SOW_v1-0" and "..._SOW_v2-0" and
    "..._SOW_v1-0-Aug-17-2026-05-12-09-5101-PM" are one document. '' for a name too short to say."""
    words = [w for w in re.findall(r"[a-z]+", _stem(filename).lower()) if w not in _LINEAGE_STOP]
    return " ".join(words) if len(words) >= 3 else ""


def _norm_words(text: Any) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower())) + " "


#: Our own naming on a file: our name in it, or the deal code our drafts are filed under ("010149 - CDW
#: Medtronic Teams Phone Migration (8.4.26_v1).docx", "PurTeraIT_BitGo_..._SOW_v1-0.docx").
_OUR_NAME = re.compile(r"pur\s*-?\s*terr?a|^\d{6}\s*-\s*(?!hs-)", re.I)


def _our_name(filename: Any) -> bool:
    return bool(_OUR_NAME.search(_stem(filename)))


def _blank_form(doc: dict[str, Any]) -> bool:
    """A blank form someone sent us to fill (``is_ours``: a template is not our paper unless it is ours)."""
    role, name = str(doc.get("role") or "").lower(), str(doc.get("filename") or "").lower()
    return ("template" in role or "template" in name) and "purtera" not in role


def _own_arrival(doc: dict[str, Any]) -> float | None:
    return epoch(doc.get("sent_at") if "sent_at" in doc else doc.get("arrival"))


def _side_origin(side: str, by: str) -> dict[str, Any]:
    direction = {"ours": "internal", "partner": "partner_inbound", "theirs": "inbound", "vendor": "inbound"}[side]
    return {"direction": direction, "sender_email": None, "via": by}


def document_authors(documents: Iterable[dict[str, Any]], rows: Iterable[dict[str, Any]] | None = None,
                     people: People | None = None) -> dict[str, dict[str, Any]]:
    """Who wrote each document, where that is not who carried it. ``doc_id -> {side, why, direction}``;
    a document left out is its carrier's (``sender`` rule 4).

    1. A CRM record (a ``-hs-meeting-/-hs-note-/-hs-call-`` export with no sender, or a note the CRM itself
       wrote, "HubSpot user") is ours and internal: the call recap of our own call is not the customer's.
    2. A file whose role names its author is that author's: our SOW draft a partner forwarded back stays
       ours; the customer's RFQ we uploaded to a note is the customer's.
    3. A later revision of such a file (the same name but for version and date) is the same author's.
    4. A file that reached us through our own upload or note (no outbound carrier), that a customer's or
       partner's message names by most of its name's words (three or more, company names aside)
       ("For reference - Teams Voice Site Migration Build Workbook ..."), is that sender's: an attachment
       keeps the sender of the message that first carried it, however it was forwarded or re-uploaded.
    5. A file uploaded in the same batch as a file rule 2 or 4 placed with the other side (one note, one
       email's attachments) is theirs too, unless it is our own paper kind or names its own author.
    ``direction`` is the document's direction once its author is known (''= unchanged): a CRM record is
    internal, a file of theirs that only our upload carried came in from them."""
    docs = [d for d in documents if isinstance(d, dict)]
    out: dict[str, dict[str, Any]] = {}
    oinfo = {}
    for d in docs:
        o = d.get("origin") if isinstance(d.get("origin"), dict) else {}
        oinfo[str(d.get("doc_id") or "")] = (o, doc_direction(o.get("direction"), o.get("sender_email")))
    lineage: dict[str, str] = {}
    pending: list[dict[str, Any]] = []
    for d in docs:
        doc, kind = str(d.get("doc_id") or ""), str(d.get("kind") or "").strip().lower()
        o, direction = oinfo[doc]
        name = str(d.get("filename") or "")
        has_sender = bool(address(o.get("sender_email")) or str(o.get("sender_name") or "").strip())
        if (kind in _MESSAGE_KINDS or _CRM_RECORD.search(name)) and (
                _CRM_SYSTEM.match(str(o.get("sender_name") or "")) and not address(o.get("sender_email"))
                or _CRM_RECORD.search(name) and not has_sender):
            out[doc] = {"side": "ours", "why": "crm_record", "direction": "internal"}
            continue
        if kind in ("email", "meeting"):
            continue
        side = role_author(d.get("role"))
        if side in ("theirs", "partner") and kind in _OUR_PAPER and not _blank_form(d) and direction not in (
                "inbound", "partner_inbound") and party(o.get("sender_email"))[0] not in ("theirs", "partner"):
            # Our own pricing workbook, carried by us, whose role sentence names the customer it prices
            # ("Onsite Level of Effort estimation ... supplied by Medtronic"): the carrier stands.
            side = ""
        came_in = direction in ("inbound", "partner_inbound") or party(o.get("sender_email"))[0] in (
            "theirs", "partner", "vendor")
        if side == "ours" and came_in and not _our_name(name):
            # A role sentence can misread the customer's own form as ours (010082: Sodexo's Smarthands SOW
            # request, "provided by PurTera-IT"). Over a carrier from outside it needs our own naming, or a
            # revision of the same draft that our side carried (below).
            pending.append(d)
            continue
        if side:
            out[doc] = {"side": side, "why": "role_author",
                        "direction": "" if side == "ours" or came_in else _side_origin(side, "")["direction"]}
            if _lineage(name):
                lineage.setdefault(_lineage(name), side)
    ours_carried = {_lineage(d.get("filename")) for d in docs
                    if _lineage(d.get("filename")) and oinfo[str(d.get("doc_id") or "")][1] in ("internal", "outbound")}
    for d in pending:
        if _lineage(d.get("filename")) in ours_carried:
            out[str(d.get("doc_id") or "")] = {"side": "ours", "why": "role_author", "direction": ""}
            lineage.setdefault(_lineage(d.get("filename")), "ours")
    for d in docs:
        doc = str(d.get("doc_id") or "")
        if doc in out or str(d.get("kind") or "").strip().lower() in EVENT_DOC_KINDS:
            continue
        side = lineage.get(_lineage(d.get("filename")))
        if side:
            _, direction = oinfo[doc]
            out[doc] = {"side": side, "why": "lineage",
                        "direction": "" if side == "ours" or direction in ("inbound", "partner_inbound")
                        else _side_origin(side, "")["direction"]}
    # Messages of the other side, by author: the text each one wrote (its own lines, quoted or not).
    said: dict[tuple[str, str], list[str]] = {}
    for r in rows or ():
        author = r.get("message_author")
        if not author:
            continue
        p = party(author)
        if not p[0] and people is not None:
            p = party(people.find(author))
        if p[0] in ("theirs", "partner"):
            said.setdefault((p[0], address(author) or str(author)), []).append(str(r.get("text") or ""))
    texts = [(side, _norm_words(" ".join(xs))) for (side, _), xs in said.items()]
    # A company's name says nothing about which file is meant ("Medtronic - Onsite LOE.xlsx").
    orgs = {a.split("@", 1)[1].split(".", 1)[0] for a in
            [address(r.get("message_author")) for r in rows or ()] + [address(oinfo[k][0].get("sender_email"))
                                                                       for k in oinfo] if a}
    for d in docs:
        doc, kind = str(d.get("doc_id") or ""), str(d.get("kind") or "").strip().lower()
        if doc in out or kind in EVENT_DOC_KINDS or kind in _OUR_PAPER and not _blank_form(d):
            continue
        o, direction = oinfo[doc]
        if direction not in ("internal", "") or party(o.get("sender_email"))[0] in ("theirs", "partner"):
            continue                       # it came in from them already, or we sent it out ourselves
        if _CALL_NAME.search(_stem(d.get("filename"))):
            continue                       # a call's own notes, named like the invite that announced it
        words = _stem_words(d.get("filename")) - orgs - {"purterait"}
        if len(words) < 3:
            continue
        best = None
        for side, body in texts:
            hits = sum(1 for w in words if f" {w} " in body or f" {w}s " in body)
            if hits >= 3 and hits * 3 >= 2 * len(words) and (
                    best is None or hits > best[0] or hits == best[0] and side == "theirs"):
                best = (hits, side)
        if best:
            out[doc] = {"side": best[1], "why": "named_in_their_message", "direction": _side_origin(best[1], "")["direction"]}
    # The batch a placed file arrived in.
    placed = [(str(d.get("doc_id") or ""), d) for d in docs if out.get(str(d.get("doc_id") or ""), {}).get("side")
              in ("theirs", "partner") and out[str(d.get("doc_id") or "")]["why"] != "crm_record"]
    for d in docs:
        doc, kind = str(d.get("doc_id") or ""), str(d.get("kind") or "").strip().lower()
        if doc in out or kind in EVENT_DOC_KINDS or kind in _OUR_PAPER and not _blank_form(d):
            continue
        o, direction = oinfo[doc]
        if direction not in ("internal", ""):
            continue
        # One batch: the same uploader (or, where neither upload records one, none), within _BATCH_S.
        # A document's own arrival only (``sent_at``): an arrival filled in from the document before it is
        # a guess, not a batch.
        t, who = _own_arrival(d), address(o.get("sender_email")) or str(o.get("sender_name") or "")
        if t is None:
            continue
        for pd, p in placed:
            po, _ = oinfo[pd]
            pt = _own_arrival(p)
            if pt is not None and abs(pt - t) <= _BATCH_S and (
                    address(po.get("sender_email")) or str(po.get("sender_name") or "")) == who:
                out[doc] = {"side": out[pd]["side"], "why": "same_batch", "direction": out[pd]["direction"]}
                break
    return out


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
    authors = document_authors(docs, rows, people)
    for r in rows:
        doc = str(r.get("doc_id") or "")
        o = origin.get(doc, {})
        who = authors.get(doc)
        r.update(sender(said_by=r.get("said_by"), message_author=r.get("message_author"),
                        speaker_name=r.get("speaker_name"),
                        origin=_side_origin(who["side"], who["why"]) if who else o, people=people))
        r["doc_direction"] = (who or {}).get("direction") or doc_direction(o.get("direction"), o.get("sender_email"))
    marks = milestones(docs, rows, authors)
    for r in rows:
        r["stage_at"] = stage_at(r.get("entered_at"), marks)
        for k in SENDER_INPUTS:
            r.pop(k, None)
    return rows


__all__ = ["OUR_DOMAINS", "PARTNER_DOMAINS", "SIDES_FULL", "ROLES_FULL", "ORG_KINDS", "DIRECTIONS",
           "STAGES", "SENDER_INPUTS", "People", "party", "doc_direction", "sender", "epoch", "is_ours",
           "milestones", "stage_at", "annotate", "PRICE", "QUOTE_SENT", "sends_quote", "role_author",
           "document_authors", "PARTNER_NAMES"]
