"""Whose each line is, and when it was written: the quoting cut, shared by training and serving.

Training's two views (``runE/views.py``) and the compile's quoting read call this one module, so a
line is a quoting input on both sides by the same rule. Each side only gathers what it can see into
the shapes below; every decision is made here.

A document (``docinfo[doc_id]``):
    direction   the page's / manifest's origin direction (``inbound``, ``purtera_outbound``, ``internal`` ...)
    sender      the origin's sender address (else its name)
    role        the lifecycle's role sentence (the page's ``docRole``)
    kind        what it is (a SOW draft, a Deal Kit, ``email`` ...); training's label doc_kind read when it has one
    carrier     ``("email", direction, sender)`` when an email carried it, else None
    carrier_at  that email's time (epoch seconds), else 0
    message     an email, CRM note, call or meeting
    filename    the source file's name

A line (``atoms``): ``label_key``, ``atom_id``, ``doc_id``, ``text``, ``entered_at`` (its document's
arrival, epoch or ISO), ``doc_kind``, ``speaker_side``, ``section``. ``lines[atom_id]`` is the message
a mail line was written in: ``(author, at, quoted, message index)``.

A document is OURS (answer key, never an input) when nothing shows it came from the customer or
partner (no inbound carrier, no inbound message that names it) AND it is our pricing or scoping paper,
unless its role names another party as author. A message line is OURS when its own author is PurTera
and its message carries a price or sends our quote (an internal note that talks rates counts). The CUT
is the earliest of: our paper's arrival, our first priced outbound message. A line is a quoting input
when it is not ours and its own time (the quoted message's date for a quoted line) is before the cut.
"""
from __future__ import annotations

import collections
import datetime as dt
import email.utils
import os
import re
from typing import Any, Iterable

from .sender import epoch, party, sends_quote, to_supplier_only

PAPER = {"sow_draft", "sow_signed", "signed_sow", "sow", "deal_kit", "estimating_calc", "cost_breakdown",
         "rate_card", "quote", "pricing_workbook", "change_order", "loe", "proposal", "estimate"}
SOW_LIKE = {"sow_draft", "sow_signed", "signed_sow", "sow", "change_order", "proposal"}
INBOUND = {"inbound", "customer_inbound", "partner_inbound", "vendor_inbound"}
#: Our own pricing workbooks: never an input line in either view (owner T1.6, 2026-10-23).
DK_KINDS = {"deal_kit", "estimating_calc", "cost_breakdown", "rate_card", "pricing_workbook"}
PRICE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,.]*\s*(?:/\s?(?:hr|hour|site|tech|day)|per\s+(?:hour|hr|site|"
                   r"tech|technician|day|visit|location|drop|ap|camera|unit))\b|\b(?:usd|eur)\s?\d", re.I)
OWN_PAPER = re.compile(r"\bdeal ?kit\b|statement of work|\bsow\b|cost rates|sel+ rates|project financials|"
                       r"\bestimating calc|\bcost breakdown|\brate card", re.I)
RATE_TALK = re.compile(r"\brates?\b|\bmargin\b|\bmarkup\b|\bROM\b", re.I)
# A CRM note that pastes the customer's or partner's own message: a forwarded header from another domain, or
# a greeting addressed to us.
PASTED_INBOUND = re.compile(r"from:\s*[^|\n]*@(?!purtera)|\b(?:hello|hi|dear|hey)\s+(?:the\s+)?purtera\b|"
                            r"\bpurtera\s+team\b", re.I)
PROVIDER = re.compile(r"\b(?:supplied|provided|prepared|issued)\s+by\s+([A-Za-z][\w&.-]*)", re.I)
# Our own estimate said in words on a call, a recap or a note (010149: our LOE walkthrough, "a 15% uplift
# for the fixed fee stuff", "two techs there for five days"): an our-speaker line that states LOE, hours,
# buffer, uplift, minutes per device, tech counts or a total price is ours (the answer key), not an input;
# so is a recap line whose same_as chain ends on one. The partner's lines stay inputs; the cut does not move.
_NUM = (r"(?:\d[\d,.]*%?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
        r"forty|fifty|sixty|hundred|half(?:\s+an?)?|an?|couple(?:\s+of)?)")
ESTIMATE_TALK = re.compile(
    r"\bLOE\b|\blevel[- ]of[- ]effort\b|\bbuffer\b|\buplift\b|\bcontingency\b|\bdelta\b|"
    rf"\b{_NUM}\s*(?:-|to|or)?\s*(?:{_NUM}\s*)?(?:man[- ]?)?(?:hours?|hrs?)\b|"
    r"\b(?:survey|migration|hypercare|install(?:ation)?|base|total|labor|onsite|on-site|project|billable|estimated|"
    r"allocated|tech|technician|travel)\s+hours?\b|\bhours?\s+(?:per|a|each|for each)\s+(?:site|device|location|"
    r"store|day|tech)\b|\bhours?\s+(?:that\s+)?(?:we|i)\s+(?:have\s+)?(?:allocated|estimated|quoted|budgeted|included)\b|"
    r"\b(?:adjusted|added|cut|reduced|padded)\s+(?:the\s+)?hours\b|"
    r"\b(?:survey|migration|hypercare)\s+(?:will\s+be|is|at)\s+(?:zero|\d)|"
    rf"\b(?:{_NUM}\s*)?(?:minutes?|mins?)\s+(?:per|a|an|each|for each)\s+\w+|"
    rf"\b{_NUM}\s*(?:-|to|or)?\s*(?:{_NUM}\s*)?(?:techs?|technicians?|engineers?|field techs?)\b|"
    r"\bnumber of (?:techs?|technicians?|engineers?)\b|\btechs?\s+(?:per|a|each|on)\s+(?:site|location|day)\b|"
    r"\btotal (?:price|cost|fee)\b", re.I)
ESTIMATE_KINDS = ("meeting", "note", "crm_note")
# A line in the buyer's voice is the buyer's, even pasted into our own note under our name (000123).
BUYER_VOICE = re.compile(r"\b(?:want|willing|able|have|need|going|prefer)\s+to\s+pay\b|\bwhichever\b[^.]{0,60}\bcheaper\b|"
                         r"\bopen to (?:either|both)\b|\bour budget\b|\bwe(?:'d| would) (?:be )?(?:ship|reus)", re.I)
STOP = {"the", "and", "for", "with", "from", "this", "that", "copy", "final", "draft", "updated", "update",
        "new", "docx", "xlsx", "pdf", "csv", "purtera", "file", "2026", "2025", "version"}
_MESSAGE_KINDS = ("email", "note", "meeting", "crm_note")
MESSAGE_NAME = re.compile(r"-hs-(note|email|meeting|call|task)-|fireflies|\.eml$|transcript", re.I)
_CONTAINER_NAME = re.compile(r"-hs-(note|email|meeting|call|task)-|fireflies|\.eml$", re.I)
OURS_DIRECTIONS = ("internal", "outbound", "purtera_outbound")
SUPPLIER_READS = {"third_party", "vendor"}
#: The paper kinds the document-kind head answers (ml/c3/judgment_heads.json ``doc_kind``, card
#: ``jdg:doc_kind``). A document's kind comes from that head's reading when a base serves it, and in training
#: from a label kind only when it is one of these (the head's own gold); anything else is no kind read.
DOC_KINDS = ("email", "call_transcript", "meeting_recap", "crm_note", "request_form", "rfp", "site_list",
             "site_survey_report", "site_checklist", "equipment_list", "vendor_quote_bom", "vendor_datasheet",
             "drawing", "customer_layout_deck", "install_guide", "process_document", "purchase_order", "invoice",
             "tax_form_w9", "credit_application", "nda", "partner_sow", "sow_template", "sow_draft", "sow_signed",
             "change_request", "quote", "deal_kit", "estimating_calc", "pricing_workbook", "photo", "other")
DOC_KIND_HEAD = "jdg:doc_kind"


#: The labels' older names for a head kind (jdg:doc_kind's answers fold them: "synonyms folded").
DOC_KIND_SYNONYMS = {"signed_sow": "sow_signed", "meeting_record": "meeting_recap", "crm_meeting_recap": "meeting_recap",
                     "crm_meeting_record": "meeting_recap", "transcript": "call_transcript",
                     "intake_record": "request_form", "intake_request": "request_form",
                     "intake_request_screenshot": "request_form", "site_photo": "photo", "image": "photo",
                     "email_image": "photo", "planning_brief_answer": "other", "document_comment": "other",
                     "case_manifest": "other"}


def head_kind(value: Any) -> str:
    """The document-kind head's answer for a kind (a label's older name folded to it); '' when the head has
    no such answer."""
    v = str(value or "").strip().lower()
    v = DOC_KIND_SYNONYMS.get(v, v)
    return v if v in DOC_KINDS else ""

# Owner rulings on single documents, by artifact id: "ours" (answer key, never an input) or "theirs" (an
# input). Flip one here when the owner rules; both sides read it.
RULINGS = {
    # 010045 "BB Kiosk SOW.pdf": docRole says provided by PurTera-IT; labels and text read it as a forwarded
    # install guide. Owner ruled THEIRS (the customer's runbook we probably retyped): an input.
    "art_4e0b98f29cdd4175": "theirs",     # owner ruling 2026-10-23
}


def t(value: Any) -> float:
    """Epoch seconds of a time (number, ISO or RFC date), 0.0 when unknown."""
    return epoch(value) or 0.0


def is_ours_addr(s: Any) -> bool:
    return "purtera" in str(s or "").lower()


def parse_when(s: Any) -> float:
    """A message date. RFC dates carry their zone; Outlook's quoted 'Tuesday, August 18, 2026 7:09 PM' does
    not, and is read as UTC (hours off at worst; the cut compares days of mail, not minutes)."""
    if not s:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    d = None
    if re.search(r"[+-]\d{4}\b|\bGMT\b|\bUTC\b", s):
        try:
            d = email.utils.parsedate_to_datetime(s)
        except (TypeError, ValueError, IndexError):
            d = None
    if d is None:
        try:
            from dateutil import parser as dparser
        except ImportError:  # pragma: no cover - the worker ships dateutil; without it only ISO/RFC dates read
            return epoch(s) or 0.0
        try:
            d = dparser.parse(re.sub(r"^(?:EDT|EST|CDT|CST|PDT|PST|MDT|MST|CEST|CET|BST)\s+", "", s), fuzzy=True)
        except (ValueError, OverflowError):
            return 0.0
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def zoned(s: Any) -> bool:
    return bool(s) and (isinstance(s, (int, float)) or bool(re.search(r"[+-]\d{4}\b|\bGMT\b|\bUTC\b|Z$", str(s).strip())))


def _addr(s: Any) -> str:
    m = re.search(r"[\w.%+-]+@[\w.-]+\.[a-z]{2,}", str(s or ""), re.I)
    return m.group(0).lower() if m else re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def stem_words(name: Any) -> set[str]:
    base = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", os.path.basename(str(name or "")))
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", base)} - STOP


def zone_fix(lines: dict[str, tuple], texts_of: dict[str, tuple[str, bool]]) -> None:
    """A quoted copy's Outlook date has no zone ("Monday, May 4, 2026 at 11:00 AM", read as UTC: four hours
    early in New York). When the same author's same line also appears with a zoned date within 14 hours
    (the message itself), that is its time (000061: our quote went out 15:00:56Z, not 11:00Z). In place."""
    zoned_at = collections.defaultdict(list)
    for aid, (text, z) in texts_of.items():
        ln = lines.get(aid)
        if z and ln and ln[1] > 0:
            zoned_at[(_addr(ln[0]), _norm(text))].append(ln[1])
    shift = {}
    for aid, (text, z) in texts_of.items():
        ln = lines.get(aid)
        if z or not ln or ln[1] <= 0:
            continue
        near = [x for x in zoned_at.get((_addr(ln[0]), _norm(text)), ()) if abs(x - ln[1]) <= 14 * 3600]
        if near:
            shift[(_addr(ln[0]), ln[1])] = min(near, key=lambda x: abs(x - ln[1]))
    # Every line of that quoted message (its header, greeting, signature) takes the same time.
    for aid, (text, z) in texts_of.items():
        ln = lines.get(aid)
        if not z and ln and (_addr(ln[0]), ln[1]) in shift:
            lines[aid] = (ln[0], shift[(_addr(ln[0]), ln[1])], ln[2], ln[3])


class Prov:
    def __init__(self) -> None:
        self.doc: dict[str, dict[str, Any]] = {}   # doc_id -> side, arrival, paper, ours, why, kind, message ...
        self.atom: dict[str, dict[str, Any]] = {}  # label_key -> at, author_ours, quoted, msg, doc
        self.cut = float("inf")
        self.cut_why = ""
        self.ours_atoms: set[str] = set()
        self.to_supplier: set[str] = set()         # our priced lines to a vendor / subcontractor (cost side)
        self.suppliers: set[str] = set()
        self.estimate_atoms: set[str] = set()      # our estimate said on a call / recap / note (ESTIMATE_TALK)
        self.estimate_chain: set[str] = set()      # recap copies whose same_as chain ends on one
        self.other_job: dict[str, str] = {}        # doc_id -> why (document_job.set_aside)
        self.source = ""


def supplier_domains(atoms: Iterable[dict[str, Any]], supplier_of: dict[str, str]) -> set[str]:
    """Outside domains the deal's own record places with a supplier: most lines that name one of its
    addresses read supplier third_party or vendor (010224, 010045). ``supplier_of``: label_key -> the
    line's supplier read. Never the customer or a reseller, whose lines read customer / partner."""
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for a in atoms:
        v = supplier_of.get(a["label_key"])
        if not v:
            continue
        for dom in set(re.findall(r"[\w.%+-]+@([\w-]+(?:\.[\w-]+)*\.[a-z]{2,})", str(a.get("text") or "").lower())):
            if party("x@" + dom)[0] == "theirs":
                votes[dom][v] += 1
    return {d for d, c in votes.items()
            if sum(c[x] for x in SUPPLIER_READS) * 2 > sum(c.values()) and sum(c[x] for x in SUPPLIER_READS) >= 3}


def decide(atoms: list[dict[str, Any]], docinfo: dict[str, dict[str, Any]], lines: dict[str, tuple],
           *, texts_of: dict[str, tuple[str, bool]] | None = None, to_of: dict[str, Any] | None = None,
           links: Iterable[dict[str, Any]] = (), suppliers: Iterable[str] = (),
           other_job: dict[str, str] | None = None, source: str = "",
           kinds: dict[str, str] | None = None) -> Prov:
    """Provenance of every document and line of one deal (see the module doc for the shapes). ``kinds``:
    doc_id -> the document-kind head's answer (``jdg:doc_kind``); where given it is the document's kind,
    over whatever the side gathered."""
    P = Prov()
    P.source = source
    P.other_job = dict(other_job or {})
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for a in atoms:
        by_doc[a.get("doc_id")].append(a)
    for d, k in (kinds or {}).items():
        if head_kind(k) and d in by_doc:
            docinfo.setdefault(d, {})["kind"] = head_kind(k)
    for d in by_doc:
        docinfo.setdefault(d, {})
        for k, v in (("direction", ""), ("sender", ""), ("role", ""), ("kind", ""), ("carrier", None),
                     ("carrier_at", 0.0), ("message", False), ("filename", "")):
            docinfo[d].setdefault(k, v)
    stamp = {d: min((t(a.get("entered_at")) for a in xs), default=0.0) for d, xs in by_doc.items()}
    to_of = to_of or {}
    if texts_of:
        zone_fix(lines, texts_of)
    # A message container (email, CRM note, call), recognized by its atoms' kind or its export name. This only
    # says the lines are conversation; whose they are comes from each line's author.
    for d, xs in by_doc.items():
        i = docinfo[d]
        if any(a.get("doc_kind") in _MESSAGE_KINDS for a in xs) or _CONTAINER_NAME.search(i["filename"] or d or ""):
            i["message"] = True

    def author_ours(author, doc):
        if author:
            return is_ours_addr(author)
        i = docinfo[doc]
        return is_ours_addr(i["sender"]) or i["direction"] in ("purtera_outbound", "outbound", "internal")

    # Per line: its own time and author.
    for a in atoms:
        ln = lines.get(a.get("atom_id"))
        d = a.get("doc_id")
        if ln and ln[1] > 0:
            # A quoted message cannot be younger than the email that carries it.
            at, author, quoted, msg = min(ln[1], t(a.get("entered_at")) or ln[1]), ln[0], ln[2], ln[3]
        else:
            at, author, quoted, msg = t(a.get("entered_at")), (ln[0] if ln else ""), bool(ln and ln[2]), (ln[3] if ln else None)
        P.atom[a["label_key"]] = {"at": at, "author_ours": author_ours(author, d), "quoted": quoted,
                                  "msg": (d, msg), "doc": d}
    # Documents: carrier side, a theirs message naming the file, authorship.
    theirs_msgs = []
    for d, xs in by_doc.items():
        i = docinfo[d]
        if i["message"] or i["direction"] in INBOUND:
            text = " ".join(str(a.get("text") or "") for a in xs).lower()
            if i["direction"] in INBOUND or PASTED_INBOUND.search(text):
                theirs_msgs.append((stamp[d], d, text))
    for d, xs in by_doc.items():
        i = docinfo[d]
        arrival = i["carrier_at"] or stamp[d]
        side, why = "ours_side", "no inbound carrier"
        if i["carrier"] and i["carrier"][1] in INBOUND:
            side, why = "theirs", f"carried by an inbound email ({i['carrier'][2]})"
        elif i["direction"] in INBOUND:
            side, why = "theirs", f"direction {i['direction']}"
        elif not i["message"] and i["kind"] not in PAPER:
            # A file uploaded with no carrier that an inbound or forwarded message announces is that sender's,
            # and is as old as that message (010269: the partner's guide and forms re-uploaded with our CALC).
            words = stem_words(i["filename"])
            for tm, m, text in sorted(theirs_msgs):
                hits = sum(1 for w in words if w in text)
                if m != d and len(words) >= 2 and hits >= 2 and hits * 2 >= len(words) and tm <= arrival:
                    side, why, arrival = "theirs", f"named in an inbound/forwarded message {m}", min(arrival, tm)
                    break
        role = i["role"].lower()
        named = [x.lower() for x in PROVIDER.findall(role)]
        other_author = bool(named) and not any(x.startswith(("pur", "sales")) for x in named) and "purtera" not in role
        priced = sum(1 for a in xs if PRICE.search(str(a.get("text") or ""))) >= 2
        paper = i["kind"] in PAPER
        if not paper and i["kind"] in ("", "file") and priced and not i["message"]:
            # No kind read anywhere: our pricing paper by its own headings or name (010353's "VC Links.xlsx").
            heads = " ".join({str(a.get("section") or "") for a in xs}) + " " + i["filename"]
            if OWN_PAPER.search(heads):
                paper, i["kind"] = True, "deal_kit" if re.search(r"deal ?kit|rates", heads, re.I) else "sow_draft"
        if side == "theirs" and paper and not i["message"] and xs and all(a.get("speaker_side") == "ours" for a in xs):
            # Our own draft a partner forwarded back (c3_record.sender.document_authors reads its author as us).
            side, why, other_author = "ours_side", "our paper forwarded back (author ours)", False
        ours = (side == "ours_side" and not i["message"] and not other_author
                and ("template" not in role or "purtera" in role)
                and paper and d not in P.other_job)
        if side == "ours_side" and not i["message"] and not ours and paper:
            why = "another party's authorship (docRole)" if other_author else "a blank template"
        ruling = RULINGS.get(d)
        if ruling:
            ours, why = ruling == "ours", f"owner ruling: {ruling}"
            side = "ours_side" if ours else "theirs"
        P.doc[d] = {"side": side, "arrival": arrival, "paper": paper, "priced": priced, "ours": ours,
                    "why": why if not ours else f"ours: {why}; kind {i['kind'] or '-'}" + ("; priced" if priced else ""),
                    "kind": i["kind"], "message": i["message"], "direction": i["direction"],
                    "filename": i["filename"]}
        if ours:
            if arrival < P.cut:
                P.cut, P.cut_why = arrival, f"our {i['kind'] or 'paper'} {d}"
            for a in xs:
                P.ours_atoms.add(a["label_key"])
    # A file's lines are as old as the file's arrival (its carrier, or the inbound message that names it).
    for d, xs in by_doc.items():
        if not P.doc[d]["message"]:
            for a in xs:
                P.atom[a["label_key"]]["at"] = P.doc[d]["arrival"]
    fwd_docs = {m for _, m, _ in theirs_msgs}
    # Our priced lines to a supplier (every outside recipient a vendor or subcontractor) are our cost side,
    # never our quote (owner ruling, 010224). Their quoted copies in later mail keep that.
    P.suppliers = set(suppliers)
    to_supplier = set()
    for a in atoms:
        if to_supplier_only(to_of.get(a.get("atom_id")), P.suppliers):
            ln = lines.get(a.get("atom_id"))
            to_supplier.add((_addr(ln[0] if ln else ""), _norm(a.get("text"))))
    P.to_supplier = {a["label_key"] for a in atoms
                     if (_addr((lines.get(a.get("atom_id")) or ("",))[0]), _norm(a.get("text"))) in to_supplier}
    # Messages: our own priced lines; an outbound one sets the cut.
    msgs = collections.defaultdict(list)
    for a in atoms:
        p = P.atom[a["label_key"]]
        if docinfo[a["doc_id"]]["message"] or a.get("doc_kind") in _MESSAGE_KINDS:
            msgs[p["msg"]].append(a)
    for (d, _mi), xs in msgs.items():
        own = [a for a in xs if P.atom[a["label_key"]]["author_ours"]]
        if not own or d in P.other_job:
            continue
        i = docinfo[d]
        meeting = (i["kind"] or "").startswith("meeting") or "fireflies" in i["filename"] or "-meeting-" in i["filename"]
        if meeting:
            continue                      # a call's lines are speech; who priced is not in the line's author
        if d in fwd_docs and not any(lines.get(a.get("atom_id")) for a in own):
            continue                      # a CRM note pasting an inbound email: the lines are the sender's
        internal = i["direction"] == "internal" and not any(P.atom[a["label_key"]]["quoted"] for a in own)
        # A priced line, or our quote sent in words (c3_record.sender.sends_quote, the rule stage_at uses).
        hit = [a for a in own if a["label_key"] not in P.to_supplier and (
            sends_quote(a.get("text")) or (internal and RATE_TALK.search(str(a.get("text") or ""))))]
        if not hit:
            continue
        for a in own:
            P.ours_atoms.add(a["label_key"])
        tm = min(P.atom[a["label_key"]]["at"] for a in own)
        outbound = not internal and (i["direction"] in ("purtera_outbound", "outbound") or any(
            P.atom[a["label_key"]]["quoted"] for a in own) or i["direction"] in INBOUND)
        if outbound and 0 < tm < P.cut:
            P.cut, P.cut_why = tm, f"our priced message in {d}"
    estimate_lines(P, atoms, docinfo, links)
    return P


def estimate_lines(P: Prov, atoms: list[dict[str, Any]], docinfo: dict[str, dict[str, Any]],
                   links: Iterable[dict[str, Any]] = ()) -> None:
    """ESTIMATE_TALK: our speaker's call, recap and note lines that state our estimate, and the recap lines
    whose same_as chain ends on one, are ours (never inputs). The cut stays where it is."""
    for a in atoms:
        d = a.get("doc_id")
        if d in P.other_job or a["label_key"] in P.ours_atoms:
            continue
        kind = str(a.get("doc_kind") or "")
        if kind not in ESTIMATE_KINDS and not (docinfo.get(d, {}).get("message") and kind != "email"):
            continue
        text = str(a.get("text") or "")
        if a.get("speaker_side") == "ours" and (ESTIMATE_TALK.search(text) or PRICE.search(text)) \
                and not BUYER_VOICE.search(text):
            P.estimate_atoms.add(a["label_key"])
    same = collections.defaultdict(set)
    for k in links or ():
        f, to = str(k.get("from_key") or ""), k.get("to_label_key")
        if k.get("relation") == "same_as" and to and f.startswith("lbl_"):
            same[f].add(to)
    for a in atoms:
        k = a["label_key"]
        if k in P.estimate_atoms or k in P.ours_atoms or a.get("doc_id") in P.other_job:
            continue
        if str(a.get("doc_kind") or "") not in ESTIMATE_KINDS or a.get("speaker_side") != "ours":
            continue                       # only our own recap / note copies; another party's line stays theirs
        seen, todo = set(), [k]
        while todo:
            for x in same.get(todo.pop(), ()):
                if x not in seen:
                    seen.add(x)
                    todo.append(x)
        if seen & P.estimate_atoms:
            P.estimate_chain.add(k)
    P.ours_atoms |= P.estimate_atoms | P.estimate_chain


def is_input(P: Prov, a: dict[str, Any]) -> bool:
    """Not ours, and written before the cut. A customer's or partner's line stamped the same instant as the
    cut (a batch upload with our paper) came first: the intake was there before we priced it."""
    p = P.atom[a["label_key"]]
    if a["label_key"] in P.ours_atoms or P.doc[a["doc_id"]]["ours"] or p["at"] <= 0 or a["doc_id"] in P.other_job:
        return False
    theirs = P.doc[a["doc_id"]]["side"] == "theirs" or (P.doc[a["doc_id"]]["message"] and not p["author_ours"])
    return p["at"] < P.cut or (theirs and p["at"] == P.cut)


def theirs_line(P: Prov, a: dict[str, Any]) -> bool:
    """A customer's or partner's line: its document is theirs, or it is another party's message line."""
    d = P.doc[a["doc_id"]]
    return d["side"] == "theirs" or (d["message"] and not P.atom[a["label_key"]]["author_ours"])


def pricing_workbook_docs(P: Prov, atoms: Iterable[dict[str, Any]]) -> set[str]:
    """Our own pricing workbooks: Deal Kit / CALC / LOE / cost breakdown documents; never a customer's form."""
    kinds: dict[Any, set] = collections.defaultdict(set)
    for a in atoms:
        kinds[a.get("doc_id")].add(a.get("doc_kind"))
    out = set()
    for doc, ks in kinds.items():
        x = P.doc.get(doc) or {}
        if x.get("kind") == "rfp" or x.get("side") == "theirs":
            continue
        if x.get("kind") in DK_KINDS or (ks and ks <= DK_KINDS):
            out.add(doc)
    return out


def pricing_workbook_keys(P: Prov, atoms: list[dict[str, Any]]) -> set[str]:
    """Label keys of our pricing-workbook lines: every line of a workbook document, plus Deal Kit lines in a
    mixed bucket (010353: workbook cells and SOW lines share an empty document id). Never an input."""
    docs = pricing_workbook_docs(P, atoms)
    out = set()
    for a in atoms:
        x = P.doc.get(a.get("doc_id")) or {}
        if a.get("doc_id") in docs or (a.get("doc_kind") in DK_KINDS and x.get("kind") != "rfp" and x.get("side") != "theirs"):
            out.add(a["label_key"])
    return out


def quoting_inputs(P: Prov, atoms: list[dict[str, Any]], *, exclude: Iterable[str] = (),
                   as_of: Any = None) -> list[int]:
    """Indices of the quoting view's input lines: ``is_input``, not a pricing-workbook line (``exclude``:
    label keys), and ARRIVED before the cut (its document's arrival, the ``entered_at`` serving sends), a
    customer's line stamped the same instant as our paper included. A pre-cut message quoted inside a
    post-cut email is not an input: a read at the cut has no such email. ``as_of`` (a quoting read's date)
    moves the cut earlier, never later."""
    cut = P.cut
    lim = t(as_of) if as_of not in (None, "") else 0.0
    if lim > 0 and lim < cut:
        P.cut, P.cut_why = lim, "the read's as_of"
        cut = lim
    ex = set(exclude)
    keep = []
    for i, a in enumerate(atoms):
        if a["label_key"] in ex or not is_input(P, a):
            continue
        w = t(a.get("entered_at"))
        if w < cut or (w == cut and theirs_line(P, a)):
            keep.append(i)
    return keep


def estimator_sources(atoms: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]]) -> tuple[float, str, set[str]]:
    """The same rules for an estimator file (atoms only, doc_id = filename) with the deal's parser manifests
    (direction, sender_domain, authored_at per file). Returns (cut, why, set of ours label_keys)."""
    import json

    by = collections.defaultdict(list)
    for a in atoms:
        by[a.get("doc_id")].append(a)
    cut, why, ours = float("inf"), "", set()
    for d, xs in by.items():
        art = artifacts.get(d) or {}
        direction = str(art.get("direction") or "")
        ours_side = direction in OURS_DIRECTIONS or is_ours_addr(art.get("sender_domain")) and direction not in INBOUND
        if not ours_side or RULINGS.get(d) == "theirs":
            continue
        message = bool(MESSAGE_NAME.search(d or ""))
        internal = direction == "internal"
        meta = art.get("metadata")
        try:
            title = str((json.loads(meta) if isinstance(meta, str) else meta or {}).get("title") or "")
        except ValueError:
            title = ""
        if message and internal and PASTED_INBOUND.search(title + " " + " ".join(a["text"] for a in xs)):
            continue                      # our note pasting the customer's message: the lines are theirs
        hit = [a for a in xs if PRICE.search(a["text"]) or (message and internal and RATE_TALK.search(a["text"]))]
        if not message and len(hit) >= 2:
            ours |= {a["label_key"] for a in xs}
            tm = min(t(a.get("entered_at")) for a in xs)
            if 0 < tm < cut:
                cut, why = tm, f"our priced file {d[:60]}"
        elif message and hit:
            ours |= {a["label_key"] for a in xs}
            tm = min(t(a.get("entered_at")) for a in hit)
            if not internal and 0 < tm < cut:
                cut, why = tm, f"our priced message {d[:60]}"
    return cut, why, ours


__all__ = ["DOC_KINDS", "DOC_KIND_HEAD", "DOC_KIND_SYNONYMS", "head_kind", "PAPER", "SOW_LIKE", "INBOUND", "DK_KINDS", "PRICE", "RULINGS", "Prov", "decide", "is_input",
           "theirs_line", "quoting_inputs", "pricing_workbook_docs", "pricing_workbook_keys", "supplier_domains",
           "estimate_lines", "estimator_sources", "parse_when", "zoned", "zone_fix", "is_ours_addr", "t"]
