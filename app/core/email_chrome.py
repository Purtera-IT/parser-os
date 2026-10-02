"""Mail chrome: the lines of an email that are furniture, not something dropped.

A signature ("PurTera-IT.com<https://urldefense.com/...>", "Email:
x@y.com<mailto:x@y.com>", "Mobile: 555 ..."), the header rows of a quoted
message ("From:", "Sent:", "To:", "Cc:", "Subject:"), the "On ... wrote:"
line above it, and the greeting and sign-off around a body are not content a
gate threw away. Listed as ``suppressed`` they read as possible misses: on
deal 000132 every email carried 33-64 of them, and the labeler's Missed tab
offered each as something the parser lost.

So they get their own class, ``chrome``, recorded with a reason (still
visible, never silent) and never counted as suppressed or as unread. A line
that only LOOKS like chrome because it is short or has a link in it is not
this: "Diagram: <https://...>" names a document the job depends on, and
"40 drops," is a fact. Only the shapes below qualify.
"""
from __future__ import annotations

import re
from typing import Any

#: Reasons an admission regex gave a line it refused that mean "mail chrome".
#: ``banter`` is talk, not furniture, and stays where it is.
ADMISSION_CHROME_REASONS = frozenset({
    "greeting", "signoff", "signature", "quote_attribution", "forward_marker",
    "footer", "link_only", "automated_sender", "signature_image", "signature_logo",
})

#: Chrome that never becomes an atom at all: the compiler moves it to the
#: ledger (stage ``chrome``) on arrival, so it is listed with its reason and
#: no head, roll-up or atom list ever carries it. A signature logo / badge /
#: address banner read off an inline image (010087: typed scope under
#: "Equipment list"), and an e-signature page stamp ("Docusign Envelope ID:
#: 3F2A...", 010087: a deal_metadata atom on every page of a signed SOW).
DIVERTED_CHROME_REASONS = frozenset({"signature_image", "signature_logo"})
CHROME_STAGE = "chrome"

# Leading furniture before the label: quote marks, bullets, arrows, bold.
_LEAD_RE = re.compile(r"^(?:[>\s]|[←-⇿➔-➿⬀-⯿•·▪◦‣*_|~-])+")
_HTML_BOLD_RE = re.compile(r"</?(?:b|strong|span|font)[^>]*>", re.I)

_HEADER_RE = re.compile(
    r"^(?:from|sent|to|cc|bcc|date|subject|importance|reply-to)\s*:(?:\s|$)", re.I,
)
_ATTRIBUTION_RE = re.compile(
    r"^(?:on\b.{4,200}\bwrote\s*:|-{2,}\s*(?:original|forwarded) message\s*-{2,}|"
    r"begin forwarded message\s*:?)\s*$",
    re.I,
)
_GREETING_LEAD_RE = re.compile(
    r"^(?:hi|hey|hiya|hello|dear|greetings|good\s+(?:morning|afternoon|evening))\b", re.I,
)
_GREETING_CLOSE_RE = re.compile(r"\s*[,:\-–—!]+\s*$")
_SIGNOFF_RE = re.compile(
    r"^(?:thanks|thank\s+you|thanks\s+(?:so\s+much|again|a\s+lot|much)|many\s+thanks|"
    r"regards|best|best\s+regards|kind\s+regards|warm\s+regards|warmest\s+regards|"
    r"sincerely|cheers|respectfully|talk\s+soon|appreciate\s+it|much\s+appreciated|"
    r"all\s+the\s+best|take\s+care|yours(?:\s+(?:truly|sincerely))?)\s*[,.!]*\s*$",
    re.I,
)
_CONTACT_LABEL_RE = re.compile(
    r"^(?:e-?mail|email\s+address|phone|tel|telephone|mobile|cell|office|direct|"
    r"main|fax|web|website|[emptofcw])\s*[:.]\s*(?P<rest>\S.*)$",
    re.I,
)
_LINK_TOKEN_RE = re.compile(
    r"(?:<?(?:https?://|mailto:)\S+>?|\burl=\S+|\[cid:[^\]]*\]|<mailto:[^>]*>)", re.I,
)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_DOMAIN_RE = re.compile(r"^(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|io|co|us|biz)$", re.I)
_PHONE_RE = re.compile(r"^[+(]?\d[\d\s().\-x]{8,}$", re.I)


def _clean(text: str) -> str:
    t = _HTML_BOLD_RE.sub("", str(text or "")).replace("\xa0", " ").strip()
    t = _LEAD_RE.sub("", t).strip()
    # "**From:** X" / "*Sent:* Y"
    t = re.sub(r"^[*_]{1,2}([A-Za-z][A-Za-z -]*:)[*_]{1,2}", r"\1", t)
    return t.strip()


def _contact_value(rest: str) -> bool:
    """The value after a contact label is an address, a number, a domain or a
    link -- not a sentence."""
    r = _LINK_TOKEN_RE.sub(" ", rest).strip(" <>;,|")
    if not r:
        return True
    parts = [p.strip(" <>;,|") for p in re.split(r"\s*[|;/]\s*|\s{2,}", r) if p.strip(" <>;,|")]
    out = []
    for p in parts:
        m = _CONTACT_LABEL_RE.match(p)
        out.append((m.group("rest") if m else p).strip(" <>;,|"))
    return all(_EMAIL_RE.match(p) or _DOMAIN_RE.match(p) or _PHONE_RE.match(p) for p in out)


def _is_greeting(t: str) -> bool:
    s = t.rstrip()
    if _GREETING_LEAD_RE.match(s) and _GREETING_CLOSE_RE.search(s):
        return 1 <= len(_GREETING_CLOSE_RE.sub("", s).split()) <= 7
    if not s.endswith(","):
        return False
    words = s.rstrip(",").split()
    if not (1 <= len(words) <= 4):
        return False
    return bool(_GREETING_LEAD_RE.match(s))


def chrome_reason(text: str) -> str | None:
    """Why ``text`` is mail chrome, or ``None`` when it is not.

    Reasons: ``quoted_header``, ``quote_attribution``, ``greeting``,
    ``signoff``, ``signature_contact``, ``wrapped_link``.
    """
    t = _clean(text)
    if not t:
        return None
    if _HEADER_RE.match(t):
        return "quoted_header"
    if _ATTRIBUTION_RE.match(t):
        return "quote_attribution"
    if _is_greeting(t):
        return "greeting"
    if _SIGNOFF_RE.match(t):
        return "signoff"
    m = _CONTACT_LABEL_RE.match(t)
    if m and _contact_value(m.group("rest")):
        return "signature_contact"
    if _LINK_TOKEN_RE.search(t):
        # "PurTera-IT.com<https://urldefense.com/...>": the words left once
        # the links are gone are at most the anchor itself. "Diagram:
        # <https://...>" names something and is NOT chrome.
        rest = _LINK_TOKEN_RE.sub(" ", t).strip(" <>;,|-–—")
        if not rest:
            return "wrapped_link"
        if rest.endswith(":"):
            return None
        if len(rest.split()) <= 1 and len(rest) <= 40:
            return "signature_contact" if (_EMAIL_RE.match(rest) or _PHONE_RE.match(rest)) else "wrapped_link"
        return None
    if _EMAIL_RE.match(t) or _DOMAIN_RE.match(t) or _PHONE_RE.match(t):
        return "signature_contact"
    return None


def _is_email_atom(atom: Any) -> bool:
    for ref in (getattr(atom, "source_refs", None) or [])[:1]:
        at = getattr(ref, "artifact_type", None)
        if str(getattr(at, "value", at) or "") == "email":
            return True
        fn = str(getattr(ref, "filename", "") or "").lower()
        if fn.endswith((".eml", ".msg")):
            return True
    val = getattr(atom, "value", None)
    return isinstance(val, dict) and "message_index" in val


def diverted_chrome_reason(atom: Any) -> str | None:
    """Why an atom a parser emitted is chrome that must not stay an atom."""
    val = getattr(atom, "value", None)
    val = val if isinstance(val, dict) else {}
    regex_reason = str(val.get("admission_regex") or "")
    if regex_reason in DIVERTED_CHROME_REASONS:
        return regex_reason
    if str(val.get("rejected_by") or "") == "doc_stamp":
        return "esign_stamp"
    return None


def divert_chrome(atoms: list[Any], ledger: list[Any]) -> list[Any]:
    """``atoms`` without the diverted chrome, which joins ``ledger`` at stage
    ``chrome`` with its reason (see ``DIVERTED_CHROME_REASONS``)."""
    from app.core.suppression_ledger import capture_suppressed, merge_suppressed

    by_reason: dict[str, list[Any]] = {}
    for a in atoms:
        why = diverted_chrome_reason(a)
        if why:
            by_reason.setdefault(why, []).append(a)
    if not by_reason:
        return atoms
    gone = {id(a) for grp in by_reason.values() for a in grp}
    for why, grp in by_reason.items():
        merge_suppressed(ledger, capture_suppressed(grp, [], stage=CHROME_STAGE, reason=why))
    return [a for a in atoms if id(a) not in gone]


def atom_chrome_reason(atom: Any) -> str | None:
    """Why a dropped atom is mail chrome rather than a dropped fact.

    A line an email parser's admission regex refused carries its reason; any
    other email line is read by shape. Atoms from other documents are never
    chrome here: "Phone: 555 ..." on a site survey is a site contact.
    """
    diverted = diverted_chrome_reason(atom)
    if diverted:
        return diverted
    val = getattr(atom, "value", None)
    val = val if isinstance(val, dict) else {}
    regex_reason = str(val.get("admission_regex") or "")
    if regex_reason in ADMISSION_CHROME_REASONS:
        return regex_reason
    if not _is_email_atom(atom):
        return None
    return chrome_reason(str(getattr(atom, "raw_text", "") or getattr(atom, "text", "") or ""))


__all__ = ["ADMISSION_CHROME_REASONS", "CHROME_STAGE", "DIVERTED_CHROME_REASONS",
           "atom_chrome_reason", "chrome_reason", "divert_chrome", "diverted_chrome_reason"]
