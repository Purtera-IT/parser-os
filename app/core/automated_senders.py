"""Automated senders: machines that send mail, never people on the deal.

"From: Adobe Sign <echosign@echosign.com>" became a stakeholder, and with it
an entity, a roster row and a "who said it" for every line of a completion
notice. An e-signature service, a no-reply mailbox, a calendar bot or a
bounce daemon is a CHANNEL: what it says ("SOW 010215 is signed") is a fact
about the deal and stays a normal atom, but the sender is nobody.

Two tests, either is enough:
  * the address's DOMAIN is a known e-signature / SaaS notification domain;
  * the address's LOCAL PART is shaped like a robot (noreply, no-reply,
    donotreply, notifications, mailer-daemon, bounce, calendar-notification).
A display name like "Carl Painter via DocuSign" is still the robot: the
person named in it signed, but did not write the message.
"""
from __future__ import annotations

import re
from email.utils import parseaddr

#: Domains whose mail is always machine-sent (suffix match, so
#: ``dse_na3@docusign.net`` and ``mail.docusign.net`` both match).
AUTOMATED_SENDER_DOMAINS: frozenset[str] = frozenset({
    # Sending domains only: a company's staff domain (zoom.us, box.com,
    # salesforce.com) can be a real customer's, so it is never listed.
    "echosign.com", "adobesign.com", "adobe-sign.com",
    "docusign.net",
    "hellosign.com", "mail.dropboxsign.com", "pandadoc.net", "signnow.com",
    "rightsignature.com",
    "notifications.hubspot.com", "hubspotemail.net",
    "calendar.google.com",
})

#: Local parts shaped like a robot. Anchored on separators so "renotify" or
#: "bob.bounceback" style real names are not caught by accident.
_ROBOT_LOCAL_RE = re.compile(
    r"(?:^|[._+-])(?:"
    r"no[._-]?reply|do[._-]?not[._-]?reply|dont[._-]?reply|"
    r"notifications?|notify|mailer[._-]?daemon|postmaster|bounces?|"
    r"calendar[._-]?notification|calendar[._-]?server|"
    r"echosign|docusign|dse(?:_[a-z0-9]+)?|automated|auto[._-]?mail(?:er)?"
    r")(?:$|[._+-])",
    re.I,
)

#: Display names that are a product, not a person.
_ROBOT_NAME_RE = re.compile(
    r"\b(?:adobe\s+(?:acrobat\s+)?sign|echosign|docusign|hellosign|dropbox\s+sign|"
    r"pandadoc|signnow|hubspot|google\s+calendar|mailer[- ]daemon|"
    r"mail\s+delivery\s+(?:system|subsystem)|microsoft\s+outlook)\b",
    re.I,
)


def _domain_is_automated(domain: str) -> bool:
    d = domain.lower().strip(".>")
    return any(d == s or d.endswith("." + s) for s in AUTOMATED_SENDER_DOMAINS)


def is_automated_address(address: str) -> bool:
    """Is this e-mail address a machine's?"""
    addr = str(address or "").strip().strip("<>").lower()
    if "@" not in addr:
        return False
    local, _, domain = addr.rpartition("@")
    return _domain_is_automated(domain) or bool(_ROBOT_LOCAL_RE.search(local))


def is_automated_sender(value: str, name: str | None = None) -> bool:
    """``value`` is an address or a ``"Name" <addr>`` header value."""
    raw = str(value or "")
    nm, addr = parseaddr(raw)
    if not addr and "@" in raw:
        addr = raw
    if addr and is_automated_address(addr):
        return True
    label = (name if name is not None else nm) or ""
    return bool(label and _ROBOT_NAME_RE.search(label))


__all__ = [
    "AUTOMATED_SENDER_DOMAINS",
    "is_automated_address",
    "is_automated_sender",
]
