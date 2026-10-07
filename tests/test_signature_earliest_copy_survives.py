"""A signature folds onto its EARLIEST copy, not the newest email's.

Every email a person sends repeats their signature, and every reply quotes
the older ones again. The quoted copies folded onto the newest email's own
copy, so the July signature (the first time the deal had that title, company
and phone) was dropped as a copy of a September one. The survivor of a fold
is the earliest-dated copy; later copies fold onto it.

Synthetic mail only.
"""
from __future__ import annotations

from datetime import date
from email.message import EmailMessage
from pathlib import Path

from app.core.suppression_ledger import SURVIVOR_KEY

BUYER_SIG = ["Lee Buyer", "Account Manager – Example Partner", "➔ Email: lee@partner.example",
             "➔ Phone: (555) 010-2299"]
SELLER_SIG = ["Dana Seller", "Vice President of Field Services", "dana@seller.example", "555.010.7788"]


def _quoted(frm: str, sent: str, body: list[str], sig: list[str]) -> list[str]:
    return ["", "", f"From: {frm}", f"Sent: {sent}", "To: Lee Buyer <lee@partner.example>",
            "Subject: RE: Field support proposal", "", *body, *sig]


JULY = _quoted("Lee Buyer <lee@partner.example>", "Thursday, July 30, 2026 12:58 PM",
               ["Hi Dana,", "The client asked whether the weekly visit can move to Thursdays from August."], BUYER_SIG)
JUNE = _quoted("Dana Seller <dana@seller.example>", "Thursday, June 4, 2026 6:07 PM",
               ["Hi Lee,", "Attached is the quote for the weekly field visits across both sites."], SELLER_SIG)
AUG = _quoted("Lee Buyer <lee@partner.example>", "Wednesday, August 19, 2026 11:02 AM",
              ["Hi Dana,", "Still waiting on the client to confirm the start date for the visits."], BUYER_SIG)


def _eml(tmp: Path, name: str, frm: str, date_: str, msgid: str, body: list[str], reply_to: str | None = None):
    m = EmailMessage()
    m["From"] = frm
    m["To"] = "lee@partner.example"
    m["Subject"] = "RE: Field support proposal"
    m["Date"] = date_
    m["Message-ID"] = msgid
    if reply_to:
        m["In-Reply-To"] = reply_to
        m["References"] = reply_to
    m.set_content("\n".join(body) + "\n")
    (tmp / name).write_bytes(bytes(m))


def _compile(tmp: Path):
    from app.core.compiler import compile_project

    history = AUG + JULY + JUNE
    _eml(tmp, "thread-1.eml", "Dana Seller <dana@seller.example>", "Tue, 22 Sep 2026 16:29:56 +0000", "<t1@x>",
         ["Hi Lee,", "Checking in again on the field visits below, any word from the client?", *SELLER_SIG,
          *history])
    _eml(tmp, "thread-2.eml", "Lee Buyer <lee@partner.example>", "Tue, 22 Sep 2026 21:20:19 +0000", "<t2@x>",
         ["Hi Dana,", "I reached out to the client today and expect an answer tomorrow.", *BUYER_SIG,
          "", "", "From: Dana Seller <dana@seller.example>", "Sent: Tuesday, September 22, 2026 12:29 PM",
          "To: Lee Buyer <lee@partner.example>", "Subject: RE: Field support proposal", "",
          "Hi Lee,", "Checking in again on the field visits below, any word from the client?", *SELLER_SIG,
          *history], reply_to="<t1@x>")
    return compile_project(tmp, project_id="p", allow_errors=True, use_cache=False)


def _message_day(atom):
    """The day the line's message was sent (a quote's own "Sent:")."""
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    v = atom.value or {}
    et = v.get("email_thread") or {}
    raw = str((et.get("message") or {}).get("sent_at") or v.get("authored_at") or et.get("date") or "")
    try:
        return parsedate_to_datetime(raw).date()
    except Exception:
        pass
    try:
        return datetime.strptime(raw, "%A, %B %d, %Y %I:%M %p").date()
    except ValueError:
        return None


def _signature_lines(atoms, name: str):
    return [a for a in atoms if a.raw_text.strip() == name or a.raw_text.startswith(name + " | ")]


def test_signature_copies_fold_onto_the_earliest_message(tmp_path):
    r = _compile(tmp_path)
    by_id = {a.id: a for a in list(r.atoms) + list(r.suppressed_atoms)}
    for name, first in (("Lee Buyer", date(2026, 7, 30)), ("Dana Seller", date(2026, 6, 4))):
        folded = [a for a in _signature_lines(r.suppressed_atoms, name)
                  if (a.value or {}).get("quoted") and ((a.value or {}).get(SURVIVOR_KEY) or {}).get("atom_id")]
        assert folded, f"fixture: quoted copies of {name}'s signature fold"
        for a in folded:
            surv = by_id[(a.value or {})[SURVIVOR_KEY]["atom_id"]]
            # Never a later message's copy: the earliest message's own line.
            assert _message_day(surv) == first, (a.raw_text, _message_day(a), _message_day(surv))
        # The earliest copy itself stands.
        assert any(_message_day(a) == first for a in _signature_lines(r.atoms, name)), name
