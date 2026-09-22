"""Email patterns found reviewing deal 010289 (access-control thread with CDW).

Synthetic mail, same shapes:
* a "Location:" block (name / street / phone) is a job site, not a person
* "Provided by us:" / "Provided by Club/installer:" lists keep every item,
  carry the label as the intro line, and say who provides them -- "us" is the
  SENDER's organisation
* dash-glued bullets ("-Relay", "-Mag Lock Cable") are list items: never
  dropped as a name or as chatter
* a quoted "From: X | Sent: Y" routing atom goes away when X's original
  message is in the thread; it stays when the original is missing
"""
from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from app.core.atom_substance_gate import drop_email_non_scope
from app.core.email_threading import dedup_quoted_history, thread_emails
from app.parsers.email_parser import EmailParser

ASK = """Hey AJ,

Here are the details for the small job.

Provided by us:

-PC with Access Control Software

-Relay

-Local and Remote Extender

Provided by Club/installer:

-Mag Lock Cable

-Power Supply for mag lock

Diagram:

https://example.com/diagram.png

Thank you,

Alec Burns
Senior Client Executive
Cell: 281-840-3437 | Email: alecbur@vendor-partner.example
"""

ANSWER = """Sorry, left that part off.

Location:

Nesfield Performance Bethesda

7832 Wisconsin Ave Bethesda MD 20814

P: 240.652.2808

Thank you,

Alec Burns
Senior Client Executive
Cell: 281-840-3437 | Email: alecbur@vendor-partner.example

________________________________
From: AJ Evans <aj@purtera-it.com>
Sent: Wednesday, September 2, 2026 11:33 AM
To: Alec <alec@vendor-partner.example>
Subject: RE: Access Control

Where is this site located?

________________________________
From: Alec <alec@vendor-partner.example>
Sent: Wednesday, September 2, 2026 10:38 AM
To: AJ Evans <aj@purtera-it.com>
Subject: Access Control

Here are the details for the small job.
"""


def _eml(tmp: Path, name: str, body: str, *, frm: str, date: str, subject: str, msgid: str, reply_to: str | None = None) -> Path:
    m = EmailMessage()
    m["From"] = frm
    m["To"] = "aj@purtera-it.com"
    m["Subject"] = subject
    m["Date"] = date
    m["Message-ID"] = msgid
    if reply_to:
        m["In-Reply-To"] = reply_to
        m["References"] = reply_to
    m.set_content(body)
    p = tmp / name
    p.write_bytes(bytes(m))
    return p


def _parse(p: Path):
    return EmailParser().parse(p)


def test_location_block_is_a_named_site_with_its_phone(tmp_path):
    atoms = _parse(_eml(tmp_path, "a.eml", ANSWER, frm="alec@vendor-partner.example",
                        date="Wed, 02 Sep 2026 16:40:40 +0000", subject="RE: Access Control", msgid="<m3@x>"))
    sites = [a for a in atoms if a.atom_type.value == "physical_site"]
    assert len(sites) == 1
    v = sites[0].value
    assert "7832 Wisconsin Ave" in sites[0].raw_text and "MD" in sites[0].raw_text
    assert v["site_name"] == "Nesfield Performance Bethesda"
    assert v["site_phone"] == "240.652.2808"
    assert not any("Nesfield" in a.raw_text for a in atoms if a.atom_type.value == "stakeholder")
    assert not any(a.raw_text.strip() in ("Location:", "P: 240.652.2808") for a in atoms)


def test_supply_lists_keep_every_item_with_label_and_who_provides(tmp_path):
    atoms = _parse(_eml(tmp_path, "b.eml", ASK, frm="Alec <alec@vendor-partner.example>",
                        date="Wed, 02 Sep 2026 14:38:56 +0000", subject="Access Control", msgid="<m1@x>"))
    items = {a.raw_text: a.value for a in atoms if (a.value or {}).get("list_item")}
    assert set(items) >= {"PC with Access Control Software", "Relay", "Local and Remote Extender",
                          "Mag Lock Cable", "Power Supply for mag lock"}
    assert items["Relay"]["list_label"] == "Provided by us:"
    assert items["Relay"]["provided_by"]["party"] == "sender"
    assert "alec@vendor-partner.example" in items["Relay"]["provided_by"]["sender"]
    assert items["Mag Lock Cable"]["provided_by"] == {
        "label": "Club/installer", "party": "named", "sender": "Alec <alec@vendor-partner.example>"}
    assert items["Relay"]["lead_in"] == ["Provided by us:"]
    # the labels themselves are intro lines, not facts
    assert not any(a.raw_text.strip() in ("Provided by us:", "Provided by Club/installer:", "Diagram:") for a in atoms)
    # and the email gate keeps one-word list items
    kept, dropped = drop_email_non_scope([a for a in atoms if a.raw_text == "Relay"])
    assert kept and not dropped


def test_quoted_headers_go_when_the_original_is_in_the_thread(tmp_path):
    ask = _eml(tmp_path, "1.eml", "Here are the details for the small job.", frm="Alec <alec@vendor-partner.example>",
               date="Wed, 02 Sep 2026 14:38:56 +0000", subject="Access Control", msgid="<m1@x>")
    ans = _eml(tmp_path, "3.eml", ANSWER, frm="alec@vendor-partner.example",
               date="Wed, 02 Sep 2026 16:40:40 +0000", subject="RE: Access Control", msgid="<m3@x>", reply_to="<m1@x>")
    atoms = _parse(ask) + _parse(ans)
    atoms, _ = thread_emails(atoms, project_id="p")
    kept, dropped = dedup_quoted_history(atoms, project_id="p")
    hdr = lambda xs: [a.raw_text for a in xs if (a.value or {}).get("kind") == "quoted_message_header"]  # noqa: E731
    # Alec's 10:38 (14:38 UTC) original is in the thread -> its quoted header goes
    assert any("alec@vendor-partner.example" in t for t in hdr(dropped))
    # AJ's 11:33 message is NOT in the thread -> its quoted header stays (attribution)
    assert any("aj@purtera-it.com" in t for t in hdr(kept))
