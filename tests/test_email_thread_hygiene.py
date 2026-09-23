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


def test_short_quoted_list_items_collapse_to_the_original(tmp_path):
    # Live 010289: "Relay" came back quoted in nine replies. One word is too
    # short to collapse on text; under its list label it is the same fact.
    ask = _eml(tmp_path, "1.eml", ASK, frm="Alec <alec@vendor-partner.example>",
               date="Wed, 02 Sep 2026 14:38:56 +0000", subject="Access Control", msgid="<m1@x>")
    quoted = "Got it.\n\n" + "________________________________\nFrom: Alec <alec@vendor-partner.example>\n" \
        "Sent: Wednesday, September 2, 2026 10:38 AM\nTo: AJ Evans <aj@purtera-it.com>\nSubject: Access Control\n\n" + ASK
    r1 = _eml(tmp_path, "2.eml", quoted, frm="AJ Evans <aj@purtera-it.com>",
              date="Wed, 02 Sep 2026 15:00:00 +0000", subject="RE: Access Control", msgid="<m2@x>", reply_to="<m1@x>")
    r2 = _eml(tmp_path, "3.eml", quoted, frm="AJ Evans <aj@purtera-it.com>",
              date="Wed, 02 Sep 2026 15:30:00 +0000", subject="RE: Access Control", msgid="<m4@x>", reply_to="<m2@x>")
    atoms = _parse(ask) + _parse(r1) + _parse(r2)
    atoms, _ = thread_emails(atoms, project_id="p")
    kept, _ = dedup_quoted_history(atoms, project_id="p")
    assert [a.raw_text for a in kept].count("Relay") == 1


def test_a_named_site_keeps_name_and_phone_through_dedup(tmp_path):
    from app.core.semantic_dedup import _PHYSICAL_SITE_ALLOWED_FIELDS

    atoms = _parse(_eml(tmp_path, "a.eml", ANSWER.replace("Location:\n\n", "Location:\n"), frm="alec@vendor-partner.example",
                        date="Wed, 02 Sep 2026 16:40:40 +0000", subject="RE: Access Control", msgid="<m3@x>"))
    site = next(a for a in atoms if a.atom_type.value == "physical_site").value
    assert site["facility_name"] == site["name"] == "Nesfield Performance Bethesda"
    assert site["phone"] == "240.652.2808"
    assert {"site_name", "site_phone", "facility_name", "phone"} <= _PHYSICAL_SITE_ALLOWED_FIELDS


def test_a_pasted_note_list_splits_on_inline_dashes(tmp_path):
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "010289-hs-note-1-The Ask.txt"
    p.write_text(
        "HubSpot Note: The Ask\nHubSpot Note ID: 1\nDate: 2026-09-08T14:39:55.273Z\nAuthor: AJ Evans\n"
        "Author-Email: aj@purtera-it.com\n\nThe Ask Hey AJ, here are the details. Provided by us: -PC with Access Control "
        "Software -Relay -Local and Remote Extender Provided by Club/installer: -Mag Lock Cable -Power Supply for mag lock\n",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse(p)
    items = {a.raw_text: (a.value or {}).get("list_label") for a in atoms if (a.value or {}).get("list_item")}
    assert items.get("Relay") == "Provided by us:"
    assert items.get("Mag Lock Cable") == "Provided by Club/installer:"
    assert not any(a.raw_text.startswith("Provided by") for a in atoms)


def test_sentences_of_one_paragraph_keep_the_order_they_were_written(tmp_path):
    """Live 010288 showed a paragraph's three sentences back to front. They
    share a line number, so the reading-order sort tied and fell through to
    the atom id -- which is a hash."""
    from app.core.orbitbrief_envelope import _in_reading_order

    body = (
        "Hey AJ,\n\n"
        "Here are the details for the small job. If this is a successful implementation, it could lead to "
        "many more of the same opportunity. If you all would be able to do something like this, I will get "
        "a conversation going with the club owner.\n\nThanks,\n\nAlec\n"
    )
    atoms = _parse(_eml(tmp_path, "ask.eml", body, frm="Alec <alec@vendor-partner.example>",
                        date="Wed, 02 Sep 2026 14:38:56 +0000", subject="Access Control", msgid="<m1@x>"))
    ordered = [a.raw_text for a in _in_reading_order(atoms, [{"artifact_id": atoms[0].artifact_id, "authored_at": ""}])]
    first = next(i for i, t in enumerate(ordered) if t.startswith("Here are the details"))
    second = next(i for i, t in enumerate(ordered) if t.startswith("If this is a successful"))
    third = next(i for i, t in enumerate(ordered) if t.startswith("If you all would be able"))
    assert first < second < third

    # and each one says where it sits, so nothing downstream has to guess
    seqs = {}
    for a in atoms:
        loc = (a.source_refs[0].locator if a.source_refs else {}) or {}
        if a.raw_text.startswith(("Here are the details", "If this is a successful", "If you all would be")):
            seqs[a.raw_text[:20]] = loc.get("sentence_index", 0)
    assert sorted(seqs.values()) == [0, 1, 2]
