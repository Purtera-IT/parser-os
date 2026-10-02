"""The original owns its text, not the email that quoted it.

Live 000132: HubSpot note 110373542233 (scope paragraph + six site cities)
was written first; a later email QUOTED it. pasted_note_dedup assumed every
shared text was a note pasted from an email, so the note's body and its city
list were folded onto the quoting email and the note kept only its title.
"""
from __future__ import annotations

from pathlib import Path

from app.core.pasted_note_dedup import collapse_pasted_note_duplicates
from tests.test_pasted_note_dedup import LINES, _atom

NOTE_FILE = """HubSpot Note: Scope for IT support
HubSpot Note ID: 110373542233
Date: 2026-05-29T18:58:14.007Z
Author: Trent Torrence
Author-Email: t@purtera-it.com

Scope for IT support

Maintenance and support of the technical IT infrastructure across all offices, including the network and the servers on site.
Locations
Delphos, OH
Hudson, WI
Plymouth, MI
Troy, MI
Tupelo, MS
Wilmington, DE
"""

QUOTING_EMAIL = """From: Customer Ops <ops@customer.com>
To: Trent Torrence <t@purtera-it.com>
Subject: RE: IT support scope
Date: Mon, 1 Jun 2026 10:00:00 -0400
Message-ID: <abc@customer.com>
Content-Type: text/plain; charset=utf-8

Hi Trent, see the scope below, please confirm pricing.

Thanks,
Ops

-----Original Message-----
From: Trent Torrence <t@purtera-it.com>
Sent: Friday, May 29, 2026 2:00 PM
Subject: IT support scope

Maintenance and support of the technical IT infrastructure across all offices, including the network and the servers on site.
Locations
Delphos, OH
Hudson, WI
Plymouth, MI
Troy, MI
Tupelo, MS
Wilmington, DE
"""


def _note(lines, date=None):
    atoms = [_atom(t, "art_note", {"kind": "hubspot_note_body"}, ["hubspot_note_parser"]) for t in lines]
    if date:
        atoms.append(_atom(f"note_id=1 | date={date}", "art_note",
                           {"kind": "hubspot_note_meta", "date": date}, ["hubspot_note_parser"]))
    return atoms


def _mail(lines, **extra):
    return [_atom(t, "art_mail", {"kind": "email_body_line", "message_index": 0, **extra}) for t in lines]


def test_a_mail_that_quotes_the_note_folds_onto_the_note():
    note, mail = _note(LINES), _mail(LINES, quoted=True)
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert {a.artifact_id for a in dropped} == {"art_mail"}
    assert all(a in kept for a in note), "the note is the original and keeps every line"
    assert note[0].value["also_in_email"] == ["art_mail"]


def test_the_earlier_dated_document_is_the_original():
    note = _note(LINES, date="2026-05-29T18:58:14Z")
    mail = _mail(LINES, email_thread={"date": "Mon, 1 Jun 2026 10:00:00 -0400"})
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert {a.artifact_id for a in dropped} == {"art_mail"}

    # ...and the 010289 direction still holds when the mail came first.
    note = _note(LINES, date="2026-06-05T10:00:00Z")
    mail = _mail(LINES, email_thread={"date": "Mon, 1 Jun 2026 10:00:00 -0400"})
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert {a.artifact_id for a in dropped} == {"art_note"}


def test_a_city_line_in_a_non_mail_document_is_not_a_paste_original():
    # A SOW row stamped into a mail thread is not the line a PM pasted.
    note = _note(["Delphos, OH", "Plymouth, MI", "Hudson, WI"])
    sow = [_atom(t, "art_sow", {"kind": "docx_table_row"}) for t in ["Delphos, OH", "Plymouth, MI", "Hudson, WI"]]
    sow.append(_atom("Mail line in the sow thread here.", "art_sow",
                     {"kind": "email_body_line", "message_index": 0}))
    kept, dropped = collapse_pasted_note_duplicates(sow + note)
    assert dropped == []


def test_compile_keeps_the_note_body_when_an_email_quotes_it(tmp_path: Path):
    from app.core.compiler import compile_project

    (tmp_path / "000132-hs-note-110373542233.txt").write_text(NOTE_FILE, encoding="utf-8")
    (tmp_path / "reply.eml").write_text(QUOTING_EMAIL, encoding="utf-8")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)

    note_ids = {a.artifact_id for a in r.atoms if "hubspot_note_id" in (a.value or {})}
    assert len(note_ids) == 1
    note_id = next(iter(note_ids))
    note_texts = {a.raw_text for a in r.atoms if a.artifact_id == note_id}
    for city in ("Delphos, OH", "Hudson, WI", "Troy, MI", "Wilmington, DE"):
        assert city in note_texts, f"{city} must stay with the note"
    assert any(t.startswith("Maintenance and support") for t in note_texts)
    pasted = [a for a in r.suppressed_atoms
              if (a.value or {}).get("_suppression", {}).get("stage") == "pasted_note_dedup"]
    assert pasted and all(a.artifact_id != note_id for a in pasted)
