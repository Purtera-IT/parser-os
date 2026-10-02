"""A quoted message is said by ITS author, and every message keeps its header.

Live 010087: Trent's 7/8 reply quoted Stephanie's 7/7 email. Her lines were
split into their own message, but each was credited ``said_by`` Trent (the
file's sender), and the "From: Stephanie | Sent: ..." header that names her
message was dropped as OCR debris -- names and addresses are not dictionary
words -- so no email section showed a sender or a date.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

from app.core.atom_substance_gate import apply_substance_gate
from app.core.deal_parties import parties_for_message, stamp_parties
from app.parsers.email_parser import EmailParser

REPLY = """From: Trent Torrence <t@purtera-it.com>
To: Stephanie Hechsel <stephanie.hechsel@amtivo.com>
Cc: Sean Moore <sean.moore@amtivo.com>
Subject: RE: Equipment list
Date: Wed, 8 Jul 2026 10:12:00 -0400
Message-ID: <o2@purtera-it.com>
Content-Type: text/plain; charset=utf-8

Thanks, we will review the list and come back with pricing by Friday.

From: Stephanie Hechsel <stephanie.hechsel@amtivo.com>
Sent: Tuesday, July 7, 2026 3:36 PM
To: Trent Torrence <t@purtera-it.com>
Subject: Equipment list

Please find the equipment list below.
- 4 Cisco C9300-48P switches
"""


def test_a_quoted_line_is_said_by_its_own_messages_author():
    tb = {"sender": "Trent Torrence <t@purtera-it.com>", "to": ["Stephanie <stephanie.hechsel@amtivo.com>"],
          "message": {"index": 1, "author": "Stephanie Hechsel <stephanie.hechsel@amtivo.com>", "quoted": True}}
    info = parties_for_message(tb)
    assert info["said_by"]["email"] == "stephanie.hechsel@amtivo.com"
    assert "said_to" not in info  # the reply's recipients are not hers
    own = parties_for_message({**tb, "message": {"index": 0, "author": "Trent Torrence <t@purtera-it.com>"}})
    assert own["said_by"]["email"] == "t@purtera-it.com"
    assert own["said_to"][0]["email"] == "stephanie.hechsel@amtivo.com"
    a = NS(value={"email_thread": tb})
    assert stamp_parties([a]) == 1 and a.value["said_by"]["name"] == "Stephanie Hechsel"


def test_every_messages_routing_header_survives_the_substance_gate(tmp_path: Path):
    p = tmp_path / "reply.eml"
    p.write_text(REPLY, encoding="utf-8")
    atoms = EmailParser().parse_artifact("p", "art", p)
    kept, _ = apply_substance_gate(atoms)
    kinds = {a.value.get("kind"): a for a in kept if isinstance(a.value, dict)}
    assert "email_header" in kinds
    q = kinds["quoted_message_header"]
    assert "Stephanie Hechsel" in q.raw_text and "July 7, 2026" in q.raw_text
    assert q.source_refs[0].locator["message_index"] == 1


NOTE = """HubSpot Note: Fwd: Equipment list
HubSpot Note ID: 112490900231
Date: 2026-07-08T14:00:00.000Z
Author: Trent Torrence
Author-Email: t@purtera-it.com

Fwd: Equipment list

Sending this over for pricing.

From: Stephanie Hechsel <stephanie.hechsel@amtivo.com>
Sent: Tuesday, July 7, 2026 3:36 PM
To: Trent Torrence <t@purtera-it.com>
Subject: Equipment list

Please find the equipment list below.
- 4 Cisco C9300-48P switches
- 2 Fortinet FG-100F firewalls
"""


def test_an_email_pasted_into_a_note_is_said_by_its_sender(tmp_path: Path):
    """010087: note 112490900231 is Stephanie's email pasted by Trent; its
    lines had no said_by at all."""
    from app.core.deal_parties import stamp_note_parties
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "010087-hs-note-112490900231-fwd.txt"
    p.write_text(NOTE, encoding="utf-8")
    atoms = HubspotNoteParser().parse_artifact("p", "note", p)
    assert stamp_note_parties(atoms) > 0
    by = {a.raw_text.lstrip("- "): (a.value.get("said_by") or {}).get("email") for a in atoms}
    assert by["4 Cisco C9300-48P switches"] == "stephanie.hechsel@amtivo.com"
    assert next(v for k, v in by.items() if k.startswith("Sending this over")) == "t@purtera-it.com"
    meta = next(a for a in atoms if a.value.get("kind") == "hubspot_note_meta")
    assert "said_by" not in meta.value
