"""An automated sender is a channel, never a person on the deal.

"From: Adobe Sign <echosign@echosign.com>" minted a stakeholder and an
entity; a DocuSign notice did the same for "Carl Painter via DocuSign". No
person or entity is minted for a machine, its header lines are kept as
chatter atoms (rejectable), and the body's facts stay normal atoms.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.automated_senders import is_automated_address, is_automated_sender
from app.core.compiler import compile_project
from app.core.entity_extraction import _emit_email_keys, _emit_person_from_contact
from app.parsers.email_parser import EmailParser

ADOBE = """From: Adobe Sign <echosign@echosign.com>
To: Trent Walsh <trent@purtera-it.com>
Cc: Carl Painter <carl@acmehealth.com>
Subject: Signed and Filed: SOW 010215 Acme Health Cabling
Date: Mon, 07 Jul 2026 09:00:00 -0400
Content-Type: text/plain; charset=utf-8

SOW 010215 Acme Health Cabling between Acme Health and PurTera IT is Signed and Filed!

From: Adobe Sign <echosign@echosign.com>

Carl Painter (carl@acmehealth.com) has signed SOW 010215 Acme Health Cabling.

Attached is a final copy of SOW 010215 Acme Health Cabling.
"""

DOCUSIGN = """From: "Carl Painter via DocuSign" <dse_NA3@docusign.net>
To: Trent Walsh <trent@purtera-it.com>
Subject: Completed: Please DocuSign: SOW 010215 Acme Health.pdf
Date: Tue, 08 Jul 2026 10:00:00 -0400
Content-Type: text/plain; charset=utf-8

Your document has been completed

All parties have completed Please DocuSign: SOW 010215 Acme Health.pdf.

The SOW was signed by Carl Painter on July 8, 2026.
"""


@pytest.mark.parametrize("value,robot", [
    ("Adobe Sign <echosign@echosign.com>", True),
    ('"Carl Painter via DocuSign" <dse_NA3@docusign.net>', True),
    ("dse@docusign.net", True),
    ("noreply@notifications.hubspot.com", True),
    ("calendar-notification@google.com", True),
    ("MAILER-DAEMON@mx.acme.com", True),
    ("no-reply@acme.com", True),
    ("donotreply@acme.com", True),
    ("bounces+123@mail.acme.com", True),
    ("Carl Painter <carl@acmehealth.com>", False),
    ("renotify.smith@acme.com", False),
    ("jane@box.com", False),
])
def test_is_automated_sender(value, robot):
    assert is_automated_sender(value) is robot


def test_entity_extraction_mints_no_robot():
    text = "From: Adobe Sign <echosign@echosign.com> | To: Trent Walsh <trent@purtera-it.com>"
    assert _emit_person_from_contact(text) == {"stakeholder:trent_walsh"}
    assert _emit_email_keys(text) == {"email:trent_purtera_it_com"}
    assert not is_automated_address("trent@purtera-it.com")


def _parse(tmp_path: Path, name: str, raw: str):
    p = tmp_path / name
    p.write_text(raw, encoding="utf-8")
    return EmailParser().parse_artifact_full(project_id="p", artifact_id=name, path=p).atoms


@pytest.mark.parametrize("name,raw,robot_key", [
    ("adobe.eml", ADOBE, "adobe_sign"),
    ("docusign.eml", DOCUSIGN, "docusign"),
])
def test_parser_keeps_facts_and_flags_the_robot(tmp_path, name, raw, robot_key):
    atoms = _parse(tmp_path, name, raw)
    people = [a for a in atoms if a.atom_type.value == "stakeholder"]
    assert not any(is_automated_address(str((a.value or {}).get("email") or "")) for a in people)
    assert not any(robot_key in k for a in atoms for k in (a.entity_keys or []))
    headers = [a for a in atoms if (a.value or {}).get("kind") == "email_header"]
    assert headers and all("chatter" in a.review_flags and a.value.get("automated_sender") for a in headers)
    facts = [a.raw_text for a in atoms if "signed" in a.raw_text.lower() and "chatter" not in a.review_flags]
    assert facts, "the body's signing fact stays a normal atom"


def test_compile_mints_no_robot_stakeholder_or_entity(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    deal = tmp_path / "deal"
    deal.mkdir()
    (deal / "adobe.eml").write_text(ADOBE, encoding="utf-8")
    (deal / "docusign.eml").write_text(DOCUSIGN, encoding="utf-8")
    # allow_errors: an unrelated pre-existing validator error (a quoted header
    # atom left auto_accepted with calibration_abstain) fails this compile on
    # the base branch too.
    r = compile_project(deal, project_id="deal", use_cache=False, allow_errors=True)
    keys = {k for a in r.atoms for k in (a.entity_keys or [])}
    assert not any("adobe" in k or "echosign" in k or "docusign" in k for k in keys)
    assert not any("adobe" in str(getattr(e, "canonical_key", "")) or "docusign" in str(getattr(e, "canonical_key", ""))
                   for e in r.entities)
    assert not [a for a in r.atoms if a.atom_type.value == "stakeholder"
                and is_automated_address(str((a.value or {}).get("email") or ""))]
    assert any("signed" in a.raw_text.lower() and "chatter" not in a.review_flags for a in r.atoms)
