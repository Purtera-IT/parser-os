"""A note that is someone else's email pasted in credits that someone.

Live 010087: Trent's HubSpot note opened "Hi Trent," -- Stephanie's email to
him -- and every atom of it was credited to Trent, the note's author.
"""

from __future__ import annotations

from app.parsers.hubspot_note_parser import HubspotNoteParser

_HEAD = (
    "HubSpot Note: Note\nHubSpot Note ID: 5\nDate: 2026-09-01\n"
    "Author: Trent Hall\nAuthor-Email: trent@purtera-it.com\n\n"
)


def _atoms(tmp_path, body: str):
    p = tmp_path / "deal-hs-note-5.txt"
    p.write_text(_HEAD + body)
    return HubspotNoteParser().parse_artifact("p", "a1", p)


def _body_atoms(atoms):
    return [a for a in atoms if (a.value or {}).get("kind") == "hubspot_note_body"]


def test_greeting_to_the_author_and_a_signature_credit_the_sender(tmp_path):
    atoms = _atoms(tmp_path, (
        "Hi Trent,\n"
        "We need the two cameras at the Raleigh office replaced next week.\n"
        "Parking is in the rear lot.\n"
        "Thanks,\n"
        "Stephanie Moore\n"
        "Facilities Manager\n"
        "stephanie.moore@acme.com\n"
    ))
    body = [a for a in _body_atoms(atoms) if "cameras" in a.raw_text or "Parking" in a.raw_text]
    assert len(body) >= 2
    for a in body:
        assert a.value["author"] == "Stephanie Moore"
        assert a.value["author_email"] == "stephanie.moore@acme.com"
        assert a.value["author_affiliation"] == "external"
        assert a.value["note_author"] == "Trent Hall"
        assert "internal_author" not in a.review_flags
    meta = next(a for a in atoms if (a.value or {}).get("kind") == "hubspot_note_meta")
    assert meta.value["author"] == "Trent Hall"


def test_sign_off_and_first_name_credit_the_sender(tmp_path):
    atoms = _atoms(tmp_path, (
        "Hi Trent,\nWe need the two cameras replaced next week.\nThanks,\nStephanie\n"
    ))
    a = next(a for a in _body_atoms(atoms) if "cameras" in a.raw_text)
    assert a.value["author"] == "Stephanie"


def test_pasted_header_block_credits_only_the_pasted_part(tmp_path):
    atoms = _atoms(tmp_path, (
        "Customer confirmed scope on the call today.\n"
        "From: Stephanie Moore <stephanie.moore@acme.com>\n"
        "Sent: Monday, September 1, 2026 9:14 AM\n"
        "To: Trent Hall <trent@purtera-it.com>\n"
        "Subject: Raleigh cameras\n"
        "We need the two cameras at the Raleigh office replaced next week.\n"
    ))
    mine = next(a for a in _body_atoms(atoms) if a.raw_text.startswith("Customer confirmed"))
    theirs = next(a for a in atoms if "cameras at the Raleigh" in a.raw_text)
    assert mine.value["author"] == "Trent Hall"
    assert theirs.value["author"] == "Stephanie Moore"
    assert theirs.value["author_email"] == "stephanie.moore@acme.com"


def test_the_author_s_own_note_is_untouched(tmp_path):
    atoms = _atoms(tmp_path, (
        "Hi team,\nSpoke with Stephanie today.\nThey need the cameras replaced.\nThanks,\nTrent\n"
    ))
    for a in _body_atoms(atoms):
        assert a.value["author"] == "Trent Hall"
        assert "pasted_email" not in a.value
