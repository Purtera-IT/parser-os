"""A pasted email's lines are said_by its sender, through a real compile.

Live 010087: a HubSpot note is a customer's email pasted by our account exec.
#289 set ``author`` to the sender, but ``said_by`` -- what the label walk
reads -- still named the note's author (side ``ours``), so 15 of 16 body
lines were credited to him. The pasted email carried the sender's name and
signature but no address; her address was in the deal's other mail.

Synthetic text mirroring the real note's shape (greeting to the note author,
body, sign-off, name twice, title line, phone rows, no address).
"""

from __future__ import annotations

from pathlib import Path

NOTE = (
    "HubSpot Note: Hi Dana,\n"
    "HubSpot Note ID: 900000001\n"
    "Date: 2026-07-08T12:29:01.028Z\n"
    "Author: Dana Whitfield\n"
    "Author-Email: dana@purtera-it.com\n"
    "\n"
    "Hi Dana,\n"
    "Hope you had a great long weekend! I have CC'd Raj, he schedules all of our logistics here at Northwind Group.\n"
    "I'm reaching out about a project for a school program we have partnered with for years.\n"
    "They need 400 laptops redeployed across 12 locations in August.\n"
    "Could you handle the logistics side as well, or would we need to coordinate that?\n"
    "Thank you,\n"
    "Morgan Ellery\n"
    "Morgan Ellery\n"
    "Client Executive - Services\n"
    "Book A Time That Works Best!\n"
    "Office: 555-010-2000\n"
    "Cell Phone: 555-010-3000\n"
)

EMAIL = (
    "From: Morgan Ellery <Morgan.Ellery@northwindgroup.com>\n"
    "To: Dana Whitfield <dana@purtera-it.com>\n"
    "Subject: Laptop redeployment\n"
    "Date: Tue, 14 Jul 2026 10:00:00 -0500\n"
    "\n"
    "Hi Dana,\n"
    "The school confirmed the 12 locations for the laptop redeployment.\n"
    "Thank you,\n"
    "Morgan Ellery\n"
)


def _note_atoms(tmp_path: Path, files: dict[str, str]):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    project = tmp_path / "deal"
    project.mkdir()
    for name, text in files.items():
        (project / name).write_text(text, encoding="utf-8")
    result = compile_project(project_dir=project, project_id="010900", use_cache=False)
    env = build_orbitbrief_envelope(project_dir=project, compile_result=result)
    return [a for a in env["atoms"] if (a.get("structured") or {}).get("pasted_email")]


def test_pasted_email_said_by_is_the_sender_not_the_note_author(tmp_path):
    atoms = _note_atoms(tmp_path, {"010900-hs-note-900000001-Hi Dana_.txt": NOTE})
    assert len(atoms) >= 5
    for a in atoms:
        by = a["structured"]["said_by"]
        assert by["name"] == "Morgan Ellery", a["text"]
        assert by["side"] == "theirs"
        assert by["role_guess"] != "internal"
        assert by["email"] != "dana@purtera-it.com"
        assert a["structured"]["note_author"] == "Dana Whitfield"


def test_pasted_sender_takes_address_and_company_from_the_deal(tmp_path):
    atoms = _note_atoms(tmp_path, {
        "010900-hs-note-900000001-Hi Dana_.txt": NOTE,
        "010900-hs-email-1.eml": EMAIL,
    })
    assert atoms
    for a in atoms:
        by = a["structured"]["said_by"]
        assert by["email"] == "morgan.ellery@northwindgroup.com", a["text"]
        assert by["company"] == "northwindgroup.com"
        assert by["side"] == "theirs"


def test_own_lines_above_a_pasted_header_stay_with_the_note_author(tmp_path):
    from app.core.deal_parties import stamp_note_parties
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "010900-hs-note-2.txt"
    p.write_text(
        "HubSpot Note: Fwd\nHubSpot Note ID: 2\nDate: 2026-07-08\n"
        "Author: Dana Whitfield\nAuthor-Email: dana@purtera-it.com\n\n"
        "Customer confirmed scope on the call today.\n"
        "From: Morgan Ellery\n"
        "Sent: Monday, July 6, 2026 9:14 AM\n"
        "To: Dana Whitfield <dana@purtera-it.com>\n"
        "Subject: Laptops\n"
        "We need the 400 laptops redeployed in August.\n",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("p", "n2", p)
    stamp_note_parties(atoms)
    mine = next(a for a in atoms if a.raw_text.startswith("Customer confirmed"))
    theirs = next(a for a in atoms if "400 laptops" in a.raw_text)
    assert mine.value["said_by"]["email"] == "dana@purtera-it.com"
    assert theirs.value["said_by"]["name"] == "Morgan Ellery"
    assert theirs.value["said_by"]["side"] == "theirs"
