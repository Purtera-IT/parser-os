"""Live 000132: two HubSpot notes filed as message "2 of 3" of an email thread.

Trent Torrence's notes 110373542233 and 110373542437 share their scope list
and their "Locations" city list with Christopher Picchietti's forward. The
note's copies folded onto the email lines, which then CITED the note -- and
the envelope read each note's thread block off those citing email atoms. Both
notes came out with Christopher as sender and "2 of 3" as position, so the
labeling walk merged them under one email header.

Also here: a " - "-separated list on one line read in hash order, and a
"City, ST" list became prose instead of sites.
"""
from __future__ import annotations

from pathlib import Path

from app.core.schemas import AtomType
from app.parsers.email_parser import EmailParser
from app.parsers.hubspot_note_parser import HubspotNoteParser

SERVICES = [
    "Maintenance and support of the technical IT infrastructure in the areas of network, server & virtualization",
    "Support for physical network components (LAN, WLAN)",
    "Support for physical server and NAS environments",
    "Support for the virtual server environment based on VMware vSphere",
    "Support for operating system patches and upgrades (Windows + Linux)",
    "End-user support including O365/M365 clients",
    "Troubleshooting and incident response",
    "On-site visit once per week (4-8 hours as needed)",
    "24/7 on-call availability",
    "Coordination with local stakeholders",
]
CITIES = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS", "Wilmington, DE"]
LOCATIONS = "Locations\n" + "\n".join(CITIES)

NOTE_1 = (
    "HubSpot Note: Maintenance and support of the technical IT infrastructure in the areas of ne…\n"
    "HubSpot Note ID: 110373542233\n"
    "Date: 2026-05-29T18:58:14.007Z\n"
    "Author: Trent Torrence\n"
    "Author-Email: t@purtera-it.com\n\n"
    "Maintenance and support of the technical IT infrastructure in the areas of ne…\n\n"
    + " - ".join(SERVICES) + "\n" + LOCATIONS + "\n"
)
NOTE_2 = (
    "HubSpot Note: Maintenance and support of their technical IT infrastructure covering network…\n"
    "HubSpot Note ID: 110373542437\n"
    "Date: 2026-05-29T18:58:26.171Z\n"
    "Author: Trent Torrence\n"
    "Author-Email: t@purtera-it.com\n\n"
    "Maintenance and support of their technical IT infrastructure covering network…\n\n"
    "Maintenance and support of their technical IT infrastructure covering network, server, and virtualization\n"
    + "\n".join("- " + s for s in SERVICES[1:]) + "\n"
)
# Christopher's own body carries the list: the mail is the original and the
# note's copies fold onto it.
FORWARD = (
    "From: Christopher Picchietti <christopher.picchietti@cdw.com>\n"
    "To: Trent Torrence <t@purtera-it.com>\n"
    "Subject: Fw: 000132 - Multi site technical IT support\n"
    "Date: Fri, 29 May 2026 14:58:00 -0400\n"
    "Message-ID: <fw1@cdw.com>\n"
    "Content-Type: text/plain; charset=utf-8\n\n"
    "Hi Trent,\n\nCan you quote this:\n\n"
    + "\n".join(SERVICES[:1] + ["- " + s for s in SERVICES[1:]]) + "\n" + LOCATIONS + "\n"
)
REPLY = (
    "From: Trent Torrence <t@purtera-it.com>\n"
    "To: Christopher Picchietti <christopher.picchietti@cdw.com>\n"
    "Subject: RE: Fw: 000132 - Multi site technical IT support\n"
    "Date: Tue, 22 Sep 2026 12:29:00 -0400\n"
    "Message-ID: <re1@purtera-it.com>\n"
    "In-Reply-To: <fw1@cdw.com>\n"
    "References: <fw1@cdw.com>\n"
    "Content-Type: text/plain; charset=utf-8\n\n"
    "Hi Chris,\n\nGood afternoon. Just tried calling you. Pricing attached.\n\nTrent Torrence\n"
)


def _note_atoms(tmp_path: Path, text: str, name: str = "000132-hs-note-1.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return HubspotNoteParser().parse_artifact("p", "art_note", p)


def _compile(tmp_path: Path):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    (tmp_path / "000132-hs-note-110373542233-maint.txt").write_text(NOTE_1, encoding="utf-8")
    (tmp_path / "000132-hs-note-110373542437-maint.txt").write_text(NOTE_2, encoding="utf-8")
    (tmp_path / "000132-hs-email-1.eml").write_text(FORWARD, encoding="utf-8")
    (tmp_path / "000132-hs-email-2.eml").write_text(REPLY, encoding="utf-8")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    return build_orbitbrief_envelope(project_dir=tmp_path, compile_result=r)


# -- 1. a note never wears an email's thread block -------------------------

def test_notes_do_not_take_the_thread_of_the_email_that_absorbed_their_lines(tmp_path: Path):
    env = _compile(tmp_path)
    docs = {d["filename"]: d for d in env["documents"]}
    for name, doc in docs.items():
        if "-hs-note-" in name:
            assert doc.get("email_thread") is None, f"{name} is a note, not a message of a thread"
    fw = docs["000132-hs-email-1.eml"]["email_thread"]
    assert fw and "christopher.picchietti" in fw["sender"]
    # The fold really happened: the email's lines cite the note.
    note_id = docs["000132-hs-note-110373542233-maint.txt"]["artifact_id"]
    assert any(note_id in ((a.get("structured") or {}).get("also_in_note") or []) for a in env["atoms"])
    # ...and the site list survives as sites, on the original.
    sites = {a["text"] for a in env["atoms"] if a["atom_type"] == "physical_site"}
    assert set(CITIES) <= sites


def test_document_thread_reads_only_the_documents_own_atoms():
    from types import SimpleNamespace as NS

    from app.core.orbitbrief_envelope import _document_thread

    mail_line = NS(artifact_id="art_mail", structured=None,
                   value={"email_thread": {"thread_id": "thr_1", "thread_index": 2, "sender": "c@cdw.com"}})
    note_line = NS(artifact_id="art_note", structured=None, value={"kind": "hubspot_note_body"})
    # The note's atom list holds the mail line because the mail line cites it.
    assert _document_thread([mail_line, note_line], artifact_id="art_note") is None
    assert _document_thread([mail_line], artifact_id="art_mail")["thread_id"] == "thr_1"
    # A note atom that took an email_thread in a dedup merge is still a note.
    note_line.value["email_thread"] = dict(mail_line.value["email_thread"])
    assert _document_thread([note_line], artifact_id="art_note", is_message=False) is None


# -- 2. reading order ------------------------------------------------------

def test_a_one_line_dash_list_keeps_its_order(tmp_path: Path):
    atoms = _note_atoms(tmp_path, NOTE_1)
    body = [a for a in atoms if a.atom_type == AtomType.scope_item and a.raw_text in SERVICES[1:]]
    assert [a.raw_text for a in body] == SERVICES[1:]
    cols = [a.source_refs[0].locator["char_start"] for a in body]
    assert cols == sorted(cols) and len(set(cols)) == len(cols)

    from app.core.orbitbrief_envelope import _in_reading_order

    shuffled = sorted(body, key=lambda a: a.id)  # hash order: what readers used to get
    assert [a.raw_text for a in _in_reading_order(shuffled, [])] == SERVICES[1:]


def test_a_full_sentence_is_located_on_its_own_line_not_the_truncated_title(tmp_path: Path):
    atoms = _note_atoms(tmp_path, NOTE_2)
    full = next(a for a in atoms if a.raw_text.endswith("server, and virtualization"))
    lines = NOTE_2.splitlines()
    ln = full.source_refs[0].locator["line_start"]
    assert lines[ln - 1].endswith("server, and virtualization")


# -- 3. a "City, ST" list is a list of sites -------------------------------

def test_note_city_list_lines_are_sites(tmp_path: Path):
    atoms = _note_atoms(tmp_path, NOTE_1)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert [a.raw_text for a in sites] == CITIES
    delphos = sites[0]
    assert delphos.entity_keys == ["site:delphos_oh"]
    assert delphos.value["city"] == "Delphos" and delphos.value["state"] == "OH"
    assert delphos.value["list_label"] == "Locations"
    lines = NOTE_1.splitlines()
    for a in sites:  # each one points at its own line
        assert lines[a.source_refs[0].locator["line_start"] - 1] == a.raw_text
    # The heading is still an atom; nothing is left unread.
    assert any(a.raw_text == "Locations" for a in atoms)


def test_email_city_list_lines_are_sites(tmp_path: Path):
    p = tmp_path / "fw.eml"
    p.write_text(FORWARD, encoding="utf-8")
    atoms = EmailParser().parse_artifact("p", "art_mail", p)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert [a.raw_text for a in sites] == CITIES
    assert sites[-1].entity_keys[-1] == "site:wilmington_de"
    assert sites[0].value["kind"] == "email_body_line" and sites[0].value["message_index"] == 0


def test_one_city_line_alone_is_not_a_list():
    from app.core.city_site_list import find_city_site_lists

    assert find_city_site_lists(["Thanks for the call.", "Troy, MI", "See you Monday."]) == []
    found = find_city_site_lists(["Sites:", "- Lima, OH", "- Troy, MI", "", "Nashville"])
    assert [(c.city, c.state, c.label) for c in found] == [("Lima", "OH", "Sites"), ("Troy", "MI", "Sites")]
