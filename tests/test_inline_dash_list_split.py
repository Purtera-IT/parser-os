"""A bullet list flattened onto one line is one item per " - " segment.

Live 000132: a HubSpot note's scope -- "Maintenance and support of ... network,
server & virtualization - Support for physical network components (LAN, WLAN)
- ... - 24/7 on-call availability - Coordination with local stakeholders" --
stayed one atom, so nine scope items could not be labelled or answered.
"""
from __future__ import annotations

from pathlib import Path

from app.core.sentences import split_inline_dash_list
from app.parsers.email_parser import EmailParser
from app.parsers.hubspot_note_parser import HubspotNoteParser

LEAD = "Maintenance and support of the technical IT infrastructure in the areas of network, server & virtualization"
ITEMS = [
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
PARAGRAPH = " - ".join([LEAD, *ITEMS])


def test_split_inline_dash_list_splits_a_real_list():
    assert split_inline_dash_list(PARAGRAPH) == [LEAD, *ITEMS]


def test_split_inline_dash_list_leaves_prose_alone():
    assert split_inline_dash_list("We need the router - if it arrives - in rack 2 by Friday.") == []
    assert split_inline_dash_list("On-site visit once per week (4-8 hours as needed)") == []
    assert split_inline_dash_list("Pricing - see attached") == []
    # Three dashes, but the segments are sentences, not items.
    assert split_inline_dash_list(
        "Call me - we talked about it. Then we decided otherwise - and that was it. "
        "No - nothing else - really."
    ) == []


def test_hubspot_note_body_inline_list_becomes_items(tmp_path: Path):
    p = tmp_path / "000132-hs-note-110373542233.txt"
    p.write_text(
        "HubSpot Note: Maintenance and support\nHubSpot Note ID: 110373542233\n"
        "Date: 2026-05-29T18:58:14.007Z\nAuthor: Trent Torrence\n\n"
        f"{PARAGRAPH}\nLocations\nDelphos, OH\n",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("p", "art_note", p)
    texts = {a.raw_text for a in atoms}
    assert LEAD in texts
    for item in ITEMS:
        assert item in texts, item
    assert PARAGRAPH not in texts


def test_email_body_inline_list_becomes_items(tmp_path: Path):
    p = tmp_path / "scope.eml"
    p.write_text(
        "From: Customer Ops <ops@customer.com>\nTo: t@purtera-it.com\nSubject: Scope\n"
        "Date: Mon, 1 Jun 2026 10:00:00 -0400\n\n"
        f"Hi Trent,\n\n{PARAGRAPH}\n\nThanks,\nOps\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact(project_id="p", artifact_id="art_mail", path=p)
    texts = {a.raw_text for a in atoms}
    for item in ITEMS:
        assert item in texts, item
    assert not any(PARAGRAPH in t for t in texts)
