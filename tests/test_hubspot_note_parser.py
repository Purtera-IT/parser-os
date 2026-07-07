from __future__ import annotations

from pathlib import Path

from app.core.note_provenance_backfill import ensure_hubspot_note_provenance
from app.core.schemas import AtomType
from app.parsers.hubspot_note_parser import HubspotNoteParser, parse_hubspot_note_text
from app.parsers.registry import choose_parser


def test_hubspot_note_parser_routes_before_transcript(tmp_path: Path) -> None:
    p = tmp_path / "010058-hs-note-112019851881-Fred ROM.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: Fred at NMC gave us 1500 ROM- PK called CDW and told them 2k ROM. GOOD 2 GO",
                "HubSpot Note ID: 112019851881",
                "Date: 2026-06-29T14:40:09.904Z",
                "Author: Chase Smith",
                "",
                "Fred at NMC gave us 1500 ROM- PK called CDW and told them 2k ROM. GOOD 2 GO",
            ]
        ),
        encoding="utf-8",
    )
    parser, match, _ = choose_parser(p)
    assert parser is not None
    assert match.parser_name == "hubspot_note"
    atoms = parser.parse_artifact("deal-1", "art_rom", p)
    types = {a.atom_type for a in atoms}
    assert AtomType.commercial_total in types
    assert AtomType.scope_item in types
    assert any("1500" in a.raw_text for a in atoms)


def test_hubspot_note_parser_emits_scope_without_utterance_segmentation(tmp_path: Path) -> None:
    p = tmp_path / "010058-hs-note-111648788885-hardware.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: 4 e7 aps. 2 udm beast for routers. Has some set up.",
                "HubSpot Note ID: 111648788885",
                "Date: 2026-06-24T17:54:42.831Z",
                "Author: Patrick Kelly",
                "",
                "4 e7 aps. 2 udm beast for routers. Has some set up. two 48 port switches and 2nvr.",
            ]
        ),
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_hw", p)
    assert len(atoms) >= 2
    assert any(a.atom_type == AtomType.scope_item for a in atoms)
    body_atoms = [a for a in atoms if "udm beast" in a.raw_text.lower()]
    assert body_atoms


def test_hubspot_note_parser_extracts_physical_site_with_city(tmp_path: Path) -> None:
    # Address-only note (company lead-in + "PA15212-5359" with no space) must ingest as
    # a physical_site atom carrying structured city/state/zip so site_facility_head can
    # derive a "<City> Office" facility name. Universal: any address-bearing note works.
    p = tmp_path / "010058-hs-note-111645120815-GECKO ROBOTICS.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: GECKO ROBOTICS",
                "HubSpot Note ID: 111645120815",
                "Date: 2026-06-24T16:05:00.000Z",
                "Author: Patrick Kelly",
                "",
                "GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359",
            ]
        ),
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_addr", p)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    site = sites[0]
    assert site.value["city"] == "PITTSBURGH"
    assert site.value["state"] == "PA"
    assert site.value["zip"] == "15212"
    assert site.value["street_address"] == "100 S COMMONS STE 145"

    # End-to-end: the head derives the friendly "<City> Office" name from that locality.
    from app.core.site_facility_head import decide_site_facility_label

    decision = decide_site_facility_label(site)
    assert decision.facility_name == "Pittsburgh Office"


def test_parse_hubspot_note_text_splits_headers() -> None:
    parsed = parse_hubspot_note_text(
        "HubSpot Note: Title\nHubSpot Note ID: 99\nDate: 2026-01-01\nAuthor: Pat\n\nBody line."
    )
    assert parsed["note_id"] == "99"
    assert parsed["author"] == "Pat"
    assert parsed["body"] == "Body line."


class _Atom:
    def __init__(self, artifact_id: str, text: str):
        self.id = f"atm_{artifact_id}"
        self.artifact_id = artifact_id
        self.atom_type = type("T", (), {"value": "scope_item"})()
        self.raw_text = text
        self.text = text
        self.value = {"text": text}
        self.entity_keys = []
        self.source_refs = []


def test_note_provenance_backfill_mints_pointer_when_note_has_no_atoms(tmp_path: Path) -> None:
    note = tmp_path / "010058-hs-note-111645581827-Jacob.txt"
    note.write_text(
        "\n".join(
            [
                "HubSpot Note: Jacob told him- configured remotely",
                "HubSpot Note ID: 111645581827",
                "Date: 2026-06-24T16:21:08.067Z",
                "Author: Patrick Kelly",
                "",
                "Jacob told him- that is alr phyiscally intsalled just needs to be configured.",
            ]
        ),
        encoding="utf-8",
    )
    atoms = [
        _Atom(
            "art_pdf",
            "Jacob told him- that is alr phyiscally intsalled just needs to be configured. Meraki network.",
        )
    ]
    out, minted = ensure_hubspot_note_provenance(
        atoms,
        project_id="deal-1",
        artifact_paths={"art_note": note},
    )
    assert minted == 1
    note_atoms = [a for a in out if getattr(a, "artifact_id", "") == "art_note"]
    assert len(note_atoms) == 1
    assert note_atoms[0].value.get("duplicate_of") == "atm_art_pdf"
