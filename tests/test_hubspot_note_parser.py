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


def test_parse_hubspot_note_text_handles_headerless_title_body() -> None:
    # The deal-uploads pipeline writes notes as "title\n\nbody" with NO HubSpot
    # export headers. The header state machine used to drop every line here,
    # producing zero atoms (ok_empty). The body (and its address) must survive.
    parsed = parse_hubspot_note_text(
        "GECKO ROBOTICS\n\nGECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359"
    )
    assert parsed["title"] == "GECKO ROBOTICS"
    assert "PITTSBURGH" in parsed["body"]


def test_hubspot_note_parser_extracts_physical_site_with_city(tmp_path: Path) -> None:
    # Header-less address note (the real on-blob shape: company lead-in +
    # "PA15212-5359" with no space before the ZIP) must ingest as a physical_site
    # atom carrying structured city/state/zip so site_facility_head can derive a
    # "<City> Office" facility name. Universal: any address-bearing note works.
    p = tmp_path / "010058-hs-note-111645120815-GECKO ROBOTICS.txt"
    p.write_text(
        "GECKO ROBOTICS\n\nGECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_addr", p)
    assert atoms, "header-less address note must not parse to zero atoms (ok_empty)"
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


def test_note_provenance_backfill_remints_address_note_as_physical_site(tmp_path: Path) -> None:
    """Trent-style all-lowercase address notes must remint as physical_site."""
    from app.core.schemas import AtomType

    note = tmp_path / "010097-hs-note-112676376893-ashley.txt"
    note.write_text(
        "\n".join(
            [
                "HubSpot Note: 100 south ashley drive suite 500 tampa fl 33602",
                "HubSpot Note ID: 112676376893",
                "Date: 2026-07-09T16:35:02.874Z",
                "Author: Trent Torrence",
                "Author-Email: t@purtera-it.com",
                "",
                "100 south ashley drive suite 500 tampa fl 33602",
            ]
        ),
        encoding="utf-8",
    )
    out, minted = ensure_hubspot_note_provenance(
        [],
        project_id="deal-stinson",
        artifact_paths={"art_note": note},
    )
    assert minted == 1
    note_atoms = [a for a in out if getattr(a, "artifact_id", "") == "art_note"]
    assert len(note_atoms) == 1
    site = note_atoms[0]
    assert site.atom_type == AtomType.physical_site
    assert site.value["city"].lower() == "tampa"
    assert site.value["state"] == "FL"
    assert site.value["zip"] == "33602"
    assert "hubspot_note_physical_site" in site.review_flags
    assert "site:tampa_fl_33602" in site.entity_keys


def test_note_provenance_remints_site_when_deal_metadata_remains(tmp_path: Path) -> None:
    """Address notes must remint physical_site even if deal_metadata survived dedup."""
    from app.core.schemas import (
        ArtifactType,
        AtomType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )

    note = tmp_path / "010097-hs-note-112676376893-ashley.txt"
    note.write_text(
        "\n".join(
            [
                "HubSpot Note: 100 south ashley drive suite 500 tampa fl 33602",
                "HubSpot Note ID: 112676376893",
                "Date: 2026-07-09T16:35:02.874Z",
                "Author: Trent Torrence",
                "",
                "100 south ashley drive suite 500 tampa fl 33602",
            ]
        ),
        encoding="utf-8",
    )
    leftover = EvidenceAtom(
        id="atm_meta",
        project_id="deal-stinson",
        artifact_id="art_note",
        atom_type=AtomType.deal_metadata,
        raw_text="hubspot note provenance",
        normalized_text="hubspot note provenance",
        value={"field_name": "hubspot_note_provenance", "text": "hubspot note provenance"},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="s1",
                artifact_id="art_note",
                artifact_type=ArtifactType.txt,
                filename=note.name,
                locator={},
                extraction_method="t",
                parser_version="t",
            )
        ],
        authority_class=AuthorityClass.meeting_note,
        confidence=0.5,
        review_status=ReviewStatus.needs_review,
        review_flags=["note_provenance_backfill"],
        parser_version="t",
    )
    out, minted = ensure_hubspot_note_provenance(
        [leftover],
        project_id="deal-stinson",
        artifact_paths={"art_note": note},
    )
    assert minted == 1
    sites = [a for a in out if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    assert sites[0].value["zip"] == "33602"
    assert "hubspot_note_physical_site" in sites[0].review_flags
