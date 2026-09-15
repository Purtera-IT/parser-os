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
    # The city verbatim; the rule no longer composes "<City> Office", which
    # invented a name absent from every document. See test_deal_kit_heads.
    assert decision.facility_name == "Pittsburgh"


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


def test_note_typed_one_label_per_line_keeps_each_line_as_an_item(tmp_path: Path) -> None:
    # 000020 Binghamton shape: labels on their own lines (one carries a store
    # number, one an apostrophe), a two-line address with no ZIP, task lines
    # that contain commas, and sign-off sentences under the last list.
    p = tmp_path / "000020-hs-note-107431630257-work order.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: We have a WO for you",
                "HubSpot Note ID: 107431630257",
                "Date: 2026-04-03T11:21:02.144Z",
                "Author: Trent Torrence",
                "",
                "We have a WO for you. Could you let me know the cost?",
                "Address:",
                "1179 Vestal Ave, Suite 1",
                "Binghamton, NY",
                "Here\u2019s the scope:",
                "Project: Consolidate the 1518 location tech into the existing 1517 subnet",
                "Hardware involved:",
                "Medicine Shoppe 1517 SonicWall (192.168.133.50) firewall",
                "Medicine Shoppe 1518 Ubiquiti router (192.168.132.120)",
                "Medicine Shoppe 1517 CCM-1517-BOTTOM Meraki switch",
                "Onsite work at 1517:",
                "Coordinate scheduling with the remote team and client",
                "Remove the 1518 SonicWall from service, label it, and store as a backup",
                "Reset Ubiquiti gateway and switch for the new subnet",
                "Remote work:",
                "Guide onsite tech in bringing devices online in the new subnet",
                "Update/configure settings in SonicWall, Meraki, and/or Ubiquiti (subnet, ISP, mapping, etc.)",
                "Test systems with the client and troubleshoot as needed",
                "The tech will have to work with A1 engineer throughout the WO.",
                "If you have any questions, feel free to reach out.",
            ]
        ),
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_wo", p)
    items = [a.raw_text for a in atoms if (a.value or {}).get("kind") == "note_field_item"]
    assert "Remove the 1518 SonicWall from service, label it, and store as a backup" in items
    assert "Update/configure settings in SonicWall, Meraki, and/or Ubiquiti (subnet, ISP, mapping, etc.)" in items
    onsite = [a for a in atoms if (a.value or {}).get("parent_field") == "Onsite work at 1517"
              and (a.value or {}).get("kind") == "note_field_item"]
    assert len(onsite) == 3
    hardware = [a for a in atoms if (a.value or {}).get("parent_field") == "Hardware involved"]
    assert len(hardware) == 3 and all(a.atom_type == AtomType.scope_item for a in hardware)
    for fragment in ("label it", "ISP", "mapping", "feel free to reach out"):
        assert fragment not in items
    trailer = [a for a in atoms if (a.value or {}).get("kind") == "note_field_trailer"]
    assert len(trailer) == 1 and trailer[0].raw_text.startswith("The tech will have to work")
    assert trailer[0].atom_type == AtomType.constraint
    # No whole-section atom repeating every line of a one-per-line list.
    assert not [a for a in atoms if (a.value or {}).get("kind") == "note_field"
                and (a.value or {}).get("field_name") in ("Onsite work at 1517", "Remote work", "Hardware involved")]
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    assert sites[0].value["city"] == "Binghamton" and sites[0].value["state"] == "NY"
    assert sites[0].value["street_address"] == "1179 Vestal Ave, Suite 1"
    assert "scope" not in sites[0].raw_text.lower()


def test_one_line_form_note_still_splits_inline_fields(tmp_path: Path) -> None:
    # The 010297 shape the inline splitter exists for: every field on one line.
    p = tmp_path / "010297-hs-note-1-form.txt"
    p.write_text(
        "HubSpot Note: form\nHubSpot Note ID: 1\nDate: 2026-06-01T00:00:00Z\nAuthor: A\n\n"
        "Address: 500 Main Street, Springfield, IL 62701 Duration: 4 hours Scope of work: mount displays, run HDMI, test audio",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_form", p)
    fields = {(a.value or {}).get("field_name") for a in atoms if (a.value or {}).get("kind") == "note_field"}
    assert {"Duration", "Scope of work"} <= fields


_SF_ASK = (
    "Hi Trent, Hope all is well! My customer, Checkout LLC, is renovating their office in San Fransico. "
    "Most of their IT team is remote, so they are looking for a service partner to install their new "
    "Samsung 65' display. Would PurTeraIT be able to do this?"
)
_SF_ADDRESS_AND_SIGNATURE = [
    "Adress: CHECKOUT SAN FRANCISCO",
    "30 HOTALING PL FL 3",
    "SAN FRANCISCO, CA 94111-2201",
    "Sarah Halpern",
    "Account Manager | NYC Financial Services | CDW",
    "72 Madison Avenue | New York, NY 10016",
    "Direct: 212.894.0736 | Toll Free Number: 877.842.7372",
]
_SF_HEADERS = "HubSpot Note: Hi Trent,\nHubSpot Note ID: 108135392133\nDate: 2026-04-16T18:12:09.247Z\nAuthor: Trent Torrence\n\n"


def _sf_atoms(tmp_path: Path, body: str):
    p = tmp_path / "000036-hs-note-108135392133-Hi Trent_.txt"
    p.write_text(_SF_HEADERS + body, encoding="utf-8")
    return HubspotNoteParser().parse_artifact("deal-1", "art_sf", p)


def _assert_sf(atoms) -> None:
    # The request before the first label is the only statement of the work; it must survive.
    assert any("install their new Samsung 65' display" in a.raw_text
               and a.atom_type == AtomType.scope_item for a in atoms)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    v = sites[0].value
    assert (v["city"].upper(), v["state"], v["zip"]) == ("SAN FRANCISCO", "CA", "94111")
    assert "Halpern" not in sites[0].raw_text and "New York" not in sites[0].raw_text


def test_request_before_labels_and_signature_after_address_line_broken(tmp_path: Path) -> None:
    # 000036 San Fran TV mount, the copy with the author's line breaks.
    _assert_sf(_sf_atoms(tmp_path, "\n".join(["Hi Trent,", _SF_ASK.removeprefix("Hi Trent, ")] + _SF_ADDRESS_AND_SIGNATURE)))


def test_request_before_labels_and_signature_after_address_flattened(tmp_path: Path) -> None:
    # The same note as mirrored flattened onto one line.
    _assert_sf(_sf_atoms(tmp_path, " ".join([_SF_ASK] + _SF_ADDRESS_AND_SIGNATURE)))


def test_label_whose_line_is_a_full_statement_does_not_swallow_the_recap(tmp_path: Path) -> None:
    # 000061 MBrany meeting recap: "Label: sentence." lines, then the whole
    # recap body under the last one.
    p = tmp_path / "000061-hs-note-108983476067-APs Relocation.txt"
    p.write_text(
        "HubSpot Note: APs Relocation\nHubSpot Note ID: 108983476067\nDate: 2026-04-30T18:30:00Z\nAuthor: HubSpot user\n\n"
        + "\n".join([
            "APs Relocation: Plan to lower 5-10 APs to enhance signal coverage in a 35ft ceiling office; site survey needed.",
            "aj@purtera-it.com",
            "Site Survey Details: Expected four-hour survey will confirm AP count, cabling types, and verify obstacles for installations.",
            "Next Steps: Client to provide floor plans; survey will finalize scope and enable quick quote delivery after that.",
            "Project Scope and Site Preparation",
            "The existing APs are mounted too high, causing signal inefficiency for users below",
            "The office's network closet is about 150 feet from the AP area, impacting cable runs",
        ]),
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_recap", p)
    items_of_next_steps = " ".join(a.raw_text for a in atoms if (a.value or {}).get("parent_field") == "Next Steps")
    assert "mounted too high" not in items_of_next_steps and "150 feet" not in items_of_next_steps
    prose = [a.raw_text for a in atoms if a.atom_type == AtomType.scope_item and (a.value or {}).get("kind") == "hubspot_note_body"]
    assert any("mounted too high" in t for t in prose)
    assert any("150 feet from the AP area" in t for t in prose)
    assert any(a.raw_text == "aj@purtera-it.com" and a.atom_type == AtomType.deal_metadata for a in atoms)


def test_note_metadata_atom_is_marked_non_deal(tmp_path: Path) -> None:
    from app.core.span_admission import _is_protected_email_atom

    p = tmp_path / "000020-hs-note-1-meta.txt"
    p.write_text("HubSpot Note: X\nHubSpot Note ID: 1\nDate: 2026-04-03T11:21:02Z\nAuthor: Trent Torrence\n\nReset the gateway after hours.", encoding="utf-8")
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_meta", p)
    meta = [a for a in atoms if (a.value or {}).get("field_name") == "hubspot_note_meta"]
    assert len(meta) == 1 and meta[0].value["non_deal"] is True
    assert _is_protected_email_atom(meta[0])
