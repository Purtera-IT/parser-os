"""Regression tests for the four ways a large retail roster used to lose sites.

Grounded in deal 000043 (Clayton Homes Retail Onsite): 437 real store locations
across three workbooks, of which only 20 reached the brief. Each test below pins
one link in that chain. The shapes are generic — a title-banner sheet, a jargon
ID column, an operational status column, a blank display name — because every
multi-site retail/branch roster ships at least one of them.
"""
from __future__ import annotations

from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import semantic_dedup_atoms
from app.core.site_geo_fallback import geo_fallback_sites
from app.parsers.site_roster_extractor import (
    detect_site_id_column,
    extract_site_roster,
    find_header_row,
    looks_like_site_roster,
    map_columns_to_fields,
)

# A store roster as real workbooks ship it: jargon ID column ("HC"), an
# operational status column, snake_case headers, and one branch that has an
# address but no branded display name yet.
_COLUMNS = [
    "HC", "Zone", "Site_Name", "Region", "Manager_Name",
    "Address", "City", "ST", "ZIP", "Migration Status",
]
_ROWS = [
    {"HC": "31", "Zone": "2", "Site_Name": "Homes of Knoxville", "Region": "TEN",
     "Manager_Name": "James McManus", "Address": "4606 Clinton Hwy",
     "City": "Knoxville", "ST": "TN", "ZIP": "37912", "Migration Status": "Complete"},
    {"HC": "268", "Zone": "4", "Site_Name": "Homes of Texarkana", "Region": "TXN",
     "Manager_Name": "Taylor Bradshaw", "Address": "4401 South Lake Drive",
     "City": "Texarkana", "ST": "TX", "ZIP": "75501", "Migration Status": "In Progress"},
    {"HC": "1054", "Zone": "4", "Site_Name": "Homes of Corsicana", "Region": "TXN",
     "Manager_Name": "Morgan Hughey", "Address": "5656 South I-45 West",
     "City": "Corsicana", "ST": "TX", "ZIP": "75109", "Migration Status": "Not Started"},
    # Unbranded store: no display name, real address.
    {"HC": "2712", "Zone": "", "Site_Name": "", "Region": "",
     "Manager_Name": "Joe Manis", "Address": "16021 Elmore Rd",
     "City": "Laurinburg", "ST": "NC", "ZIP": "28352", "Migration Status": ""},
]


# ── 1. An operational Status column must not veto a roster ───────


def test_status_column_does_not_veto_a_dominant_roster():
    """"Migration Status" used to hard-reject the sheet, losing every site."""
    assert looks_like_site_roster(columns=_COLUMNS, rows=_ROWS) is True


def test_asset_inventory_is_still_rejected_outright():
    """The soft guard must not weaken the asset-inventory ghost-site block."""
    cols = ["Asset ID", "Facility", "Street Address", "City", "IP Address", "Serial"]
    rows = [{"Asset ID": "AST-001", "Facility": "Knoxville DC",
             "Street Address": "1 Main St", "City": "Knoxville",
             "IP Address": "10.0.0.1", "Serial": "SN123"}]
    assert looks_like_site_roster(columns=cols, rows=rows) is False


def test_weak_evidence_plus_soft_signal_still_rejects():
    """One incidental roster field is not enough to override a BOM header."""
    cols = ["Line item", "Description", "Location", "Quantity", "Unit price"]
    rows = [{"Line item": "1", "Description": "Switch", "Location": "MDF",
             "Quantity": "5", "Unit price": "$1,000"}]
    assert looks_like_site_roster(columns=cols, rows=rows) is False


# ── 2. Title banners must not be read as the header ──────────────


def test_find_header_row_skips_title_banner():
    rows = [
        ["Site Master", "", "", "", ""],
        ["All in-scope sites with dispatch metadata", "", "", "", ""],
        ["HC", "Site_Name", "Address", "City", "ZIP"],
        ["31", "Homes of Knoxville", "4606 Clinton Hwy", "Knoxville", "37912"],
    ]
    assert find_header_row(rows) == 2


def test_find_header_row_falls_back_to_first_nonblank():
    rows = [["", ""], ["Widget", "5"], ["Gadget", "7"]]
    assert find_header_row(rows) == 1


def test_snake_case_headers_map_to_canonical_fields():
    fields = set(map_columns_to_fields(["Site_Name", "Street_Address"]).values())
    assert {"facility_name", "street_address"} <= fields


# ── 3. The ID column is found by shape when no header names it ───


def test_detect_site_id_column_finds_jargon_id_column():
    field_map = map_columns_to_fields(_COLUMNS)
    assert detect_site_id_column(_COLUMNS, _ROWS, field_map) == 0


def test_detect_site_id_column_ignores_human_names():
    """A name column is unique and short but carries no digit — never an id."""
    cols = ["First", "Facility", "Address"]
    rows = [{"First": "David", "Facility": "A", "Address": "1 Main St"},
            {"First": "Taylor", "Facility": "B", "Address": "2 Main St"},
            {"First": "Morgan", "Facility": "C", "Address": "3 Main St"}]
    assert detect_site_id_column(cols, rows, map_columns_to_fields(cols)) is None


def test_detect_site_id_column_ignores_uniform_width_digits():
    """A 10-digit unique column is a phone number, not a store code."""
    cols = ["Contact number", "Facility", "Address"]
    rows = [{"Contact number": "2547995522", "Facility": "A", "Address": "1 Main St"},
            {"Contact number": "9038385994", "Facility": "B", "Address": "2 Main St"},
            {"Contact number": "9038872114", "Facility": "C", "Address": "3 Main St"}]
    assert detect_site_id_column(cols, rows, map_columns_to_fields(cols)) is None


def test_bare_numeric_codes_are_namespaced_by_their_column():
    out = extract_site_roster(columns=_COLUMNS, rows=_ROWS)
    assert [r.site_id for r in out] == ["HC-31", "HC-268", "HC-1054", "HC-2712"]


# ── 4. An address alone anchors a site ───────────────────────────


def test_row_with_address_but_no_name_is_kept():
    out = extract_site_roster(columns=_COLUMNS, rows=_ROWS)
    assert len(out) == len(_ROWS)
    unbranded = out[-1]
    assert not unbranded.facility_name
    assert unbranded.street_address == "16021 Elmore Rd"


def test_row_with_nothing_is_still_dropped():
    cols = ["HC", "Site_Name", "Address", "City", "ZIP"]
    rows = [
        {"HC": "31", "Site_Name": "Knoxville", "Address": "4606 Clinton Hwy",
         "City": "Knoxville", "ZIP": "37912"},
        {"HC": "34", "Site_Name": "Middlesboro", "Address": "625 N. 12th St.",
         "City": "Middlesboro", "ZIP": "40965"},
        {"HC": "37", "Site_Name": "Somerset", "Address": "4860 South Hwy 27",
         "City": "Somerset", "ZIP": "42501"},
        {"HC": "", "Site_Name": "", "Address": "", "City": "Knoxville", "ZIP": ""},
    ]
    assert len(extract_site_roster(columns=cols, rows=rows)) == 3


# ── 5. Roster rows survive semantic_dedup / never trigger geo guesses ──


def _roster_atom(idx: int, *, site_id: str | None, name: str, address: str) -> EvidenceAtom:
    aid = f"atm_roster_{idx}"
    return EvidenceAtom(
        id=aid,
        project_id="p",
        artifact_id="art_1",
        atom_type=AtomType.physical_site,
        raw_text=f"{name} | {address}",
        normalized_text=f"{name} | {address}".lower(),
        value={"kind": "physical_site", "id": site_id, "site_id": site_id,
               "name": name, "facility_name": name,
               "address": address, "street_address": address},
        entity_keys=[],
        source_refs=[SourceRef(
            id=f"src_{idx}",
            artifact_id="art_1",
            artifact_type=ArtifactType.xlsx,
            filename="Exhibit A - Retail Locations.xlsx",
            locator={"sheet": "Pending", "row": idx, "extraction": "xlsx_site_roster_v1"},
            extraction_method="xlsx_site_roster_v1",
            parser_version="test",
        )],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.85,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def _roster_atoms(with_ids: bool) -> list[EvidenceAtom]:
    return [
        _roster_atom(i, site_id=(f"HC-{i}" if with_ids else None),
                     name=f"Homes of Town {i}", address=f"{100 + i} Main St")
        for i in range(1, 26)
    ]


def test_semantic_dedup_keeps_every_roster_row_with_ids():
    atoms = _roster_atoms(with_ids=True)
    kept = [a for a in semantic_dedup_atoms(atoms)
            if a.atom_type is AtomType.physical_site]
    assert len(kept) == len(atoms)


def test_semantic_dedup_keeps_roster_rows_that_have_no_id_column():
    """The regression that cost deal 000043 all but 20 of its 437 sites: an
    id-less roster row was classed as a prose ghost and silently dropped."""
    atoms = _roster_atoms(with_ids=False)
    kept = [a for a in semantic_dedup_atoms(atoms)
            if a.atom_type is AtomType.physical_site]
    assert len(kept) == len(atoms)


def test_geo_fallback_stays_quiet_when_a_roster_exists():
    """A deal with a real roster must never get "City, ST ZIP" guesses on top,
    even when the roster ships no ID column."""
    atoms = _roster_atoms(with_ids=False)
    atoms[0].raw_text = "site located at Santa Fe, NM 87506"
    assert geo_fallback_sites(atoms, project_id="p") == []
