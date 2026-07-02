"""Geo-fallback site extraction.

Grounded in the Yonah deal: the only locational anchor is
``location Santa Fe, NM 87506`` in Notes.pdf, no street address. The
fallback turns that into one needs_review physical_site so the deal
anchors instead of going blank-RED — but only when no real site exists.
"""

from __future__ import annotations

from app.core.schemas import AtomType
from app.core.site_geo_fallback import geo_fallback_sites


class _Atom:
    def __init__(self, atom_type, text, value=None, artifact_id="art"):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.value = value or {}
        self.source_refs = []
        self.artifact_id = artifact_id


def test_santa_fe_anchor_emits_physical_site() -> None:
    atoms = [
        _Atom("open_question", "location Santa Fe, NM 87506 What size TVs?"),
        _Atom("scope_item", "Property has 23 dwellings, approx 8 second story"),
    ]
    out = geo_fallback_sites(atoms, project_id="yonah")
    assert len(out) == 1
    site = out[0]
    assert site.atom_type == AtomType.physical_site
    assert site.value["city"] == "Santa Fe"
    assert site.value["state"] == "NM"
    assert site.value["zip"] == "87506"
    assert site.value["site_id"] == "santa_fe_nm_87506"
    assert site.value["inferred"] is True
    assert "site:santa_fe_nm_87506" in site.entity_keys
    assert "geo_fallback_site" in site.review_flags


def test_hubspot_note_compact_state_zip_address_emits_physical_site() -> None:
    atoms = [
        _Atom("scope_item", "GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359")
    ]
    out = geo_fallback_sites(atoms, project_id="gecko")
    assert len(out) == 1
    site = out[0]
    assert site.value["street_address"] == "100 S COMMONS STE 145"
    assert site.value["city"] == "PITTSBURGH"
    assert site.value["state"] == "PA"
    assert site.value["zip"] == "15212"
    assert "site:pittsburgh_pa_15212" in site.entity_keys


def test_no_fallback_when_two_structured_sites_exist() -> None:
    atoms = [
        _Atom(
            "physical_site",
            "12575 Oakland Park Blvd, Highland Park, MI 48203",
            value={
                "id": "site-a",
                "site_id": "site-a",
                "street_address": "12575 Oakland Park Blvd",
                "city": "Highland Park",
                "state": "MI",
                "zip": "48203",
            },
        ),
        _Atom(
            "physical_site",
            "200 Main St, Detroit, MI 48201",
            value={
                "id": "site-b",
                "site_id": "site-b",
                "street_address": "200 Main St",
                "city": "Detroit",
                "state": "MI",
            },
        ),
    ]
    assert geo_fallback_sites(atoms, project_id="p") == []


def test_fallback_still_runs_with_one_weak_site() -> None:
    """MBrany class: one misparsed site must not block discovering more anchors."""
    atoms = [
        _Atom("physical_site", "Park BLvd. Highland Park, MI 48203", value={"id": "x", "site_id": "x"}),
        _Atom("open_question", "location Santa Fe, NM 87506"),
    ]
    out = geo_fallback_sites(atoms, project_id="yonah")
    assert len(out) >= 1
    assert any(a.value.get("city") == "Santa Fe" for a in out)


def test_invalid_state_rejected() -> None:
    # "ZZ" is not a US state — no false site.
    atoms = [_Atom("note", "Springfield, ZZ 99999 is fictional")]
    assert geo_fallback_sites(atoms, project_id="p") == []


def test_dedup_by_address_not_just_zip() -> None:
    atoms = [
        _Atom("note", "12575 Oakland Park Blvd, Highland Park, MI 48203"),
        _Atom("note", "200 Main St, Highland Park, MI 48203"),
    ]
    out = geo_fallback_sites(atoms, project_id="p")
    assert len(out) == 2


def test_dedup_same_address() -> None:
    atoms = [
        _Atom("note", "location Santa Fe, NM 87506 first mention"),
        _Atom("note", "again Santa Fe, NM 87506 second mention"),
    ]
    out = geo_fallback_sites(atoms, project_id="p")
    assert len(out) == 1


def test_multiword_city() -> None:
    atoms = [_Atom("note", "Site at San Luis Obispo, CA 93401 confirmed")]
    out = geo_fallback_sites(atoms, project_id="p")
    assert len(out) == 1 and out[0].value["city"] == "San Luis Obispo"


def test_empty_when_no_anchor() -> None:
    atoms = [_Atom("scope_item", "Install 23 TVs across the dwellings")]
    assert geo_fallback_sites(atoms, project_id="p") == []
