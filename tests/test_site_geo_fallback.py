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
    assert site.value["site_id"] == "santa_fe_87506"
    assert site.value["inferred"] is True
    assert "site:santa_fe_87506" in site.entity_keys
    assert "geo_fallback_site" in site.review_flags


def test_no_fallback_when_real_site_exists() -> None:
    atoms = [
        _Atom("physical_site", "Memorial Hospital", value={"id": "MEM-01", "site_id": "MEM-01"}),
        _Atom("open_question", "location Santa Fe, NM 87506"),
    ]
    assert geo_fallback_sites(atoms, project_id="yonah") == []


def test_invalid_state_rejected() -> None:
    # "ZZ" is not a US state — no false site.
    atoms = [_Atom("note", "Springfield, ZZ 99999 is fictional")]
    assert geo_fallback_sites(atoms, project_id="p") == []


def test_dedup_by_zip() -> None:
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
