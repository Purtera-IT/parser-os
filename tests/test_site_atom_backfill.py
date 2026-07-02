"""Tests for site_atom_backfill."""

from __future__ import annotations

from app.core.schemas import EntityRecord, ReviewStatus
from app.core.site_atom_backfill import backfill_physical_sites_from_entities


class _Atom:
    def __init__(self, atom_type: str, text: str, entity_keys=None):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.entity_keys = entity_keys or []
        self.source_refs = []
        self.artifact_id = "art1"


def test_backfill_mints_from_site_entity() -> None:
    atoms = [_Atom("scope_item", "Work at 12575 Oakland Park Blvd, Highland Park, MI 48203")]
    entities = [
        EntityRecord(
            id="ent1",
            project_id="p1",
            entity_type="site",
            canonical_key="site:highland_park_mi",
            canonical_name="Highland Park, MI 48203",
            aliases=[],
            source_atom_ids=[],
            confidence=0.8,
            review_status=ReviewStatus.auto_accepted,
        )
    ]
    out, n = backfill_physical_sites_from_entities(atoms, entities, project_id="p1")
    assert n == 1
    phys = [a for a in out if str(getattr(a, "atom_type", "")).endswith("physical_site") or getattr(a.atom_type, "value", "") == "physical_site"]
    assert len(phys) == 1
    assert phys[0].value["city"] == "Highland Park"
    assert phys[0].value["state"] == "MI"


def test_backfill_noop_when_roster_sufficient() -> None:
    class _Site:
        atom_type = "physical_site"
        value = {"city": "A", "state": "MI"}
        entity_keys = ["site:highland_park_mi"]

    atoms = [_Site()]
    entities = [
        EntityRecord(
            id="ent1",
            project_id="p1",
            entity_type="site",
            canonical_key="site:highland_park_mi",
            canonical_name="Highland Park, MI",
            aliases=[],
            source_atom_ids=[],
            confidence=0.8,
            review_status=ReviewStatus.auto_accepted,
        )
    ]
    out, n = backfill_physical_sites_from_entities(atoms, entities, project_id="p1")
    assert n == 0
    assert out is atoms


def test_backfill_noop_when_alias_site_keys_share_one_roster_location() -> None:
    """MBrany-class: semantic_dedup leaves one site; alias site:* keys must not re-mint."""
    class _Site:
        atom_type = "physical_site"
        value = {
            "site_id": "HIGHLAND-PARK-MI-48203",
            "street_address": "12575 Oakland Park BLvd",
            "city": "Highland Park",
            "state": "MI",
            "zip": "48203",
        }
        entity_keys = ["site:highland_park_mi_48203"]
        raw_text = "12575 Oakland Park BLvd, Highland Park, MI 48203"
        text = raw_text
        source_refs = []
        artifact_id = "art1"

    atoms = [
        _Site(),
        _Atom(
            "scope_item",
            "Location: Mobis North America Work, 12575 Oakland Park BLvd., Highland Park, MI 48203",
            entity_keys=["site:12575_oakland_park_blvd"],
        ),
    ]
    entities = [
        EntityRecord(
            id="ent1",
            project_id="p1",
            entity_type="site",
            canonical_key="site:highland_park_mi_48203",
            canonical_name="Highland Park, MI 48203",
            aliases=[],
            source_atom_ids=[],
            confidence=0.8,
            review_status=ReviewStatus.auto_accepted,
        ),
        EntityRecord(
            id="ent2",
            project_id="p1",
            entity_type="site",
            canonical_key="site:12575_oakland_park_blvd",
            canonical_name="12575 Oakland Park Blvd",
            aliases=[],
            source_atom_ids=[],
            confidence=0.8,
            review_status=ReviewStatus.auto_accepted,
        ),
    ]
    out, n = backfill_physical_sites_from_entities(atoms, entities, project_id="p1")
    assert n == 0
    phys = [
        a
        for a in out
        if str(getattr(a, "atom_type", "")).endswith("physical_site")
        or getattr(a.atom_type, "value", "") == "physical_site"
    ]
    assert len(phys) == 1
    assert phys[0].value["site_id"] == "HIGHLAND-PARK-MI-48203"


def test_backfill_runs_when_roster_insufficient() -> None:
    class _Site:
        atom_type = "physical_site"
        value = {"city": "Highland Park", "state": "MI", "zip": "48203"}
        entity_keys = ["site:highland_park_mi_48203"]
        raw_text = "Highland Park, MI 48203"
        text = raw_text
        source_refs = []
        artifact_id = "art1"

    atoms = [
        _Site(),
        _Atom("scope_item", "Work at 12575 Oakland Park Blvd, Alpharetta, GA 30009"),
    ]
    entities = [
        EntityRecord(
            id="ent1",
            project_id="p1",
            entity_type="site",
            canonical_key="site:highland_park_mi_48203",
            canonical_name="Highland Park, MI 48203",
            aliases=[],
            source_atom_ids=[],
            confidence=0.8,
            review_status=ReviewStatus.auto_accepted,
        ),
        EntityRecord(
            id="ent2",
            project_id="p1",
            entity_type="site",
            canonical_key="site:alpharetta_ga_30009",
            canonical_name="Alpharetta, GA 30009",
            aliases=[],
            source_atom_ids=[],
            confidence=0.8,
            review_status=ReviewStatus.auto_accepted,
        ),
    ]
    out, n = backfill_physical_sites_from_entities(atoms, entities, project_id="p1")
    assert n >= 1
    phys = [
        a
        for a in out
        if str(getattr(a, "atom_type", "")).endswith("physical_site")
        or getattr(a.atom_type, "value", "") == "physical_site"
    ]
    assert len(phys) >= 2


def test_post_backfill_dedup_collapses_same_place_mints() -> None:
    """MBrany-class: backfill mints after dedup; post-backfill dedup must collapse."""
    from app.core.semantic_dedup import _dedupe_physical_site_atoms
    from app.core.site_atom_backfill import _mint_physical_site

    class _Roster:
        atom_type = "physical_site"
        value = {
            "site_id": "HIGHLAND-PARK-MI-48203",
            "id": "HIGHLAND-PARK-MI-48203",
            "street_address": "12575 Oakland Park BLvd",
            "city": "Highland Park",
            "state": "MI",
            "zip": "48203",
            "name": "12575 Oakland Park BLvd, Highland Park, MI 48203",
        }
        entity_keys = ["site:highland_park_mi_48203"]
        raw_text = "12575 Oakland Park BLvd, Highland Park, MI 48203"
        text = raw_text
        source_refs = []
        artifact_id = "art1"
        review_flags = []

    anchor = _Atom(
        "scope_item",
        "Location: Mobis North America Work, 12575 Oakland Park BLvd., Highland Park, MI 48203",
        entity_keys=["site:12575_oakland_park_blvd"],
    )
    mint_street = _mint_physical_site(
        project_id="p1",
        site_key="site:12575_oakland_park_blvd",
        display_name="12575 Oakland Park Blvd",
        geo={
            "street_address": "Location: Mobis North America Work, 12575 Oakland Park BLvd.",
            "city": "Highland Park",
            "state": "MI",
            "zip": "48203",
        },
        source_atom=anchor,
        reason="entity_backfill",
    )
    mint_slug = _mint_physical_site(
        project_id="p1",
        site_key="site:highland_park_mi_48203",
        display_name="highland park mi 48203",
        geo={},
        source_atom=anchor,
        reason="entity_backfill",
    )
    out = _dedupe_physical_site_atoms([_Roster(), mint_street, mint_slug])
    phys = [
        a
        for a in out
        if str(getattr(a, "atom_type", "")).endswith("physical_site")
        or getattr(a.atom_type, "value", "") == "physical_site"
    ]
    assert len(phys) == 1
    assert phys[0].value["site_id"] == "HIGHLAND-PARK-MI-48203"


def test_physical_site_dedup_prefers_address_over_name_only_alias() -> None:
    from app.core.semantic_dedup import _dedupe_physical_site_atoms

    class _Site:
        atom_type = "physical_site"
        source_refs = []
        artifact_id = "art1"
        review_flags = []

        def __init__(self, value, raw_text, keys):
            self.value = value
            self.raw_text = raw_text
            self.text = raw_text
            self.entity_keys = keys

    address_site = _Site(
        {
            "site_id": "PITTSBURGH-PA-15212",
            "id": "PITTSBURGH-PA-15212",
            "street_address": "100 S COMMONS STE 145",
            "address": "100 S COMMONS STE 145",
            "city": "PITTSBURGH",
            "state": "PA",
            "zip": "15212",
            "name": "100 S COMMONS STE 145, PITTSBURGH, PA 15212",
        },
        "100 S COMMONS STE 145, PITTSBURGH, PA 15212",
        ["site:pittsburgh_pa_15212"],
    )
    name_only = _Site(
        {
            "site_id": "GECKO-ROBOTICS-PITTSBURGH-OFFICE-WORKSHOP",
            "id": "GECKO-ROBOTICS-PITTSBURGH-OFFICE-WORKSHOP",
            "name": "gecko robotics pittsburgh office workshop",
            "facility_name": "gecko robotics pittsburgh office workshop",
        },
        "Gecko Robotics requested assistance configuring a new Ubiquiti deployment for a new office/workshop location.",
        ["site:gecko_robotics_pittsburgh_office_workshop"],
    )

    phys = [
        a
        for a in _dedupe_physical_site_atoms([name_only, address_site])
        if str(getattr(a, "atom_type", "")).endswith("physical_site")
        or getattr(a.atom_type, "value", "") == "physical_site"
    ]

    assert len(phys) == 1
    assert phys[0].value["site_id"] == "PITTSBURGH-PA-15212"
    assert phys[0].value["street_address"] == "100 S COMMONS STE 145"
    assert "gecko robotics pittsburgh office workshop" in phys[0].value.get("aliases", [])


def test_physical_site_dedup_keeps_authoritative_name_only_roster_site() -> None:
    from app.core.semantic_dedup import _dedupe_physical_site_atoms

    class _Ref:
        filename = "customer_site_roster.xlsx"

    class _Site:
        atom_type = "physical_site"
        artifact_id = "art1"
        review_flags = []

        def __init__(self, value, raw_text, keys, source_refs=None):
            self.value = value
            self.raw_text = raw_text
            self.text = raw_text
            self.entity_keys = keys
            self.source_refs = source_refs or []

    address_site = _Site(
        {
            "site_id": "PITTSBURGH-PA-15212",
            "id": "PITTSBURGH-PA-15212",
            "street_address": "100 S COMMONS STE 145",
            "city": "PITTSBURGH",
            "state": "PA",
            "zip": "15212",
            "name": "100 S COMMONS STE 145, PITTSBURGH, PA 15212",
        },
        "100 S COMMONS STE 145, PITTSBURGH, PA 15212",
        ["site:pittsburgh_pa_15212"],
    )
    roster_site = _Site(
        {
            "site_id": "ATL-HQ-01",
            "id": "ATL-HQ-01",
            "name": "Atlanta HQ",
            "facility_name": "Atlanta HQ",
        },
        "site roster row: ATL-HQ-01 Atlanta HQ",
        ["site:atl_hq_01"],
        [_Ref()],
    )

    phys = [
        a
        for a in _dedupe_physical_site_atoms([address_site, roster_site])
        if str(getattr(a, "atom_type", "")).endswith("physical_site")
        or getattr(a.atom_type, "value", "") == "physical_site"
    ]

    assert {p.value["site_id"] for p in phys} == {"PITTSBURGH-PA-15212", "ATL-HQ-01"}
