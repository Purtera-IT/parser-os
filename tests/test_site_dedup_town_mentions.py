"""Two mentions of one place, minted from a call with no street, are one site.

Live 000061 (compile 9a6aacfc, 2026-09-15): the roster was empty, so backfill
minted "highland park warehouse office" and "office in Highland Park Michigan"
from two sentences of the same call. Both are the customer's one office; the
Deal Kit saw two sites and could place its survey on neither.
"""
from app.core.schemas import AtomType
from app.core.semantic_dedup import _dedupe_physical_site_atoms


class _Site:
    atom_type = AtomType.physical_site
    entity_keys: list = []
    source_refs: list = []
    review_flags: list = []
    raw_text = ""
    artifact_id = "art1"

    def __init__(self, name, city="Highland Park", state="MI", street=None, zip_=None):
        slug = name.upper().replace(" ", "-")
        self.value = {"site_id": slug, "id": slug, "name": name, "city": city, "state": state}
        if street:
            self.value["street_address"] = street
        if zip_:
            self.value["zip"] = zip_
        self.raw_text = name


def _sites(out):
    return sorted(a.value["name"] for a in out if a.atom_type == AtomType.physical_site)


def test_one_office_named_twice_from_the_call_is_one_site():
    out = _dedupe_physical_site_atoms([_Site("highland park warehouse office"), _Site("office in Highland Park Michigan")])
    assert len(_sites(out)) == 1


def test_two_facilities_in_one_town_stay_two():
    out = _dedupe_physical_site_atoms([_Site("Highland Park office"), _Site("Highland Park warehouse")])
    assert _sites(out) == ["Highland Park office", "Highland Park warehouse"]


def test_a_town_only_mention_attaches_to_the_named_site_there():
    out = _dedupe_physical_site_atoms([_Site("Highland Park"), _Site("Highland Park warehouse office")])
    assert len(_sites(out)) == 1


def test_sites_that_know_their_street_never_meet_on_the_town():
    a = _Site("Academy of Early Learning", city="Marion", state="SC", street="719 N Main St")
    b = _Site("Easterling Primary", city="Marion", state="SC", street="1103 N Main St")
    out = _dedupe_physical_site_atoms([a, b])
    assert len(_sites(out)) == 2
