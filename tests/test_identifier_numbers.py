"""A number that names something (store 1518) is not a headline count."""
from types import SimpleNamespace

from app.core.atom_type_sanity import identifier_numbers, surface_headline_quantities
from app.core.schemas import AtomType


def _scope(text):
    return SimpleNamespace(atom_type=AtomType.scope_item, raw_text=text, value={}, artifact_id="art", source_refs=[], entity_keys=[])


_BINGHAMTON = [
    "Consolidate the 1518 location tech into the existing 1517 subnet",
    "Medicine Shoppe 1518 Ubiquiti router (192.168.132.120)",
    "Medicine Shoppe 1517 CCM-1517-BOTTOM Meraki switch",
    "Remove the 1518 SonicWall from service, label it, and store as a backup",
    "Remove 1518 Wi-Fi or combine it with 1517 Wi-Fi as needed",
]


def test_store_numbers_are_names():
    assert {1517, 1518} <= identifier_numbers(_BINGHAMTON)


def test_counts_followed_by_a_plural_stay_counts():
    texts = ["Install 12 cameras in the lobby", "12 access points on floor 2", "San Fran (2 techs x 2 hours)"]
    ids = identifier_numbers(texts)
    assert 12 not in ids and 2 not in ids


def test_surfacing_skips_named_numbers_but_keeps_real_counts():
    atoms = [_scope(t) for t in _BINGHAMTON] + [_scope("The rollout adds 24 routers across the district")]
    surfaced = surface_headline_quantities(atoms, project_id="p")
    counts = {a.value["quantity"] for a in surfaced}
    assert 1518 not in counts and 1517 not in counts
    assert 24 in counts
