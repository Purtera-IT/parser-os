"""A taught merge has to move the site COUNT, not just the entity roster.

`fuse_alias_groups` collapses EntityRecords and folds the other keys into that
record's aliases. It never rewrites an atom's `entity_keys` — and
`build_site_readiness` reads exactly those. So a merge could be judged, agreed
and applied to the entity roster while the number a PM is looking at never
moved. That is what "One site" was failing to do on deal 010302.
"""

from __future__ import annotations

from app.core.orbitbrief_core import _site_canonical_map, build_site_readiness


class _Atom:
    _n = 0

    def __init__(self, entity_keys, value, atom_type="physical_site"):
        _Atom._n += 1
        self.id = f"atm_{_Atom._n}"
        self.atom_type = atom_type
        self.entity_keys = entity_keys
        self.value = value
        self.text = ""
        self.raw_text = ""
        self.artifact_id = "art_1"
        self.confidence = 0.9
        self.authority_class = "machine_extractor"


A = "site:palo_alto_ca_94304"
B = "site:symphony_ai_hillview_office"


def _rows(atoms, groups):
    out = build_site_readiness(atoms=atoms, edges=[], alias_groups=groups)
    return {r["site"] for r in out["sites"]}, out["site_count"]


def test_a_taught_group_collapses_two_rows_into_one() -> None:
    atoms = [
        _Atom([A], {"id": "PALO-ALTO-CA-94304", "city": "Palo Alto", "state": "CA"}),
        _Atom([B], {"id": "SymphonyAI-Hillview-Office"}),
    ]
    keys, count = _rows(atoms, [frozenset({A, B})])
    assert count == 1, keys
    # Alphabetically first survives, matching fuse_alias_groups.
    assert keys == {A}


def test_without_a_group_they_stay_two() -> None:
    """No merge is invented — a group has to have been judged."""
    atoms = [
        _Atom([A], {"id": "PALO-ALTO-CA-94304"}),
        _Atom([B], {"id": "SymphonyAI-Hillview-Office"}),
    ]
    _, count = _rows(atoms, [])
    assert count == 2


def test_the_canonical_key_matches_fuse_alias_groups() -> None:
    """Both must pick the same survivor, or the roster and the entity records
    disagree about which key a site is filed under."""
    assert _site_canonical_map([], [frozenset({B, A})]) == {A: A, B: A}


def test_a_group_of_one_is_not_a_merge() -> None:
    assert _site_canonical_map([], [frozenset({A})]) == {}


def test_non_site_keys_are_ignored() -> None:
    assert _site_canonical_map([], [frozenset({"stakeholder:a", "stakeholder:b"})]) == {}
    assert _site_canonical_map([], [frozenset({"site:", A})]) == {}
