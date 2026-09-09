"""A site row is filled by every atom that knows a field, not just the first.

Deal 02557291 published 2205 Gregg St twice after the address-identity merge
shipped, and the merge was demonstrably correct: replayed against the finished
envelope it grouped all three keys. Live it grouped only two.

The difference was ATOM ORDER. Several physical_site atoms can name one site
key and they do not all carry the same fields -- one holds the street address,
another only a name. `site_rows` took the FIRST atom to mention a key and
discarded the rest, so an address-less atom reaching `site:site_1` first
shadowed the atom that knew the street, and a row with no address cannot be
compared to any other row by address.

Order differs between the mid-pipeline atom list this runs on and the finished
envelope, which is exactly why replaying the envelope "fixed" a bug that was
still live.
"""

from __future__ import annotations

from app.core.entity_resolution import collect_site_alias_groups


class _Atom:
    def __init__(self, keys, value, atom_type="physical_site"):
        self.entity_keys = list(keys)
        self.value = dict(value)
        self.atom_type = atom_type
        self.raw_text = ""
        self.normalized_text = ""
        self.id = f"atm_{abs(hash((tuple(keys), tuple(sorted(value.items())))))}"


#: Knows the key, knows nothing else. This is the shadowing atom.
NAME_ONLY = _Atom(["site:site_1"], {"id": "SITE-1", "name": "Site 1"})

#: The same place, with the address that makes it identifiable.
WITH_ADDRESS = _Atom(
    ["site:site_1"],
    {"id": "SITE-1", "name": "Site 1", "address": "2205 Gregg St",
     "city": "Columbia", "state": "SC", "zip": "29201"},
)

#: The roster row for the same building, under its own key.
ROSTER_ROW = _Atom(
    ["site:2205_gregg_st"],
    {"id": "2205-GREGG-ST", "address": "2205 Gregg St",
     "city": "Columbia", "state": "SC", "zip": "29201"},
)


def _grouped(atoms):
    return [sorted(g) for g in collect_site_alias_groups(atoms)]


def test_an_address_less_atom_cannot_shadow_the_one_that_knows_the_street():
    """The live ordering: the field-poor atom arrives first."""
    assert _grouped([NAME_ONLY, WITH_ADDRESS, ROSTER_ROW]) == [
        ["site:2205_gregg_st", "site:site_1"]
    ]


def test_the_other_order_gives_the_same_answer():
    """The replay ordering, which is how this bug stayed hidden."""
    assert _grouped([WITH_ADDRESS, NAME_ONLY, ROSTER_ROW]) == [
        ["site:2205_gregg_st", "site:site_1"]
    ]


def test_every_permutation_agrees():
    """Order independence is the property, not a lucky arrangement."""
    from itertools import permutations

    answers = {tuple(_grouped(list(p))[0]) for p in permutations([NAME_ONLY, WITH_ADDRESS, ROSTER_ROW])}
    assert answers == {("site:2205_gregg_st", "site:site_1")}


def test_a_key_no_atom_can_place_still_groups_with_nothing():
    """Filling never invents: a key nobody gave an address to stays alone."""
    lonely = _Atom(["site:mystery"], {"id": "MYSTERY", "name": "Mystery"})
    assert _grouped([lonely, ROSTER_ROW]) == []
