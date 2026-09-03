"""A per-site document speaks for its own rows -- and only if it is unambiguous."""

from __future__ import annotations

from app.core.site_provenance_join import (
    document_site_map,
    join_atoms_to_document_site,
)


class _Atom:
    def __init__(self, artifact_id, atom_type, entity_keys=None):
        self.artifact_id = artifact_id
        self.atom_type = atom_type
        self.entity_keys = list(entity_keys or [])
        self.decision_provenance = None


def _sow(art, school):
    """One per-site SOW: the site row, plus rows that never repeat the name."""
    return [
        _Atom(art, "physical_site", [f"site:{school}"]),
        _Atom(art, "site_access_window", ["stakeholder:smarthands_statement"]),
        _Atom(art, "scope_item"),
        _Atom(art, "contract_term"),
    ]


def test_the_rows_of_a_per_site_sow_get_that_site():
    atoms = _sow("art_johnakin", "johnakin_middle_school")
    assert join_atoms_to_document_site(atoms) == 3
    for a in atoms:
        assert "site:johnakin_middle_school" in a.entity_keys


def test_the_link_records_that_it_came_from_the_document():
    atoms = _sow("art_johnakin", "johnakin_middle_school")
    join_atoms_to_document_site(atoms)
    window = atoms[1]
    assert window.decision_provenance["source"] == "document_scope"
    # The site atom itself was not touched -- it asserted its own identity.
    assert atoms[0].decision_provenance is None


def test_a_multi_site_document_propagates_nothing():
    """The guard that carries the weight: a locations list must not smear."""
    art = "art_locations_list"
    atoms = [
        _Atom(art, "physical_site", ["site:johnakin_middle_school"]),
        _Atom(art, "physical_site", ["site:palmetto_middle_school"]),
        _Atom(art, "scope_item"),
    ]
    assert document_site_map(atoms) == {}
    assert join_atoms_to_document_site(atoms) == 0
    assert atoms[2].entity_keys == []


def test_a_document_with_no_site_propagates_nothing():
    atoms = [_Atom("art_msa", "contract_term"), _Atom("art_msa", "scope_item")]
    assert join_atoms_to_document_site(atoms) == 0


def test_an_atom_that_names_its_own_site_is_left_alone():
    """A direct assertion outranks provenance."""
    art = "art_johnakin"
    atoms = [
        _Atom(art, "physical_site", ["site:johnakin_middle_school"]),
        _Atom(art, "scope_item", ["site:palmetto_middle_school"]),
    ]
    join_atoms_to_document_site(atoms)
    assert atoms[1].entity_keys == ["site:palmetto_middle_school"]
    assert atoms[1].decision_provenance is None


def test_ten_per_site_sows_stay_ten_distinct_sites():
    """The template rows are identical across SOWs; provenance keeps them apart."""
    atoms = []
    for s in ("johnakin", "palmetto", "marion_high", "mullins_high"):
        atoms += _sow(f"art_{s}", s)
    join_atoms_to_document_site(atoms)
    windows = [a for a in atoms if a.atom_type == "site_access_window"]
    assert len({k for w in windows for k in w.entity_keys if k.startswith("site:")}) == 4
