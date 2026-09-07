"""The join that carries geography from a site atom onto its roster row.

This passthrough silently matched nothing for most sites: it keyed atoms by
``slugify(value["id"])`` while ``build_site_readiness`` keys rows from
``entity_keys``. Measured on a 140-envelope corpus sample (2026-09-07), city
was present on 72% of ``physical_site`` atoms and on 4.7% of roster rows.
"""

from __future__ import annotations

from app.core.orbitbrief_envelope import apply_site_attributes


def _atom(entity_keys, value):
    return {"atom_type": "physical_site", "entity_keys": entity_keys, "value": value}


def test_geography_joins_on_entity_keys_not_the_atom_id() -> None:
    """The regression: id is a facility code, the row is keyed by address."""
    rows = [{"site": "site:address_2970_brandywine_rd_ste_200"}]
    atoms = [
        _atom(
            ["site:address_2970_brandywine_rd_ste_200"],
            {
                "id": "HQ-01",  # nothing like the row key
                "city": "Atlanta",
                "state": "GA",
                "zip": "30641",
                "address": "2970 Brandywine Rd",
            },
        )
    ]
    matched, unmatched = apply_site_attributes(rows, atoms)
    assert (matched, unmatched) == (1, 0)
    assert rows[0]["city"] == "Atlanta"
    assert rows[0]["state"] == "GA"
    assert rows[0]["zip"] == "30641"


def test_id_derived_slug_still_matches() -> None:
    """The original path keeps working when entity_keys are absent."""
    rows = [{"site": "site:hq_01"}]
    atoms = [_atom([], {"id": "HQ-01", "city": "Atlanta", "state": "GA"})]
    matched, unmatched = apply_site_attributes(rows, atoms)
    assert (matched, unmatched) == (1, 0)
    assert rows[0]["city"] == "Atlanta"


def test_existing_values_are_never_overwritten() -> None:
    rows = [{"site": "site:a", "city": "Decatur"}]
    atoms = [_atom(["site:a"], {"id": "a", "city": "Atlanta", "state": "GA"})]
    apply_site_attributes(rows, atoms)
    assert rows[0]["city"] == "Decatur"  # the row already knew
    assert rows[0]["state"] == "GA"      # the blank still fills


def test_unmatched_atoms_are_counted_not_swallowed() -> None:
    """A join that lands on nothing must say so."""
    rows = [{"site": "site:a"}]
    atoms = [_atom(["site:elsewhere"], {"id": "zzz", "city": "Atlanta"})]
    matched, unmatched = apply_site_attributes(rows, atoms)
    assert (matched, unmatched) == (0, 1)
    assert "city" not in rows[0]


def test_non_site_atoms_are_ignored() -> None:
    rows = [{"site": "site:a"}]
    atoms = [{"atom_type": "scope_item", "entity_keys": ["site:a"], "value": {"city": "Nowhere"}}]
    assert apply_site_attributes(rows, atoms) == (0, 0)
    assert "city" not in rows[0]


def test_names_extend_aliases_without_duplicating() -> None:
    rows = [{"site": "site:a", "aliases": ["HQ"]}]
    atoms = [_atom(["site:a"], {"id": "a", "names": ["HQ", "Headquarters"]})]
    apply_site_attributes(rows, atoms)
    assert rows[0]["aliases"] == ["HQ", "Headquarters"]
