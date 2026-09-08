"""One site, one key.

`build_site_readiness` re-derived a site key from `value["id"]` instead of
using the atom's own entity key. When the two normalisers disagreed, the row
was keyed under something nothing else in the system knew:

    atom graph / entities / joins : site:symphony_ai_hillview_office
    site_readiness               : site:symphonyai_hillview_office

An orphaned row has no address (the attribute passthrough joins on
entity_keys), no anchor, and cannot be merged (the fusion pass works on the
atom graph). Live deal 010302 showed all three at once.
"""

from __future__ import annotations

from app.core.orbitbrief_core import build_site_readiness


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


def _keys(atoms):
    out = build_site_readiness(atoms=atoms, edges=[])
    return {r["site"] for r in out["sites"]}


def test_the_atoms_own_key_wins_over_a_re_derived_slug() -> None:
    atom = _Atom(
        ["site:symphony_ai_hillview_office"],
        {"id": "SymphonyAI-Hillview-Office", "facility_name": "SymphonyAI Hillview Office"},
    )
    assert _keys([atom]) == {"site:symphony_ai_hillview_office"}


def test_an_atom_with_no_site_key_still_gets_a_row() -> None:
    """The id-derived slug remains the fallback, so nothing that used to
    appear disappears."""
    atom = _Atom([], {"id": "ATL-HQ-01", "facility_name": "Atlanta HQ"})
    assert _keys([atom]) == {"site:atl_hq_01"}


def test_an_atom_with_no_site_id_is_still_skipped() -> None:
    """The ghost-site guard is unchanged: an atom with no explicit code would
    otherwise be keyed off an address string."""
    atom = _Atom(["site:address_1180_peachtree_st"], {"facility_name": "Somewhere"})
    assert _keys([atom]) == set()


def test_two_atoms_of_one_site_produce_one_row() -> None:
    """The point: agreeing on the key is what stops one place being two."""
    a = _Atom(["site:symphony_ai_hillview_office"], {"id": "SymphonyAI-Hillview-Office"})
    b = _Atom(
        ["site:symphony_ai_hillview_office"],
        {"id": "SYMPHONYAI HILLVIEW OFFICE", "city": "Palo Alto", "state": "CA"},
    )
    assert _keys([a, b]) == {"site:symphony_ai_hillview_office"}
