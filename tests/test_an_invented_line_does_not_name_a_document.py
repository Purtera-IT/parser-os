"""A line the parser MADE may not say what a document is about.

010452's SOW resolved to `site:floor_east_brooklyn_ny_11245` -- a JPMorgan lockbox letterhead the
geo fallback minted -- and `join_atoms_to_document_site` stamped that key onto 154 of its atoms,
"Date: 4/6/2026" and "2.0 | Scope of Work" included, with the rationale "the document this atom
came from resolves to exactly one physical site". The job's own Green Bay address reached 4 atoms.
A Green Bay job was relabelled, document-wide, as an address in Brooklyn.

Measured over four deals, this changes exactly that one document: 010171 has no single-site
document, and 010447 and 010358 keep theirs unchanged.
"""
import types

from app.core.site_provenance_join import document_site_map, join_atoms_to_document_site


def _atom(aid, atom_type, keys, art="art_1", flags=(), text="x"):
    return types.SimpleNamespace(id=aid, atom_type=atom_type, entity_keys=list(keys),
                                 review_flags=list(flags), artifact_id=art, raw_text=text, text=text,
                                 value=None, decision_provenance=None)


def test_a_minted_letterhead_does_not_name_the_document():
    made = _atom("a1", "physical_site", ["site:floor_east_brooklyn_ny_11245"], flags=["geo_fallback_site"])
    row = _atom("a2", "scope_item", [])
    assert document_site_map([made, row]) == {}
    assert join_atoms_to_document_site([made, row]) == 0
    assert row.entity_keys == []


def test_a_real_site_line_still_names_its_document():
    real = _atom("a1", "physical_site", ["site:akron_oh_44301"])
    row = _atom("a2", "scope_item", [])
    assert document_site_map([real, row]) == {"art_1": "site:akron_oh_44301"}
    assert join_atoms_to_document_site([real, row]) == 1
    assert "site:akron_oh_44301" in row.entity_keys
    assert row.decision_provenance["source"] == "document_scope"


def test_a_real_line_beside_a_minted_one_still_speaks_for_the_document():
    real = _atom("a1", "physical_site", ["site:akron_oh_44301"])
    made = _atom("a2", "physical_site", ["site:floor_east_brooklyn_ny_11245"], flags=["geo_fallback_site"])
    row = _atom("a3", "scope_item", [])
    assert document_site_map([real, made, row]) == {"art_1": "site:akron_oh_44301"}


def test_two_real_sites_still_make_the_document_ambiguous():
    a = _atom("a1", "physical_site", ["site:akron_oh_44301"])
    b = _atom("a2", "physical_site", ["site:canton_oh_44702"])
    assert document_site_map([a, b]) == {}


def test_a_geo_mention_site_does_not_name_a_document_either():
    made = _atom("a1", "physical_site", ["site:somewhere"], flags=["geo_mention_site"])
    assert document_site_map([made, _atom("a2", "scope_item", [])]) == {}
