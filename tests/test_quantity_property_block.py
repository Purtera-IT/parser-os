"""A labelled quantity row is read by shape, kept per site, and kept from the LLM."""
from __future__ import annotations
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus
from app.core.semantic_dedup import dedupe_stakeholder_atoms, semantic_dedup_atoms
from app.core.site_provenance_join import join_atoms_to_document_site
from app.parsers.quantity_property_block import quantities_from_property_row, quantity_from_cell

def _row(*c): return {str(i): v for i, v in enumerate(c)}

def test_the_real_row_reads_one_clock():
    q = quantities_from_property_row(_row("Hardware to be Installed (By Category)", "1 UKG DX Clock", "1 UKG DX Clock", "1 UKG DX Clock"))
    assert q == [{"kind": "bom_line", "quantity": 1, "item": "UKG DX Clock", "description": "1 UKG DX Clock",
                  "label": "Hardware to be Installed (By Category)"}]

def test_shape_refuses_durations_addresses_codes_dates():
    for bad in ("1 hr 28 min", "601 Gurley St", "94575001", "8/19/26", "843-423-8335 x3069", "30 600 East Northside Avenue 76"):
        assert quantity_from_cell(bad) is None, bad

def test_a_bare_number_without_a_label_is_not_taken():
    assert quantities_from_property_row(_row("", "1 UKG DX Clock")) == []
    assert quantities_from_property_row(_row("2 Routers", "1 UKG DX Clock")) == []  # label cell is itself a quantity

def _atom(aid, atype, value, keys=()):
    return EvidenceAtom(id=f"atm_{aid}_{atype.value}", project_id="p", artifact_id=aid, atom_type=atype,
        raw_text=str(value.get("description") or value.get("name") or aid), normalized_text="x", value=value,
        authority_class=AuthorityClass.contractual_scope, confidence=0.9, review_status=ReviewStatus.auto_accepted,
        entity_keys=list(keys), parser_version="t")

def test_ten_identical_clock_rows_keep_ten_site_keys():
    atoms = []
    for s in ("johnakin", "palmetto", "marion_high"):
        atoms.append(_atom(f"d_{s}", AtomType.physical_site, {"kind": "physical_site", "id": s, "site_id": s, "name": s}, [f"site:{s}"]))
        atoms.append(_atom(f"d_{s}", AtomType.bom_line, {"kind": "bom_line", "quantity": 1, "item": "UKG DX Clock", "description": "1 UKG DX Clock"}))
    atoms = semantic_dedup_atoms(atoms); join_atoms_to_document_site(atoms); atoms = dedupe_stakeholder_atoms(atoms)
    bom = [a for a in atoms if a.atom_type == AtomType.bom_line]
    assert len(bom) == 1 and {k for k in bom[0].entity_keys if k.startswith("site:")} == {"site:johnakin", "site:palmetto", "site:marion_high"}

def test_a_quantity_row_is_deflected_from_the_llm(monkeypatch):
    import app.core.typed_atom_classifier as tac
    from tests.test_typed_classifier_contact_deflect import _patched, _Atom
    seen = _patched(monkeypatch, tac)
    a = _Atom(AtomType.scope_item, {"kind": "table_row", "cells": {"R": "Hardware to be Installed (By Category)", "R__1": "1 UKG DX Clock", "R__2": "1 UKG DX Clock", "R__3": "1 UKG DX Clock"}})
    tac.classify_atoms([a]); assert sum(len(b) for b in seen) == 0
