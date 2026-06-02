"""v57 pre-LLM dedup: identical text emitted under the two generic
catch-all types (scope_item / entity) collapses to ONE atom before the
expensive per-atom LLM stages run.

Structured types (raw_table_row, bom_line, physical_site) and the
meaningful customer_instruction type are NOT folded — they legitimately
co-exist as distinct facets / classifications of the same source span.
"""

from __future__ import annotations

from app.core.entity_resolution import collapse_duplicate_atoms


class _Atom:
    def __init__(self, atom_type, text, confidence=0.8, artifact_id="art1"):
        self.atom_type = atom_type
        self.artifact_id = artifact_id
        self.raw_text = text
        self.normalized_text = text
        self.confidence = confidence


def _types(atoms):
    return sorted(a.atom_type for a in atoms)


def test_scope_item_and_entity_same_text_collapse():
    txt = "Install one 55-inch display in the main lobby."
    atoms = [
        _Atom("scope_item", txt, confidence=0.7),
        _Atom("entity", txt, confidence=0.9),
    ]
    out = collapse_duplicate_atoms(atoms)
    assert len(out) == 1
    # Higher-confidence copy survives (sorted desc before dedup).
    assert out[0].atom_type == "entity"


def test_customer_instruction_not_folded_into_generic():
    txt = "Please remove the West Wing from scope."
    atoms = [
        _Atom("scope_item", txt, confidence=0.9),
        _Atom("customer_instruction", txt, confidence=0.7),
    ]
    out = collapse_duplicate_atoms(atoms)
    # customer_instruction is meaningful — must survive alongside.
    assert len(out) == 2
    assert "customer_instruction" in _types(out)


def test_structured_facets_not_folded():
    txt = "Technician #1- TV Install | $98.00 | Per Hour | 55 | $5,390.00"
    atoms = [
        _Atom("raw_table_row", txt),
        _Atom("bom_line", txt),
    ]
    out = collapse_duplicate_atoms(atoms)
    # Distinct structured facets of one row — both survive.
    assert len(out) == 2


def test_different_text_generic_atoms_kept():
    atoms = [
        _Atom("scope_item", "Mount a projector in Room A."),
        _Atom("entity", "Install speakers in Room B."),
    ]
    out = collapse_duplicate_atoms(atoms)
    assert len(out) == 2
