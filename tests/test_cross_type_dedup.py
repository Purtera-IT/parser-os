"""Cross-type dedup: one source sentence, emitted under many atom types,
collapses to the single most-specific type.

Grounded in the real Yonah deal, where the DOCX scope doc emitted
"Technician #1- TV Install | $98.00 | Per Hour | 55 | $5,390.00" as a
scope_item AND a task AND a service_line AND a raw_table_row, and
"ESTIMATED TOTAL FEES ... $21,560.00" three times. Those four/three
copies inflated scope_truth and every scorecard. semantic_dedup keys
*with* atom_type so it can't fold them; cross_type_dedup does.
"""

from __future__ import annotations

from app.core.semantic_dedup import cross_type_dedup_atoms


class _Atom:
    def __init__(self, atom_type, text, confidence=0.8):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.confidence = confidence
        self.source_refs = [f"src::{atom_type}"]
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def _types(atoms):
    return sorted(a.atom_type for a in atoms)


# ── the real technician line: 4 types → 1 service_line ──────────────


def test_technician_line_collapses_to_service_line() -> None:
    txt = "Technician #1- TV Install | $98.00 | Per Hour | 55 | $5,390.00"
    atoms = [
        _Atom("scope_item", txt),
        _Atom("task", txt),
        _Atom("service_line", txt),
        _Atom("raw_table_row", txt),
    ]
    out = cross_type_dedup_atoms(atoms)
    assert len(out) == 1
    assert out[0].atom_type == "service_line"
    # All four source_refs preserved on the survivor.
    assert len(out[0].source_refs) == 4


# ── totals line: differing money tokens still collapse ──────────────


def test_estimated_total_collapses_despite_money_tokens() -> None:
    atoms = [
        _Atom("raw_table_row", "ESTIMATED TOTAL FEES | $21,560.00"),
        _Atom("scope_item", "ESTIMATED TOTAL FEES | 21560"),
        _Atom("service_line", "ESTIMATED TOTAL FEES"),
    ]
    out = cross_type_dedup_atoms(atoms)
    assert len(out) == 1
    assert out[0].atom_type == "service_line"


# ── access constraint: requirement + constraint → constraint ────────


def test_access_requirement_and_constraint_collapse_to_constraint() -> None:
    txt = "Provide access to all 23 dwellings and all installation locations."
    atoms = [_Atom("requirement", txt), _Atom("constraint", txt)]
    out = cross_type_dedup_atoms(atoms)
    assert len(out) == 1
    assert out[0].atom_type == "constraint"


# ── raw_table_row always loses to any typed atom ────────────────────


def test_raw_table_row_loses_to_typed() -> None:
    txt = "Some bom item description here"
    out = cross_type_dedup_atoms(
        [_Atom("raw_table_row", txt), _Atom("bom_line", txt)]
    )
    assert len(out) == 1 and out[0].atom_type == "bom_line"


# ── distinct facts are NOT merged ───────────────────────────────────


def test_distinct_text_not_merged() -> None:
    atoms = [
        _Atom("task", "Remove the existing TV and mount from the location."),
        _Atom("task", "Clean each work area upon completion."),
        _Atom("requirement", "Nearby power is available at each location."),
    ]
    out = cross_type_dedup_atoms(atoms)
    assert len(out) == 3


# ── same type, same text is left for semantic_dedup (untouched here) ─


def test_same_type_group_passes_through() -> None:
    txt = "Install each new display using the provided method."
    atoms = [_Atom("task", txt), _Atom("task", txt)]
    out = cross_type_dedup_atoms(atoms)
    # Cross-type pass only folds across DIFFERENT types; same-type dupes
    # are semantic_dedup's responsibility.
    assert len(out) == 2


# ── ties broken by confidence within equal priority ─────────────────


def test_tie_broken_by_confidence() -> None:
    txt = "Mounting bracket assembly for wall display"
    lo = _Atom("bom_line", txt, confidence=0.5)
    hi = _Atom("service_line", txt, confidence=0.9)
    out = cross_type_dedup_atoms([lo, hi])
    assert len(out) == 1 and out[0] is hi


# ── short text is left alone (too little signal to key safely) ──────


def test_short_text_passthrough() -> None:
    atoms = [_Atom("scope_item", "TVs"), _Atom("task", "TVs")]
    out = cross_type_dedup_atoms(atoms)
    assert len(out) == 2
