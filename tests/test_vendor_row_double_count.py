"""The vendor total must count each quote ROW once.

The quote parser emits two quantity-bearing atoms per spreadsheet row — the
``vendor_line_item`` (the reconciliation node) and a redundant ``quantity``
atom kept for the legacy quantity_claim consumers. Both satisfy
``graph_builder._is_qty_node``, so the roster-vs-vendor material comparison
used to sum both and report exactly double the vendor quantity: an RJ45 line
of 68 reconciled as 136, and that doubled figure was rendered to PMs as
``vendor_primary_sum`` in the packet certificate.

These are the fast unit-level guards. The end-to-end proof on the real COPPER
pack lives in ``test_roster_vendor_quantity_edges.py`` (a full compile, minutes
long, skipped when the validation pack isn't on disk).
"""
from __future__ import annotations

from app.core.graph_builder import (
    _collapse_duplicate_row_nodes,
    _roster_vendor_material_totals,
)
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _qty_atom(
    atom_id: str,
    *,
    atom_type: AtomType,
    authority: AuthorityClass,
    quantity: float,
    normalized_item: str,
    source_row_key: str | None,
) -> EvidenceAtom:
    value: dict[str, object] = {
        "quantity": quantity,
        "normalized_item": normalized_item,
        "included": True,
        "inclusion_status": "included",
    }
    if source_row_key is not None:
        value["source_row_key"] = source_row_key
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=f"{normalized_item} {quantity}",
        normalized_text=f"{normalized_item} {quantity}".lower(),
        value=value,
        entity_keys=[],
        source_refs=[SourceRef(
            id=stable_id("src", atom_id),
            artifact_id="art_1",
            artifact_type=ArtifactType.xlsx,
            filename="quote.xlsx",
            locator={"sheet": "Quote", "row": 4},
            extraction_method="test",
            parser_version="test",
        )],
        receipts=[],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def _vendor_row_pair(row: int, qty: float, item: str) -> list[EvidenceAtom]:
    """One quote row as the parser really emits it: line + redundant quantity."""
    key = f"quote.xlsx:Quote:row_{row}"
    return [
        _qty_atom(f"atm_line_{row}", atom_type=AtomType.vendor_line_item,
                  authority=AuthorityClass.vendor_quote, quantity=qty,
                  normalized_item=item, source_row_key=key),
        _qty_atom(f"atm_qty_{row}", atom_type=AtomType.quantity,
                  authority=AuthorityClass.vendor_quote, quantity=qty,
                  normalized_item=item, source_row_key=key),
    ]


def test_collapse_keeps_the_line_over_the_redundant_quantity_atom():
    collapsed = _collapse_duplicate_row_nodes(_vendor_row_pair(4, 68, "rj45 terminations"))
    assert len(collapsed) == 1
    assert collapsed[0].atom_type is AtomType.vendor_line_item


def test_collapse_keeps_distinct_rows_apart():
    atoms = _vendor_row_pair(4, 68, "rj45 terminations") + _vendor_row_pair(5, 12, "rj45 terminations")
    collapsed = _collapse_duplicate_row_nodes(atoms)
    assert {a.id for a in collapsed} == {"atm_line_4", "atm_line_5"}


def test_collapse_never_merges_atoms_without_a_row_key():
    """Roster-side quantity atoms carry no source_row_key — leave them alone."""
    atoms = [
        _qty_atom("atm_r1", atom_type=AtomType.quantity,
                  authority=AuthorityClass.approved_site_roster, quantity=40,
                  normalized_item="rj45 terminations", source_row_key=None),
        _qty_atom("atm_r2", atom_type=AtomType.quantity,
                  authority=AuthorityClass.approved_site_roster, quantity=32,
                  normalized_item="rj45 terminations", source_row_key=None),
    ]
    assert len(_collapse_duplicate_row_nodes(atoms)) == 2


def test_vendor_total_is_not_doubled():
    roster = [_qty_atom("atm_roster", atom_type=AtomType.quantity,
                        authority=AuthorityClass.approved_site_roster, quantity=72,
                        normalized_item="rj45 terminations", source_row_key=None)]
    roster[0].value["aggregate"] = True
    atoms = roster + _vendor_row_pair(4, 68, "rj45 terminations")
    anchor, roster_total, primary, vendor_total, _excluded = (
        _roster_vendor_material_totals(atoms, "rj45")
    )
    assert anchor is not None
    assert roster_total == 72.0
    assert vendor_total == 68.0, "68 counted twice would give 136"
    assert len(primary) == 1
    assert primary[0].atom_type is AtomType.vendor_line_item


def test_multi_line_vendor_total_sums_each_row_once():
    roster = [_qty_atom("atm_roster", atom_type=AtomType.quantity,
                        authority=AuthorityClass.approved_site_roster, quantity=100,
                        normalized_item="rj45 terminations", source_row_key=None)]
    roster[0].value["aggregate"] = True
    atoms = (roster
             + _vendor_row_pair(4, 68, "rj45 terminations")
             + _vendor_row_pair(9, 12, "rj45 terminations"))
    _anchor, _rt, primary, vendor_total, _exc = _roster_vendor_material_totals(atoms, "rj45")
    assert vendor_total == 80.0
    assert len(primary) == 2
