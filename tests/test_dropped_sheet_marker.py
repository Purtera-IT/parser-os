"""Whole-sheet DROP → retained suppression marker (upgrade #5).

When the xlsx sheet-role router classifies a whole sheet DROP (empty / cover /
lookup-helper noise), the sheet must NOT vanish silently. The parser emits ONE
``dropped_sheet`` marker carrying the sheet's rows, stamped
``suppressed:sheet_router``; the compiler later diverts pre-suppressed parser
atoms into ``CompileResult.suppressed_atoms``. This keeps an omission complaint
("you missed the Lookup tab") localizable and auditable.

Hermetic: drives the parser's sheet router directly with an all-blank sheet
(guaranteed EMPTY → DROP) and a real scope sheet. No network, no workbook file.
"""

from __future__ import annotations

from app.core.schemas import ArtifactType, AtomType, ReviewStatus
from app.core.suppression_ledger import (
    SUPPRESSION_FLAG_PREFIX,
    capture_suppressed,
    merge_suppressed,
)
from app.parsers.xlsx_parser import XlsxParser


def _rows_for_drop():
    # Non-empty list, but every cell blank → SheetRole.EMPTY → DROP.
    return [[None, None, None], ["", "", ""]]


def _parse(rows, sheet_name="Lookup"):
    return XlsxParser()._parse_sheet_rows(
        project_id="p1",
        artifact_id="a1",
        filename="book.xlsx",
        artifact_type=ArtifactType.xlsx,
        sheet_name=sheet_name,
        rows=rows,
    )


def test_dropped_sheet_emits_suppressed_marker():
    atoms = _parse(_rows_for_drop(), sheet_name="Lookup")
    assert len(atoms) == 1
    m = atoms[0]
    assert m.atom_type is AtomType.dropped_sheet
    assert any(
        str(f).startswith(SUPPRESSION_FLAG_PREFIX) for f in m.review_flags
    )
    assert m.review_status is ReviewStatus.needs_review
    # Content retained for recoverability + structured suppression provenance.
    assert m.value["sheet_name"] == "Lookup"
    assert m.value["_suppression"]["stage"] == "sheet_router"
    assert "rows" in m.value
    # A suppressed marker must never carry trust.
    assert m.confidence == 0.0


def test_marker_diverts_cleanly_via_ledger():
    """The compiler's diversion is: pull atoms whose review_flags carry the
    suppression prefix into the sidecar, keep the rest. Mirror that here and
    assert the marker leaves the accepted set but survives in the ledger."""
    drop_marker = _parse(_rows_for_drop())[0]

    class _Atom:
        def __init__(self, aid, flags):
            self.id = aid
            self.review_flags = flags
            self.value = {}

    accepted = [_Atom("k1", []), drop_marker, _Atom("k2", ["low_confidence_floor"])]

    pre_suppressed = [
        a
        for a in accepted
        if any(
            str(f).startswith(SUPPRESSION_FLAG_PREFIX)
            for f in (getattr(a, "review_flags", None) or [])
        )
    ]
    ledger: list = []
    merge_suppressed(ledger, pre_suppressed)
    sup_ids = {str(getattr(a, "id", "")) for a in pre_suppressed}
    kept = [a for a in accepted if str(getattr(a, "id", "")) not in sup_ids]

    assert [a.id for a in kept] == ["k1", "k2"]  # marker removed from scope
    assert drop_marker in ledger  # but retained for audit


def test_scope_sheet_has_no_marker():
    """A real scope sheet routes SCOPE and emits content atoms — never a
    dropped_sheet marker."""
    rows = [
        ["Item", "Description", "Qty"],
        ["1", "Install 24-port switch in MDF closet at ATL-HQ", "12"],
        ["2", "Run Cat6A drops to each workstation per floor plan", "240"],
    ]
    atoms = _parse(rows, sheet_name="Scope")
    assert all(a.atom_type is not AtomType.dropped_sheet for a in atoms)
