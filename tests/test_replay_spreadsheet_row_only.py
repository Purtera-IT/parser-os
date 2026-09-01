"""A spreadsheet atom that cites a row but no cells is still verifiable.

_verify_spreadsheet_row was only reached when the locator carried `columns`, so
an atom citing just a row fell past every branch to "No verifier available".

On deal 010215 that was 89 atoms, and not one atom from the deal's own Deal Kit
was verified (0 of 55), nor from the Sodexo Breakdown (0 of 8). Those are the
two documents carrying the money -- pricing, totals and service lines going
completely unchecked while reading as clean.
"""
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.source_replay import replay_source_ref


def _book(tmp_path: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Assumptions"
    ws["A1"] = "Assumption"
    ws["A2"] = "Union labor is not in scope."
    ws["B2"] = "standard business hours"
    p = tmp_path / "kit.xlsx"
    wb.save(p)
    return p


def _ref(locator: dict) -> SourceRef:
    return SourceRef(
        id="src_1", artifact_id="art_1", artifact_type=ArtifactType.xlsx,
        filename="kit.xlsx", locator=locator, extraction_method="xlsx", parser_version="t",
    )


def _atom(text: str, ref: SourceRef) -> EvidenceAtom:
    # The schema requires at least one SourceRef, which is the right invariant:
    # an atom with no provenance is exactly what receipts exist to prevent.
    return EvidenceAtom(
        id="atm_1", project_id="p", artifact_id="art_1",
        atom_type=AtomType.pricing_assumption, raw_text=text, normalized_text=text.lower(),
        value={}, entity_keys=[], source_refs=[ref],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.9, review_status=ReviewStatus.auto_accepted, parser_version="t",
    )


class TestRowOnlyLocator:
    def test_a_row_without_cited_columns_now_verifies(self, tmp_path):
        p = _book(tmp_path)
        ref = _ref({"sheet": "Assumptions", "row": 2, "extraction": "xlsx_block_row_v1"})
        r = replay_source_ref(_atom("Union labor is not in scope.", ref), ref, {"art_1": p})
        assert r.replay_status == "verified"

    def test_a_singular_col_is_treated_as_a_citation(self, tmp_path):
        p = _book(tmp_path)
        ref = _ref({"sheet": "Assumptions", "row": 2, "col": "A"})
        r = replay_source_ref(_atom("Union labor is not in scope.", ref), ref, {"art_1": p})
        assert r.replay_status == "verified"

    def test_a_row_that_does_not_say_it_still_fails(self, tmp_path):
        # Widening the route must not turn every atom into a pass.
        p = _book(tmp_path)
        ref = _ref({"sheet": "Assumptions", "row": 2})
        r = replay_source_ref(_atom("Weekend premium applies to all sites.", ref), ref, {"art_1": p})
        assert r.replay_status == "failed"

    def test_an_out_of_range_row_still_fails(self, tmp_path):
        p = _book(tmp_path)
        ref = _ref({"sheet": "Assumptions", "row": 999})
        r = replay_source_ref(_atom("Union labor is not in scope.", ref), ref, {"art_1": p})
        assert r.replay_status == "failed"
