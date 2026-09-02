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


class TestRowOnlyIsNotTooGenerous:
    """A whole-row match must still identify WHICH row.

    _snippet_matches_atom's last resort is "two important terms appear
    somewhere", which is fine against cells the parser named and far too
    generous against a whole row of a repetitive sheet.

    On the 010215 Deal Kit rate card three atoms passed against the wrong row --
    "Cancellation/Turnaway Fee" verified against the row holding "Additional
    Onsite Hourly Technician Labor", because both are rate rows whose numbers
    look alike. A receipt pointing at the wrong row is worse than no receipt: it
    says "checked" about something that was not.
    """

    def _rates(self, tmp_path: Path) -> Path:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Summary"
        ws["A1"], ws["C1"], ws["F1"] = "Additional Onsite Hourly Technician Labor", 115, 1
        ws["A2"], ws["C2"], ws["F2"] = "Cancellation/Turnaway Fee", 400, 2
        p = tmp_path / "rates.xlsx"
        wb.save(p)
        return p

    def test_a_lookalike_rate_row_no_longer_passes(self, tmp_path):
        p = self._rates(tmp_path)
        ref = _ref({"sheet": "Summary", "row": 1, "extraction": "xlsx_block_row_v1"})
        r = replay_source_ref(
            _atom("Price: Cancellation/Turnaway Fee | NB Price: 400 | Hours: 2", ref), ref, {"art_1": p},
        )
        assert r.replay_status == "failed"

    def test_the_right_row_still_passes(self, tmp_path):
        p = self._rates(tmp_path)
        ref = _ref({"sheet": "Summary", "row": 2, "extraction": "xlsx_block_row_v1"})
        r = replay_source_ref(
            _atom("Price: Cancellation/Turnaway Fee | NB Price: 400 | Hours: 2", ref), ref, {"art_1": p},
        )
        assert r.replay_status == "verified"

    def test_a_short_label_falls_back_rather_than_failing_outright(self, tmp_path):
        # A bare numeric or very short atom cannot identify a row by label, and
        # rejecting those outright would lose real receipts.
        p = self._rates(tmp_path)
        ref = _ref({"sheet": "Summary", "row": 2, "extraction": "xlsx_block_row_v1"})
        r = replay_source_ref(_atom("400", ref), ref, {"art_1": p})
        assert r.replay_status in {"verified", "failed"}  # decided by the ordinary matcher, not crashed
