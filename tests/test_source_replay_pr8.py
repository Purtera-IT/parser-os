"""Regression tests for PR8 source-replay improvements.

Two specific things to guard:

* ``_replay_norm`` strips Unicode combining marks so atoms whose
  ``normalized_text`` lost diacritics still verify against the raw
  source.
* ``_verify_spreadsheet_row`` uses the full-row fallback when the
  cited cells alone don't satisfy the atom — common when the parser
  cited Severity but the atom text was authored from Mitigation.
"""
from __future__ import annotations

from pathlib import Path

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.source_replay import (
    _replay_norm,
    _spreadsheet_full_row_text,
    _verify_spreadsheet_row,
)


def _atom(text: str, normalized: str | None = None) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm",
        project_id="P",
        artifact_id="A",
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=normalized or text.lower(),
        value={"text": text},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="src",
                artifact_id="A",
                artifact_type=ArtifactType.csv,
                filename="x.csv",
                locator={"sheet": "Sheet1", "row": 2, "columns": {"sev": "A"}},
                extraction_method="test",
                parser_version="test",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_replay_norm_strips_combining_marks():
    assert _replay_norm("café") == _replay_norm("cafe")
    assert _replay_norm("M\xa0Smith") == _replay_norm("M Smith")


def test_replay_norm_handles_none_and_empty():
    assert _replay_norm("") == ""
    assert _replay_norm(None) == ""  # type: ignore[arg-type]


def test_spreadsheet_full_row_text_csv(tmp_path: Path):
    p = tmp_path / "risk.csv"
    p.write_text(
        "Severity,Impact,Mitigation\n"
        "High,Outage,Add WAN failover at District Core\n",
        encoding="utf-8",
    )
    text = _spreadsheet_full_row_text(p, "Sheet1", 2)
    assert "High" in text
    assert "Outage" in text
    assert "Add WAN failover" in text


def test_spreadsheet_full_row_text_handles_missing_row(tmp_path: Path):
    p = tmp_path / "risk.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert _spreadsheet_full_row_text(p, "Sheet1", 99) == ""


def test_full_row_fallback_verifies_when_cited_cell_misses(tmp_path: Path):
    """Atom text was authored from the Mitigation column but the
    parser only cited Severity. Cell-only matching fails; full-row
    fallback succeeds and returns ``verified``."""
    p = tmp_path / "risk.csv"
    p.write_text(
        "Severity,Impact,Mitigation\n"
        "High,Outage,Add WAN failover at District Core\n",
        encoding="utf-8",
    )
    atom = _atom("Add WAN failover at District Core")
    # Override locator to cite ONLY column A (Severity).
    atom.source_refs[0].locator["columns"] = {"severity": "A"}
    receipt = _verify_spreadsheet_row(atom, atom.source_refs[0], p)
    assert receipt.replay_status == "verified"
    assert "full-row" in (receipt.reason or "").lower()
