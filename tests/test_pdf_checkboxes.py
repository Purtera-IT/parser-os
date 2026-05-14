"""Regression tests for PR7: PDF checkbox / workflow / visual atoms.

These exercise the pure-text helpers directly (no fitz needed) so the
test runs fast and stable. The integration with parse_artifact is
covered by existing PDF parser tests; here we just guarantee the
extraction logic is correct.
"""
from __future__ import annotations

from app.core.schemas import AtomType, ReviewStatus
from app.parsers.orbitbrief_pdf import (
    _checkbox_atoms_from_text,
    _workflow_atoms_from_text,
    _visual_review_atom,
)


def _kw():
    return {
        "project_id": "P",
        "artifact_id": "A",
        "filename": "f.pdf",
        "parser_version": "test",
    }


def test_checked_checkboxes_become_scope_items():
    text = "☒ Provide WAN failover\n☑ Quote includes 24x7 NOC\n[X] After-hours dispatch"
    atoms = _checkbox_atoms_from_text(page_number=2, text=text, **_kw())
    assert len(atoms) == 3
    assert all(a.atom_type == AtomType.scope_item for a in atoms)
    assert all(a.value["checked"] is True for a in atoms)
    assert all(a.review_status == ReviewStatus.auto_accepted for a in atoms)


def test_unchecked_checkboxes_become_exclusions_with_review_flag():
    text = "☐ 8x5 service desk\n[ ] Optional after-hours\n( ) Microsoft Sentinel"
    atoms = _checkbox_atoms_from_text(page_number=4, text=text, **_kw())
    assert len(atoms) == 3
    assert all(a.atom_type == AtomType.exclusion for a in atoms)
    assert all(a.value["checked"] is False for a in atoms)
    assert all(
        "unchecked_checkbox_not_scope" in a.review_flags for a in atoms
    )
    assert all(a.review_status == ReviewStatus.needs_review for a in atoms)


def test_workflow_atoms_emit_when_three_or_more_workflow_steps():
    text = "Detect → Triage → Contain → Recover → Notify"
    atoms = _workflow_atoms_from_text(page_number=1, text=text, **_kw())
    assert len(atoms) == 5
    assert all(a.atom_type == AtomType.action_item for a in atoms)
    assert {a.raw_text for a in atoms} == {
        "Detect",
        "Triage",
        "Contain",
        "Recover",
        "Notify",
    }


def test_workflow_atoms_skip_when_under_three_workflow_steps():
    text = "Detect → Triage"
    atoms = _workflow_atoms_from_text(page_number=1, text=text, **_kw())
    assert atoms == []


def test_visual_review_atom_marks_review_needed():
    atom = _visual_review_atom(
        page_number=7,
        reason="low_text_page_42_chars",
        **_kw(),
    )
    assert atom.atom_type == AtomType.open_question
    assert "visual_evidence_not_fully_extracted" in atom.review_flags
    assert atom.review_status == ReviewStatus.needs_review
    assert atom.value["page"] == 7
    assert atom.value["reason"] == "low_text_page_42_chars"
