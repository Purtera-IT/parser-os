from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.source_replay import replay_atom_receipts


def _artifact_map_from_result(project_dir: Path, result) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for atom in result.atoms:
        for source_ref in atom.source_refs:
            mapping[source_ref.artifact_id] = project_dir / source_ref.filename
    return mapping


def test_demo_project_atoms_all_have_receipts(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    assert result.atoms
    assert all(atom.receipts for atom in result.atoms)
    assert all(len(atom.receipts) >= len(atom.source_refs) for atom in result.atoms)


def test_spreadsheet_atom_receipt_verifies_row(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    target = next(
        atom
        for atom in result.atoms
        if atom.source_refs[0].filename == "site_list.xlsx"
        and atom.value.get("quantity") == 50
    )
    receipt = target.receipts[0]
    assert receipt.replay_status == "verified"
    assert receipt.extracted_snippet is not None
    assert "Site=Main Campus" in receipt.extracted_snippet
    assert "Quantity=50" in receipt.extracted_snippet


def test_transcript_atom_receipt_verifies_line_range(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    target = next(
        atom
        for atom in result.atoms
        if atom.source_refs[0].filename == "kickoff_transcript.txt"
        and "escort access after 5pm" in atom.normalized_text
    )
    receipt = target.receipts[0]
    assert receipt.replay_status == "verified"
    assert receipt.locator.get("line_start") == 2
    assert "escort access after 5pm" in (receipt.extracted_snippet or "").lower()


def test_email_atom_receipts_verify_quoted_and_current_locators(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    email_atoms = [atom for atom in result.atoms if atom.source_refs[0].filename == "customer_email.txt"]
    assert email_atoms
    quoted = [atom for atom in email_atoms if atom.source_refs[0].locator.get("quoted") is True]
    current = [atom for atom in email_atoms if atom.source_refs[0].locator.get("quoted") is False]
    assert quoted and current
    assert all(atom.receipts[0].replay_status == "verified" for atom in quoted)
    assert all(atom.receipts[0].replay_status == "verified" for atom in current)


def test_modified_artifact_after_compile_causes_replay_failure(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    target = next(
        atom
        for atom in result.atoms
        if atom.source_refs[0].filename == "kickoff_transcript.txt"
        and "west wing" in atom.normalized_text.lower()
        and atom.source_refs[0].locator.get("line_start") == 3
    )
    transcript = demo_project / "kickoff_transcript.txt"
    content = transcript.read_text(encoding="utf-8")
    content = content.replace("Please remove West Wing from scope for now.", "We discussed timeline only.")
    transcript.write_text(content, encoding="utf-8")

    replayed = replay_atom_receipts(target, _artifact_map_from_result(demo_project, result))
    assert replayed
    assert any(receipt.replay_status == "failed" for receipt in replayed)


def test_unsupported_docx_tracked_deletion_creates_warning_not_crash(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert any("unsupported receipt" in warning.lower() for warning in result.warnings)
    deleted_atoms = [atom for atom in result.atoms if "tracked_change_deleted_text" in atom.review_flags]
    assert deleted_atoms
    assert any(receipt.replay_status == "unsupported" for atom in deleted_atoms for receipt in atom.receipts)


def test_no_atom_silently_skips_receipt_creation(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    for atom in result.atoms:
        source_ref_ids = {source_ref.id for source_ref in atom.source_refs}
        receipt_ref_ids = {receipt.source_ref_id for receipt in atom.receipts}
        assert source_ref_ids.issubset(receipt_ref_ids)
