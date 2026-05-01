from __future__ import annotations

from pathlib import Path

from app.testing.mutators import write_docx_fixture
from app.parsers.docx_parser import DocxParser


def test_docx_adversarial_cases(tmp_path: Path) -> None:
    path = tmp_path / "sow.docx"
    write_docx_fixture(
        path,
        included_site="Main Campus",
        excluded_site="West Wing",
        scoped_device="IP Camera",
        mutation="scope_in_table",
    )
    atoms = DocxParser().parse_artifact("proj", "art", path)
    assert atoms
    assert any(atom.atom_type.value == "scope_item" for atom in atoms)
    assert any(atom.atom_type.value == "exclusion" for atom in atoms)
    deleted = [atom for atom in atoms if "tracked_change_deleted_text" in atom.review_flags]
    assert deleted
    assert all(atom.authority_class.value == "deleted_text" for atom in deleted)
    assert all(atom.review_status.value == "rejected" for atom in deleted)
