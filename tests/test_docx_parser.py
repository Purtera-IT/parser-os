from __future__ import annotations

from pathlib import Path

from app.core.schemas import AtomType, AuthorityClass, ReviewStatus
from app.parsers.docx_parser import DocxParser
from scripts.make_demo_fixtures import create_demo_project


def test_docx_parser_with_tracked_deletion(tmp_path: Path) -> None:
    base_root = tmp_path / "repo"
    base_root.mkdir(parents=True, exist_ok=True)
    project_dir = create_demo_project(base_root)
    file_path = project_dir / "sow_draft.docx"

    atoms = DocxParser().parse_artifact(
        project_id="proj_1",
        artifact_id="art_docx_1",
        path=file_path,
    )

    assert atoms
    assert all(atom.source_refs for atom in atoms)

    scope_atoms = [a for a in atoms if a.atom_type == AtomType.scope_item]
    assert any("scope includes installation of ip cameras" in a.normalized_text for a in scope_atoms)

    exclusion_atoms = [a for a in atoms if a.atom_type == AtomType.exclusion]
    assert any("av displays are excluded from scope" in a.normalized_text for a in exclusion_atoms)

    constraint_atoms = [a for a in atoms if a.atom_type == AtomType.constraint]
    assert any("customer is responsible for providing lift access" in a.normalized_text for a in constraint_atoms)

    deleted_atoms = [a for a in atoms if "install av displays in conference rooms" in a.normalized_text]
    assert deleted_atoms
    assert all(a.authority_class == AuthorityClass.deleted_text for a in deleted_atoms)
    assert all(a.review_status == ReviewStatus.rejected for a in deleted_atoms)
    assert all("tracked_change_deleted_text" in a.review_flags for a in deleted_atoms)
    assert all(a.source_refs[0].locator.get("tracked_change") == "deleted" for a in deleted_atoms)
    assert all(a.authority_class == AuthorityClass.deleted_text for a in deleted_atoms)
