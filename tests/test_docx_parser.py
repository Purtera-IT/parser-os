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


def _emit(text: str, *, heading: bool = False):
    return DocxParser()._emit_atoms_for_text(
        project_id="p",
        artifact_id="a",
        filename="sow.docx",
        text=text,
        paragraph_index=0,
        table_index=None,
        row=None,
        cell=None,
        tracked_change=None,
        heading=heading,
    )


def test_overview_prose_without_scope_verb_is_captured_not_dropped() -> None:
    # The SOW overview sentence carries the deal's headline quantity but uses
    # no scope/install/exclude verb — the lexical classifier matches nothing.
    # It must be captured (fail open), not silently dropped.
    text = (
        "The customer requires onsite field services support to replace "
        "approximately 110 existing TVs and mounts across 23 dwellings."
    )
    atoms = _emit(text)
    assert len(atoms) == 1
    a = atoms[0]
    assert a.atom_type == AtomType.scope_item
    assert "prose_fallback_capture" in a.review_flags
    assert a.value.get("prose_fallback") is True
    assert "110" in a.raw_text


def test_short_heading_fragment_still_dropped() -> None:
    # A bare title / short label is not load-bearing prose and stays dropped,
    # so the fail-open rule does not flood the graph with headings.
    assert _emit("Project Overview", heading=True) == []
    assert _emit("Scope") == []


def test_matched_scope_prose_keeps_full_confidence() -> None:
    # A paragraph that DOES match a lexical pattern is unaffected by the
    # fallback path (full confidence, no fallback flag).
    atoms = _emit("Installation of IP cameras at the main campus is in scope.")
    assert atoms
    assert all("prose_fallback_capture" not in a.review_flags for a in atoms)
