from __future__ import annotations

import shutil
from pathlib import Path

from openpyxl import load_workbook

from app.core.compiler import compile_project


def _copy_demo_to_tmp(demo_project: Path, tmp_path: Path) -> Path:
    target = tmp_path / "demo_copy"
    shutil.copytree(demo_project, target)
    return target


def test_same_project_replay_has_same_output_signature(demo_project: Path) -> None:
    first = compile_project(demo_project, project_id="demo_project")
    second = compile_project(demo_project, project_id="demo_project")
    assert first.manifest is not None and second.manifest is not None
    assert first.manifest.output_signature == second.manifest.output_signature


def test_modifying_fixture_value_changes_input_and_output_signature(demo_project: Path, tmp_path: Path) -> None:
    baseline = compile_project(demo_project, project_id="demo_project")
    project_copy = _copy_demo_to_tmp(demo_project, tmp_path)

    quote_path = project_copy / "vendor_quote.xlsx"
    workbook = load_workbook(quote_path)
    sheet = workbook.active
    sheet["C2"] = 73
    workbook.save(quote_path)

    changed = compile_project(project_copy, project_id="demo_project")
    assert baseline.manifest is not None and changed.manifest is not None
    assert baseline.manifest.input_signature != changed.manifest.input_signature
    assert baseline.manifest.output_signature != changed.manifest.output_signature


def test_transcript_whitespace_change_keeps_signatures_when_semantics_unchanged(
    demo_project: Path,
    tmp_path: Path,
) -> None:
    baseline = compile_project(demo_project, project_id="demo_project")
    project_copy = _copy_demo_to_tmp(demo_project, tmp_path)

    transcript = project_copy / "kickoff_transcript.txt"
    content = transcript.read_text(encoding="utf-8")
    # Add harmless whitespace noise; parser normalizes these forms.
    content = content.replace("Main Campus requires escort access after 5pm.", "  Main Campus requires escort access after 5pm.   ")
    transcript.write_text(content, encoding="utf-8")

    replay = compile_project(project_copy, project_id="demo_project")
    assert baseline.manifest is not None and replay.manifest is not None
    assert baseline.manifest.input_signature != replay.manifest.input_signature
    assert baseline.manifest.output_signature == replay.manifest.output_signature
