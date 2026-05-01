from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.core.compiler import compile_project
from app.eval.coverage import build_coverage_report, build_segment_coverage_index


def _artifact_id(result, filename: str) -> str:
    assert result.manifest is not None
    for row in result.manifest.artifact_fingerprints:
        if row.filename == filename:
            return row.artifact_id
    raise AssertionError(f"Artifact not found for filename={filename}")


def test_demo_project_coverage_report_writes_json(demo_project: Path, tmp_path: Path) -> None:
    result = compile_project(demo_project, project_id="coverage_demo", allow_errors=True, allow_unverified_receipts=True)
    compile_json = tmp_path / "compile.json"
    compile_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "coverage.json"
    subprocess.run(
        [sys.executable, "scripts/evidence_coverage_report.py", "--compile-result", str(compile_json), "--out", str(out)],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["project_id"] == "coverage_demo"
    assert isinstance(payload["artifact_reports"], list)


def test_spreadsheet_total_row_segment_indexed(demo_project: Path) -> None:
    """TOTAL row appears in coverage index; xlsx v2 may cover it as aggregate governing quantities."""
    result = compile_project(demo_project, project_id="coverage_total", allow_errors=True, allow_unverified_receipts=True)
    payload = json.loads(result.model_dump_json())
    segment_index = build_segment_coverage_index(payload)
    site_artifact_id = _artifact_id(result, "site_list.xlsx")
    rows = segment_index.get(site_artifact_id, [])
    total_rows = [row for row in rows if "total" in row.text_preview.lower()]
    assert total_rows
    allowed = {"ignored", "covered", "partial", "unsupported"}
    assert all(row.coverage_status in allowed for row in total_rows)


def test_transcript_open_question_segment_covered(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="coverage_transcript", allow_errors=True, allow_unverified_receipts=True)
    payload = json.loads(result.model_dump_json())
    segment_index = build_segment_coverage_index(payload)
    transcript_id = _artifact_id(result, "kickoff_transcript.txt")
    transcript_rows = segment_index.get(transcript_id, [])
    question_rows = [row for row in transcript_rows if "open question" in row.text_preview.lower()]
    assert question_rows
    assert any(row.coverage_status == "covered" for row in question_rows)


def test_random_text_artifact_shows_ignored_or_unsupported_segments(demo_project: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(demo_project, project)
    (project / "random.txt").write_text(
        "CONFIDENTIAL DISCLAIMER ONLY\n\nThis is free-form random narrative with no extraction cues.",
        encoding="utf-8",
    )
    result = compile_project(project, project_id="coverage_random", allow_errors=True, allow_unverified_receipts=True)
    payload = json.loads(result.model_dump_json())
    report = build_coverage_report(payload)
    random_report = next(row for row in report.artifact_reports if row.filename == "random.txt")
    assert random_report.ignored_count + random_report.unsupported_count >= 1


def test_coverage_rates_bounded_zero_to_one(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="coverage_bounds", allow_errors=True, allow_unverified_receipts=True)
    payload = json.loads(result.model_dump_json())
    report = build_coverage_report(payload)
    assert 0.0 <= report.overall_coverage_rate <= 1.0
    assert all(0.0 <= row.coverage_rate <= 1.0 for row in report.artifact_reports)


def test_recommended_improvements_generated_for_low_coverage_artifact(demo_project: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(demo_project, project)
    (project / "random.txt").write_text(
        "This message contains generic boilerplate.\n\nBest regards,\nTeam\n\nNo structured extraction hints.",
        encoding="utf-8",
    )
    result = compile_project(project, project_id="coverage_reco", allow_errors=True, allow_unverified_receipts=True)
    payload = json.loads(result.model_dump_json())
    report = build_coverage_report(payload)
    assert report.recommended_parser_improvements
