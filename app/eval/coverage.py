from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.segments import ArtifactSegment
from app.parsers.segmenters import (
    segment_docx,
    segment_email,
    segment_quote,
    segment_text,
    segment_transcript,
    segment_xlsx,
)

CoverageStatus = Literal["covered", "candidate_rejected", "ignored", "unsupported"]

LOCATOR_KEYS = {
    "line_start",
    "line_end",
    "sheet",
    "row",
    "message_index",
    "paragraph_index",
    "table_index",
    "change_index",
    "block_index",
    "utterance_index",
}


class SegmentCoverage(BaseModel):
    segment_id: str
    artifact_id: str
    segment_type: str
    text_preview: str
    has_candidate: bool
    has_accepted_atom: bool
    atom_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    coverage_status: CoverageStatus
    reason: str | None = None


class ArtifactCoverageReport(BaseModel):
    artifact_id: str
    filename: str
    segment_count: int
    covered_count: int
    candidate_rejected_count: int
    ignored_count: int
    unsupported_count: int
    coverage_rate: float = Field(ge=0.0, le=1.0)
    top_ignored_segments: list[SegmentCoverage] = Field(default_factory=list)


class ProjectCoverageReport(BaseModel):
    project_id: str
    artifact_reports: list[ArtifactCoverageReport] = Field(default_factory=list)
    overall_coverage_rate: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    recommended_parser_improvements: list[str] = Field(default_factory=list)


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _is_boilerplate(text: str, segment_type: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return True
    boilerplate_tokens = [
        "sent from my iphone",
        "confidential",
        "privileged and confidential",
        "do not print this email",
        "best regards",
        "thanks,",
        "kind regards",
    ]
    if any(token in lowered for token in boilerplate_tokens):
        return True
    if segment_type == "spreadsheet_row" and any(token in lowered for token in ("total", "subtotal", "grand total")):
        return True
    return False


def _coerce_segment(row: dict[str, Any]) -> ArtifactSegment | None:
    try:
        return ArtifactSegment.model_validate(row)
    except Exception:
        return None


def _locator_matches(source_locator: dict[str, Any], segment_locator: dict[str, Any]) -> bool:
    if not isinstance(source_locator, dict) or not isinstance(segment_locator, dict):
        return False

    for key in LOCATOR_KEYS:
        if key in source_locator and key in segment_locator and source_locator[key] != segment_locator[key]:
            return False

    s_start = source_locator.get("line_start")
    s_end = source_locator.get("line_end")
    g_start = segment_locator.get("line_start")
    g_end = segment_locator.get("line_end")
    if isinstance(s_start, int) and isinstance(g_start, int):
        s_end = s_end if isinstance(s_end, int) else s_start
        g_end = g_end if isinstance(g_end, int) else g_start
        if s_end < g_start or g_end < s_start:
            return False

    source_subset = {k: source_locator[k] for k in source_locator if k in LOCATOR_KEYS}
    segment_subset = {k: segment_locator[k] for k in segment_locator if k in LOCATOR_KEYS}
    if source_subset and segment_subset:
        common = set(source_subset).intersection(segment_subset)
        if not common:
            return False
    return True


def _generate_segments_for_artifact(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
) -> list[ArtifactSegment]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".csv"}:
        if any(token in name for token in ("quote", "po", "vendor")):
            return segment_quote(project_id=project_id, artifact_id=artifact_id, path=path)
        return segment_xlsx(project_id=project_id, artifact_id=artifact_id, path=path)
    if suffix == ".docx":
        return segment_docx(project_id=project_id, artifact_id=artifact_id, path=path)
    if suffix == ".eml":
        return segment_email(project_id=project_id, artifact_id=artifact_id, path=path)
    if suffix in {".txt", ".md", ".vtt", ".srt"}:
        if "transcript" in name or "kickoff" in name or "meeting" in name:
            return segment_transcript(project_id=project_id, artifact_id=artifact_id, path=path)
        if "email" in name:
            return segment_email(project_id=project_id, artifact_id=artifact_id, path=path)
        return segment_text(project_id=project_id, artifact_id=artifact_id, path=path)
    return []


def _artifact_rows(compile_payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = compile_payload.get("manifest")
    if isinstance(manifest, dict):
        fingerprints = manifest.get("artifact_fingerprints")
        if isinstance(fingerprints, list):
            return [row for row in fingerprints if isinstance(row, dict)]
    return []


def _build_segment_inventory(compile_payload: dict[str, Any], warnings: list[str]) -> dict[str, list[ArtifactSegment]]:
    provided = compile_payload.get("segments")
    if isinstance(provided, list):
        segments_by_artifact: dict[str, list[ArtifactSegment]] = {}
        for row in provided:
            if not isinstance(row, dict):
                continue
            seg = _coerce_segment(row)
            if seg is None:
                continue
            segments_by_artifact.setdefault(seg.artifact_id, []).append(seg)
        if segments_by_artifact:
            return segments_by_artifact

    project_dir_raw = compile_payload.get("project_dir")
    if not isinstance(project_dir_raw, str):
        warnings.append("coverage: project_dir missing; unable to regenerate segments from files.")
        return {}
    project_dir = Path(project_dir_raw)
    if not project_dir.exists():
        warnings.append(f"coverage: project_dir not found: {project_dir}")
        return {}

    segments_by_artifact: dict[str, list[ArtifactSegment]] = {}
    for artifact in _artifact_rows(compile_payload):
        artifact_id = str(artifact.get("artifact_id", ""))
        filename = str(artifact.get("filename", ""))
        if not artifact_id or not filename:
            continue
        path = project_dir / filename
        if not path.exists():
            warnings.append(f"coverage: artifact file missing for segment regeneration: {filename}")
            continue
        segments_by_artifact[artifact_id] = _generate_segments_for_artifact(
            project_id=str(compile_payload.get("project_id", "project")),
            artifact_id=artifact_id,
            path=path,
        )
    return segments_by_artifact


def build_coverage_report(compile_payload: dict[str, Any]) -> ProjectCoverageReport:
    project_id = str(compile_payload.get("project_id", "unknown_project"))
    warnings: list[str] = []
    artifact_rows = _artifact_rows(compile_payload)
    atoms = [row for row in (compile_payload.get("atoms") or []) if isinstance(row, dict)]
    candidates = [row for row in (compile_payload.get("candidates") or []) if isinstance(row, dict)]
    candidates.extend([row for row in (compile_payload.get("rejected_candidates") or []) if isinstance(row, dict)])
    parser_routing = {}
    manifest = compile_payload.get("manifest")
    if isinstance(manifest, dict):
        parser_routing = {
            str(row.get("artifact_id")): row
            for row in (manifest.get("parser_routing") or [])
            if isinstance(row, dict)
        }

    segments_by_artifact = _build_segment_inventory(compile_payload, warnings)

    atom_refs: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for atom in atoms:
        atom_id = str(atom.get("id", ""))
        for source_ref in atom.get("source_refs") or []:
            if not isinstance(source_ref, dict):
                continue
            artifact_id = str(source_ref.get("artifact_id", ""))
            locator = source_ref.get("locator") if isinstance(source_ref.get("locator"), dict) else {}
            atom_refs.setdefault(artifact_id, []).append((atom_id, locator))

    candidate_refs: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        status = str(candidate.get("validation_status", "pending"))
        for source_ref in candidate.get("source_refs") or []:
            if not isinstance(source_ref, dict):
                continue
            artifact_id = str(source_ref.get("artifact_id", ""))
            locator = source_ref.get("locator") if isinstance(source_ref.get("locator"), dict) else {}
            candidate_refs.setdefault(artifact_id, []).append((candidate_id, locator, status))

    reports: list[ArtifactCoverageReport] = []
    recommendations: list[str] = []
    total_segments = 0
    total_covered = 0

    for artifact in artifact_rows:
        artifact_id = str(artifact.get("artifact_id", ""))
        filename = str(artifact.get("filename", ""))
        segments = segments_by_artifact.get(artifact_id, [])
        if not segments:
            warnings.append(f"coverage: no segments generated for artifact {filename}")
        segment_coverages: list[SegmentCoverage] = []
        for segment in segments:
            matched_atom_ids = sorted(
                {
                    atom_id
                    for atom_id, locator in atom_refs.get(artifact_id, [])
                    if _locator_matches(locator, segment.locator)
                }
            )
            matched_candidates = [
                (candidate_id, status)
                for candidate_id, locator, status in candidate_refs.get(artifact_id, [])
                if _locator_matches(locator, segment.locator)
            ]
            matched_candidate_ids = sorted({candidate_id for candidate_id, _ in matched_candidates})
            has_atom = bool(matched_atom_ids)
            has_candidate = bool(matched_candidate_ids)
            unsupported = str((parser_routing.get(artifact_id) or {}).get("chosen_parser", "")) == "none"
            reason: str | None = None
            if has_atom:
                status: CoverageStatus = "covered"
            elif has_candidate:
                status = "candidate_rejected"
                reason = "Candidates were produced for this segment but none were accepted into evidence."
            elif unsupported:
                status = "unsupported"
                reason = "No parser matched this artifact."
            else:
                status = "ignored"
                if _is_boilerplate(segment.text, segment.segment_type):
                    reason = "Ignored boilerplate/non-actionable content."
                else:
                    reason = "No candidate or atom extracted for this segment."

            segment_coverages.append(
                SegmentCoverage(
                    segment_id=segment.id,
                    artifact_id=artifact_id,
                    segment_type=segment.segment_type,
                    text_preview=_preview(segment.text),
                    has_candidate=has_candidate,
                    has_accepted_atom=has_atom,
                    atom_ids=matched_atom_ids,
                    candidate_ids=matched_candidate_ids,
                    coverage_status=status,
                    reason=reason,
                )
            )

        segment_count = len(segment_coverages)
        covered_count = sum(1 for row in segment_coverages if row.coverage_status == "covered")
        candidate_rejected_count = sum(1 for row in segment_coverages if row.coverage_status == "candidate_rejected")
        ignored_count = sum(1 for row in segment_coverages if row.coverage_status == "ignored")
        unsupported_count = sum(1 for row in segment_coverages if row.coverage_status == "unsupported")
        coverage_rate = round(covered_count / segment_count, 6) if segment_count else 0.0
        top_ignored = sorted(
            [row for row in segment_coverages if row.coverage_status in {"ignored", "unsupported"}],
            key=lambda row: (-len(row.text_preview), row.segment_id),
        )[:5]

        reports.append(
            ArtifactCoverageReport(
                artifact_id=artifact_id,
                filename=filename,
                segment_count=segment_count,
                covered_count=covered_count,
                candidate_rejected_count=candidate_rejected_count,
                ignored_count=ignored_count,
                unsupported_count=unsupported_count,
                coverage_rate=coverage_rate,
                top_ignored_segments=top_ignored,
            )
        )

        total_segments += segment_count
        total_covered += covered_count

        if segment_count > 0 and coverage_rate < 0.5:
            lower_name = filename.lower()
            if lower_name.endswith((".xlsx", ".csv")):
                recommendations.append(
                    f"Low coverage in {filename}: expand header aliases and row extraction rules for sparse spreadsheet segments."
                )
            elif "transcript" in lower_name or "meeting" in lower_name:
                recommendations.append(
                    f"Low coverage in {filename}: extend transcript cue patterns for questions/decisions and speaker-role mapping."
                )
            elif lower_name.endswith(".txt"):
                recommendations.append(
                    f"Low coverage in {filename}: improve text/email parser routing and add domain-pack phrase patterns."
                )
            else:
                recommendations.append(
                    f"Low coverage in {filename}: add parser rules or domain patterns for uncovered segment types."
                )

    overall = round(total_covered / total_segments, 6) if total_segments else 0.0
    return ProjectCoverageReport(
        project_id=project_id,
        artifact_reports=sorted(reports, key=lambda row: row.filename),
        overall_coverage_rate=overall,
        warnings=sorted(set(warnings)),
        recommended_parser_improvements=sorted(set(recommendations)),
    )


def build_segment_coverage_index(compile_payload: dict[str, Any]) -> dict[str, list[SegmentCoverage]]:
    """Return raw segment-level coverage rows keyed by artifact_id.

    This is mainly useful for diagnostics/tests where detailed segment status
    assertions are needed beyond aggregate artifact counters.
    """

    artifact_rows = _artifact_rows(compile_payload)
    warnings: list[str] = []
    segments_by_artifact = _build_segment_inventory(compile_payload, warnings)
    atoms = [row for row in (compile_payload.get("atoms") or []) if isinstance(row, dict)]
    candidates = [row for row in (compile_payload.get("candidates") or []) if isinstance(row, dict)]
    candidates.extend([row for row in (compile_payload.get("rejected_candidates") or []) if isinstance(row, dict)])
    parser_routing = {}
    manifest = compile_payload.get("manifest")
    if isinstance(manifest, dict):
        parser_routing = {
            str(row.get("artifact_id")): row
            for row in (manifest.get("parser_routing") or [])
            if isinstance(row, dict)
        }

    atom_refs: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for atom in atoms:
        atom_id = str(atom.get("id", ""))
        for source_ref in atom.get("source_refs") or []:
            if isinstance(source_ref, dict):
                atom_refs.setdefault(str(source_ref.get("artifact_id", "")), []).append(
                    (atom_id, source_ref.get("locator") if isinstance(source_ref.get("locator"), dict) else {})
                )
    candidate_refs: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        status = str(candidate.get("validation_status", "pending"))
        for source_ref in candidate.get("source_refs") or []:
            if isinstance(source_ref, dict):
                candidate_refs.setdefault(str(source_ref.get("artifact_id", "")), []).append(
                    (candidate_id, source_ref.get("locator") if isinstance(source_ref.get("locator"), dict) else {}, status)
                )

    index: dict[str, list[SegmentCoverage]] = {}
    for artifact in artifact_rows:
        artifact_id = str(artifact.get("artifact_id", ""))
        rows: list[SegmentCoverage] = []
        for segment in segments_by_artifact.get(artifact_id, []):
            matched_atom_ids = sorted(
                {
                    atom_id
                    for atom_id, locator in atom_refs.get(artifact_id, [])
                    if _locator_matches(locator, segment.locator)
                }
            )
            matched_candidates = [
                candidate_id
                for candidate_id, locator, _ in candidate_refs.get(artifact_id, [])
                if _locator_matches(locator, segment.locator)
            ]
            has_atom = bool(matched_atom_ids)
            has_candidate = bool(matched_candidates)
            unsupported = str((parser_routing.get(artifact_id) or {}).get("chosen_parser", "")) == "none"
            if has_atom:
                status: CoverageStatus = "covered"
                reason = None
            elif has_candidate:
                status = "candidate_rejected"
                reason = "Candidates were produced for this segment but none were accepted into evidence."
            elif unsupported:
                status = "unsupported"
                reason = "No parser matched this artifact."
            else:
                status = "ignored"
                reason = (
                    "Ignored boilerplate/non-actionable content."
                    if _is_boilerplate(segment.text, segment.segment_type)
                    else "No candidate or atom extracted for this segment."
                )
            rows.append(
                SegmentCoverage(
                    segment_id=segment.id,
                    artifact_id=artifact_id,
                    segment_type=segment.segment_type,
                    text_preview=_preview(segment.text),
                    has_candidate=has_candidate,
                    has_accepted_atom=has_atom,
                    atom_ids=matched_atom_ids,
                    candidate_ids=sorted(set(matched_candidates)),
                    coverage_status=status,
                    reason=reason,
                )
            )
        index[artifact_id] = sorted(rows, key=lambda row: row.segment_id)
    return index


def load_compile_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Compile result payload must be a JSON object.")
    return payload
