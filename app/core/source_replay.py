from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Literal

from docx import Document
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string

from app.core.normalizers import normalize_text
from app.core.segments import ArtifactSegment
from app.core.schemas import EvidenceAtom, EvidenceReceipt, SourceRef

VERIFIER_VERSION = "source_replay_v1"
_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "will",
    "was",
    "are",
    "after",
    "before",
    "please",
}


def _receipt(
    atom: EvidenceAtom,
    source_ref: SourceRef,
    replay_status: Literal["verified", "failed", "unsupported"],
    reason: str,
    extracted_snippet: str | None = None,
) -> EvidenceReceipt:
    return EvidenceReceipt(
        atom_id=atom.id,
        artifact_id=source_ref.artifact_id,
        filename=source_ref.filename,
        source_ref_id=source_ref.id,
        replay_status=replay_status,
        extracted_snippet=extracted_snippet,
        locator=dict(source_ref.locator),
        reason=reason,
        verifier_version=VERIFIER_VERSION,
    )


def _atom_match_candidates(atom: EvidenceAtom) -> list[str]:
    candidates = [normalize_text(atom.raw_text), normalize_text(atom.normalized_text)]
    for key in atom.entity_keys:
        _, _, tail = key.partition(":")
        if tail:
            candidates.append(normalize_text(tail.replace("_", " ")))
    for value in atom.value.values():
        if isinstance(value, (str, int, float)):
            candidates.append(normalize_text(str(value)))
    return sorted({c for c in candidates if c})


def _important_terms(atom: EvidenceAtom) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(atom.normalized_text))
    terms = [t for t in tokens if (t.isdigit() or len(t) >= 4) and t not in _STOP_WORDS]
    if not terms:
        terms = [t for t in tokens if t not in _STOP_WORDS]
    deduped: list[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped[:8]


def _snippet_matches_atom(atom: EvidenceAtom, snippet: str) -> bool:
    snippet_norm = normalize_text(snippet)
    if not snippet_norm:
        return False
    if atom.normalized_text and normalize_text(atom.normalized_text) in snippet_norm:
        return True
    candidates = _atom_match_candidates(atom)
    if any(candidate in snippet_norm for candidate in candidates):
        return True
    terms = _important_terms(atom)
    if not terms:
        return False
    matched = sum(1 for term in terms if term in snippet_norm)
    required = 1 if len(terms) <= 2 else 2
    return matched >= required


def _verify_spreadsheet_row(atom: EvidenceAtom, source_ref: SourceRef, path: Path) -> EvidenceReceipt:
    locator = source_ref.locator
    sheet = locator.get("sheet")
    row_number = locator.get("row")
    columns = locator.get("columns") or {}
    if not isinstance(row_number, int) or row_number < 1:
        return _receipt(atom, source_ref, "failed", "Spreadsheet locator missing valid row number")

    row_values: dict[str, str] = {}
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        worksheet = workbook[sheet] if isinstance(sheet, str) and sheet in workbook.sheetnames else workbook.active
        max_row = worksheet.max_row or 0
        if row_number > max_row:
            return _receipt(atom, source_ref, "failed", f"Spreadsheet row {row_number} out of range")
        for key, col_letter in columns.items():
            try:
                col_index = column_index_from_string(str(col_letter))
                value = worksheet.cell(row=row_number, column=col_index).value
                row_values[str(key)] = "" if value is None else str(value)
            except Exception:
                continue
    else:
        delimiter = "," if suffix == ".csv" else None
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            if delimiter is None:
                rows = [re.split(r"[,\t|]", line.rstrip("\n")) for line in handle.readlines()]
            else:
                rows = list(csv.reader(handle))
        if row_number > len(rows):
            return _receipt(atom, source_ref, "failed", f"Spreadsheet row {row_number} out of range")
        row = rows[row_number - 1]
        for key, col_letter in columns.items():
            try:
                index = column_index_from_string(str(col_letter)) - 1
                row_values[str(key)] = row[index] if index < len(row) else ""
            except Exception:
                continue

    snippet_parts = [f"{str(key).replace('_', ' ').title()}={str(value).strip()}" for key, value in row_values.items()]
    extracted_snippet = " | ".join(part for part in snippet_parts if part and not part.endswith("="))
    if not extracted_snippet:
        return _receipt(atom, source_ref, "failed", "Spreadsheet row found but no cited cells were readable")
    if _snippet_matches_atom(atom, extracted_snippet):
        return _receipt(atom, source_ref, "verified", "Spreadsheet row and cited cells verified", extracted_snippet)
    return _receipt(
        atom,
        source_ref,
        "failed",
        "Spreadsheet row exists but cited cells do not match atom content",
        extracted_snippet,
    )


def _verify_line_range(atom: EvidenceAtom, source_ref: SourceRef, path: Path) -> EvidenceReceipt:
    locator = source_ref.locator
    line_start = locator.get("line_start")
    line_end = locator.get("line_end")
    if not isinstance(line_start, int) or not isinstance(line_end, int) or line_start < 1 or line_end < line_start:
        return _receipt(atom, source_ref, "failed", "Line-range locator missing or invalid")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if line_end > len(lines):
        return _receipt(atom, source_ref, "failed", f"Line range {line_start}-{line_end} is out of bounds")
    snippet = "\n".join(lines[line_start - 1 : line_end])
    if _snippet_matches_atom(atom, snippet):
        return _receipt(atom, source_ref, "verified", "Line-range snippet verified", snippet)
    return _receipt(atom, source_ref, "failed", "Line-range snippet did not match atom content", snippet)


def _verify_docx_locator(atom: EvidenceAtom, source_ref: SourceRef, path: Path) -> EvidenceReceipt:
    locator = source_ref.locator
    tracked_change = locator.get("tracked_change")
    if tracked_change == "deleted":
        return _receipt(
            atom,
            source_ref,
            "unsupported",
            "Tracked deletion replay via OOXML is not enabled in this verifier",
        )

    document = Document(path)
    paragraph_index = locator.get("paragraph_index")
    table_index = locator.get("table_index")
    row = locator.get("row")
    cell = locator.get("cell")

    if isinstance(paragraph_index, int):
        if paragraph_index < 0 or paragraph_index >= len(document.paragraphs):
            return _receipt(atom, source_ref, "failed", f"Paragraph index {paragraph_index} out of range")
        snippet = document.paragraphs[paragraph_index].text
        if _snippet_matches_atom(atom, snippet):
            return _receipt(atom, source_ref, "verified", "DOCX paragraph locator verified", snippet)
        return _receipt(atom, source_ref, "failed", "DOCX paragraph did not match atom content", snippet)

    if isinstance(table_index, int) and isinstance(row, int) and isinstance(cell, int):
        if table_index < 0 or table_index >= len(document.tables):
            return _receipt(atom, source_ref, "failed", f"Table index {table_index} out of range")
        table = document.tables[table_index]
        if row < 0 or row >= len(table.rows):
            return _receipt(atom, source_ref, "failed", f"Table row {row} out of range")
        if cell < 0 or cell >= len(table.rows[row].cells):
            return _receipt(atom, source_ref, "failed", f"Table cell {cell} out of range")
        snippet = table.rows[row].cells[cell].text
        if _snippet_matches_atom(atom, snippet):
            return _receipt(atom, source_ref, "verified", "DOCX table locator verified", snippet)
        return _receipt(atom, source_ref, "failed", "DOCX table cell did not match atom content", snippet)

    return _receipt(atom, source_ref, "unsupported", "DOCX locator type is not currently supported")


def replay_source_ref(atom: EvidenceAtom, source_ref: SourceRef, artifact_paths: dict[str, Path]) -> EvidenceReceipt:
    path = artifact_paths.get(source_ref.artifact_id)
    if path is None or not path.exists():
        return _receipt(atom, source_ref, "unsupported", "Original artifact file is not available for replay")

    suffix = path.suffix.lower()
    try:
        if source_ref.locator.get("row") is not None and source_ref.locator.get("columns"):
            return _verify_spreadsheet_row(atom, source_ref, path)
        if source_ref.locator.get("line_start") is not None and source_ref.locator.get("line_end") is not None:
            return _verify_line_range(atom, source_ref, path)
        if suffix == ".docx":
            return _verify_docx_locator(atom, source_ref, path)
        if suffix in {".txt", ".md", ".eml", ".vtt", ".srt", ".json"}:
            return _verify_line_range(atom, source_ref, path)
    except Exception as exc:  # pragma: no cover
        return _receipt(atom, source_ref, "failed", f"Source replay failed with exception: {exc}")

    return _receipt(atom, source_ref, "unsupported", "No verifier available for this locator/artifact type")


def replay_atom_receipts(atom: EvidenceAtom, artifact_paths: dict[str, Path]) -> list[EvidenceReceipt]:
    receipts = [replay_source_ref(atom, source_ref, artifact_paths) for source_ref in atom.source_refs]
    return sorted(receipts, key=lambda receipt: receipt.source_ref_id)


def attach_receipts_to_atoms(atoms: list[EvidenceAtom], artifact_paths: dict[str, Path]) -> list[EvidenceAtom]:
    for atom in atoms:
        atom.receipts = replay_atom_receipts(atom, artifact_paths)
    return atoms


def summarize_receipts(atoms: list[EvidenceAtom]) -> dict[str, int]:
    summary = {"verified": 0, "unsupported": 0, "failed": 0}
    for atom in atoms:
        for receipt in atom.receipts:
            if receipt.replay_status in summary:
                summary[receipt.replay_status] += 1
    return summary


def verify_segment(segment: ArtifactSegment, artifact_path: Path) -> EvidenceReceipt:
    artifact_paths = {segment.artifact_id: artifact_path}
    synthetic_atom = EvidenceAtom.model_validate(
        {
            "id": f"segment_atom_{segment.id}",
            "project_id": segment.project_id,
            "artifact_id": segment.artifact_id,
            "atom_type": "entity",
            "raw_text": segment.text,
            "normalized_text": segment.normalized_text,
            "value": {"segment_type": segment.segment_type},
            "entity_keys": [],
            "source_refs": [segment.source_ref.model_dump(mode="json")],
            "authority_class": "machine_extractor",
            "confidence": 1.0,
            "review_status": "auto_accepted",
            "review_flags": [],
            "parser_version": segment.source_ref.parser_version,
        }
    )
    return replay_source_ref(synthetic_atom, segment.source_ref, artifact_paths)
