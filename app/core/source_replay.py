from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from docx import Document
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string

from app.core.normalizers import normalize_text
from app.core.segments import ArtifactSegment
from app.core.schemas import EvidenceAtom, EvidenceReceipt, SourceRef


# ────────────── workbook cache (insanity-perf) ──────────────
# openpyxl ``load_workbook`` is the dominant cost in source_replay
# when a project has many spreadsheet atoms. Each atom triggered an
# independent load of the same .xlsx file — on STRESS_MULTI_CAM,
# 1349 spreadsheet atoms × ~100 ms per load = ~135 s of source_replay
# wall time. Cache by (path, mtime, size) so each unique workbook is
# loaded at most once per compile, while invalidating if the file
# changes on disk between calls.
_WORKBOOK_CACHE: dict[tuple[str, float, int], Any] = {}


def _load_workbook_cached(path: Path):
    """Return a memoized workbook for ``path``. Cached by (path, mtime,
    size) so editing the file between compiles invalidates the entry."""
    try:
        st = path.stat()
        key = (str(path), st.st_mtime, st.st_size)
    except FileNotFoundError:
        return None
    cached = _WORKBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        wb = load_workbook(path, data_only=True)
    except Exception:
        return None
    _WORKBOOK_CACHE[key] = wb
    # Soft cap: keep the cache to ~16 distinct workbooks. Realistic
    # projects compile <16 unique .xlsx files.
    if len(_WORKBOOK_CACHE) > 16:
        # Drop the oldest entry. dict preserves insertion order so the
        # first key is the oldest.
        oldest = next(iter(_WORKBOOK_CACHE))
        if oldest != key:
            _WORKBOOK_CACHE.pop(oldest, None)
    return wb


def clear_workbook_cache() -> None:  # pragma: no cover — used by tests
    _WORKBOOK_CACHE.clear()


def _replay_norm(text: str) -> str:
    """Normalize text for replay matching.

    Strips Unicode combining marks (so "café" matches "cafe", and
    "M\xa0Smith" matches "M Smith") then defers to the canonical
    ``normalize_text``. The PR8 spec calls this out specifically: a
    lot of "failed" replay receipts in the corpus were just NFKD
    drift between parser-extracted text and the spreadsheet/PDF
    re-read.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return normalize_text(text)

VERIFIER_VERSION = "source_replay_v2"
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
    """Phrase-level candidates the snippet must contain to count as a hit.

    A candidate is "evidence-grade" only if it is *substantive enough*
    that finding it inside the snippet is meaningful on its own. Single
    short tokens like ``customer``, ``vendor``, ``owner`` (which leak
    out of ``atom.value['owner']`` / role tags) cause false-positive
    line-range matches: any speaker line that happens to contain the
    word "Customer" would falsely verify an atom whose actual text was
    deleted from the file. We require either whitespace (i.e. a real
    phrase) or length >= 12 (i.e. a long-enough single token like a
    site/part identifier) to qualify.
    """
    candidates = [normalize_text(atom.raw_text), normalize_text(atom.normalized_text)]
    for key in atom.entity_keys:
        _, _, tail = key.partition(":")
        if tail:
            candidates.append(normalize_text(tail.replace("_", " ")))
    for value in atom.value.values():
        if isinstance(value, (str, int, float)):
            candidates.append(normalize_text(str(value)))
    return sorted({
        c for c in candidates
        if c and (
            " " in c                       # multi-word phrases are always evidence
            or any(ch.isdigit() for ch in c)  # part numbers, quantities, measurements
            or len(c) >= 12                # long unique tokens (UUIDs, hashes, slugs)
        )
    })


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
    snippet_norm = _replay_norm(snippet)
    if not snippet_norm:
        return False
    if atom.normalized_text and _replay_norm(atom.normalized_text) in snippet_norm:
        return True
    candidates = [_replay_norm(c) for c in _atom_match_candidates(atom)]
    if any(c and c in snippet_norm for c in candidates):
        return True
    terms = [_replay_norm(t) for t in _important_terms(atom)]
    terms = [t for t in terms if t]
    if not terms:
        return False
    matched = sum(1 for term in terms if term in snippet_norm)
    required = 1 if len(terms) <= 2 else 2
    return matched >= required


def _spreadsheet_full_row_text(
    path: Path, sheet: str | None, row_number: int
) -> str:
    """Return ``"v1 | v2 | v3 ..."`` for every non-empty cell in
    ``row_number`` (1-based). Used as a fallback when the cited
    columns alone don't satisfy ``_snippet_matches_atom``: the row
    might have been authored with the relevant fact in a column the
    parser didn't cite (e.g. the parser cited Severity, but the
    important text is in the Mitigation column on the same row).
    """
    suffix = path.suffix.lower()
    parts: list[str] = []
    if suffix == ".xlsx":
        wb = _load_workbook_cached(path)
        if wb is None:
            return ""
        ws = (
            wb[sheet]
            if isinstance(sheet, str) and sheet in wb.sheetnames
            else wb.active
        )
        max_row = ws.max_row or 0
        if max_row and row_number > max_row:
            return ""
        for cell in ws[row_number]:
            if cell.value is None:
                continue
            text = str(cell.value).strip()
            if text:
                parts.append(text)
    else:
        delimiter = "," if suffix == ".csv" else None
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                if delimiter is None:
                    rows = [
                        re.split(r"[,\t|]", line.rstrip("\n"))
                        for line in fh.readlines()
                    ]
                else:
                    rows = list(csv.reader(fh))
        except Exception:
            return ""
        if row_number > len(rows):
            return ""
        for cell in rows[row_number - 1]:
            text = str(cell or "").strip()
            if text:
                parts.append(text)
    return " | ".join(parts)


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
        # Cached loader (insanity-perf). On STRESS_MULTI_CAM this
        # alone cuts source_replay from ~135 s to ~3 s.
        workbook = _load_workbook_cached(path)
        if workbook is None:
            return _receipt(atom, source_ref, "failed", f"Spreadsheet file unreadable: {path}")
        worksheet = workbook[sheet] if isinstance(sheet, str) and sheet in workbook.sheetnames else workbook.active
        max_row = worksheet.max_row or 0
        if max_row and row_number > max_row:
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

    # Sort columns alphabetically so the snippet text is deterministic across
    # runs.  Parser-side dict insertion order has been observed to drift (e.g.
    # openpyxl backed by C accelerators), which silently broke output_signature
    # reproducibility — see manifest.compute_output_signature.
    snippet_parts = [
        f"{str(key).replace('_', ' ').title()}={str(value).strip()}"
        for key, value in sorted(row_values.items())
    ]
    extracted_snippet = " | ".join(part for part in snippet_parts if part and not part.endswith("="))
    if not extracted_snippet:
        return _receipt(atom, source_ref, "failed", "Spreadsheet row found but no cited cells were readable")
    if _snippet_matches_atom(atom, extracted_snippet):
        return _receipt(atom, source_ref, "verified", "Spreadsheet row and cited cells verified", extracted_snippet)

    # PR8 — full-row fallback. The row exists but the cited cells
    # alone don't carry the atom's text. This commonly happens when
    # the parser cited only one column (e.g. Severity) but the atom
    # text was authored from a different column on the same row
    # (e.g. Mitigation). Read the full row and try once more before
    # giving up — true mismatches still fail.
    full_row = _spreadsheet_full_row_text(path, sheet if isinstance(sheet, str) else None, row_number)
    if full_row and _snippet_matches_atom(atom, full_row):
        return _receipt(
            atom,
            source_ref,
            "verified",
            "Spreadsheet row verified via full-row fallback",
            full_row,
        )

    # Insanity-pass — value-dict match. The quote_parser synthesizes
    # atoms with text like ``"Line item TesiraFORTÉ X 400-SUP Biamp ..."``
    # whose "Line item" prefix doesn't appear in any cell, so the
    # cell-based matcher fails even though the row IS the source.
    # If the atom carries a value-dict with cell-derived fields
    # (part_number, description, manufacturer, etc.), check that at
    # least one of those values is present in the full row text.
    if full_row and isinstance(atom.value, dict):
        cell_like_keys = (
            "part_number", "description", "manufacturer", "vendor",
            "model", "asset_id", "serial", "section", "item",
            "support_note", "lead_time",
        )
        norm_row = _replay_norm(full_row)
        for key in cell_like_keys:
            v = atom.value.get(key)
            if not isinstance(v, str) or len(v.strip()) < 4:
                continue
            if _replay_norm(v) in norm_row:
                return _receipt(
                    atom,
                    source_ref,
                    "verified",
                    f"Spreadsheet row verified via value-dict field {key!r}",
                    full_row,
                )

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


def _verify_pdf_block(atom: EvidenceAtom, source_ref: SourceRef, path: Path) -> EvidenceReceipt:
    """Verify an OrbitBrief PDF atom against the persisted structured doc.

    The OrbitBrief PDF parser writes a deterministic
    ``<stem>.derived/structured.json`` next to every parsed PDF.  We
    look up the atom's ``block_id`` (or ``row_index`` for table cells,
    or ``bullet_path`` for bullets) inside that doc and check that the
    atom's text actually came from there.  This makes PDF receipts
    first-class — same status as XLSX rows and DOCX paragraphs.
    """
    locator = source_ref.locator
    block_id = locator.get("block_id")
    if not block_id:
        return _receipt(atom, source_ref, "unsupported", "PDF locator missing block_id")
    structured = _load_structured_doc(path)
    if structured is None:
        return _receipt(
            atom,
            source_ref,
            "unsupported",
            "Structured PDF doc not present (run orbitbrief_pdf parser first)",
        )
    block = _find_block(structured, block_id)
    if block is None:
        return _receipt(atom, source_ref, "failed", f"block_id {block_id} not found in structured doc")

    snippet = _block_to_snippet(block, locator)
    if not snippet:
        return _receipt(atom, source_ref, "failed", "Located PDF block contained no text")
    if _snippet_matches_atom(atom, snippet):
        return _receipt(atom, source_ref, "verified", "PDF block locator verified", snippet)
    return _receipt(atom, source_ref, "failed", "PDF block found but text did not match atom", snippet)


def _structured_doc_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}.derived") / "structured.json"


def _load_structured_doc(pdf_path: Path) -> dict[str, Any] | None:
    out = _structured_doc_path_for(pdf_path)
    if not out.is_file():
        return None
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_block(structured: dict[str, Any], block_id: str) -> dict[str, Any] | None:
    for page in structured.get("pages", []) or []:
        match = _walk_sections_for_block(page.get("sections", []) or [], block_id)
        if match is not None:
            return match
    return None


def _walk_sections_for_block(sections: list[dict[str, Any]], block_id: str) -> dict[str, Any] | None:
    for section in sections:
        for block in section.get("blocks", []) or []:
            if block.get("id") == block_id:
                return block
        nested = _walk_sections_for_block(section.get("subsections", []) or [], block_id)
        if nested is not None:
            return nested
    return None


def _block_to_snippet(block: dict[str, Any], locator: dict[str, Any]) -> str:
    kind = block.get("kind")
    if kind == "paragraph":
        return str(block.get("text") or "")
    if kind == "note":
        return str(block.get("text") or "")
    if kind == "bullet_list":
        bullet_path = locator.get("bullet_path")
        if isinstance(bullet_path, list) and bullet_path:
            item = _bullet_at_path(block.get("items", []) or [], bullet_path)
            if item is not None:
                return str(item.get("text") or "")
        if locator.get("bullet_role") == "intro":
            return str(block.get("intro") or "")
        # Fallback: stitch the whole list together so the snippet still
        # contains the atom text somewhere.
        parts: list[str] = []
        intro = block.get("intro")
        if intro:
            parts.append(str(intro))
        parts.extend(_flatten_bullets(block.get("items", []) or []))
        return " | ".join(parts)
    if kind == "table":
        row_index = locator.get("row_index")
        rows = block.get("rows", []) or []
        if isinstance(row_index, int) and 0 <= row_index < len(rows):
            cells = rows[row_index]
            if isinstance(cells, dict):
                return " | ".join(f"{k}: {v}" for k, v in cells.items() if v is not None)
        return str(block.get("raw_text") or "")
    return ""


def _bullet_at_path(items: list[dict[str, Any]], path: list[int]) -> dict[str, Any] | None:
    if not path:
        return None
    current = items
    node: dict[str, Any] | None = None
    for index in path:
        if not isinstance(index, int) or index < 0 or index >= len(current):
            return None
        node = current[index]
        if not isinstance(node, dict):
            return None
        current = node.get("children", []) or []
    return node


def _flatten_bullets(items: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = item.get("text")
        if text:
            out.append(str(text))
        children = item.get("children")
        if children:
            out.extend(_flatten_bullets(children))
    return out


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
        if suffix == ".pdf":
            return _verify_pdf_block(atom, source_ref, path)
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
