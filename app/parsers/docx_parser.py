from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docx import Document

from app.core.ids import stable_id
from app.core.normalizers import normalize_entity_key, normalize_text
from app.core.segments import ArtifactSegment
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserOutput,
    ReviewStatus,
    SourceRef,
    ParserCapability,
    ParserMatch,
)
from app.parsers.base import BaseParser
from app.parsers.segmenters import segment_docx
from app.parsers.structured_projection import (
    derived_files_for,
    make_bullet_list,
    make_page,
    make_paragraph,
    make_section,
    make_structured_document,
    make_table,
    stamp_section_and_block_ids,
)
from app.domain.schemas import DomainPack

STRUCTURED_SCHEMA_DOCX = "orbitbrief.docx.structured.v1"

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

SCOPE_PATTERNS = [r"\bscope includes\b", r"\binclude\b", r"\binstallation\b", r"\binstall\b"]
EXCLUSION_PATTERNS = [r"\bexclude\b", r"\bexcluded\b", r"\bout of scope\b", r"\bnot in scope\b"]
CONSTRAINT_PATTERNS = [
    r"\baccess\b",
    r"\bcustomer is responsible\b",
    r"\bescort required\b",
    r"\bbadge required\b",
]
ASSUMPTION_PATTERNS = [r"\bassum(?:e|ption|ing)\b"]


class DocxParser(BaseParser):
    parser_name = "docx"
    parser_version = "docx_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".docx"],
        supported_artifact_types=[ArtifactType.docx],
        emitted_atom_types=[AtomType.scope_item, AtomType.exclusion, AtomType.constraint, AtomType.assumption, AtomType.open_question],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        confidence = 0.94 if suffix == ".docx" else 0.0
        reasons = ["docx_extension"] if suffix == ".docx" else []
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.docx,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def segment_artifact(self, project_id: str, artifact_id: str, path: Path) -> list[ArtifactSegment]:
        return segment_docx(project_id=project_id, artifact_id=artifact_id, path=path, parser_version=self.parser_version)

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom]:
        return self.parse_artifact_full(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        ).atoms

    def parse_artifact_full(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        document = Document(path)
        atoms: list[EvidenceAtom] = []

        # v48 FIX 1: Build the set of paragraph object IDs that live inside
        # table cells BEFORE the paragraph loop runs. python-docx's
        # document.paragraphs includes table-cell paragraphs, so without this
        # guard every table cell is processed TWICE: once as a standalone
        # paragraph atom (losing row context) and once inside the table loop
        # below (correct, with full row structure). Double-emission produces
        # garbage atoms ("Access Constraint") and silently drops full-row
        # structured content when site_roster fires.
        _table_para_ids: set[int] = set()
        for _tbl in document.tables:
            for _row in _tbl.rows:
                for _cell in _row.cells:
                    for _para in _cell.paragraphs:
                        _table_para_ids.add(id(_para))

        for idx, paragraph in enumerate(document.paragraphs):
            # v48 FIX 1: skip paragraphs that belong to a table cell.
            # Their content is handled by the table loop below with proper
            # row structure and column context preserved.
            if id(paragraph) in _table_para_ids:
                continue
            text = paragraph.text.strip()
            if not text:
                continue
            is_heading = bool(paragraph.style and paragraph.style.name.lower().startswith("heading"))
            atoms.extend(
                self._emit_atoms_for_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    text=text,
                    paragraph_index=idx,
                    table_index=None,
                    row=None,
                    cell=None,
                    tracked_change=None,
                    heading=is_heading,
                )
            )

        # Build all-document text once for ``kind=physical_site`` declarations.
        # v48 FIX 1: exclude table-cell paragraphs so the surrounding-text
        # heuristic stays accurate (otherwise table text bleeds in twice).
        document_text = " ".join(
            (p.text or "").strip()
            for p in document.paragraphs
            if id(p) not in _table_para_ids
        )

        for table_idx, table in enumerate(document.tables):
            # Build a column/rows view of the table for the site_roster
            # extractor (and a per-row atom emitter below).
            table_rows: list[list[str]] = []
            for row_cells in table.rows:
                table_rows.append([c.text.strip() for c in row_cells.cells])

            # Site-roster fast path — when the first non-empty row
            # looks like roster headers + a roster-specific column,
            # emit one physical_site atom per data row.
            roster_atoms = self._maybe_emit_docx_site_roster_atoms(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                table_index=table_idx,
                rows=table_rows,
                surrounding_text=document_text,
            )
            if roster_atoms:
                atoms.extend(roster_atoms)
                # v48 FIX 3: Only skip per-row fallback when the roster
                # extraction is HIGH-confidence (≥2 atoms, all confidence
                # ≥0.75). Low-confidence rosters fall through so the
                # per-row emitter ALSO processes the table — duplicate
                # coverage beats silent data loss. entity_resolution
                # collapses the dupes later.
                _high_conf_roster = (
                    len(roster_atoms) >= 2
                    and all(getattr(a, "confidence", 0.0) >= 0.75 for a in roster_atoms)
                )
                if _high_conf_roster:
                    continue
                # Low confidence: fall through to per-row emission below.

            # Per-row atom — concatenate all cells so the row's
            # context (Site + Part + Qty) survives as one atom for
            # entity extraction. Skip the all-cell header row when
            # row 0 looks like field labels.
            header_cells = table_rows[0] if table_rows else []
            for row_idx, row_cells in enumerate(table.rows):
                cell_texts = [c.text.strip() for c in row_cells.cells if c.text.strip()]
                if not cell_texts:
                    continue
                # Skip the header row (first row) when it looks like
                # column labels — its cells are repeated below as
                # data labels.
                if row_idx == 0 and len(cell_texts) >= 2 and all(
                    len(c) <= 30 for c in cell_texts
                ):
                    continue
                row_text = " | ".join(cell_texts)
                # v49.2: emit a raw_table_row atom alongside the legacy
                # row blob. The centralized _enrich_table_atoms() in
                # entity_extraction will classify all raw_table_row
                # atoms in one pass using the column schema registry.
                if header_cells and row_idx > 0:
                    _row_cells_full = [c.text.strip() for c in row_cells.cells]
                    _rtr_id = stable_id("atm", artifact_id, "raw_table_row", table_idx, row_idx)
                    _rtr_src = SourceRef(
                        id=stable_id("src", _rtr_id),
                        artifact_id=artifact_id,
                        artifact_type=ArtifactType.docx,
                        filename=path.name,
                        locator={"table_index": table_idx, "row": row_idx, "extraction": "raw_table_row_v49_2"},
                        extraction_method="raw_table_row_v49_2",
                        parser_version=self.parser_version,
                    )
                    atoms.append(EvidenceAtom(
                        id=_rtr_id,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.raw_table_row,
                        raw_text=row_text[:4000],
                        normalized_text=row_text.lower()[:4000],
                        value={
                            "_columns": list(header_cells),
                            "_row": _row_cells_full,
                            "_table_idx": table_idx,
                            "_row_idx": row_idx,
                            "_filename": path.name,
                            "_artifact_type": "docx",
                        },
                        entity_keys=[],
                        source_refs=[_rtr_src],
                        receipts=[],
                        authority_class=AuthorityClass.contractual_scope,
                        confidence=0.80,
                        confidence_raw=0.80,
                        calibrated_confidence=0.80,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[],
                        parser_version=self.parser_version,
                    ))
                # Table rows carry structured data even without
                # scope/constraint verbs — emit unconditionally as
                # a scope_item (the classifier path is for prose).
                row_atom_id = stable_id(
                    "atm", artifact_id, "docx_row",
                    table_idx, row_idx, row_text
                )
                row_src = SourceRef(
                    id=stable_id("src", row_atom_id),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.docx,
                    filename=path.name,
                    locator={
                        "table_index": table_idx,
                        "row": row_idx,
                        "extraction": "docx_table_row_v1",
                    },
                    extraction_method="docx_table_row_v1",
                    parser_version=self.parser_version,
                )
                atoms.append(
                    EvidenceAtom(
                        id=row_atom_id,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.scope_item,
                        raw_text=row_text,
                        normalized_text=row_text.lower(),
                        value={
                            "kind": "table_row",
                            "columns": header_cells,
                            "cells": dict(zip(header_cells, cell_texts)) if header_cells else {f"col_{i}": v for i, v in enumerate(cell_texts)},
                        },
                        entity_keys=[],
                        source_refs=[row_src],
                        receipts=[],
                        authority_class=AuthorityClass.contractual_scope,
                        confidence=0.85,
                        confidence_raw=0.85,
                        calibrated_confidence=0.85,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[],
                        parser_version=self.parser_version,
                    )
                )

        atoms.extend(
            self._extract_tracked_change_atoms(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                path=path,
            )
        )

        # Comments — sidebar Word comments often carry scope-relevant
        # notes ("we're cutting this from scope", "verify w/ vendor",
        # "approved by Jane"). Extract them so they don't get
        # silently dropped.
        atoms.extend(
            self._extract_comment_atoms(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                path=path,
            )
        )

        structured_doc = self._build_structured_doc(filename=path.name, document=document)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _maybe_emit_docx_site_roster_atoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        table_index: int,
        rows: list[list[str]],
        surrounding_text: str,
    ) -> list[EvidenceAtom]:
        """Emit ``physical_site`` entity atoms when a DOCX table looks
        like a site roster (header row with Site ID + Facility +
        Address / MDF / Access / Escort columns).
        """
        try:
            from app.parsers.site_roster_extractor import (
                extract_site_roster,
                looks_like_site_roster,
                map_columns_to_fields,
            )
        except Exception:  # pragma: no cover
            return []
        if not rows or len(rows) < 2:
            return []
        header = [(c or "").strip() for c in rows[0]]
        data_rows: list[dict[str, str]] = []
        for r in rows[1:]:
            cells = {
                header[i] if i < len(header) and header[i] else f"col_{i}":
                (r[i] if i < len(r) else "") or ""
                for i in range(max(len(header), len(r)))
            }
            if any(v.strip() for v in cells.values()):
                data_rows.append(cells)
        if not data_rows:
            return []
        try:
            if not looks_like_site_roster(
                columns=header, rows=data_rows, surrounding_text=surrounding_text
            ):
                return []
            field_map = map_columns_to_fields(header)
            roster_specific = {
                "facility_name", "street_address", "mdf_idf",
                "access_window", "escort_owner", "city_state",
            }
            if not (set(field_map.values()) & roster_specific):
                return []
            roster_rows = extract_site_roster(
                columns=header, rows=data_rows, surrounding_text=surrounding_text
            )
        except Exception:  # pragma: no cover
            return []
        if not roster_rows:
            return []

        out: list[EvidenceAtom] = []
        for site_row in roster_rows:
            sid = (site_row.site_id or "").strip()
            canon_id = sid or site_row.facility_name or ""
            if not canon_id:
                continue
            row_index = site_row.row_index + 1  # +1 for header
            text_parts: list[str] = []
            for label, val in [
                ("site_id", sid or site_row.site_id),
                ("facility", site_row.facility_name),
                ("address", site_row.street_address),
                ("mdf_idf", site_row.mdf_idf),
                ("access", site_row.access_window),
                ("escort", site_row.escort_owner),
            ]:
                if val:
                    text_parts.append(f"{label}: {val}")
            row_text = " | ".join(text_parts) or canon_id
            entity_keys: list[str] = []
            if sid:
                slug = re.sub(r"[^a-z0-9]+", "_", sid.lower()).strip("_")
                if slug:
                    entity_keys.append(f"site:{slug}")
            atom_id = stable_id(
                "atm", artifact_id, "docx_site_roster",
                table_index, row_index, canon_id
            )
            src = SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.docx,
                filename=filename,
                locator={
                    "table_index": table_index,
                    "row": row_index,
                    "extraction": "docx_site_roster_v1",
                },
                extraction_method="docx_site_roster_v1",
                parser_version=self.parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    # v53.2 ROOT-CAUSE FIX: physical_site (not entity).
                    atom_type=AtomType.physical_site,
                    raw_text=row_text,
                    normalized_text=row_text.lower(),
                    value={
                        "kind": "physical_site",
                        "id": sid or site_row.site_id,
                        "site_id": sid or site_row.site_id,
                        "name": site_row.facility_name,
                        "facility_name": site_row.facility_name,
                        "address": site_row.street_address,
                        "street_address": site_row.street_address,
                        "mdf_idf": site_row.mdf_idf,
                        "access_window": site_row.access_window,
                        "escort_owner": site_row.escort_owner,
                        "contact": site_row.contact,
                        "phone": site_row.phone,
                        "email": site_row.email,
                        "city_state": site_row.city_state,
                        "zip": site_row.zip,
                        "notes": site_row.notes,
                        "extras": dict(site_row.extra_fields),
                    },
                    entity_keys=sorted(set(entity_keys)),
                    source_refs=[src],
                    receipts=[],
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=site_row.confidence,
                    confidence_raw=site_row.confidence,
                    calibrated_confidence=site_row.confidence,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
        return out

    def _build_structured_doc(
        self,
        *,
        filename: str,
        document: Any,
    ) -> dict[str, Any]:
        """Build a structured doc from the DOCX, walking the body element
        sequentially so paragraphs, bullets, and tables stay in source
        order.  Headings open new sections; consecutive bullets fuse
        into a bullet_list block.
        """
        sections: list[dict[str, Any]] = []
        current_blocks: list[dict[str, Any]] = []
        current_heading = filename
        current_level = 1
        pending_bullets: list[dict[str, Any]] = []

        def flush_bullets() -> None:
            nonlocal pending_bullets
            if pending_bullets:
                current_blocks.append(make_bullet_list(items=pending_bullets))
                pending_bullets = []

        def flush_section() -> None:
            nonlocal current_blocks, current_heading, current_level
            flush_bullets()
            if current_blocks or current_heading:
                sections.append(
                    make_section(
                        heading=current_heading,
                        level=current_level,
                        blocks=current_blocks,
                    )
                )
            current_blocks = []

        for paragraph in document.paragraphs:
            text = (paragraph.text or "").strip()
            style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
            is_heading = style_name.startswith("heading")
            is_list = "list" in style_name or "bullet" in style_name
            if is_heading and text:
                flush_section()
                current_heading = text
                # Heading 1 -> level 2, Heading 2 -> level 3, etc.
                m = re.search(r"\d+", style_name)
                level = (int(m.group()) + 1) if m else 2
                current_level = max(2, min(level, 6))
                continue
            if not text:
                continue
            if is_list:
                pending_bullets.append({"text": text, "children": []})
            else:
                flush_bullets()
                current_blocks.append(make_paragraph(text))

        for table in document.tables:
            try:
                cells = [[(c.text or "").strip() for c in row.cells] for row in table.rows]
            except Exception:
                continue
            if not cells:
                continue
            columns = cells[0] or [f"col_{i + 1}" for i in range(len(cells[0]))]
            columns = [c or f"col_{i + 1}" for i, c in enumerate(columns)]
            rows: list[dict[str, Any]] = []
            for raw in cells[1:]:
                if all(not v for v in raw):
                    continue
                rows.append({columns[i]: raw[i] if i < len(raw) else "" for i in range(len(columns))})
            flush_bullets()
            current_blocks.append(make_table(columns=columns, rows=rows))

        flush_section()

        if not sections:
            sections.append(
                make_section(
                    heading=filename,
                    level=2,
                    blocks=[make_paragraph("(empty document)")],
                )
            )
        page = make_page(page=0, title=filename, sections=sections)
        return make_structured_document(
            schema_version=STRUCTURED_SCHEMA_DOCX,
            filename=filename,
            artifact_type=ArtifactType.docx.value,
            title=filename,
            metadata=[],
            pages=[page],
        )

    def _extract_comment_atoms(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        path: Path,
    ) -> list[EvidenceAtom]:
        """Pull Word comments out of word/comments.xml. Each comment
        becomes a low-confidence scope_item atom flagged so the
        reviewer can decide whether the side-note is in-scope."""
        atoms: list[EvidenceAtom] = []
        try:
            with zipfile.ZipFile(path) as zf:
                if "word/comments.xml" not in zf.namelist():
                    return []
                xml_raw = zf.read("word/comments.xml")
        except Exception:
            return []
        try:
            root = ET.fromstring(xml_raw)
        except Exception:
            return []
        for idx, node in enumerate(root.findall(".//w:comment", WORD_NS)):
            text_parts = [t.text for t in node.findall(".//w:t", WORD_NS) if t.text]
            text = " ".join(part.strip() for part in text_parts if part.strip()).strip()
            if not text:
                continue
            author = node.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}author", "")
            initials = node.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}initials", "")
            comment_id = node.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id", str(idx))
            atom_text = text if not author else f"[Comment by {author}] {text}"
            atom_id = stable_id("atm", project_id, artifact_id, "docx_comment", comment_id, atom_text)
            src = SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.docx,
                filename=filename,
                locator={
                    "comment_id": comment_id,
                    "author": author,
                    "initials": initials,
                    "extraction": "docx_comment_v1",
                },
                extraction_method="docx_comment_v1",
                parser_version=self.parser_version,
            )
            atoms.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.open_question,
                    raw_text=atom_text,
                    normalized_text=atom_text.lower(),
                    value={
                        "kind": "docx_comment",
                        "author": author,
                        "initials": initials,
                        "comment_id": comment_id,
                    },
                    entity_keys=[],
                    source_refs=[src],
                    receipts=[],
                    authority_class=AuthorityClass.meeting_note,
                    confidence=0.55,
                    confidence_raw=0.55,
                    calibrated_confidence=0.55,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["docx_sidebar_comment"],
                    parser_version=self.parser_version,
                )
            )
        return atoms

    def _extract_tracked_change_atoms(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        path: Path,
    ) -> list[EvidenceAtom]:
        atoms: list[EvidenceAtom] = []
        with zipfile.ZipFile(path) as zf:
            xml_raw = zf.read("word/document.xml")
        root = ET.fromstring(xml_raw)

        for idx, node in enumerate(root.findall(".//w:del", WORD_NS)):
            text_parts = [t.text for t in node.findall(".//w:delText", WORD_NS) if t.text]
            if not text_parts:
                text_parts = [t.text for t in node.findall(".//w:t", WORD_NS) if t.text]
            text = " ".join(part.strip() for part in text_parts if part.strip()).strip()
            if not text:
                continue
            atoms.extend(
                self._emit_atoms_for_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    text=text,
                    paragraph_index=None,
                    table_index=None,
                    row=None,
                    cell=None,
                    tracked_change="deleted",
                    heading=False,
                    tracked_index=idx,
                )
            )

        for idx, node in enumerate(root.findall(".//w:ins", WORD_NS)):
            text_parts = [t.text for t in node.findall(".//w:t", WORD_NS) if t.text]
            text = " ".join(part.strip() for part in text_parts if part.strip()).strip()
            if not text:
                continue
            atoms.extend(
                self._emit_atoms_for_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    text=text,
                    paragraph_index=None,
                    table_index=None,
                    row=None,
                    cell=None,
                    tracked_change="inserted",
                    heading=False,
                    tracked_index=idx,
                )
            )
        return atoms

    def _extract_entity_keys(self, text: str) -> list[str]:
        lowered = normalize_text(text)
        keys: list[str] = []
        if "main campus" in lowered:
            keys.append(normalize_entity_key("site", "Main Campus"))
        if "west wing" in lowered:
            keys.append(normalize_entity_key("site", "West Wing"))
        if "camera" in lowered:
            keys.append(normalize_entity_key("device", "IP Cameras"))
        return keys

    def _classify_text(self, text: str) -> list[AtomType]:
        lowered = normalize_text(text)
        atom_types: list[AtomType] = []
        if any(re.search(pattern, lowered) for pattern in SCOPE_PATTERNS):
            atom_types.append(AtomType.scope_item)
        if any(re.search(pattern, lowered) for pattern in EXCLUSION_PATTERNS):
            atom_types.append(AtomType.exclusion)
        if any(re.search(pattern, lowered) for pattern in CONSTRAINT_PATTERNS):
            atom_types.append(AtomType.constraint)
        if any(re.search(pattern, lowered) for pattern in ASSUMPTION_PATTERNS):
            atom_types.append(AtomType.assumption)
        if "?" in text:
            atom_types.append(AtomType.open_question)
        # Stakeholder / signature-block pattern: a role keyword
        # adjacent to a proper-noun name. Emit a scope_item so the
        # downstream stakeholder extractor sees the name in atom
        # entity_keys. Without this, signature paragraphs like
        # "OPTBOT - Director of Workplace Technology: Jane Roe"
        # produce zero atoms despite carrying a load-bearing
        # stakeholder.
        if not atom_types and re.search(
            r"\b(?:director|vp|cio|cto|cfo|coo|ceo|president|"
            r"manager|architect|engineer|owner|sponsor|"
            r"program\s+manager|project\s+manager|principal|"
            r"foreman|superintendent|approver|signatory|"
            r"officer|administrator|consultant)\b",
            lowered,
        ):
            # Must also contain at least one likely name token
            # (Capitalized word, 2+ chars).
            if re.search(r"\b[A-Z][a-z]{1,}\b", text):
                atom_types.append(AtomType.scope_item)
        return atom_types

    @staticmethod
    def _is_substantive_prose(text: str) -> bool:
        """Whether a paragraph is load-bearing narrative prose worth keeping
        even when it matches none of the scope/exclusion/constraint patterns.

        The lexical classifier is a *type hint*, not a keep/drop gate: an SOW
        overview sentence ("The customer requires onsite field services to
        replace approximately 110 existing TVs across 23 dwellings...") carries
        the deal's headline facts yet uses no scope verb. Dropping it silently
        loses data. This is a universal, content-derived signal — a real
        multi-word sentence — not a keyword whitelist. Downstream semantic
        classification and dedup refine the type and prune true boilerplate."""
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        if len(words) < 5:
            return False  # headings, labels, fragments — not a sentence
        # Require sentence-like shape: ends with terminal punctuation OR
        # carries a concrete fact (a digit: quantity / date / money / count).
        has_terminal = text.rstrip().endswith((".", "!", ";", ":"))
        has_number = bool(re.search(r"\d", text))
        return has_terminal or has_number

    def _emit_atoms_for_text(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        text: str,
        paragraph_index: int | None,
        table_index: int | None,
        row: int | None,
        cell: int | None,
        tracked_change: str | None,
        heading: bool,
        tracked_index: int | None = None,
    ) -> list[EvidenceAtom]:
        atom_types = self._classify_text(text)
        # Fail OPEN, not closed: a paragraph that matches no lexical pattern
        # but is substantive narrative prose is captured as a scope_item
        # (lower confidence, flagged) so no load-bearing fact is silently
        # dropped at parse time. Headings stay out of the fallback — they are
        # short titles, not facts.
        prose_fallback = False
        if not atom_types:
            if not heading and self._is_substantive_prose(text):
                atom_types = [AtomType.scope_item]
                prose_fallback = True
            else:
                return []
        locator = {
            "paragraph_index": paragraph_index,
            "table_index": table_index,
            "row": row,
            "cell": cell,
            "tracked_change": tracked_change,
        }
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, paragraph_index, table_index, row, cell, tracked_change, tracked_index),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.docx,
            filename=filename,
            locator=locator,
            extraction_method="docx_text_and_ooxml",
            parser_version=self.parser_version,
        )

        atoms: list[EvidenceAtom] = []
        for atom_type in atom_types:
            authority_class = AuthorityClass.contractual_scope if heading else AuthorityClass.meeting_note
            review_status = ReviewStatus.auto_accepted
            review_flags: list[str] = []
            confidence = 0.84
            if tracked_change == "deleted":
                authority_class = AuthorityClass.deleted_text
                review_status = ReviewStatus.rejected
                review_flags = ["tracked_change_deleted_text"]
                confidence = 0.2
            elif tracked_change == "inserted":
                confidence = 0.72
            elif prose_fallback:
                # Captured by fail-open prose rule, not a lexical match:
                # lower confidence and flag so downstream semantic stages
                # reclassify/prune. Provenance preserved, data not lost.
                confidence = 0.5
                review_flags = ["prose_fallback_capture"]
            atoms.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm",
                        project_id,
                        artifact_id,
                        atom_type.value,
                        text,
                        paragraph_index,
                        table_index,
                        row,
                        cell,
                        tracked_change,
                        tracked_index,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=text,
                    normalized_text=normalize_text(text),
                    value={"text": text, "tracked_change": tracked_change, "prose_fallback": prose_fallback},
                    entity_keys=self._extract_entity_keys(text),
                    source_refs=[source_ref],
                    authority_class=authority_class,
                    confidence=confidence,
                    review_status=review_status,
                    review_flags=review_flags,
                    parser_version=self.parser_version,
                )
            )
        return atoms
