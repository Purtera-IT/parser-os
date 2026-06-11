from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph as _DocxParagraph

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
WORD_TBL_TAG = f"{{{WORD_NS['w']}}}tbl"

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
        # Universal reading-order section map: every paragraph/table learns the
        # heading chain it lives under, so site/section attribution has real
        # signal instead of empty section_path.
        para_section, table_section, _heading_paras, para_order, table_order = self._build_section_index(document)

        # v54 ROOT-CAUSE FIX: skip paragraphs that belong to a table cell so
        # their content is handled by the table loop below with proper row
        # structure and column context preserved (the original v48 intent).
        #
        # The previous implementation precomputed a set of id(paragraph) for
        # every table-cell paragraph and skipped main-loop paragraphs whose
        # id() was in that set. That is fundamentally broken: python-docx
        # builds throwaway Paragraph proxy objects on every access and they are
        # garbage-collected immediately, so the main loop's freshly-allocated
        # proxies reused the same memory addresses → id() collisions → real
        # body paragraphs (e.g. the SOW overview carrying "~110 TVs") were
        # silently skipped as if they were table cells. We now decide table
        # membership STRUCTURALLY by walking the lxml element's ancestors,
        # which is stable and correct across python-docx versions.
        for idx, paragraph in enumerate(document.paragraphs):
            if self._paragraph_in_table(paragraph):
                continue
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
            is_heading = style_name.startswith("heading")
            is_list_item = ("list" in style_name) or ("bullet" in style_name)
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
                    is_list_item=is_list_item,
                    section_path=para_section.get(idx, []),
                )
            )

        # Build all-document text once for ``kind=physical_site`` declarations.
        # Exclude table-cell paragraphs so the surrounding-text heuristic stays
        # accurate (otherwise table text bleeds in twice).
        document_text = " ".join(
            (p.text or "").strip()
            for p in document.paragraphs
            if not self._paragraph_in_table(p)
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
                        locator={"table_index": table_idx, "row": row_idx, "extraction": "raw_table_row_v49_2", "section_path": table_section.get(table_idx, [])},
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
                        "section_path": table_section.get(table_idx, []),
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

        # Never-detected recovery: pull in content that lives inside content
        # controls / textboxes (invisible to Document.paragraphs/.tables) and
        # mark every embedded binary region. Dedup against what we already
        # emitted so AlternateContent fallbacks don't double-count.
        already_emitted = {
            a.normalized_text for a in atoms if getattr(a, "normalized_text", "")
        }
        atoms.extend(
            self._recover_nested_region_atoms(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                document=document,
                already_emitted=already_emitted,
            )
        )
        atoms.extend(
            self._emit_embedded_media_markers(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                path=path,
            )
        )

        # Reading-order sort: python-docx yields all paragraphs then all tables,
        # so a table that sits right under a heading in the document otherwise
        # surfaces far down the atom list. Re-order atoms to true document order
        # using the body sequence computed in _build_section_index. Stable sort
        # preserves the emission sub-order within one paragraph/table (e.g. the
        # raw_table_row alongside its row blob) and parks order-less atoms
        # (comments, tracked changes, recovered/embedded regions) at the end.
        def _body_key(a: Any) -> tuple[int, int]:
            try:
                refs = getattr(a, "source_refs", None) or []
                loc = (getattr(refs[0], "locator", None) or {}) if refs else {}
            except Exception:
                loc = {}
            pi = loc.get("paragraph_index") if isinstance(loc, dict) else None
            ti = loc.get("table_index") if isinstance(loc, dict) else None
            if pi is not None and pi in para_order:
                return (para_order[pi], 0)
            if ti is not None and ti in table_order:
                return (table_order[ti], 1)
            return (10**9, 2)
        atoms.sort(key=_body_key)

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
    def _paragraph_in_table(paragraph: Any) -> bool:
        """Whether a python-docx paragraph lives inside a table cell.

        Decided STRUCTURALLY by walking the underlying lxml element's
        ancestors for a ``<w:tbl>`` tag. This replaces an ``id(paragraph)``
        membership test that was unsound: python-docx creates throwaway
        Paragraph proxy objects on every access and they are GC'd at once, so
        ``id()`` values get reused across the table scan and the main loop,
        causing false-positive matches that silently dropped body paragraphs.
        Ancestry never collides and is stable across python-docx versions.
        """
        el = getattr(paragraph, "_p", None)
        if el is None:
            return False
        parent = el.getparent()
        while parent is not None:
            if parent.tag == WORD_TBL_TAG:
                return True
            parent = parent.getparent()
        return False

    @staticmethod
    def _span_id(artifact_id: str, paragraph_index, table_index, row, cell, tracked_change, tracked_index) -> str:
        """Stable id for a raw source unit, used by the span-provenance ledger."""
        if tracked_change is not None:
            return f"{artifact_id}:{tracked_change}{tracked_index}"
        if table_index is not None:
            return f"{artifact_id}:t{table_index}.r{row}.c{cell}"
        return f"{artifact_id}:p{paragraph_index}"

    @staticmethod
    def _prose_drop_reason(text: str, *, heading: bool) -> str:
        """One-line diagnosis for why the prose GATE dropped a paragraph.

        Doubles as the fix pointer in the lost-content report: it names the
        exact condition in ``_is_substantive_prose`` that rejected the line.
        """
        if heading:
            return "heading/label fragment, no scope/exclusion verb"
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        if len(words) < 5:
            return f"short fragment ({len(words)} words), no numeric fact"
        # 5+ word lines now fail open (kept as prose_fallback), so a GATE drop
        # of a multi-word line should not occur; if it ever does, surface it.
        return "multi-word line unexpectedly dropped at prose gate"

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
        has_number = bool(re.search(r"\d", text))
        if len(words) < 5:
            # Short fragment. Keep it ONLY when it states a concrete numeric
            # fact with a little surrounding context, e.g. a "label: value"
            # line like "Estimated quantity: 110 units" or "Project duration:
            # 2 weeks". These carry load-bearing deal facts yet are too short
            # to be a sentence. Bare headings/labels ("PROJECT OVERVIEW",
            # "Name:") have no digit and are dropped; a lone number with no
            # words ("110") lacks context and is dropped. This is a
            # content-derived signal, not a keyword whitelist.
            return has_number and len(words) >= 2
        # 5+ real words -> KEEP. A multi-word line is load-bearing prose: SOW
        # narrative, an exclusion bullet ("Procurement or supply of TVs,
        # mounts, brackets..."), a tooling list item ("Battery powered drill
        # or impact driver"). The deterministic GATE must NOT silently kill
        # these just because they lack terminal punctuation or a digit —
        # exclusion lists ("what's NOT in scope") and equipment lists are
        # exactly the load-bearing facts a PM needs. We fail OPEN here and let
        # the learnable decide() SEAM (semantic dedup + boilerplate drop) prune
        # true boilerplate, where a human correction can teach it. That turns a
        # silent recall loss into a visible, learnable decision. Universal,
        # content-derived (a multi-word line), not a keyword whitelist.
        return True

    @staticmethod
    def _heading_level(style_name: str | None) -> int | None:
        """Heading depth from a paragraph style name, layout-agnostic.
        'Heading 1'->1, 'Heading 2'->2, 'Title'->0; non-headings -> None."""
        s = (style_name or "").strip().lower()
        if s in ("title", "subtitle"):
            return 0
        if s.startswith("heading"):
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else 1
        return None

    def _build_section_index(
        self, document: Any
    ) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[int, tuple[int, list[str]]]]:
        """ONE reading-order pass over the document body, maintaining a heading
        stack, so EVERY paragraph and table knows the section (heading chain) it
        lives under — universal across any layout (heading→table, heading→prose→
        table, nested sub-headings, etc.). python-docx exposes paragraphs and
        tables as separate sequences, which is exactly why section context was
        being lost; walking ``body.iterchildren()`` restores true document order.

        Returns ``(para_section, table_section, heading_paras)``:
          - para_section[i]   = section_path for the i-th body paragraph
          - table_section[i]  = section_path for the i-th table
          - heading_paras[i]  = (level, ancestor_section_path) for heading paras
        Indices align with ``document.paragraphs`` / ``document.tables`` order.
        """
        para_section: dict[int, list[str]] = {}
        table_section: dict[int, list[str]] = {}
        heading_paras: dict[int, tuple[int, list[str]]] = {}
        # global reading-order sequence per body paragraph / table, so atoms can
        # be emitted/sorted into true document order (python-docx otherwise yields
        # all paragraphs, then all tables — losing the interleave).
        para_order: dict[int, int] = {}
        table_order: dict[int, int] = {}
        stack: list[tuple[int, str]] = []  # (level, heading_text)
        pidx = -1
        tidx = -1
        seq = 0
        try:
            children = list(document.element.body.iterchildren())
        except Exception:
            return para_section, table_section, heading_paras, para_order, table_order
        for child in children:
            tag = child.tag
            if tag == qn("w:p"):
                pidx += 1
                para_order[pidx] = seq
                seq += 1
                try:
                    para = _DocxParagraph(child, document)
                    text = (para.text or "").strip()
                    style = (para.style.name or "") if para.style is not None else ""
                except Exception:
                    para_section[pidx] = [t for _, t in stack]
                    continue
                lvl = self._heading_level(style)
                if lvl is not None and text:
                    while stack and stack[-1][0] >= lvl:
                        stack.pop()
                    ancestors = [t for _, t in stack]
                    heading_paras[pidx] = (lvl, ancestors)
                    para_section[pidx] = ancestors
                    stack.append((lvl, text))
                else:
                    para_section[pidx] = [t for _, t in stack]
            elif tag == qn("w:tbl"):
                tidx += 1
                table_order[tidx] = seq
                seq += 1
                table_section[tidx] = [t for _, t in stack]
        return para_section, table_section, heading_paras, para_order, table_order

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
        is_list_item: bool = False,
        section_path: list[str] | None = None,
    ) -> list[EvidenceAtom]:
        # Span-provenance ledger (passive side-channel; only active when a
        # ledger is attached). Register this raw unit so the lost-content
        # report can attribute any drop to this exact GATE.
        ledger = getattr(self, "_ledger", None)
        span_id = self._span_id(
            artifact_id, paragraph_index, table_index, row, cell, tracked_change, tracked_index
        )
        if ledger is not None:
            ledger.register_span(span_id, text)

        atom_types = self._classify_text(text)
        # Fail OPEN, not closed: a paragraph that matches no lexical pattern
        # but is substantive narrative prose is captured as a scope_item
        # (lower confidence, flagged) so no load-bearing fact is silently
        # dropped at parse time. Headings stay out — they are NOT facts, they
        # are structure: their text is preserved as section_path on every child
        # atom beneath them, so a heading never needs to be its own (content)
        # atom. (Matches the PDF parser, which treats headings as section
        # context only.)
        prose_fallback = False
        if not atom_types:
            # Bullet list items are deliberate, load-bearing content (deliverables,
            # assumptions, checklists) — fail OPEN regardless of length, even when
            # they are too short to pass the substantive-prose heuristic. Otherwise
            # a 3-word bullet like "Hardware order tracking" is silently dropped.
            if not heading and (self._is_substantive_prose(text) or is_list_item):
                atom_types = [AtomType.scope_item]
                prose_fallback = True
            else:
                if ledger is not None:
                    from app.core.span_ledger import StageKind

                    ledger.record_drop(
                        span_id=span_id,
                        stage="docx_parse.prose_gate",
                        kind=StageKind.GATE,
                        rule="_is_substantive_prose",
                        reason=self._prose_drop_reason(text, heading=heading),
                        raw_text=text,
                        artifact=artifact_id,
                    )
                return []
        locator = {
            "paragraph_index": paragraph_index,
            "table_index": table_index,
            "row": row,
            "cell": cell,
            "tracked_change": tracked_change,
            "section_path": list(section_path) if section_path else [],
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
        if ledger is not None and atoms:
            ledger.mark_represented(span_id)
        return atoms

    # -- never-detected recovery: content python-docx's body view can't see --
    def _recover_nested_region_atoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        document: Any,
        already_emitted: set[str],
    ) -> list[EvidenceAtom]:
        """Recover paragraphs/tables that live inside content controls
        (``w:sdt``) or textboxes (``w:txbxContent``).

        ``Document.paragraphs`` / ``Document.tables`` only enumerate the body's
        *direct* ``w:p`` / ``w:tbl`` children, so anything nested under an sdt
        or a textbox (executive summaries, revision histories, sales/customer
        contact blocks) is silently invisible to the main parse loop — the
        *never-detected* loss class. We walk the body lxml tree, find those
        orphaned regions, and emit atoms for them:

        * orphan **tables** -> one ``scope_item`` per row, emitted
          UNCONDITIONALLY (mirroring the main table-row path) so contact rows
          with no scope verb / digit survive the prose gate that would
          otherwise drop them;
        * orphan **paragraphs** -> routed through ``_emit_atoms_for_text``.

        Dedup is by normalized text against ``already_emitted`` so VML/DrawingML
        ``mc:AlternateContent`` fallbacks (the same textbox stored twice) don't
        double-count. This is a structural, modality-agnostic walk — no
        keyword lists, no per-deal tuning.
        """
        body = getattr(getattr(document, "element", None), "body", None)
        if body is None:
            return []

        def _localname(tag: Any) -> str:
            t = str(tag)
            return t.rsplit("}", 1)[-1] if "}" in t else t

        def _nested_ancestors(el: Any) -> set[str]:
            out: set[str] = set()
            try:
                ancestors = el.iterancestors()
            except Exception:  # pragma: no cover - non-lxml element
                return out
            for a in ancestors:
                out.add(_localname(a.tag))
            return out

        def _el_text(el: Any) -> str:
            parts = [t.text for t in el.iter(f"{{{WORD_NS['w']}}}t") if t.text]
            return " ".join(parts).strip()

        ledger = getattr(self, "_ledger", None)
        atoms: list[EvidenceAtom] = []
        tbl_tag = f"{{{WORD_NS['w']}}}tbl"
        p_tag = f"{{{WORD_NS['w']}}}p"
        tr_tag = f"{{{WORD_NS['w']}}}tr"
        tc_tag = f"{{{WORD_NS['w']}}}tc"

        base_tbl = len(document.tables)
        base_p = len(document.paragraphs)

        # --- orphan tables (e.g. the SALES / CUSTOMER CONTACTS blocks) ------
        orphan_tbl_idx = 0
        for tbl in body.iter(tbl_tag):
            anc = _nested_ancestors(tbl)
            if not ({"sdtContent", "txbxContent"} & anc):
                continue
            if "tbl" in anc:  # nested table; outer one already handled it
                continue
            for row_idx, tr in enumerate(tbl.iter(tr_tag)):
                cells = [_el_text(tc) for tc in tr.iter(tc_tag)]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                row_text = " | ".join(cells)
                norm = normalize_text(row_text)
                if norm in already_emitted:
                    continue
                already_emitted.add(norm)
                t_index = base_tbl + orphan_tbl_idx
                span_id = self._span_id(artifact_id, None, t_index, row_idx, None, None, None)
                if ledger is not None:
                    ledger.register_span(span_id, row_text)
                atom_id = stable_id("atm", artifact_id, "docx_nested_row", t_index, row_idx, row_text)
                src = SourceRef(
                    id=stable_id("src", atom_id),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.docx,
                    filename=filename,
                    locator={
                        "table_index": t_index,
                        "row": row_idx,
                        "extraction": "docx_nested_region_v1",
                        "region": "sdt" if "sdtContent" in anc else "textbox",
                    },
                    extraction_method="docx_nested_region_v1",
                    parser_version=self.parser_version,
                )
                atoms.append(EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.scope_item,
                    raw_text=row_text,
                    normalized_text=norm,
                    value={
                        "kind": "table_row",
                        "region": "sdt" if "sdtContent" in anc else "textbox",
                        "cells": {f"col_{i}": v for i, v in enumerate(cells)},
                    },
                    entity_keys=self._extract_entity_keys(row_text),
                    source_refs=[src],
                    receipts=[],
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=0.7,
                    confidence_raw=0.7,
                    calibrated_confidence=0.7,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=["recovered_nested_region"],
                    parser_version=self.parser_version,
                ))
                if ledger is not None:
                    ledger.mark_represented(span_id)
            orphan_tbl_idx += 1

        # --- orphan paragraphs (executive summary, revision notes, etc.) ----
        orphan_p_idx = 0
        for p in body.iter(p_tag):
            anc = _nested_ancestors(p)
            if not ({"sdtContent", "txbxContent"} & anc):
                continue
            if "tbl" in anc:  # already covered by its table region above
                continue
            txt = _el_text(p)
            if not txt:
                continue
            norm = normalize_text(txt)
            if norm in already_emitted:
                continue
            already_emitted.add(norm)
            recovered = self._emit_atoms_for_text(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                text=txt,
                paragraph_index=base_p + orphan_p_idx,
                table_index=None,
                row=None,
                cell=None,
                tracked_change=None,
                heading=False,
            )
            for a in recovered:
                a.review_flags = list(a.review_flags) + ["recovered_nested_region"]
            atoms.extend(recovered)
            orphan_p_idx += 1

        return atoms

    def _emit_embedded_media_markers(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        path: Path,
    ) -> list[EvidenceAtom]:
        """Emit a located ``image_marker`` / ``embedded_object_marker`` atom for
        every binary part in the .docx zip (``word/media/*``,
        ``word/embeddings/*``).

        Detection of a binary region is *total and guaranteed* even when we
        cannot yet read it: the region never silently vanishes — it becomes a
        located marker the PM sees ("image/object here, vision/OLE pass
        required"). Extraction quality is a separate, improving frontier. The
        ``region_ref`` in each marker's value lets the content census reconcile
        the region as MARKED rather than UNCOVERED.
        """
        atoms: list[EvidenceAtom] = []
        try:
            zf = zipfile.ZipFile(path)
        except Exception:  # pragma: no cover - not a zip / unreadable
            return []
        with zf:
            for name in sorted(zf.namelist()):
                if name.startswith("word/media/"):
                    kind, atype = "image_marker", "image"
                elif name.startswith("word/embeddings/"):
                    kind, atype = "embedded_object_marker", "embedded object"
                else:
                    continue
                rel = name[len("word/"):]
                try:
                    size = zf.getinfo(name).file_size
                except KeyError:  # pragma: no cover
                    size = 0
                marker_text = (
                    f"[{atype.capitalize()} awaiting OCR / vision / OLE extraction] "
                    f"{rel} in {filename} — {size:,} bytes. A vision or embedded-"
                    f"object pass is required to recover its content."
                )
                atom_id = stable_id("atm", artifact_id, "docx_media_marker", rel)
                src = SourceRef(
                    id=stable_id("src", atom_id),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.docx,
                    filename=filename,
                    locator={"region_ref": rel, "extraction": "docx_media_marker_v1"},
                    extraction_method="docx_media_marker_v1",
                    parser_version=self.parser_version,
                )
                atoms.append(EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.open_question,
                    raw_text=marker_text,
                    normalized_text=normalize_text(marker_text),
                    value={"kind": kind, "region_ref": rel, "size_bytes": size},
                    entity_keys=[],
                    source_refs=[src],
                    receipts=[],
                    authority_class=AuthorityClass.meeting_note,
                    confidence=0.5,
                    confidence_raw=0.5,
                    calibrated_confidence=0.5,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["binary_region_marker"],
                    parser_version=self.parser_version,
                ))
        return atoms
