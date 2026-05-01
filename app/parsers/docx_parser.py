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
    ReviewStatus,
    SourceRef,
    ParserCapability,
    ParserMatch,
)
from app.parsers.base import BaseParser
from app.parsers.segmenters import segment_docx
from app.domain.schemas import DomainPack

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
        del domain_pack
        document = Document(path)
        atoms: list[EvidenceAtom] = []

        for idx, paragraph in enumerate(document.paragraphs):
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

        for table_idx, table in enumerate(document.tables):
            for row_idx, row_cells in enumerate(table.rows):
                for cell_idx, cell_obj in enumerate(row_cells.cells):
                    text = cell_obj.text.strip()
                    if not text:
                        continue
                    atoms.extend(
                        self._emit_atoms_for_text(
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            text=text,
                            paragraph_index=None,
                            table_index=table_idx,
                            row=row_idx,
                            cell=cell_idx,
                            tracked_change=None,
                            heading=False,
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
        return atom_types

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
        if not atom_types:
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
                    value={"text": text, "tracked_change": tracked_change},
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
