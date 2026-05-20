"""Universality wave 2: MSG / ODT / ODS / VSDX / MPP + ZIP recursive +
Slack/Teams HTML detection.

Same pattern as ``universal_parsers.py``: tolerant of missing
deps, emits marker atoms when a backend isn't installed so the
PM sees the file in source-inventory regardless.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserCapability,
    ParserMatch,
    ParserOutput,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser


def _make_atom(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    artifact_type: ArtifactType,
    text: str,
    locator: dict[str, Any],
    extraction_method: str,
    parser_version: str,
    atom_type: AtomType = AtomType.scope_item,
    authority_class: AuthorityClass = AuthorityClass.customer_current_authored,
    confidence: float = 0.85,
    value_extra: dict[str, Any] | None = None,
    review_status: ReviewStatus = ReviewStatus.auto_accepted,
) -> EvidenceAtom:
    src = SourceRef(
        id=stable_id("src", artifact_id, str(locator), text[:80]),
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        locator=locator,
        extraction_method=extraction_method,
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=stable_id("atm", project_id, artifact_id, text[:120], str(locator)),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value_extra or {},
        entity_keys=[],
        source_refs=[src],
        authority_class=authority_class,
        confidence=confidence,
        review_status=review_status,
        parser_version=parser_version,
    )


# ────────────────────────────── MSG (Outlook) ──────────────────────────────


class MsgParser(BaseParser):
    """Parse Outlook ``.msg`` native email files.

    Three-path strategy:
      1. ``extract_msg`` library when installed — full headers + body
         + attachments inventory.
      2. ``olefile`` stdlib path when extract_msg isn't installed —
         best-effort extraction of property streams.
      3. Marker atom + install hint when neither is available.
    """
    parser_name = "msg"
    parser_version = "msg_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".msg"],
        supported_artifact_types=[ArtifactType.msg],
        emitted_atom_types=[AtomType.scope_item, AtomType.open_question],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.95 if path.suffix.lower() == ".msg" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["msg_extension"] if confidence else [],
            artifact_type=ArtifactType.msg,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        atoms: list[EvidenceAtom] = []
        # Path 1: extract_msg
        try:
            import extract_msg
            with extract_msg.openMsg(str(path)) as msg:
                subject = (msg.subject or "").strip()
                sender = (msg.sender or "").strip()
                date = str(msg.date or "")
                body = (msg.body or "").strip()
                attachments = [att.longFilename or att.shortFilename or "(unnamed)" for att in (msg.attachments or [])]
            header_text = f"From: {sender} | Subject: {subject} | Date: {date}"
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.msg, text=header_text,
                locator={"kind": "msg_header"},
                extraction_method="extract_msg",
                parser_version=self.parser_version,
                value_extra={"subject": subject, "from": sender, "date": date},
            ))
            for para_idx, para in enumerate(re.split(r"\n\s*\n", body)):
                para = para.strip()
                if not para or len(para) < 4:
                    continue
                atoms.append(_make_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=path.name,
                    artifact_type=ArtifactType.msg, text=para[:1200],
                    locator={"kind": "msg_body", "paragraph": para_idx},
                    extraction_method="extract_msg",
                    parser_version=self.parser_version,
                ))
            if attachments:
                atoms.append(_make_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=path.name,
                    artifact_type=ArtifactType.msg,
                    text=f"Attachments referenced: {', '.join(attachments)}",
                    locator={"kind": "msg_attachments"},
                    extraction_method="extract_msg",
                    parser_version=self.parser_version,
                    value_extra={"attachment_names": attachments},
                ))
            return ParserOutput(atoms=atoms, derived_files=[])
        except ImportError:
            pass
        except Exception as exc:
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.msg,
                text=f"[MSG parser error: extract_msg path failed — {type(exc).__name__}: {exc}. Marker atom emitted; install / configure extract_msg.]",
                locator={"kind": "msg_error"},
                extraction_method="extract_msg",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
                review_status=ReviewStatus.needs_review,
            ))
            return ParserOutput(atoms=atoms, derived_files=[])

        # Path 2: olefile fallback
        try:
            import olefile
            with olefile.OleFileIO(str(path)) as ole:
                streams = ole.listdir(streams=True)
                # Find common property streams
                text_blobs: list[str] = []
                for stream in streams:
                    name = "/".join(stream)
                    if not any(tok in name for tok in ("__substg1", "001E", "001F")):
                        continue
                    try:
                        with ole.openstream(stream) as s:
                            data = s.read()
                        try:
                            decoded = data.decode("utf-16-le", errors="ignore")
                        except Exception:
                            decoded = data.decode("latin-1", errors="ignore")
                        decoded = decoded.replace("\x00", "").strip()
                        if decoded and len(decoded) > 4:
                            text_blobs.append(decoded[:1200])
                    except Exception:
                        continue
                if text_blobs:
                    for i, blob in enumerate(text_blobs[:40]):
                        atoms.append(_make_atom(
                            project_id=project_id, artifact_id=artifact_id, filename=path.name,
                            artifact_type=ArtifactType.msg, text=blob,
                            locator={"kind": "msg_olefile_stream", "stream_idx": i},
                            extraction_method="olefile_fallback",
                            parser_version=self.parser_version,
                            confidence=0.60,
                        ))
                    return ParserOutput(atoms=atoms, derived_files=[])
        except ImportError:
            pass
        except Exception:
            pass

        # Marker fallback
        atoms.append(_make_atom(
            project_id=project_id, artifact_id=artifact_id, filename=path.name,
            artifact_type=ArtifactType.msg,
            text=(
                f"[Outlook .msg file awaiting parser dependency] {path.name}. "
                f"Install ``extract_msg`` (`pip install extract-msg`) to enable full "
                f"header + body + attachment extraction. The file is recognized "
                f"and surfaced in the source inventory regardless."
            ),
            locator={"kind": "msg_marker"},
            extraction_method="msg_marker",
            parser_version=self.parser_version,
            atom_type=AtomType.open_question,
            review_status=ReviewStatus.needs_review,
        ))
        return ParserOutput(atoms=atoms, derived_files=[])


# ────────────────────────────── ODT / ODS (OpenDocument) ──────────────────────────────


# ODF XML namespaces
_ODF_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
}


def _odf_text_walk(elem: ET.Element) -> str:
    """Recursively pull all text from an ODF element."""
    out: list[str] = []
    if elem.text:
        out.append(elem.text)
    for child in elem:
        out.append(_odf_text_walk(child))
        if child.tail:
            out.append(child.tail)
    return "".join(out).strip()


class OdtParser(BaseParser):
    """Parse OpenDocument Text (.odt — LibreOffice / OpenOffice writer)."""
    parser_name = "odt"
    parser_version = "odt_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".odt"],
        supported_artifact_types=[ArtifactType.odt],
        emitted_atom_types=[
            AtomType.scope_item, AtomType.exclusion, AtomType.constraint,
            AtomType.assumption, AtomType.open_question,
        ],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.93 if path.suffix.lower() == ".odt" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["odt_extension"] if confidence else [],
            artifact_type=ArtifactType.odt,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        atoms: list[EvidenceAtom] = []
        try:
            with zipfile.ZipFile(path) as z:
                content = z.read("content.xml").decode("utf-8", errors="ignore")
        except Exception as exc:
            return ParserOutput(atoms=[_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.odt,
                text=f"[ODT parser error — could not open: {exc}]",
                locator={"kind": "odt_error"},
                extraction_method="odt_zip",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
            )], derived_files=[])
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            return ParserOutput(atoms=[_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.odt,
                text=f"[ODT parser error — XML parse failed: {exc}]",
                locator={"kind": "odt_xml_error"},
                extraction_method="odt_xml",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
            )], derived_files=[])

        # Headings
        for h_idx, h in enumerate(root.iter("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}h")):
            text = _odf_text_walk(h)
            if not text:
                continue
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.odt, text=text[:280],
                locator={"kind": "odt_heading", "index": h_idx},
                extraction_method="odt_xml",
                parser_version=self.parser_version,
                value_extra={"kind": "heading"},
            ))
        # Paragraphs
        for p_idx, p in enumerate(root.iter("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p")):
            text = _odf_text_walk(p)
            if not text or len(text) < 4:
                continue
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.odt, text=text[:1200],
                locator={"kind": "odt_paragraph", "index": p_idx},
                extraction_method="odt_xml",
                parser_version=self.parser_version,
            ))
        # Tables (rare in ODT but happens)
        for t_idx, t in enumerate(root.iter("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table")):
            for r_idx, row in enumerate(t.iter("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-row")):
                for c_idx, cell in enumerate(row.iter("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-cell")):
                    text = _odf_text_walk(cell)
                    if not text:
                        continue
                    atoms.append(_make_atom(
                        project_id=project_id, artifact_id=artifact_id, filename=path.name,
                        artifact_type=ArtifactType.odt, text=text[:400],
                        locator={"kind": "odt_table_cell", "table": t_idx, "row": r_idx, "cell": c_idx},
                        extraction_method="odt_xml",
                        parser_version=self.parser_version,
                    ))
        return ParserOutput(atoms=atoms, derived_files=[])


class OdsParser(BaseParser):
    """Parse OpenDocument Spreadsheet (.ods — LibreOffice Calc)."""
    parser_name = "ods"
    parser_version = "ods_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".ods"],
        supported_artifact_types=[ArtifactType.ods],
        emitted_atom_types=[AtomType.scope_item, AtomType.quantity],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.95 if path.suffix.lower() == ".ods" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["ods_extension"] if confidence else [],
            artifact_type=ArtifactType.ods,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        atoms: list[EvidenceAtom] = []
        try:
            with zipfile.ZipFile(path) as z:
                content = z.read("content.xml").decode("utf-8", errors="ignore")
            root = ET.fromstring(content)
        except Exception as exc:
            return ParserOutput(atoms=[_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.ods,
                text=f"[ODS parser error: {exc}]",
                locator={"kind": "ods_error"},
                extraction_method="ods_zip",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
            )], derived_files=[])

        for t_idx, table in enumerate(root.iter("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table")):
            sheet_name = table.get("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name", f"sheet_{t_idx}")
            for r_idx, row in enumerate(table.iter("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-row")):
                for c_idx, cell in enumerate(row.iter("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-cell")):
                    text = _odf_text_walk(cell)
                    if not text:
                        continue
                    atoms.append(_make_atom(
                        project_id=project_id, artifact_id=artifact_id, filename=path.name,
                        artifact_type=ArtifactType.ods, text=text[:400],
                        locator={"sheet": sheet_name, "row": r_idx, "cell": c_idx},
                        extraction_method="ods_xml",
                        parser_version=self.parser_version,
                        value_extra={"kind": "ods_cell", "sheet": sheet_name},
                    ))
        return ParserOutput(atoms=atoms, derived_files=[])


# ────────────────────────────── VSDX (Visio) ──────────────────────────────


class VsdxParser(BaseParser):
    """Parse Visio .vsdx — network diagrams + flowcharts.

    VSDX is a ZIP container with per-page XML. Extracts shape
    text + connection labels which are usually the most
    informative pieces for MSP intake (device labels, site
    names, connection types).
    """
    parser_name = "vsdx"
    parser_version = "vsdx_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".vsdx", ".vsd"],
        supported_artifact_types=[ArtifactType.vsdx],
        emitted_atom_types=[AtomType.scope_item, AtomType.entity, AtomType.open_question],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        confidence = 0.95 if suffix == ".vsdx" else (0.4 if suffix == ".vsd" else 0.0)
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=[f"vsdx_extension:{suffix}"] if confidence else [],
            artifact_type=ArtifactType.vsdx,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        if path.suffix.lower() == ".vsd":
            return ParserOutput(atoms=[_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.vsdx,
                text=(
                    f"[Legacy .vsd binary Visio file — re-save as .vsdx to enable "
                    f"shape/text extraction. Marker atom emitted; file surfaces "
                    f"in source inventory.]"
                ),
                locator={"kind": "vsd_legacy_marker"},
                extraction_method="vsd_marker",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
                review_status=ReviewStatus.needs_review,
            )], derived_files=[])
        atoms: list[EvidenceAtom] = []
        try:
            with zipfile.ZipFile(path) as z:
                page_files = [n for n in z.namelist() if re.match(r"visio/pages/page\d+\.xml", n)]
                for page_idx, page_file in enumerate(sorted(page_files)):
                    try:
                        content = z.read(page_file).decode("utf-8", errors="ignore")
                        root = ET.fromstring(content)
                    except Exception:
                        continue
                    # Visio XML namespace
                    vns = "{http://schemas.microsoft.com/office/visio/2012/main}"
                    page_name = root.get("Name", f"Page {page_idx + 1}")
                    for shape_idx, shape in enumerate(root.iter(vns + "Shape")):
                        # Shape Name attribute often contains the visible label
                        name_attr = shape.get("Name", "")
                        # Text element inside shape carries the label content
                        text_elem = shape.find(vns + "Text")
                        text_content = ""
                        if text_elem is not None:
                            text_content = _odf_text_walk(text_elem)
                        full_text = (text_content or name_attr).strip()
                        if not full_text or len(full_text) < 2:
                            continue
                        atoms.append(_make_atom(
                            project_id=project_id, artifact_id=artifact_id, filename=path.name,
                            artifact_type=ArtifactType.vsdx, text=full_text[:400],
                            locator={"page": page_name, "page_index": page_idx, "shape_index": shape_idx, "shape_name": name_attr},
                            extraction_method="vsdx_xml",
                            parser_version=self.parser_version,
                            atom_type=AtomType.entity,
                            value_extra={"kind": "visio_shape", "page": page_name},
                        ))
        except Exception as exc:
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.vsdx,
                text=f"[VSDX parser error: {exc}]",
                locator={"kind": "vsdx_error"},
                extraction_method="vsdx_xml",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
            ))
        return ParserOutput(atoms=atoms, derived_files=[])


# ────────────────────────────── MPP (MS Project) ──────────────────────────────


class MppParser(BaseParser):
    """Parse Microsoft Project .mpp — schedule files.

    .mpp is a proprietary binary format. Real parsing requires
    the mpxj library which has a Java runtime dependency. This
    parser emits a marker atom with a clear install hint so the
    PM sees the file in source inventory while a follow-up job
    pulls in mpxj for real extraction.
    """
    parser_name = "mpp"
    parser_version = "mpp_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".mpp"],
        supported_artifact_types=[ArtifactType.mpp],
        emitted_atom_types=[AtomType.open_question],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=False,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.95 if path.suffix.lower() == ".mpp" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["mpp_extension"] if confidence else [],
            artifact_type=ArtifactType.mpp,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        try:
            import mpxj  # type: ignore[import-not-found]
            # If mpxj is available, defer to its reader. The Python
            # binding requires Java; on most hosts this branch will
            # never fire. The fallback marker is still useful.
            project = mpxj.Project(str(path))
            atoms: list[EvidenceAtom] = []
            for t_idx, task in enumerate(project.tasks[:200]):
                atoms.append(_make_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=path.name,
                    artifact_type=ArtifactType.mpp,
                    text=f"Task {t_idx + 1}: {task.name} | start={task.start} end={task.finish}",
                    locator={"kind": "mpp_task", "task_index": t_idx},
                    extraction_method="mpxj",
                    parser_version=self.parser_version,
                    value_extra={"name": task.name, "start": str(task.start), "end": str(task.finish)},
                ))
            return ParserOutput(atoms=atoms, derived_files=[])
        except Exception:
            pass

        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        marker = (
            f"[MS Project .mpp file awaiting native reader] {path.name} "
            f"({size:,} bytes). Install ``mpxj`` (`pip install mpxj` — requires "
            f"a Java runtime) to enable task / milestone / resource extraction. "
            f"Until then, export the schedule from MS Project as XLSX or CSV "
            f"and re-attach to the intake for full coverage."
        )
        return ParserOutput(atoms=[_make_atom(
            project_id=project_id, artifact_id=artifact_id, filename=path.name,
            artifact_type=ArtifactType.mpp,
            text=marker,
            locator={"kind": "mpp_marker"},
            extraction_method="mpp_marker",
            parser_version=self.parser_version,
            atom_type=AtomType.open_question,
            review_status=ReviewStatus.needs_review,
        )], derived_files=[])
