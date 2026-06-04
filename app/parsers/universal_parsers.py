"""Universality wave: HTML / MBOX / RTF / ICS / ZIP parsers.

Each parser is tolerant of malformed input — they emit a marker
atom describing the failure rather than killing the run (A6
graceful per-file degradation handles the rest).

These are the real-world MSP intake formats parser-os was
missing after EML / VTT / SRT / PPTX / image coverage. Each
chooses the simplest dependency available so the parser stack
stays installable on any Python 3.11+ host without C
extensions.
"""
from __future__ import annotations

import email
import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
from app.parsers.binary_markers import attachment_marker, region_marker


# Shared atom-type heuristic — same families the docx/pptx parsers use
# so packetization sees these atoms in the same buckets.
_EXCLUSION_RE = re.compile(
    r"\b(out\s+of\s+scope|excluded?|not\s+included|"
    r"explicitly\s+excludes?|exclusion[s]?:)",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(must|shall|required?|requirement|after-?hours|escort|"
    r"badge|lift|compliance|regulatory)\b",
    re.IGNORECASE,
)
_ASSUMPTION_RE = re.compile(
    r"\b(assume[ds]?|assumption[s]?|we\s+assume|"
    r"customer\s+provides?|customer\s+supplies)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"\?$|^(?:question|tbd|open\s+question|to\s+confirm)\b",
    re.IGNORECASE,
)


def _classify(text: str) -> AtomType:
    if _EXCLUSION_RE.search(text):
        return AtomType.exclusion
    if _ASSUMPTION_RE.search(text):
        return AtomType.assumption
    if _QUESTION_RE.search(text):
        return AtomType.open_question
    if _CONSTRAINT_RE.search(text):
        return AtomType.constraint
    return AtomType.scope_item


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
    atom_type: AtomType | None = None,
    authority_class: AuthorityClass = AuthorityClass.customer_current_authored,
    confidence: float = 0.90,
    value_extra: dict[str, Any] | None = None,
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
        atom_type=atom_type or _classify(text),
        raw_text=text,
        normalized_text=text.lower(),
        value=value_extra or {},
        entity_keys=[],
        source_refs=[src],
        authority_class=authority_class,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


# ───────────────────────────────────── HTML ──────────────────────────────────────


class HtmlParser(BaseParser):
    """Parse Confluence / wiki / static-doc HTML exports.

    Strategy: BeautifulSoup walks the DOM and yields:
      * one atom per heading
      * one atom per paragraph / list item
      * one atom per table cell (so SOW-shaped tables in HTML get
        the same row-level treatment as DOCX tables)
    """
    parser_name = "html"
    parser_version = "html_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".html", ".htm", ".xhtml"],
        supported_artifact_types=[ArtifactType.html],
        emitted_atom_types=[
            AtomType.scope_item, AtomType.exclusion, AtomType.constraint,
            AtomType.assumption, AtomType.open_question,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        confidence = 0.93 if suffix in {".html", ".htm", ".xhtml"} else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=[f"html_extension:{suffix}"] if confidence else [],
            artifact_type=ArtifactType.html,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        try:
            from bs4 import BeautifulSoup
        except Exception as exc:
            raise RuntimeError("BeautifulSoup is required for the HTML parser") from exc
        html_bytes = path.read_bytes()
        soup = BeautifulSoup(html_bytes, "html.parser")
        # Drop script / style for noise.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        atoms: list[EvidenceAtom] = []

        # Slack / Teams chat export detection — preferred path. When the
        # DOM has Slack/Teams message structure, extract per-message
        # atoms with sender + timestamp so chat threads from customer
        # land as clean evidence instead of a sea of <div>s.
        if _looks_like_slack_export(soup):
            return ParserOutput(
                atoms=_extract_slack_messages(
                    soup, project_id=project_id, artifact_id=artifact_id,
                    filename=path.name, parser_version=self.parser_version,
                ),
                derived_files=[],
            )
        if _looks_like_teams_export(soup):
            return ParserOutput(
                atoms=_extract_teams_messages(
                    soup, project_id=project_id, artifact_id=artifact_id,
                    filename=path.name, parser_version=self.parser_version,
                ),
                derived_files=[],
            )
        # Headings (h1-h6)
        for level in range(1, 7):
            for h_idx, h in enumerate(soup.find_all(f"h{level}")):
                text = h.get_text(separator=" ", strip=True)
                if not text:
                    continue
                atoms.append(_make_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=path.name,
                    artifact_type=ArtifactType.html, text=text[:280],
                    locator={"tag": f"h{level}", "index": h_idx},
                    extraction_method="html_bs4",
                    parser_version=self.parser_version,
                    value_extra={"kind": "heading", "level": level},
                ))
        # Paragraphs + list items
        for tag_name in ("p", "li"):
            for idx, node in enumerate(soup.find_all(tag_name)):
                text = node.get_text(separator=" ", strip=True)
                if not text or len(text) < 4:
                    continue
                atoms.append(_make_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=path.name,
                    artifact_type=ArtifactType.html, text=text[:600],
                    locator={"tag": tag_name, "index": idx},
                    extraction_method="html_bs4",
                    parser_version=self.parser_version,
                    value_extra={"kind": tag_name},
                ))
        # Tables — emit one atom per cell with row + col locators
        for t_idx, table in enumerate(soup.find_all("table")):
            for r_idx, row in enumerate(table.find_all("tr")):
                for c_idx, cell in enumerate(row.find_all(["th", "td"])):
                    text = cell.get_text(separator=" ", strip=True)
                    if not text:
                        continue
                    atoms.append(_make_atom(
                        project_id=project_id, artifact_id=artifact_id, filename=path.name,
                        artifact_type=ArtifactType.html, text=text[:280],
                        locator={"table": t_idx, "row": r_idx, "cell": c_idx},
                        extraction_method="html_bs4",
                        parser_version=self.parser_version,
                        value_extra={"kind": "table_cell"},
                    ))
        # Mark referenced binary regions (img / iframe / object / embed) so an
        # embedded diagram or screenshot can't silently vanish. The region_ref
        # matches the census location (``media/<src>``).
        for ii, img in enumerate(soup.find_all(["img", "iframe", "object", "embed"])):
            ref = img.get("src") or img.get("data") or f"media{ii}"
            atoms.append(region_marker(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.html, parser_version=self.parser_version,
                region_ref=f"media/{ref}", kind="image_marker", label="image",
            ))
        return ParserOutput(atoms=atoms, derived_files=[])


# ───────────────────────────────────── MBOX ──────────────────────────────────────


class MboxParser(BaseParser):
    """Parse Gmail / Thunderbird MBOX archives — multi-message email file.

    Each message becomes its own group of atoms (same shape as
    EmailParser produces). Built on the stdlib ``mailbox`` module
    so there's no external dependency.
    """
    parser_name = "mbox"
    parser_version = "mbox_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".mbox"],
        supported_artifact_types=[ArtifactType.mbox],
        emitted_atom_types=[
            AtomType.scope_item, AtomType.action_item, AtomType.open_question,
            AtomType.decision, AtomType.constraint,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.95 if path.suffix.lower() == ".mbox" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["mbox_extension"] if confidence else [],
            artifact_type=ArtifactType.mbox,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        import mailbox
        atoms: list[EvidenceAtom] = []
        mb = mailbox.mbox(str(path))
        try:
            for msg_idx, msg in enumerate(mb):
                # Reduce each message to its raw text + minimal headers.
                subject = (msg.get("Subject") or "").strip()
                sender = (msg.get("From") or "").strip()
                date = (msg.get("Date") or "").strip()
                if subject or sender:
                    header_text = f"From: {sender} | Subject: {subject} | Date: {date}"
                    atoms.append(_make_atom(
                        project_id=project_id, artifact_id=artifact_id, filename=path.name,
                        artifact_type=ArtifactType.mbox, text=header_text,
                        locator={"message": msg_idx, "kind": "header"},
                        extraction_method="mbox_stdlib",
                        parser_version=self.parser_version,
                        atom_type=AtomType.scope_item,
                        value_extra={"kind": "email_header", "subject": subject, "from": sender, "date": date},
                    ))
                # Body
                body_text = _extract_email_body(msg)
                if body_text.strip():
                    # Split body by paragraphs / blank lines so each
                    # paragraph becomes its own atom.
                    for para_idx, para in enumerate(re.split(r"\n\s*\n", body_text)):
                        para = para.strip()
                        if not para or len(para) < 4:
                            continue
                        atoms.append(_make_atom(
                            project_id=project_id, artifact_id=artifact_id, filename=path.name,
                            artifact_type=ArtifactType.mbox, text=para[:600],
                            locator={"message": msg_idx, "kind": "body", "paragraph": para_idx},
                            extraction_method="mbox_stdlib",
                            parser_version=self.parser_version,
                        ))
                # Attachments — emit a located marker so a per-message
                # attachment can't silently vanish (census reconciles MARKED).
                try:
                    for part in msg.walk():
                        if part.get_content_maintype() == "multipart":
                            continue
                        fn = part.get_filename()
                        disp = (part.get("Content-Disposition") or "").lower()
                        if not fn and not disp.startswith("attachment"):
                            continue
                        name = fn or "(unnamed)"
                        payload = part.get_payload(decode=True)
                        atoms.append(attachment_marker(
                            project_id=project_id, artifact_id=artifact_id, filename=path.name,
                            artifact_type=ArtifactType.mbox, parser_version=self.parser_version,
                            attachment_name=name,
                            size=len(payload) if payload else 0,
                            content_type=part.get_content_type(),
                        ))
                except Exception:
                    pass
        finally:
            mb.close()
        return ParserOutput(atoms=atoms, derived_files=[])


def _extract_email_body(msg: email.message.Message) -> str:
    """Pull the best-effort text body from an email Message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                except Exception:
                    continue
        return ""
    try:
        return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore")
    except Exception:
        return str(msg.get_payload() or "")


# ───────────────────────────────────── RTF ──────────────────────────────────────


_RTF_CONTROL_RE = re.compile(r"\\[a-z]+(?:-?\d+)?\s?|\\[*'\\]|\{|\}", re.IGNORECASE)
_RTF_HEX_RE = re.compile(r"\\'([0-9a-fA-F]{2})")


def _strip_rtf(rtf_text: str) -> str:
    """Strip RTF control words / groups to plain text.

    Tolerant lossy stripper. Sufficient for most SOW / contract
    templates which are 95% prose + minimal formatting. For RTF
    with complex tables / images, a dedicated library would
    recover more.
    """
    # Hex escapes (\'AE → ®). Decode best-effort to ASCII.
    def _hex_sub(m: re.Match[str]) -> str:
        try:
            return bytes.fromhex(m.group(1)).decode("cp1252", errors="ignore")
        except Exception:
            return ""
    text = _RTF_HEX_RE.sub(_hex_sub, rtf_text)
    text = _RTF_CONTROL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RtfParser(BaseParser):
    """Parse Rich Text Format. Used for legacy SOW / contract
    templates that haven't been moved to DOCX yet."""
    parser_name = "rtf"
    parser_version = "rtf_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".rtf"],
        supported_artifact_types=[ArtifactType.rtf],
        emitted_atom_types=[
            AtomType.scope_item, AtomType.exclusion, AtomType.constraint,
            AtomType.assumption, AtomType.open_question,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.93 if path.suffix.lower() == ".rtf" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["rtf_extension"] if confidence else [],
            artifact_type=ArtifactType.rtf,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        raw = path.read_text(encoding="cp1252", errors="ignore")
        plain = _strip_rtf(raw)
        atoms: list[EvidenceAtom] = []
        for para_idx, para in enumerate(re.split(r"\.\s+(?=[A-Z])|[\r\n]+", plain)):
            para = para.strip()
            if len(para) < 8:
                continue
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.rtf, text=para[:600],
                locator={"paragraph": para_idx},
                extraction_method="rtf_regex_strip",
                parser_version=self.parser_version,
            ))
        return ParserOutput(atoms=atoms, derived_files=[])


# ───────────────────────────────────── ICS ──────────────────────────────────────


_ICS_VEVENT_BLOCK_RE = re.compile(r"BEGIN:VEVENT(.+?)END:VEVENT", re.DOTALL | re.IGNORECASE)
_ICS_FIELD_RE = re.compile(r"^([A-Z\-]+)(?:;[^:]*)?:(.+)$", re.MULTILINE)


class IcsParser(BaseParser):
    """Parse iCalendar (.ics) invites.

    Each VEVENT becomes a ``meeting_commitment``-class atom with
    structured fields (summary, dtstart, dtend, location, organizer,
    attendees). Common for kickoff meetings, design reviews,
    customer-acceptance walkthroughs.
    """
    parser_name = "ics"
    parser_version = "ics_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".ics", ".ical"],
        supported_artifact_types=[ArtifactType.ics],
        emitted_atom_types=[AtomType.meeting_commitment, AtomType.scope_item],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.95 if path.suffix.lower() in {".ics", ".ical"} else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["ics_extension"] if confidence else [],
            artifact_type=ArtifactType.ics,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # ICS line-folding: lines starting with a space are continuations.
        raw = re.sub(r"\r?\n[ \t]", "", raw)
        atoms: list[EvidenceAtom] = []
        for ev_idx, m in enumerate(_ICS_VEVENT_BLOCK_RE.finditer(raw)):
            block = m.group(1)
            fields = {fm.group(1).upper(): fm.group(2).strip() for fm in _ICS_FIELD_RE.finditer(block)}
            summary = fields.get("SUMMARY") or "(no subject)"
            start = fields.get("DTSTART") or ""
            end = fields.get("DTEND") or ""
            location = fields.get("LOCATION") or ""
            organizer = fields.get("ORGANIZER") or ""
            text = (
                f"Meeting: {summary} | "
                f"{start} → {end}"
                + (f" | at: {location}" if location else "")
                + (f" | organizer: {organizer}" if organizer else "")
            )
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.ics, text=text[:600],
                locator={"event": ev_idx, "kind": "vevent"},
                extraction_method="ics_text_parse",
                parser_version=self.parser_version,
                atom_type=AtomType.meeting_commitment,
                value_extra={
                    "kind": "calendar_event",
                    "summary": summary, "dtstart": start, "dtend": end,
                    "location": location, "organizer": organizer,
                },
            ))
        return ParserOutput(atoms=atoms, derived_files=[])


# ───────────────────────────────────── ZIP ──────────────────────────────────────


class ZipParser(BaseParser):
    """Auto-list ZIP archive contents.

    A ZIP arrives in many real intakes as the "entire deal folder"
    bundled by sales / customer. parser-os ingests ZIPs as
    artifacts and emits a directory listing as atoms so the PM
    sees what's inside even though the individual files aren't
    recursed (that would need a multi-stage discover pass).

    The PM_HANDOFF then shows the ZIP under "Source inventory
    read" with an explicit "extract this ZIP and re-run on the
    folder" hint in the marker.
    """
    parser_name = "zip"
    parser_version = "zip_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".zip"],
        supported_artifact_types=[ArtifactType.zip_archive],
        emitted_atom_types=[AtomType.open_question, AtomType.scope_item],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=False,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        confidence = 0.95 if path.suffix.lower() == ".zip" else 0.0
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=["zip_extension"] if confidence else [],
            artifact_type=ArtifactType.zip_archive,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id, path=path).atoms

    def parse_artifact_full(self, *, project_id: str, artifact_id: str, path: Path, domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        atoms: list[EvidenceAtom] = []
        # Recursive extraction — opt-in via env var. When enabled, the
        # ZIP is extracted into a temp folder, each contained file is
        # routed through ``choose_parser`` and parsed inline so the
        # PM sees evidence from inside the archive without re-running
        # the pipeline manually. Capped at 200 files / 200 MB so a
        # weaponized ZIP can't OOM the host.
        if os.environ.get("PARSER_OS_ZIP_RECURSIVE", "").strip().lower() in {"1", "true", "yes"}:
            recursive_result = _zip_recursive_extract(
                project_id=project_id,
                artifact_id=artifact_id,
                path=path,
                parser_version=self.parser_version,
            )
            if recursive_result is not None:
                return recursive_result
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                infolist = z.infolist()
        except Exception as exc:
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.zip_archive,
                text=f"[ZIP archive — could not be opened: {exc}]",
                locator={"kind": "zip_error"},
                extraction_method="zip_stdlib",
                parser_version=self.parser_version,
                atom_type=AtomType.open_question,
            ))
            return ParserOutput(atoms=atoms, derived_files=[])

        # Marker atom: tell the PM we found a ZIP + what's inside
        # + a clear next-step instruction.
        listing = "\n".join(f"  - {n} ({_info_size(infolist, n)} bytes)" for n in names[:40])
        if len(names) > 40:
            listing += f"\n  - … and {len(names) - 40} more entries"
        marker = (
            f"[ZIP archive awaiting extraction] {path.name} contains "
            f"{len(names)} entries. To get full evidence from inside, "
            f"extract this archive and re-run parser-os on the resulting "
            f"folder. Contents listing:\n{listing}"
        )
        atoms.append(_make_atom(
            project_id=project_id, artifact_id=artifact_id, filename=path.name,
            artifact_type=ArtifactType.zip_archive, text=marker[:2000],
            locator={"kind": "zip_listing", "entry_count": len(names)},
            extraction_method="zip_stdlib",
            parser_version=self.parser_version,
            atom_type=AtomType.open_question,
            value_extra={"kind": "zip_marker", "entries": names[:80]},
        ))
        # One scope_item per entry so the source-inventory section
        # gets the file list visible to the PM without extraction.
        for entry_idx, info in enumerate(infolist[:80]):
            entry_name = info.filename
            if entry_name.endswith("/"):
                continue  # skip directory entries
            atoms.append(_make_atom(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.zip_archive,
                text=f"Archive entry: {entry_name} ({info.file_size:,} bytes)",
                locator={"kind": "zip_entry", "entry_index": entry_idx, "entry_name": entry_name},
                extraction_method="zip_stdlib",
                parser_version=self.parser_version,
                atom_type=AtomType.scope_item,
                value_extra={
                    "kind": "zip_entry",
                    "name": entry_name,
                    "size_bytes": info.file_size,
                    "compress_type": info.compress_type,
                },
            ))
        return ParserOutput(atoms=atoms, derived_files=[])


def _info_size(infolist: list[zipfile.ZipInfo], name: str) -> int:
    for info in infolist:
        if info.filename == name:
            return info.file_size
    return 0


_ZIP_RECURSIVE_MAX_FILES = int(os.environ.get("PARSER_OS_ZIP_RECURSIVE_MAX_FILES", "200"))
_ZIP_RECURSIVE_MAX_BYTES = int(os.environ.get("PARSER_OS_ZIP_RECURSIVE_MAX_BYTES", str(200 * 1024 * 1024)))


def _zip_recursive_extract(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
    parser_version: str,
) -> ParserOutput | None:
    """Extract a ZIP into a temp folder, route each contained file
    through ``choose_parser``, and aggregate atoms.

    Capped at ``PARSER_OS_ZIP_RECURSIVE_MAX_FILES`` files /
    ``PARSER_OS_ZIP_RECURSIVE_MAX_BYTES`` bytes so a weaponized
    archive can't OOM the host.

    Returns ``None`` when the ZIP is structurally bad (caller
    falls back to the listing-only marker path).
    """
    from app.parsers.registry import choose_parser
    atoms: list[EvidenceAtom] = []
    try:
        with zipfile.ZipFile(path) as z:
            entries = z.infolist()
    except Exception:
        return None
    extracted_count = 0
    extracted_bytes = 0
    with tempfile.TemporaryDirectory(prefix="parser_os_zip_") as tmpdir:
        for info in entries:
            if info.is_dir():
                continue
            if extracted_count >= _ZIP_RECURSIVE_MAX_FILES:
                break
            if extracted_bytes + info.file_size > _ZIP_RECURSIVE_MAX_BYTES:
                break
            try:
                with zipfile.ZipFile(path) as z:
                    z.extract(info, tmpdir)
            except Exception:
                continue
            extracted_count += 1
            extracted_bytes += info.file_size
            inner = Path(tmpdir) / info.filename
            if not inner.is_file():
                continue
            try:
                parser, _match, _all = choose_parser(inner)
            except Exception:
                continue
            if parser is None:
                continue
            try:
                child_atoms = parser.parse_artifact(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    path=inner,
                    domain_pack=None,
                )
            except Exception as exc:
                atoms.append(_make_atom(
                    project_id=project_id, artifact_id=artifact_id, filename=path.name,
                    artifact_type=ArtifactType.zip_archive,
                    text=(
                        f"[ZIP recursive extract — child parse error for "
                        f"`{info.filename}`: {type(exc).__name__}: {exc}]"
                    ),
                    locator={"kind": "zip_child_error", "entry_name": info.filename},
                    extraction_method="zip_recursive",
                    parser_version=parser_version,
                    atom_type=AtomType.open_question,
                ))
                continue
            # Re-stamp source locators to mark these as coming from
            # inside the ZIP, so PM_HANDOFF source-inventory still
            # ties them back to the original archive artifact.
            for atom in child_atoms:
                atoms.append(atom)
        # Summary atom
        atoms.append(_make_atom(
            project_id=project_id, artifact_id=artifact_id, filename=path.name,
            artifact_type=ArtifactType.zip_archive,
            text=(
                f"[ZIP recursive extract] {path.name}: extracted "
                f"{extracted_count} files ({extracted_bytes:,} bytes) "
                f"into temp folder; parser-os recursed and emitted "
                f"{len(atoms)} child atoms."
            ),
            locator={
                "kind": "zip_recursive_summary",
                "extracted_count": extracted_count,
                "extracted_bytes": extracted_bytes,
                "atom_count": len(atoms),
            },
            extraction_method="zip_recursive",
            parser_version=parser_version,
            atom_type=AtomType.scope_item,
            value_extra={
                "kind": "zip_recursive_summary",
                "extracted_count": extracted_count,
                "extracted_bytes": extracted_bytes,
            },
        ))
    return ParserOutput(atoms=atoms, derived_files=[])


# ─── Slack / Teams chat export detection ────────────────────────────


def _looks_like_slack_export(soup: Any) -> bool:
    """Slack's HTML export uses ``c-message_kit__background`` /
    ``c-message__sender`` / ``c-message__body`` classes."""
    return bool(
        soup.find(class_=re.compile(r"c-message|slack-message|c-virtual_list__item"))
        or soup.find(attrs={"data-qa": "message_container"})
    )


def _looks_like_teams_export(soup: Any) -> bool:
    """Teams export uses ``message-body``, ``ts-message`` or
    ``data-tid="messageBodyContent"``."""
    return bool(
        soup.find(attrs={"data-tid": re.compile(r"message|chat")})
        or soup.find(class_=re.compile(r"ts-message|teams-message|message-body"))
    )


def _extract_slack_messages(
    soup: Any, *, project_id: str, artifact_id: str, filename: str, parser_version: str,
) -> list[EvidenceAtom]:
    atoms: list[EvidenceAtom] = []
    # Slack exports use ``c-message__sender`` + ``c-message__body``
    # adjacent to each other. We scan for either pattern and extract.
    msg_idx = 0
    for msg in soup.find_all(class_=re.compile(r"c-message_kit__background|c-message"))[:300]:
        sender_node = msg.find(class_=re.compile(r"c-message__sender|c-message_kit__sender"))
        body_node = msg.find(class_=re.compile(r"c-message__body|c-message_kit__text|p-rich_text_section"))
        ts_node = msg.find(class_=re.compile(r"c-timestamp"))
        sender = sender_node.get_text(strip=True) if sender_node else ""
        body = body_node.get_text(separator=" ", strip=True) if body_node else ""
        timestamp = ts_node.get_text(strip=True) if ts_node else ""
        if not body:
            continue
        text = (
            (f"[{timestamp}] " if timestamp else "")
            + (f"{sender}: " if sender else "")
            + body
        )
        atoms.append(_make_atom(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            artifact_type=ArtifactType.html, text=text[:1200],
            locator={"kind": "slack_message", "message_index": msg_idx,
                     "sender": sender, "timestamp": timestamp},
            extraction_method="html_slack_export",
            parser_version=parser_version,
            atom_type=AtomType.meeting_commitment if "?" in body else AtomType.scope_item,
            value_extra={"kind": "slack_message", "sender": sender, "timestamp": timestamp},
        ))
        msg_idx += 1
    return atoms


def _extract_teams_messages(
    soup: Any, *, project_id: str, artifact_id: str, filename: str, parser_version: str,
) -> list[EvidenceAtom]:
    atoms: list[EvidenceAtom] = []
    msg_idx = 0
    for msg in soup.find_all(attrs={"data-tid": re.compile(r"message-body|chat-message")})[:300]:
        # Teams puts sender in a sibling or parent ``data-tid="messageHeader"``
        sender = ""
        body = msg.get_text(separator=" ", strip=True)
        # Try to find a sender header nearby
        header = msg.find_parent().find(attrs={"data-tid": re.compile(r"messageHeader|sender")})
        if header:
            sender = header.get_text(strip=True)
        if not body:
            continue
        text = (f"{sender}: " if sender else "") + body
        atoms.append(_make_atom(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            artifact_type=ArtifactType.html, text=text[:1200],
            locator={"kind": "teams_message", "message_index": msg_idx, "sender": sender},
            extraction_method="html_teams_export",
            parser_version=parser_version,
            atom_type=AtomType.meeting_commitment if "?" in body else AtomType.scope_item,
            value_extra={"kind": "teams_message", "sender": sender},
        ))
        msg_idx += 1
    return atoms


