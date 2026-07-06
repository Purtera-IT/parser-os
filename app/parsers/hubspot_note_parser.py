"""HubSpot CRM note parser — short timeline notes, not transcripts.

Exported ``*-hs-note-*.txt`` files carry deal facts in a few lines (ROM,
hardware counts, config-only scope) but lack speaker/timestamp structure.
Routing them through :class:`TranscriptParser` yields ``ok_empty`` because
utterance regexes miss ``aps``/``configuration`` and ISO dates masquerade as
timestamps.

This parser reads the note body directly (no utterance segmentation) and emits
typed atoms with ``artifact_id`` provenance for the Files UI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.core.address_parse import US_STATES, find_us_addresses_in_text
from app.core.ids import stable_id
from app.core.normalizers import normalize_text
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
from app.core.training_log import TEACHER_STORE, TrainingRow, log_rows
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser
from app.parsers.structured_projection import (
    derived_files_for,
    make_page,
    make_paragraph,
    make_section,
    make_structured_document,
    stamp_section_and_block_ids,
)

STRUCTURED_SCHEMA_HUBSPOT_NOTE = "orbitbrief.hubspot_note.structured.v1"
HUBSPOT_NOTE_RELATION = "hubspot_note_extraction"

_HS_NOTE_FILENAME_RE = re.compile(r"-hs-note-", re.I)
_HS_NOTE_HEADER_RE = re.compile(r"^HubSpot Note\s*:", re.I | re.M)
_HS_NOTE_ID_RE = re.compile(r"^HubSpot Note ID:\s*(\S+)", re.I | re.M)
_HS_DATE_RE = re.compile(r"^Date:\s*(.+)$", re.I | re.M)
_HS_AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.I | re.M)

_ROM_RE = re.compile(
    r"\b(?:ROM|rough\s+order\s+of\s+magnitude)\b|"
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b|"
    r"\b(\d{1,3}(?:,\d{3})*)\s*(?:k|K)\b",
    re.I,
)
_SCOPE_SIGNAL_RE = re.compile(
    r"\b(install|configur|ubiquiti|unifi|udm|nvr|unvr|camera|badge|okta|otka|"
    r"vlan|meraki|remote|onsite|white\s+glove|equipment|switch|router|ap\b|aps\b|"
    r"reader|enterprise|rom|good\s+2\s+go)\b",
    re.I,
)
_INSTRUCTION_RE = re.compile(
    r"\b(please|need to|must|should|customer would like|get full list|"
    r"good\s+2\s+go|approved|hold off|go ahead)\b",
    re.I,
)


def is_hubspot_note_path(path: Path, sample_text: str | None = None) -> bool:
    name = path.name.lower()
    if _HS_NOTE_FILENAME_RE.search(name):
        return True
    text = (sample_text or "")[:2000]
    return bool(_HS_NOTE_HEADER_RE.search(text))


def parse_hubspot_note_text(raw: str) -> dict[str, Any]:
    """Split HubSpot export headers from the note body."""
    lines = [ln.rstrip() for ln in (raw or "").splitlines()]
    title = ""
    note_id = ""
    date_raw = ""
    author = ""
    body_lines: list[str] = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not in_body:
            continue
        if not in_body:
            m = re.match(r"^HubSpot Note:\s*(.*)$", stripped, re.I)
            if m:
                title = m.group(1).strip()
                continue
            m = _HS_NOTE_ID_RE.match(stripped)
            if m:
                note_id = m.group(1).strip()
                continue
            m = _HS_DATE_RE.match(stripped)
            if m:
                date_raw = m.group(1).strip()
                continue
            m = _HS_AUTHOR_RE.match(stripped)
            if m:
                author = m.group(1).strip()
                continue
            if not stripped and title:
                in_body = True
                continue
            if title and not note_id and not date_raw and not author:
                body_lines.append(stripped)
                in_body = True
                continue
            if title:
                in_body = True
        if in_body and stripped:
            body_lines.append(stripped)
    body = " ".join(body_lines).strip()
    if not body and title:
        body = title
    return {
        "title": title,
        "note_id": note_id,
        "date_raw": date_raw,
        "author": author,
        "body": body,
    }


class HubspotNoteParser(BaseParser):
    parser_name = "hubspot_note"
    parser_version = "hubspot_note_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".txt"],
        supported_artifact_types=[ArtifactType.txt],
        emitted_atom_types=[
            AtomType.scope_item,
            AtomType.customer_instruction,
            AtomType.deal_metadata,
            AtomType.commercial_total,
            AtomType.physical_site,
            AtomType.constraint,
            AtomType.open_question,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del domain_pack
        if path.suffix.lower() != ".txt":
            return ParserMatch(
                parser_name=self.parser_name,
                confidence=0.0,
                reasons=["not_txt"],
                artifact_type=ArtifactType.txt,
            )
        reasons: list[str] = []
        confidence = 0.0
        if _HS_NOTE_FILENAME_RE.search(path.name):
            confidence = 0.97
            reasons.append("filename:hs-note")
        text = sample_text or ""
        if _HS_NOTE_HEADER_RE.search(text[:2000]):
            confidence = max(confidence, 0.94)
            reasons.append("header:hubspot_note")
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.txt,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact(project_id="unknown_project", artifact_id=artifact_id, path=artifact_path)

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
        raw = path.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_hubspot_note_text(raw)
        atoms = self._atoms_from_note(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=path.name,
            parsed=parsed,
        )
        structured_doc = self._build_structured_doc(filename=path.name, parsed=parsed)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _base_source_ref(self, artifact_id: str, filename: str) -> SourceRef:
        return SourceRef(
            id=stable_id("src", artifact_id, "hubspot_note"),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.txt,
            filename=filename,
            locator={"kind": "hubspot_note_body"},
            extraction_method="hubspot_note_parser",
            parser_version=self.parser_version,
        )

    def _mint_atom(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        atom_type: AtomType,
        text: str,
        value: dict[str, Any],
        source_ref: SourceRef,
        confidence: float = 0.8,
        review_flags: list[str] | None = None,
        entity_keys: list[str] | None = None,
    ) -> EvidenceAtom:
        flags = list(review_flags or [])
        if "hubspot_note_parser" not in flags:
            flags.append("hubspot_note_parser")
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, atom_type.value, text[:160]),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=text[:4000],
            normalized_text=normalize_text(text),
            value=value,
            entity_keys=list(entity_keys or []),
            source_refs=[source_ref],
            authority_class=AuthorityClass.meeting_note,
            confidence=confidence,
            review_status=ReviewStatus.auto_accepted,
            review_flags=flags,
            parser_version=self.parser_version,
        )

    def _atoms_from_note(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        parsed: dict[str, Any],
    ) -> list[EvidenceAtom]:
        body = str(parsed.get("body") or "").strip()
        title = str(parsed.get("title") or "").strip()
        note_id = str(parsed.get("note_id") or "").strip()
        author = str(parsed.get("author") or "").strip()
        date_raw = str(parsed.get("date_raw") or "").strip()
        source_ref = self._base_source_ref(artifact_id, filename)
        atoms: list[EvidenceAtom] = []
        train_rows: list[TrainingRow] = []

        meta_bits = []
        if note_id:
            meta_bits.append(f"note_id={note_id}")
        if author:
            meta_bits.append(f"author={author}")
        if date_raw:
            meta_bits.append(f"date={date_raw}")
        if meta_bits:
            meta_text = " | ".join(meta_bits)
            atoms.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.deal_metadata,
                    text=meta_text,
                    value={
                        "field_name": "hubspot_note_meta",
                        "hubspot_note_id": note_id,
                        "author": author,
                        "date": date_raw,
                        "title": title,
                        "source": "hubspot_note",
                    },
                    source_ref=source_ref,
                    confidence=0.88,
                )
            )

        if body:
            atom_types: list[AtomType] = [AtomType.scope_item]
            if _INSTRUCTION_RE.search(body):
                atom_types.append(AtomType.customer_instruction)
            if _ROM_RE.search(body):
                atom_types.append(AtomType.commercial_total)
            if body.endswith("?"):
                atom_types.append(AtomType.open_question)
            if re.search(r"\b(remote|onsite|white\s+glove|bandwidth|vlan)\b", body, re.I):
                atom_types.append(AtomType.constraint)

            deduped: list[AtomType] = []
            for at in atom_types:
                if at not in deduped:
                    deduped.append(at)

            for at in deduped:
                val: dict[str, Any] = {
                    "text": body,
                    "kind": "hubspot_note_body",
                    "hubspot_note_id": note_id,
                    "title": title,
                    "source": "hubspot_note",
                }
                if at == AtomType.commercial_total:
                    amounts = re.findall(r"\$\s*(\d[\d,]*(?:\.\d{2})?)", body)
                    k_amounts = re.findall(r"\b(\d{1,3}(?:,\d{3})*)\s*[kK]\b", body)
                    val.update(
                        {
                            "category": "ROM",
                            "currency": "USD",
                            "amounts": amounts,
                            "k_amounts": k_amounts,
                            "rom_text": body,
                        }
                    )
                atoms.append(
                    self._mint_atom(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=filename,
                        atom_type=at,
                        text=body,
                        value=val,
                        source_ref=source_ref,
                        confidence=0.84 if at == AtomType.scope_item else 0.8,
                        review_flags=["hubspot_note_training_row"] if at == AtomType.scope_item else [],
                    )
                )

            train_rows.append(
                TrainingRow(
                    relation=HUBSPOT_NOTE_RELATION,
                    label="scope_item",
                    raw_text=body[:4000],
                    label_kind="judgment",
                    teacher=TEACHER_STORE,
                    confidence=0.84,
                    deal_id=project_id,
                    project_id=project_id,
                    provenance={
                        "note_id": note_id,
                        "title": title,
                        "source": "hubspot_note_parser",
                    },
                )
            )

        # Physical sites from address-bearing notes.
        corpus = f"{title}\n{body}"
        for parsed_addr in find_us_addresses_in_text(corpus):
            if (
                not parsed_addr.city
                or not parsed_addr.state
                or parsed_addr.state not in US_STATES
                or not parsed_addr.street_address
            ):
                continue
            slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                f"{parsed_addr.city}_{parsed_addr.state}_{parsed_addr.zip or parsed_addr.street_address}".lower(),
            ).strip("_")
            display = (
                f"{parsed_addr.street_address}, {parsed_addr.city}, "
                f"{parsed_addr.state} {parsed_addr.zip or ''}"
            ).strip()
            aliases = list(dict.fromkeys(parsed_addr.aliases))
            atoms.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.physical_site,
                    text=display,
                    value={
                        "kind": "physical_site",
                        "id": slug,
                        "site_id": slug,
                        "name": display,
                        "names": list(dict.fromkeys([display, parsed_addr.city, *aliases])),
                        "aliases": aliases,
                        "street_address": parsed_addr.street_address,
                        "address": parsed_addr.street_address,
                        "city": parsed_addr.city,
                        "state": parsed_addr.state,
                        "zip": parsed_addr.zip,
                        "inferred": True,
                        "source_context": corpus[:600],
                    },
                    source_ref=source_ref,
                    confidence=0.76,
                    entity_keys=[f"site:{slug}"],
                    review_flags=["hubspot_note_physical_site"],
                )
            )

        if not atoms and (title or body):
            fallback = body or title
            atoms.append(
                self._mint_atom(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    atom_type=AtomType.deal_metadata,
                    text=fallback,
                    value={
                        "field_name": "hubspot_note_body",
                        "text": fallback,
                        "hubspot_note_id": note_id,
                        "source": "hubspot_note",
                    },
                    source_ref=source_ref,
                    confidence=0.7,
                )
            )

        if train_rows:
            log_rows(train_rows)
        return atoms

    def _build_structured_doc(self, *, filename: str, parsed: dict[str, Any]) -> dict[str, Any]:
        body = str(parsed.get("body") or "").strip() or "(empty note)"
        meta = [
            f"note_id: {parsed.get('note_id') or ''}",
            f"author: {parsed.get('author') or ''}",
            f"date: {parsed.get('date_raw') or ''}",
        ]
        page = make_page(
            page=0,
            title=str(parsed.get("title") or filename),
            metadata=meta,
            sections=[
                make_section(
                    heading="HubSpot Note",
                    level=2,
                    blocks=[make_paragraph(body)],
                )
            ],
        )
        return make_structured_document(
            schema_version=STRUCTURED_SCHEMA_HUBSPOT_NOTE,
            filename=filename,
            artifact_type=ArtifactType.txt.value,
            title=filename,
            metadata=meta,
            pages=[page],
        )


__all__ = [
    "HUBSPOT_NOTE_RELATION",
    "HubspotNoteParser",
    "is_hubspot_note_path",
    "parse_hubspot_note_text",
]
