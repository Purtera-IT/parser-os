from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from app.domain import get_active_domain_pack
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
from app.parsers.segmenters import segment_email
from app.parsers.structured_projection import (
    derived_files_for,
    make_page,
    make_paragraph,
    make_section,
    make_structured_document,
    stamp_section_and_block_ids,
)
from app.domain.schemas import DomainPack

STRUCTURED_SCHEMA_EMAIL = "orbitbrief.email.structured.v1"

BLOCK_SPLIT_RE = re.compile(r"^(On .+ wrote:|-----Original Message-----)$", flags=re.IGNORECASE)
TIME_RANGE_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\s?-\s?\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.I)

EXCLUSION_PATTERNS = [
    r"\bexclude\b",
    r"\bout of scope\b",
    r"\bnot in scope\b",
    r"\bremove .+ from scope\b",
    r"\bdo not proceed\b",
    r"\bhold off\b",
]
INSTRUCTION_PATTERNS = [
    r"\bplease add\b",
    r"\bplease remove\b",
    r"\bapproved\b",
    r"\bdo not schedule\b",
    r"\bproceed\b",
    r"\bgo ahead\b",
    r"\bhold off\b",
    r"\bplease include\b",
    r"\breduce\s+\w+\s+(?:count\s+)?(?:from|to)\b",
    r"\bchange order\b",
    r"\brevised scope\b",
    r"\bcancel(?:\s+the)?\b",
    r"\badd(?:ed)?\s+\d+\s+(?:more|additional)\b",
]
CHANGE_DELTA_PATTERN = re.compile(
    r"\b(?:from|reduce(?:d)?\s+(?:from)?)\s+(\d{1,5})\s+to\s+(\d{1,5})\b",
    re.IGNORECASE,
)
CONSTRAINT_PATTERNS = [
    r"\baccess only\b",
    r"\bescort access\b",
    r"\bescort required\b",
    r"\bbadge required\b",
    r"\bafter hours\b",
    r"\bafter\s+\d{1,2}(?::\d{2})?\s?(?:am|pm)\b",
    r"\bparking\b",
    r"\bloading dock\b",
    r"\bweekdays\b",
]


def _extract_email_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".eml":
        raw = path.read_bytes()
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        body = msg.get_body(preferencelist=("plain", "html"))
        content = body.get_content() if body is not None else raw.decode("utf-8", errors="ignore")
    else:
        content = path.read_text(encoding="utf-8", errors="ignore")
    if "<html" in content.lower():
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    return content


class EmailParser(BaseParser):
    parser_name = "email"
    parser_version = "email_parser_v1"
    internal_domains = ("purtera", "internal")
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".eml", ".txt", ".md"],
        supported_artifact_types=[ArtifactType.email, ArtifactType.txt],
        emitted_atom_types=[AtomType.exclusion, AtomType.customer_instruction, AtomType.constraint, AtomType.open_question],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del domain_pack
        suffix = path.suffix.lower()
        text = normalize_text(sample_text or "")
        reasons: list[str] = []
        confidence = 0.0
        if suffix == ".eml":
            confidence = 0.98
            reasons.append("eml_extension")
        elif suffix in {".txt", ".md"}:
            if "from:" in text and ("sent:" in text or "subject:" in text):
                confidence = 0.91
                reasons.append("email_headers_detected")
            elif " wrote:" in text:
                confidence = 0.83
                reasons.append("email_thread_marker")
            else:
                # Headerless body fallback: short .txt files whose content
                # reads as customer correspondence (instruction / exclusion
                # / constraint keywords) still need an extractor. Take a
                # low-confidence claim so other parsers can override but
                # the file isn't silently dropped.
                email_hits = sum(
                    1
                    for needle in (
                        "please add", "please remove", "please include",
                        "approved to proceed", "hold off", "go ahead",
                        "badge required", "escort required",
                    )
                    if needle in text
                )
                if email_hits >= 1:
                    confidence = 0.55
                    reasons.append(f"email_keyword_heuristic({email_hits})")
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.email if suffix == ".eml" else ArtifactType.txt,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def segment_artifact(self, project_id: str, artifact_id: str, path: Path) -> list[ArtifactSegment]:
        return segment_email(project_id=project_id, artifact_id=artifact_id, path=path, parser_version=self.parser_version)

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
        text = _extract_email_text(path)
        blocks = self._split_blocks(text)
        atoms: list[EvidenceAtom] = []
        for block in blocks:
            authority = self._authority_for_block(block)
            atoms.extend(
                self._extract_atoms_from_block(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    block=block,
                    authority=authority,
                )
            )
        structured_doc = self._build_structured_doc(filename=path.name, blocks=blocks)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _build_structured_doc(
        self,
        *,
        filename: str,
        blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Render an email thread as one page per message, newest first.
        Quoted history lives under its own subsection so an LLM can
        skip it without losing context.
        """
        pages: list[dict[str, Any]] = []
        for index, block in enumerate(blocks):
            sender = block.get("sender") or "unknown"
            sent_at = block.get("sent_at") or ""
            quoted = block.get("quoted")
            heading = f"Message {index + 1}: {sender}"
            if sent_at:
                heading = f"{heading} ({sent_at})"
            metadata: list[str] = []
            if sender and sender != "unknown":
                metadata.append(f"sender: {sender}")
            if sent_at:
                metadata.append(f"sent_at: {sent_at}")
            metadata.append(f"quoted: {quoted}")

            body_lines: list[str] = []
            quoted_lines: list[str] = []
            for line in block.get("lines", []) or []:
                stripped = line.strip()
                if stripped.startswith(">"):
                    quoted_lines.append(stripped.lstrip("> ").strip())
                elif stripped.lower().startswith(("from:", "sent:", "date:", "subject:", "to:", "cc:", "bcc:")):
                    metadata.append(stripped)
                else:
                    body_lines.append(stripped)
            body_text = "\n".join(line for line in body_lines if line).strip()
            section_blocks: list[dict[str, Any]] = []
            if body_text:
                section_blocks.append(make_paragraph(body_text))
            section = make_section(
                heading=heading,
                level=2,
                blocks=section_blocks,
                subsections=(
                    [
                        make_section(
                            heading="Quoted history",
                            level=3,
                            blocks=[
                                make_paragraph(
                                    "\n".join(line for line in quoted_lines if line)
                                )
                            ],
                        )
                    ]
                    if quoted_lines
                    else []
                ),
            )
            pages.append(
                make_page(
                    page=index,
                    title=heading,
                    metadata=metadata,
                    sections=[section],
                )
            )
        if not pages:
            pages.append(
                make_page(
                    page=0,
                    title=filename,
                    sections=[
                        make_section(
                            heading=filename,
                            level=2,
                            blocks=[make_paragraph("(empty email)")],
                        )
                    ],
                )
            )
        return make_structured_document(
            schema_version=STRUCTURED_SCHEMA_EMAIL,
            filename=filename,
            artifact_type=ArtifactType.email.value,
            title=filename,
            metadata=[f"message_count: {len(blocks)}"],
            pages=pages,
        )

    def _split_blocks(self, text: str) -> list[dict[str, Any]]:
        lines = text.splitlines()
        if not lines:
            return []
        blocks: list[dict[str, Any]] = []
        current: list[tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            is_new_message_boundary = bool(BLOCK_SPLIT_RE.match(stripped))
            is_from_after_body = (
                stripped.lower().startswith("from:")
                and current
                and any(not l.strip().lower().startswith(("from:", "sent:", "date:", "subject:")) for _, l in current)
            )
            if current and (is_new_message_boundary or is_from_after_body):
                blocks.append(self._build_block(blocks, current))
                current = []
            current.append((idx, line))
        if current:
            blocks.append(self._build_block(blocks, current))
        return blocks

    def _build_block(self, existing: list[dict[str, Any]], lines: list[tuple[int, str]]) -> dict[str, Any]:
        stripped_lines = [line.strip() for _, line in lines]
        sender = self._find_header_value(stripped_lines, "from")
        sent_at = self._find_header_value(stripped_lines, "sent") or self._find_header_value(stripped_lines, "date")
        quoted = any(line.startswith(">") for line in stripped_lines) or len(existing) > 0
        return {
            "message_index": len(existing),
            "line_start": lines[0][0],
            "line_end": lines[-1][0],
            "sender": sender or "unknown",
            "sent_at": sent_at or "",
            "quoted": quoted,
            "lines": stripped_lines,
        }

    def _find_header_value(self, lines: list[str], key: str) -> str | None:
        pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.+)$", flags=re.IGNORECASE)
        for line in lines:
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
        return None

    def _authority_for_block(self, block: dict[str, Any]) -> AuthorityClass:
        if block["quoted"]:
            return AuthorityClass.quoted_old_email
        sender = normalize_text(str(block.get("sender", "")))
        if any(domain in sender for domain in self.internal_domains):
            return AuthorityClass.machine_extractor
        return AuthorityClass.customer_current_authored

    def _extract_entity_keys(self, text: str) -> list[str]:
        keys: list[str] = []
        lowered = normalize_text(text)
        pack = get_active_domain_pack()
        if "west wing" in lowered:
            keys.append(normalize_entity_key("site", "West Wing"))
        if "main campus" in lowered:
            keys.append(normalize_entity_key("site", "Main Campus"))
        if "camera" in lowered:
            keys.append(normalize_entity_key("device", "IP Camera"))
        for canonical, aliases in pack.device_aliases.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(normalize_text(alias))}\b", lowered):
                    keys.append(f"device:{canonical}")
                    break
        return keys

    def _build_source_ref(
        self,
        artifact_id: str,
        filename: str,
        block: dict[str, Any],
    ) -> SourceRef:
        return SourceRef(
            id=stable_id("src", artifact_id, block["message_index"], block["line_start"]),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.email,
            filename=filename,
            locator={
                "message_index": block["message_index"],
                "line_start": block["line_start"],
                "line_end": block["line_end"],
                "sender": block["sender"],
                "sent_at": block["sent_at"],
                "quoted": block["quoted"],
            },
            extraction_method="thread_text_rules",
            parser_version=self.parser_version,
        )

    def _extract_atoms_from_block(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        block: dict[str, Any],
        authority: AuthorityClass,
    ) -> list[EvidenceAtom]:
        atoms: list[EvidenceAtom] = []
        source_ref = self._build_source_ref(artifact_id=artifact_id, filename=filename, block=block)
        confidence = 0.45 if authority == AuthorityClass.quoted_old_email else 0.86

        for line in block["lines"]:
            cleaned = line.lstrip("> ").strip()
            if not cleaned:
                continue
            lowered = normalize_text(cleaned)
            entity_keys = self._extract_entity_keys(cleaned)
            atom_types: list[AtomType] = []
            pack = get_active_domain_pack()
            exclusion_patterns = EXCLUSION_PATTERNS + [re.escape(normalize_text(p)) for p in pack.exclusion_patterns]
            instruction_patterns = INSTRUCTION_PATTERNS + [
                re.escape(normalize_text(p)) for p in pack.customer_instruction_patterns
            ]
            constraint_patterns = CONSTRAINT_PATTERNS + [
                re.escape(normalize_text(p))
                for rows in pack.constraint_patterns.values()
                for p in rows
            ]

            if any(re.search(pattern, lowered) for pattern in exclusion_patterns):
                atom_types.append(AtomType.exclusion)
            if any(re.search(pattern, lowered) for pattern in instruction_patterns):
                atom_types.append(AtomType.customer_instruction)
            # Change-delta presence ("from 48 to 36" anywhere in line)
            # is a strong customer_instruction signal — the email writer
            # is changing the scope by a specific delta.
            if CHANGE_DELTA_PATTERN.search(cleaned) and AtomType.customer_instruction not in atom_types:
                atom_types.append(AtomType.customer_instruction)
            if any(re.search(pattern, lowered) for pattern in constraint_patterns) or TIME_RANGE_RE.search(cleaned):
                atom_types.append(AtomType.constraint)
            if cleaned.endswith("?") or re.match(r"^(who|what|when|where|why|how|can|could|should)\b", lowered):
                atom_types.append(AtomType.open_question)

            # Pre-compute change_delta once per line so customer
            # instructions with "from X to Y" carry structured deltas.
            delta_payload = None
            delta_match = CHANGE_DELTA_PATTERN.search(cleaned)
            if delta_match:
                try:
                    from_v = int(delta_match.group(1))
                    to_v = int(delta_match.group(2))
                    delta_payload = {
                        "from": from_v,
                        "to": to_v,
                        "delta": to_v - from_v,
                    }
                except (ValueError, IndexError):
                    delta_payload = None

            for atom_type in atom_types:
                review_status = ReviewStatus.auto_accepted
                if atom_type == AtomType.open_question:
                    review_status = ReviewStatus.needs_review
                atom_value = {
                    "text": cleaned,
                    "message_index": block["message_index"],
                    "quoted": block["quoted"],
                }
                if delta_payload and atom_type == AtomType.customer_instruction:
                    atom_value["change_delta"] = delta_payload
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            block["message_index"],
                            block["line_start"],
                            atom_type.value,
                            cleaned,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=atom_type,
                        raw_text=cleaned,
                        normalized_text=normalize_text(cleaned),
                        value=atom_value,
                        entity_keys=entity_keys,
                        source_refs=[source_ref],
                        authority_class=authority,
                        confidence=confidence,
                        review_status=review_status,
                        review_flags=[],
                        parser_version=self.parser_version,
                    )
                )
        return atoms
