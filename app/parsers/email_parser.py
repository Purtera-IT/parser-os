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
from app.parsers.binary_markers import attachment_marker
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


# Subject prefixes stripped to find the conversation root: Re:, Fwd:, FW:,
# Aw: (German), Rv: (Spanish/Italian) — repeated, in any case. The HubSpot deal
# number prefix (e.g. "010065") is KEPT: it is a strong, deliberate thread key.
_SUBJECT_PREFIX_RE = re.compile(r"^\s*(re|fwd?|fw|aw|rv|tr|wg)\s*(\[\d+\])?\s*:\s*", re.IGNORECASE)
_MSGID_RE = re.compile(r"<[^>]+>")


def normalize_email_subject(subject: str) -> str:
    """Conversation key from a Subject line: strip reply/forward prefixes
    (repeatedly), collapse whitespace, lowercase. Universal — no per-deal
    vocabulary. ``"RE: Fwd: 010065 AP Swap"`` and ``"010065 AP Swap"`` map to
    the same key so a whole back-and-forth threads together even when the
    References headers are missing (common in exported / HubSpot .eml)."""
    s = (subject or "").strip()
    prev = None
    # Strip stacked prefixes ("RE: FW: ...") until stable.
    while s and s != prev:
        prev = s
        s = _SUBJECT_PREFIX_RE.sub("", s, count=1).strip()
    return re.sub(r"\s+", " ", s).strip().lower()


def _parse_date_epoch(date_raw: str) -> float:
    """Epoch seconds from an email Date header, robust to both formats we see:
    RFC 2822 ("Mon, 01 Jun 2026 09:00:00 -0400") from real mail clients, and
    ISO 8601 ("2026-06-19T12:43:58Z" / with milliseconds) from HubSpot exports.
    Returns 0.0 when unparseable so ordering degrades to encounter order."""
    if not date_raw:
        return 0.0
    # RFC 2822 first (native email format).
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(date_raw)
        if dt is not None:
            return dt.timestamp()
    except Exception:
        pass
    # ISO 8601 fallback (HubSpot). Normalise trailing Z -> +00:00 for fromisoformat.
    try:
        from datetime import datetime

        iso = date_raw.strip()
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso)
        return dt.timestamp()
    except Exception:
        return 0.0


def parse_email_thread_headers(path: Path) -> dict[str, Any]:
    """Extract RFC 5322 threading headers + ordering signal from an .eml.

    Returns a dict with: ``message_id``, ``in_reply_to``, ``references`` (list
    of message-ids, oldest→newest), ``subject``, ``subject_norm``, ``sender``,
    ``date_raw``, ``date_epoch`` (float, 0.0 when unparseable). Safe: never
    raises, returns ``{}`` for non-.eml or unreadable files."""
    if path.suffix.lower() != ".eml":
        return {}
    try:
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    except Exception:  # pragma: no cover - unreadable
        return {}

    def _ids(raw: str | None) -> list[str]:
        if not raw:
            return []
        return _MSGID_RE.findall(raw)

    msg_id_list = _ids(msg.get("message-id"))
    in_reply_to_list = _ids(msg.get("in-reply-to"))
    references = _ids(msg.get("references"))
    subject = str(msg.get("subject") or "").strip()
    sender = str(msg.get("from") or "").strip()

    date_raw = str(msg.get("date") or "").strip()
    date_epoch = _parse_date_epoch(date_raw)
    return {
        "message_id": msg_id_list[0] if msg_id_list else "",
        "in_reply_to": in_reply_to_list[0] if in_reply_to_list else "",
        "references": references,
        "subject": subject,
        "subject_norm": normalize_email_subject(subject),
        "sender": sender,
        "date_raw": date_raw,
        "date_epoch": date_epoch,
    }


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
        # Header atom — the From/To/Cc/Subject/Date line is real content (and
        # the content census inventories it). Emit it as a scope_item so the
        # header is never silently absent from the atom stream. .eml only;
        # headerless .txt/.md bodies have no structured headers to surface.
        header_atom = self._header_atom(
            project_id=project_id, artifact_id=artifact_id, path=path
        )
        if header_atom is not None:
            atoms.append(header_atom)
        # Attachments are the real deal docs more often than the body — mark
        # each one so it can't silently vanish (the file content is a separate
        # artifact; this is a located pointer the PM/census can see).
        atoms.extend(
            self._attachment_markers(
                project_id=project_id, artifact_id=artifact_id, path=path
            )
        )
        structured_doc = self._build_structured_doc(filename=path.name, blocks=blocks)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _header_atom(
        self, *, project_id: str, artifact_id: str, path: Path
    ) -> EvidenceAtom | None:
        if path.suffix.lower() != ".eml":
            return None
        try:
            msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        except Exception:  # pragma: no cover - unreadable
            return None
        parts: list[str] = []
        values: dict[str, str] = {}
        for field in ("from", "to", "cc", "subject", "date"):
            val = msg.get(field)
            if val:
                sval = str(val).strip()
                parts.append(f"{field.capitalize()}: {sval}")
                values[field] = sval
        if not parts:
            return None
        text = " | ".join(parts)
        # Threading metadata: carried on the header atom so the compiler's
        # email_threading stage can group this message into its conversation
        # (RFC In-Reply-To / References, subject_norm fallback) and propagate
        # context to every atom from this artifact. Purely additive.
        thread_meta = parse_email_thread_headers(path)
        src = SourceRef(
            id=stable_id("src", artifact_id, "email_header"),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.email,
            filename=path.name,
            locator={"kind": "email_header"},
            extraction_method="email_headers",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "email_header", text),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.scope_item,
            raw_text=text,
            normalized_text=normalize_text(text),
            value={"kind": "email_header", **values, "email_thread_meta": thread_meta},
            entity_keys=[],
            source_refs=[src],
            authority_class=AuthorityClass.customer_current_authored,
            confidence=0.86,
            review_status=ReviewStatus.auto_accepted,
            review_flags=[],
            parser_version=self.parser_version,
        )

    def _attachment_markers(
        self, *, project_id: str, artifact_id: str, path: Path
    ) -> list[EvidenceAtom]:
        if path.suffix.lower() != ".eml":
            return []
        try:
            msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        except Exception:  # pragma: no cover - unreadable
            return []
        out: list[EvidenceAtom] = []
        for ai, att in enumerate(msg.iter_attachments()):
            name = att.get_filename() or f"attachment{ai}"
            try:
                payload = att.get_payload(decode=True) or b""
                size = len(payload)
            except Exception:
                size = 0
            out.append(attachment_marker(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.email, parser_version=self.parser_version,
                attachment_name=name, size=size, content_type=att.get_content_type(),
            ))
        return out

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

            # Baseline body coverage: a body line that matched no typed
            # pattern is still real content — emit it as a scope_item so it
            # is never silently absent from the atom stream (the content
            # census inventories every body line). This mirrors the docx
            # fail-open prose gate and the MboxParser per-paragraph behavior:
            # keep + let the downstream learnable seam decide, never drop.
            if not atom_types and any(ch.isalnum() for ch in cleaned):
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            block["message_index"],
                            block["line_start"],
                            "scope_item",
                            cleaned,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.scope_item,
                        raw_text=cleaned,
                        normalized_text=normalize_text(cleaned),
                        value={
                            "text": cleaned,
                            "message_index": block["message_index"],
                            "quoted": block["quoted"],
                            "kind": "email_body_line",
                        },
                        entity_keys=entity_keys,
                        source_refs=[source_ref],
                        authority_class=authority,
                        confidence=confidence,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[],
                        parser_version=self.parser_version,
                    )
                )
        return atoms
