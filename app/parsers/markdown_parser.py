"""Markdown atom parser.

Extracts evidence atoms from ``.md`` / ``.markdown`` files. The corpus
review found that 60+ KB managed-services scope briefs in Markdown
were producing zero atoms, because parser-os had no Markdown extractor
configured at all — the entire scope brief was invisible to OrbitBrief
before the brain ever started.

This parser produces:

* one atom per heading
* one atom per bullet / numbered item
* one atom per markdown table row (skipping the separator row)
* one atom per non-empty paragraph

Each atom's classification (``atom_type`` + ``authority_class``) is
heuristic — driven by section-path + text patterns (exclusion words,
assumption words, open-question markers, quantity / money mentions,
"customer responsibility" style headings).

Locator preserves ``line_start`` / ``line_end`` / ``section_path`` /
``block_kind`` / ``block_index`` so source replay can fuzzy-match
the atom back to the raw line range.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.normalizers import normalize_entity_key, normalize_text, parse_quantity
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
from app.parsers.binary_markers import region_marker


_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(?P<text>.+?)\s*$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(?P<text>.+?)\s*$")

_EXCLUSION_RE = re.compile(
    r"\b(exclud(?:e|ed|es|ing)|out of scope|not included|not in scope|by others|nic|"
    r"remove from scope|please remove|removed?\s+from\s+the\s+scope|"
    r"cancel(?:led|ling|s)?(?:\s+the)?|cancellation|"
    r"do not include|drop\s+(?:the|from)|deletion?|"
    r"hold off|on hold|defer(?:red)?\s+from|postpone(?:d)?)\b",
    re.I,
)
_CHANGE_ORDER_RE = re.compile(
    r"\b(change\s+order|reduce\s+(?:scope|count|the\s+\w+)\s+(?:from|to)?|"
    r"revised\s+scope|approve(?:d)?\s+the\s+revised|"
    r"reduce\s+\w+\s+count\s+from\s+\d+\s+to\s+\d+|"
    r"add(?:ed)?\s+\d+\s+(?:more|additional)|"
    r"increase\s+(?:scope|count)|expand(?:ed)?\s+scope|"
    r"scope\s+(?:reduction|change|update))\b",
    re.I,
)
_CHANGE_DELTA_RE = re.compile(
    r"\b(?:from|reduce(?:d)?\s+(?:from)?)\s+(\d{1,5})\s+to\s+(\d{1,5})\b",
    re.I,
)
_ASSUMPTION_RE = re.compile(
    r"\b(assum(?:e|ed|ption|ptions)|subject to|provided by owner)\b",
    re.I,
)
_OPEN_Q_RE = re.compile(
    r"\?"
    r"|\b(tbd|to\s+be\s+confirmed|to\s+be\s+determined|unknown|"
    r"open\s+question|please\s+confirm|need(?:s)?\s+confirmation|"
    r"awaiting\s+confirmation|still\s+(?:tbd|pending|outstanding)|"
    r"to\s+clarify|need(?:s)?\s+clarification|pending\s+(?:answer|response))\b",
    re.I,
)
_CONSTRAINT_RE = re.compile(
    r"\b(access window|escort|required|must|shall|badge|after[-\s]?hours|"
    r"acceptance|completion|closeout)\b",
    re.I,
)
_QTY_RE = re.compile(
    # Allow up to ~80 non-period chars between the number and the unit so
    # we capture sentences like "186 Belden Cat6 CMP drops with RJ45 ...".
    r"\b(?P<qty>\d{1,5})\b[^.]{0,80}?\b(?P<unit>drops?|cables?|aps?|cameras?|"
    r"doors?|readers?|licenses?|devices?|ports?|switches?|sites?|"
    r"buildings?|users?|endpoints?)\b",
    re.I,
)

# PR4 (post-v3 review) — risk-table detection. Markdown table rows
# inside a "Risk Register" / "RAID" section, OR rows whose first
# column matches a RISK-id pattern, are RISK atoms regardless of
# whether they happen to contain a number that the QTY_RE would
# match. Without this, rows like
# ``| R-09-08 | camera counts are politically visible | Medium | …``
# get mis-typed as ``quantity`` (the "08" matches QTY_RE if "items"
# appears nearby).
_RISK_SECTION_RE = re.compile(r"\b(risk register|raid|risks?|issues?)\b", re.I)
_RISK_ROW_ID_RE = re.compile(r"^\s*\|\s*(?:RISK|R)-?\d{1,4}", re.I)
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")


@dataclass(frozen=True)
class MarkdownBlock:
    text: str
    line_start: int
    line_end: int
    section_path: tuple[str, ...]
    block_kind: str


class MarkdownParser(BaseParser):
    parser_name = "markdown"
    parser_version = "markdown_parser_v1"

    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".md", ".markdown"],
        supported_artifact_types=[ArtifactType.txt],
        emitted_atom_types=[
            AtomType.scope_item,
            AtomType.exclusion,
            AtomType.assumption,
            AtomType.open_question,
            AtomType.constraint,
            AtomType.quantity,
            AtomType.vendor_line_item,
            AtomType.customer_instruction,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(
        self,
        path: Path,
        sample_text: str | None,
        domain_pack: DomainPack | None,
    ) -> ParserMatch:
        del sample_text, domain_pack
        if path.suffix.lower() in {".md", ".markdown"}:
            return ParserMatch(
                parser_name=self.parser_name,
                confidence=0.96,
                reasons=["markdown_extension"],
                artifact_type=ArtifactType.txt,
            )
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=0.0,
            reasons=[],
            artifact_type=ArtifactType.txt,
        )

    # The base class's ``parse_artifact`` calls this for parsers that
    # only implement the legacy flat-list signature; we override
    # ``parse_artifact`` directly so we keep the structured ParserOutput.
    def parse(self, artifact_path: Path) -> list[Any]:  # pragma: no cover - legacy seam
        artifact_id = stable_id("art", str(artifact_path))
        return self._parse(
            project_id="unknown_project",
            artifact_id=artifact_id,
            path=artifact_path,
        ).atoms

    def parse_artifact(  # type: ignore[override]
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        return self._parse(project_id=project_id, artifact_id=artifact_id, path=path)

    # ───── internals ─────

    def _parse(self, *, project_id: str, artifact_id: str, path: Path) -> ParserOutput:
        text = path.read_text(encoding="utf-8", errors="replace")
        atoms: list[EvidenceAtom] = []
        for idx, block in enumerate(_iter_markdown_blocks(text)):
            atoms.extend(
                self._emit_atoms_for_block(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    block=block,
                    block_index=idx,
                )
            )
        # Image references (``![alt](src)``) are binary regions — emit a
        # located marker so a linked diagram / screenshot can't silently
        # vanish. region_ref matches the census location (``image/<src>``).
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            atoms.append(region_marker(
                project_id=project_id, artifact_id=artifact_id, filename=path.name,
                artifact_type=ArtifactType.txt, parser_version=self.parser_version,
                region_ref=f"image/{m.group(1)}", kind="image_marker", label="image",
            ))
        return ParserOutput(atoms=atoms)

    @staticmethod
    def _split_into_qty_sentences(text: str) -> list[tuple[str, int]]:
        """Return ``[(sentence_text, offset)]`` when ``text`` has 2+
        sentences that each carry their own ``_QTY_RE`` match.

        Otherwise return ``[]`` so the caller keeps the original block
        intact. Sentences are split on ``.``/``?``/``!`` followed by
        whitespace + uppercase. Empty list means "don't split".
        """
        if not text or len(text) < 30:
            return []
        # Sentence-end pattern: punctuation then space-and-capital.
        boundaries = list(re.finditer(r"[.?!]\s+(?=[A-Z])", text))
        if not boundaries:
            return []
        sentences: list[tuple[str, int]] = []
        start = 0
        for b in boundaries:
            end = b.end()
            sentences.append((text[start:end].strip(), start))
            start = end
        if start < len(text):
            sentences.append((text[start:].strip(), start))
        # Only return when 2+ sentences AND at least 2 carry either a
        # qty match OR a clear scope-impacting signal (exclusion,
        # change order, delta). Single-signal paragraphs stay merged.
        if len(sentences) < 2:
            return []
        def _is_scope_impacting(s: str) -> bool:
            if _QTY_RE.search(s):
                return True
            if _EXCLUSION_RE.search(s):
                return True
            if _CHANGE_ORDER_RE.search(s):
                return True
            if _CHANGE_DELTA_RE.search(s):
                return True
            return False
        if sum(1 for s, _ in sentences if _is_scope_impacting(s)) < 2:
            return []
        return sentences

    def _emit_atoms_for_sentence(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        sub_block: MarkdownBlock,
        block_index: int,
        sentence_index: int,
    ) -> list[EvidenceAtom]:
        atom_types = _classify_block(
            sub_block.text,
            sub_block.section_path,
            block_kind=sub_block.block_kind,
        )
        if not atom_types:
            atom_types = [AtomType.scope_item]
        locator = {
            "line_start": sub_block.line_start,
            "line_end": sub_block.line_end,
            "section_path": list(sub_block.section_path),
            "block_kind": sub_block.block_kind,
            "block_index": block_index,
            "sentence_index": sentence_index,
        }
        source_ref = SourceRef(
            id=stable_id(
                "src", artifact_id, "md", sub_block.line_start, sub_block.line_end, block_index, sentence_index
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.txt,
            filename=filename,
            locator=locator,
            extraction_method="markdown_sentence_split",
            parser_version=self.parser_version,
        )
        entity_keys = _entity_keys_from_text(sub_block.text)
        authority_class = _authority_for_block(sub_block.text, sub_block.section_path)
        out: list[EvidenceAtom] = []
        for atom_type in atom_types:
            value: dict[str, Any] = {
                "text": sub_block.text,
                "section_path": list(sub_block.section_path),
                "block_kind": sub_block.block_kind,
                "sentence_index": sentence_index,
            }
            qty_match = _QTY_RE.search(sub_block.text)
            if atom_type is AtomType.quantity and qty_match:
                value["quantity"] = parse_quantity(qty_match.group("qty"))
                value["unit"] = qty_match.group("unit").lower()
            if atom_type is AtomType.constraint:
                sla = _extract_sla_value(sub_block.text)
                if sla:
                    value["sla"] = sla
            if atom_type is AtomType.customer_instruction:
                delta_match = _CHANGE_DELTA_RE.search(sub_block.text)
                if delta_match:
                    try:
                        from_v = int(delta_match.group(1))
                        to_v = int(delta_match.group(2))
                        value["change_delta"] = {
                            "from": from_v,
                            "to": to_v,
                            "delta": to_v - from_v,
                        }
                    except (ValueError, IndexError):
                        pass
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "markdown",
                        atom_type.value, sub_block.line_start, sub_block.line_end,
                        sentence_index, sub_block.text,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=sub_block.text,
                    normalized_text=normalize_text(sub_block.text),
                    value=value,
                    entity_keys=entity_keys,
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=authority_class,
                    confidence=0.91,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
        return out

    def _emit_atoms_for_block(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        block: MarkdownBlock,
        block_index: int,
    ) -> list[EvidenceAtom]:
        atom_types = _classify_block(
            block.text,
            block.section_path,
            block_kind=block.block_kind,
        )
        if not atom_types:
            atom_types = [AtomType.scope_item]

        # Self-contradiction unlock: when a paragraph block contains
        # MULTIPLE sentences and each sentence carries its own quantity
        # match, split the block into one atom per sentence. This lets
        # the cross-doc binding fire intra-doc edges when the same
        # device is mentioned with conflicting counts ("24 cameras...
        # 30 cameras... 28 cameras."). Tables / bullets are untouched.
        if block.block_kind == "paragraph":
            sentences = self._split_into_qty_sentences(block.text)
            if len(sentences) >= 2:
                out: list[EvidenceAtom] = []
                for s_idx, (s_text, s_offset) in enumerate(sentences):
                    sub_block = MarkdownBlock(
                        text=s_text,
                        line_start=block.line_start,
                        line_end=block.line_end,
                        section_path=block.section_path,
                        block_kind="paragraph",
                    )
                    out.extend(self._emit_atoms_for_sentence(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=filename,
                        sub_block=sub_block,
                        block_index=block_index,
                        sentence_index=s_idx,
                    ))
                return out

        locator = {
            "line_start": block.line_start,
            "line_end": block.line_end,
            "section_path": list(block.section_path),
            "block_kind": block.block_kind,
            "block_index": block_index,
        }

        source_ref = SourceRef(
            id=stable_id(
                "src", artifact_id, "md", block.line_start, block.line_end, block_index
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.txt,
            filename=filename,
            locator=locator,
            extraction_method="markdown_line_block",
            parser_version=self.parser_version,
        )

        entity_keys = _entity_keys_from_text(block.text)
        authority_class = _authority_for_block(block.text, block.section_path)

        out: list[EvidenceAtom] = []
        for atom_type in atom_types:
            value: dict[str, Any] = {
                "text": block.text,
                "section_path": list(block.section_path),
                "block_kind": block.block_kind,
            }
            qty_match = _QTY_RE.search(block.text)
            if atom_type is AtomType.quantity and qty_match:
                value["quantity"] = parse_quantity(qty_match.group("qty"))
                value["unit"] = qty_match.group("unit").lower()

            # SLA structured payload — uptime, response/resolution
            # times, service credits, support coverage hours. Attached
            # to constraint atoms so a PM can render an SLA tab
            # directly without re-parsing the raw text.
            if atom_type is AtomType.constraint:
                sla = _extract_sla_value(block.text)
                if sla:
                    value["sla"] = sla

            # Change-order delta payload — "reduce 24 to 18" /
            # "from 12 to 8" surfaces as {from: 24, to: 18, delta: -6}
            # on the customer_instruction atom so a PM rollup can show
            # scope changes with explicit deltas.
            if atom_type is AtomType.customer_instruction:
                delta_match = _CHANGE_DELTA_RE.search(block.text)
                if delta_match:
                    try:
                        from_v = int(delta_match.group(1))
                        to_v = int(delta_match.group(2))
                        value["change_delta"] = {
                            "from": from_v,
                            "to": to_v,
                            "delta": to_v - from_v,
                        }
                    except (ValueError, IndexError):
                        pass

            # Stakeholder-row payload. When a markdown table row
            # carries a name + (email or phone) — the unmistakable
            # contact-row shape — parse the cells, write structured
            # role/email/phone into the value, and emit a
            # ``stakeholder:<slug>`` entity_key so the downstream PM
            # rollup can group risks/actions/decisions by the owner.
            if block.block_kind == "table_row":
                section_blob_lower = " / ".join(block.section_path).lower()
                if _looks_like_stakeholder_row(block.text, section_blob_lower, block.block_kind):
                    cells = [c.strip() for c in block.text.strip("|").split("|")]
                    value["table_cells"] = cells
                    name_cell = next((c for c in cells if _looks_like_person_name(c)), None)
                    email_cell = next((c for c in cells if _EMAIL_RE.search(c)), None)
                    phone_cell = next((c for c in cells if _PHONE_RE.search(c)), None)
                    role_cell = next(
                        (c for c in cells if c and c != name_cell and c != email_cell and c != phone_cell),
                        None,
                    )
                    if name_cell:
                        value["name"] = name_cell
                        slug = re.sub(r"[^a-z0-9]+", "_", name_cell.lower()).strip("_")
                        if slug:
                            stakeholder_key = f"stakeholder:{slug}"
                            if stakeholder_key not in entity_keys:
                                entity_keys = sorted(set(entity_keys) | {stakeholder_key})
                    if role_cell:
                        value["role"] = role_cell
                    if email_cell:
                        m = _EMAIL_RE.search(email_cell)
                        if m:
                            value["email"] = m.group(0)
                            email_key = f"email:{m.group(0).lower()}"
                            if email_key not in entity_keys:
                                entity_keys = sorted(set(entity_keys) | {email_key})
                    if phone_cell:
                        m = _PHONE_RE.search(phone_cell)
                        if m:
                            value["phone"] = m.group(0)

            # PR4 — risk-row payload. When a markdown table row gets
            # typed as risk, parse the | … | … | cells so downstream
            # consumers can read severity / impact / mitigation
            # without re-tokenizing.
            if atom_type is AtomType.risk and block.block_kind == "table_row":
                cells = [c.strip() for c in block.text.strip("|").split("|")]
                value["table_cells"] = cells
                if cells:
                    value["risk_id"] = cells[0]
                if len(cells) >= 3:
                    value["risk_summary"] = cells[1]
                    value["severity"] = cells[2]
                if len(cells) >= 4:
                    # Risk register layouts:
                    #   | ID | Risk | Severity | Owner | Mitigation |
                    # so cells[3] is the Owner. Promote to a
                    # structured ``owner`` field and emit a
                    # stakeholder entity_key so the downstream PM
                    # rollup can group risks by responsible party.
                    raw_owner = cells[3].strip()
                    if raw_owner and not _looks_like_severity_or_probability(raw_owner):
                        value["owner"] = raw_owner
                        slug = re.sub(r"[^a-z0-9]+", "_", raw_owner.lower()).strip("_")
                        if slug and slug not in {"owner", "tbd", "unknown", "n_a"}:
                            stakeholder_key = f"stakeholder:{slug}"
                            if stakeholder_key not in entity_keys:
                                entity_keys = sorted(set(entity_keys) | {stakeholder_key})
                    else:
                        value["impact_or_probability"] = cells[3]
                if len(cells) >= 5:
                    value["mitigation"] = cells[4]

            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm",
                        project_id,
                        artifact_id,
                        "markdown",
                        atom_type.value,
                        block.line_start,
                        block.line_end,
                        block.text,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=block.text,
                    normalized_text=normalize_text(block.text),
                    value=value,
                    entity_keys=entity_keys,
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=authority_class,
                    confidence=0.91,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
        return out


# ────────────────────────────── markdown tokenizer ─────────────────────


def _iter_markdown_blocks(text: str):
    lines = text.splitlines()
    section_stack: list[tuple[int, str]] = []
    paragraph: list[str] = []
    paragraph_start = 0

    def section_path() -> tuple[str, ...]:
        return tuple(title for _, title in section_stack)

    def flush_paragraph(end_line: int):
        nonlocal paragraph, paragraph_start
        if paragraph:
            joined = " ".join(x.strip() for x in paragraph if x.strip()).strip()
            if joined:
                yield MarkdownBlock(
                    text=joined,
                    line_start=paragraph_start,
                    line_end=end_line,
                    section_path=section_path(),
                    block_kind="paragraph",
                )
        paragraph = []
        paragraph_start = 0

    for i, line in enumerate(lines, start=1):
        heading = _HEADING_RE.match(line)
        if heading:
            yield from flush_paragraph(i - 1)
            level = len(heading.group(1))
            title = heading.group("title").strip()
            section_stack = [(lvl, name) for lvl, name in section_stack if lvl < level]
            section_stack.append((level, title))
            yield MarkdownBlock(title, i, i, section_path(), "heading")
            continue

        bullet = _BULLET_RE.match(line) or _NUMBERED_RE.match(line)
        if bullet:
            yield from flush_paragraph(i - 1)
            yield MarkdownBlock(
                text=bullet.group("text").strip(),
                line_start=i,
                line_end=i,
                section_path=section_path(),
                block_kind="bullet",
            )
            continue

        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            yield from flush_paragraph(i - 1)
            is_separator = re.match(
                r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$",
                stripped,
            )
            if not is_separator:
                yield MarkdownBlock(stripped, i, i, section_path(), "table_row")
            continue

        if not stripped:
            yield from flush_paragraph(i - 1)
            continue

        if not paragraph:
            paragraph_start = i
        paragraph.append(line)

    yield from flush_paragraph(len(lines))


# ────────────────────────────── classifiers ────────────────────────────


_SLA_UPTIME_RE = re.compile(r"\b(?:uptime|availability)\s*(?:sla)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_SLA_RESPONSE_RE = re.compile(r"\b(?:response\s+time|response)\s*[:=]?\s*(\d+)\s*(hour|hr|min|minute|day)s?", re.IGNORECASE)
_SLA_RESOLUTION_RE = re.compile(r"\b(?:resolution|mttr|mean\s+time\s+to\s+(?:repair|resolve))\s*[:=]?\s*[<>]?\s*(\d+)\s*(hour|hr|min|minute|day)s?", re.IGNORECASE)
_SLA_CREDIT_RE = re.compile(r"\b(?:service\s+credit|credit)s?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_SLA_COVERAGE_RE = re.compile(r"\b(\d{1,2})\s*x\s*(\d{1,2})\b", re.IGNORECASE)


def _extract_sla_value(text: str) -> dict:
    """Return structured SLA fields parsed from constraint text."""
    out: dict = {}
    m = _SLA_UPTIME_RE.search(text)
    if m:
        out["uptime_pct"] = float(m.group(1))
    m = _SLA_RESPONSE_RE.search(text)
    if m:
        out["response_time"] = f"{m.group(1)} {m.group(2).lower()}s"
    m = _SLA_RESOLUTION_RE.search(text)
    if m:
        out["resolution_time"] = f"{m.group(1)} {m.group(2).lower()}s"
    m = _SLA_CREDIT_RE.search(text)
    if m:
        out["service_credit_pct"] = float(m.group(1))
    m = _SLA_COVERAGE_RE.search(text)
    if m:
        out["coverage_hours_per_week"] = f"{m.group(1)}x{m.group(2)}"
    return out


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
_NAME_RE = re.compile(r"^\s*[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,3}\s*$")
_STAKEHOLDER_SECTION_RE = re.compile(
    r"\b(stakeholder|contact|directory|roster|team|attendee|distribution)s?\b",
    re.IGNORECASE,
)
_ROLE_TITLE_TOKENS = frozenset({
    "manager", "managers", "director", "directors", "sponsor", "sponsors",
    "officer", "officers", "supervisor", "supervisors", "lead", "leads",
    "coordinator", "coordinators", "administrator", "administrators",
    "owner", "owners", "approver", "approvers", "engineer", "engineers",
    "architect", "architects", "analyst", "analysts", "consultant", "consultants",
    "specialist", "specialists", "head", "chief", "principal", "principals",
    "pm", "po", "vp", "ceo", "cto", "cio", "ciso", "cfo", "coo", "svp", "evp",
    "site", "network", "project", "technical", "security", "vendor", "customer",
})


def _looks_like_person_name(cell: str) -> bool:
    """True when ``cell`` looks like an actual person's name (not a
    role title or compound role/site phrase)."""
    if not _NAME_RE.match(cell):
        return False
    tokens = cell.strip().split()
    if not tokens or len(tokens) > 4:
        return False
    lowered = {t.lower() for t in tokens}
    # If any token is a role/title word, it's a role phrase not a name.
    if lowered & _ROLE_TITLE_TOKENS:
        return False
    return True


def _looks_like_stakeholder_row(text: str, section_blob: str, block_kind: str) -> bool:
    """True when a markdown table row looks like a stakeholder
    directory entry (role / name / email / phone columns)."""
    if block_kind != "table_row":
        return False
    cells = [c.strip() for c in text.strip("|").split("|")]
    if len(cells) < 2:
        return False
    has_email = any(_EMAIL_RE.search(c) for c in cells)
    has_phone = any(_PHONE_RE.search(c) for c in cells)
    has_name = any(_looks_like_person_name(c) for c in cells)
    in_stakeholder_section = bool(_STAKEHOLDER_SECTION_RE.search(section_blob))
    # Stakeholder row signals: name + (email OR phone) anywhere
    # OR we're inside a Stakeholders section with a name.
    return (has_name and (has_email or has_phone)) or (in_stakeholder_section and has_name)


_SEVERITY_PROB_TOKEN = re.compile(
    r"^\s*(?:critical|high|medium|med|low|info|"
    r"very\s+high|very\s+low|"
    r"\d{1,3}\s*%|"
    r"likely|unlikely|certain|possible|rare|frequent"
    r")\s*$",
    re.IGNORECASE,
)


def _looks_like_severity_or_probability(text: str) -> bool:
    return bool(_SEVERITY_PROB_TOKEN.match(text or ""))


def _looks_like_markdown_risk_row(
    text: str, section_blob: str, block_kind: str
) -> bool:
    """PR4 (post-v3 review) — true when this block is a markdown
    table row from a Risk Register / RAID section."""
    if block_kind != "table_row":
        return False
    if _RISK_SECTION_RE.search(section_blob):
        return True
    if _RISK_ROW_ID_RE.search(text):
        return True
    cells = [c.strip() for c in text.strip("|").split("|")]
    blob = " ".join(cells[:5]).lower()
    return bool(
        "severity" in blob
        and ("impact" in blob or "probability" in blob or "risk" in blob)
    )


def _classify_block(
    text: str,
    section_path: tuple[str, ...],
    *,
    block_kind: str = "",
) -> list[AtomType]:
    section_blob = " / ".join(section_path).lower()
    blob = f"{section_blob} {text}".lower()

    # PR4 — risk rows always classify as risk and never as quantity,
    # even if a column happens to contain "5 items" / "Medium 3".
    if _looks_like_markdown_risk_row(text, section_blob, block_kind):
        return [AtomType.risk]

    types: list[AtomType] = []
    if _EXCLUSION_RE.search(blob):
        types.append(AtomType.exclusion)
    if _ASSUMPTION_RE.search(blob):
        types.append(AtomType.assumption)
    if _OPEN_Q_RE.search(blob):
        types.append(AtomType.open_question)
    if _CONSTRAINT_RE.search(blob):
        types.append(AtomType.constraint)
    if _QTY_RE.search(text):
        types.append(AtomType.quantity)
    if _MONEY_RE.search(text):
        types.append(AtomType.vendor_line_item)
    if "customer responsibility" in blob or "owner responsibility" in blob:
        types.append(AtomType.customer_instruction)
    # Change-order signal: when the section path includes "change
    # order" or the text matches a delta phrase ("reduce 24 to 18",
    # "approved revised scope of 18", "added 2 since SOW"), emit a
    # customer_instruction atom so the packetizer routes this to the
    # change-order packet family.
    if _CHANGE_ORDER_RE.search(blob) or "change order" in section_blob or _CHANGE_DELTA_RE.search(text):
        if AtomType.customer_instruction not in types:
            types.append(AtomType.customer_instruction)
    return list(dict.fromkeys(types))


def _authority_for_block(text: str, section_path: tuple[str, ...]) -> AuthorityClass:
    blob = " ".join(section_path).lower() + " " + text.lower()
    if any(x in blob for x in ["customer", "owner", "client", "district", "agency"]):
        return AuthorityClass.customer_current_authored
    if any(x in blob for x in ["vendor", "quote", "pricing", "proposal", "bom"]):
        return AuthorityClass.vendor_quote
    return AuthorityClass.contractual_scope


def _entity_keys_from_text(text: str) -> list[str]:
    keys: list[str] = []
    lowered = text.lower()
    for vendor in [
        "cisco", "belden", "panduit", "commscope", "genetec", "axis", "meraki",
        "aruba", "apc", "biamp", "qsc", "extron",
    ]:
        if vendor in lowered:
            keys.append(normalize_entity_key("vendor", vendor))

    site_match = re.search(
        r"\b([A-Z][A-Za-z0-9.'-]+(?:\s+[A-Z][A-Za-z0-9.'-]+){0,6}\s+"
        r"(School|Campus|Building|Center|Centre|Auditorium|Courthouse|Library|Hospital|Facility))\b",
        text,
    )
    if site_match:
        phrase = site_match.group(1)
        # Role-marker guard: "Site Manager Main Campus" / "Network
        # Director West Wing" describe a ROLE responsible for a place,
        # not a site name. Skip when the phrase contains a strong role
        # marker (Manager, Director, Sponsor, Officer, Lead, ...).
        phrase_lower_tokens = {t.lower() for t in phrase.split()}
        role_markers = {
            "manager", "managers", "director", "directors", "sponsor", "sponsors",
            "officer", "officers", "supervisor", "supervisors", "lead", "leads",
            "coordinator", "coordinators", "administrator", "administrators",
            "owner", "owners", "approver", "approvers", "engineer", "engineers",
            "architect", "architects", "analyst", "analysts",
        }
        if not (phrase_lower_tokens & role_markers):
            site_key = normalize_entity_key("site", phrase)
            if site_key:
                keys.append(site_key)
    return sorted(set(keys))
