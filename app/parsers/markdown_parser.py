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


_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(?P<text>.+?)\s*$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(?P<text>.+?)\s*$")

_EXCLUSION_RE = re.compile(
    r"\b(exclud(?:e|ed|es|ing)|out of scope|not included|not in scope|by others|nic)\b",
    re.I,
)
_ASSUMPTION_RE = re.compile(
    r"\b(assum(?:e|ed|ption|ptions)|subject to|provided by owner)\b",
    re.I,
)
_OPEN_Q_RE = re.compile(
    r"\?|\b(tbd|to be confirmed|unknown|pending|clarify|confirm)\b",
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
        return ParserOutput(atoms=atoms)

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
                    value["impact_or_probability"] = cells[3]

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
        site_key = normalize_entity_key("site", site_match.group(1))
        if site_key:
            keys.append(site_key)
    return sorted(set(keys))
