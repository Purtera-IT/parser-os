"""OrbitBrief PDF parser — wires the color-driven page OS extractor into Parser OS.

Pipeline per artifact::

    PDF -> orbitbrief_page_os.detect (per page)
        -> overlay JSON payload
        -> structured_extract.extract_structured (per page)
        -> orbitbrief.pdf.structured.v1 document (all pages, hierarchical)
        -> structured.md projection (LLM-friendly mirror of the JSON)
        -> EvidenceAtom stream (one atom per content block, typed by
           section context so OrbitBrief knows what each chunk *means*)

The structured JSON + markdown pair is the "perfect compressible
OrbitBrief input format" for a single PDF.  See
``app.core.orbitbrief_envelope`` for the project-level envelope that
fuses every parser's structured projection into a single
``orbitbrief.input.v1`` payload an open-source LLM can swallow in one
prompt.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser

PARSER_NAME = "orbitbrief_pdf"
PARSER_VERSION = "orbitbrief_pdf_v3"
STRUCTURED_SCHEMA_VERSION = "orbitbrief.pdf.structured.v1"
DERIVED_DIR_SUFFIX = ".derived"
STRUCTURED_FILENAME = "structured.json"
STRUCTURED_MARKDOWN_FILENAME = "structured.md"
EXTRACTION_METHOD = "orbitbrief_pdf_color_driven_v1"

DEFAULT_BLOCK_CONFIDENCE = 0.88
DEFAULT_NOTE_CONFIDENCE = 0.78
TABLE_ROW_CONFIDENCE = 0.92  # tables are the most-trustworthy structure on a page

PDF_MAGIC = b"%PDF-"


class OrbitBriefPdfParser(BaseParser):
    """Parses ``.pdf`` artifacts into the OrbitBrief structured schema and EvidenceAtoms."""

    parser_name = PARSER_NAME
    parser_version = PARSER_VERSION
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".pdf"],
        supported_artifact_types=[ArtifactType.pdf],
        emitted_atom_types=[
            AtomType.scope_item,
            AtomType.assumption,
            AtomType.constraint,
        ],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=True,
    )

    def match(
        self,
        path: Path,
        sample_text: str | None,
        domain_pack: DomainPack | None,
    ) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        reasons: list[str] = []
        confidence = 0.0
        if suffix == ".pdf":
            reasons.append("pdf_extension")
            confidence = 0.95
        # Magic-byte sniff so a PDF dropped with the wrong extension still
        # routes here.  Cheap (5 bytes) and unambiguous.
        try:
            with path.open("rb") as fh:
                head = fh.read(len(PDF_MAGIC))
            if head == PDF_MAGIC:
                reasons.append("pdf_magic_bytes")
                confidence = max(confidence, 0.90)
        except OSError:
            pass
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.pdf,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        structured_doc = build_structured_document(path)
        write_structured_doc(path, structured_doc)
        write_structured_markdown(path, structured_doc)
        atoms = list(
            atoms_from_structured_doc(
                structured_doc=structured_doc,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                parser_version=self.parser_version,
            )
        )
        # Surface the derived artifacts in the parser output so the
        # compiler-level cache captures them and replays them on every
        # cache hit.  This guarantees ``<stem>.derived/structured.json``
        # and ``structured.md`` are always present after a compile, even
        # for cache-hot artifacts.
        derived = derived_dir_for(path)
        return ParserOutput(
            atoms=atoms,
            derived_files=[
                {
                    "relative_path": f"{derived.name}/{STRUCTURED_FILENAME}",
                    "content_kind": "json",
                    "content_json": structured_doc,
                },
                {
                    "relative_path": f"{derived.name}/{STRUCTURED_MARKDOWN_FILENAME}",
                    "content_kind": "markdown",
                    "content_text": structured_doc_to_markdown(structured_doc),
                },
            ],
        )


# ──────────────────────── public helpers ─────────────────────────────────


def build_structured_document(pdf_path: Path) -> dict[str, Any]:
    """Build the full multi-page OrbitBrief structured document for a PDF.

    Output schema is ``orbitbrief.pdf.structured.v1``:

        {
          "schema_version": "orbitbrief.pdf.structured.v1",
          "source": {"filename", "page_count"},
          "document": {"title", "metadata"},
          "pages": [
              {
                "page": int,
                "title": str | None,
                "metadata": [str, ...],
                "outline": [{"level", "heading", "block_count"}],
                "sections": [
                    {
                      "heading", "level",
                      "blocks": [
                          {"id", "kind": "paragraph", "text"},
                          {"id", "kind": "bullet_list", "intro"?, "items": [...]},
                          {"id", "kind": "table", "columns", "rows"},
                          {"id", "kind": "note", "text"},
                      ],
                      "subsections": [...]
                    },
                    ...
                ],
              },
              ...
          ],
        }
    """
    from orbitbrief_page_os.segmentation.core.config import Cfg
    from orbitbrief_page_os.segmentation.core.pipeline import build_pipeline
    from orbitbrief_page_os.segmentation.detect_standalone import _box_to_dict
    from orbitbrief_page_os.segmentation.structured_extract import extract_structured

    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - env-specific
        raise RuntimeError(
            "PyMuPDF (fitz) is required for the OrbitBrief PDF parser"
        ) from exc

    pdf_path = Path(pdf_path)
    cfg = Cfg()
    pipeline = build_pipeline()

    pages: list[dict[str, Any]] = []
    document_title: str | None = None
    document_metadata: list[str] = []
    seen_metadata: set[str] = set()

    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)

    for page_index in range(page_count):
        state = pipeline.run(str(pdf_path), page_index=page_index, cfg=cfg)
        result = state.result
        assert result is not None, "overlay pipeline produced no result"
        payload = {
            "pdf": str(pdf_path),
            "page": page_index,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "debug_stats": result.debug_stats,
            "boxes": [_box_to_dict(b) for b in result.boxes],
        }
        struct = extract_structured(payload, pdf_path=pdf_path)
        page_doc = (struct.get("document") or {})
        page_title = page_doc.get("title")
        page_meta = list(page_doc.get("metadata") or [])

        if not document_title and page_title:
            document_title = page_title
        for entry in page_meta:
            if not entry:
                continue
            key = normalize_text(entry)
            if not key or key in seen_metadata:
                continue
            seen_metadata.add(key)
            document_metadata.append(entry)

        sections = list(struct.get("sections") or [])
        _stamp_section_and_block_ids(sections, page_index)

        pages.append(
            {
                "page": page_index,
                "title": page_title,
                "metadata": page_meta,
                "outline": list(struct.get("outline") or []),
                "sections": sections,
            }
        )

    return {
        "schema_version": STRUCTURED_SCHEMA_VERSION,
        "source": {
            "filename": pdf_path.name,
            "page_count": page_count,
        },
        "document": {
            "title": document_title,
            "metadata": document_metadata,
        },
        "pages": pages,
    }


def write_structured_doc(pdf_path: Path, structured_doc: dict[str, Any]) -> Path:
    """Persist the structured doc to ``<pdf>.derived/structured.json``."""
    derived_dir = derived_dir_for(pdf_path)
    derived_dir.mkdir(parents=True, exist_ok=True)
    out = derived_dir / STRUCTURED_FILENAME
    out.write_text(
        json.dumps(structured_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def write_structured_markdown(pdf_path: Path, structured_doc: dict[str, Any]) -> Path:
    """Persist the LLM-friendly markdown projection next to the JSON.

    The markdown mirrors the JSON structure 1:1 with stable HTML anchors
    (``<a id="blk_..."></a>`` / ``<a id="sec_..."></a>``) so an LLM can
    cite a region by anchor and a UI can scroll to the same place.
    """
    derived_dir = derived_dir_for(pdf_path)
    derived_dir.mkdir(parents=True, exist_ok=True)
    out = derived_dir / STRUCTURED_MARKDOWN_FILENAME
    out.write_text(structured_doc_to_markdown(structured_doc), encoding="utf-8")
    return out


def derived_dir_for(pdf_path: Path) -> Path:
    """Return the canonical derived-artifact directory for ``pdf_path``.

    Convention: a sibling directory named ``<stem>.derived`` next to the
    PDF.  For uploaded artifacts that lands under the project's
    ``.purtera_artifacts/<project>/<sha>.derived/``; for ad-hoc files it
    sits next to the source PDF.
    """
    pdf_path = Path(pdf_path)
    return pdf_path.with_name(f"{pdf_path.stem}{DERIVED_DIR_SUFFIX}")


def overlay_payload_and_extraction(
    pdf_path: str | Path,
    *,
    page_index: int = 0,
    overlay_dir: Path | None = None,
    file_stem: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Run overlay detection + legacy text extraction for a single page.

    Mirrors what ``detect_standalone --json-out --extraction-out`` does
    so callers and tests can compute or persist the same overlay JSON +
    text extraction artifacts (PNG, ``.overlay.json``,
    ``.extraction.json``, ``.extraction.md``).

    Returns ``(overlay_payload, extraction_doc, written_paths)``.  When
    ``overlay_dir`` is ``None`` no files are written; when supplied,
    the four artifacts above are written under ``overlay_dir`` using
    ``file_stem`` (or ``"<pdf-stem>_p{NNNN}"``) as the filename root.
    """
    from orbitbrief_page_os.segmentation.core.config import Cfg
    from orbitbrief_page_os.segmentation.core.pipeline import (
        build_pipeline,
        render_overlay,
    )
    from orbitbrief_page_os.segmentation.detect_standalone import _box_to_dict
    from orbitbrief_page_os.segmentation.extract_overlay_text import (
        extract_from_overlay_json,
        write_extraction_artifacts,
    )

    pdf_path = Path(pdf_path).resolve()
    cfg = Cfg()
    pipeline = build_pipeline()
    state = pipeline.run(str(pdf_path), page_index=page_index, cfg=cfg)
    result = state.result
    rgb = state.rgb
    assert result is not None and rgb is not None, "overlay pipeline produced no result"

    payload = {
        "pdf": str(pdf_path),
        "page": page_index,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "debug_stats": result.debug_stats,
        "boxes": [_box_to_dict(b) for b in result.boxes],
    }
    doc = extract_from_overlay_json(payload, pdf_path=pdf_path)

    written: list[Path] = []
    if overlay_dir is not None:
        overlay_dir = Path(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or f"{pdf_path.stem}_p{page_index:04d}"
        png = overlay_dir / f"{stem}.png"
        ov_js = overlay_dir / f"{stem}.overlay.json"
        # write_extraction_artifacts always appends ``.extraction.json`` /
        # ``.extraction.md`` to its base path, so pass the bare stem to
        # avoid producing ``stem.extraction.extraction.json``.
        ex_base = overlay_dir / stem
        render_overlay(rgb, result, png, draw_labels=False)
        ov_js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ex_paths = write_extraction_artifacts(ex_base, doc)
        written = [png.resolve(), ov_js.resolve()] + sorted(
            Path(p).resolve() for p in ex_paths.values()
        )

    return payload, doc, written


# ──────────────────────── atom emission ──────────────────────────────────


def atoms_from_structured_doc(
    *,
    structured_doc: dict[str, Any],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    """Stream EvidenceAtoms from a structured document.

    One atom per content block (paragraph, bullet item, table row, or
    note).  Headings are not atoms — they become locator context
    (``section_path``) on the atoms beneath them so OrbitBrief can
    re-classify or re-aggregate without re-parsing.
    """
    for page in structured_doc.get("pages", []):
        page_index = int(page.get("page", 0))
        yield from _atoms_for_sections(
            sections=page.get("sections", []) or [],
            section_path=[],
            page_index=page_index,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
        )


def _atoms_for_sections(
    *,
    sections: Iterable[dict[str, Any]],
    section_path: list[str],
    page_index: int,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    for section in sections:
        heading = section.get("heading")
        path = section_path + ([heading] if heading else [])
        for block in section.get("blocks", []) or []:
            yield from _atoms_for_block(
                block=block,
                section_path=path,
                page_index=page_index,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
            )
        yield from _atoms_for_sections(
            sections=section.get("subsections", []) or [],
            section_path=path,
            page_index=page_index,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
        )


def _atoms_for_block(
    *,
    block: dict[str, Any],
    section_path: list[str],
    page_index: int,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> Iterator[EvidenceAtom]:
    kind = block.get("kind")
    block_id = block.get("id") or stable_id("blk", page_index, kind or "?", id(block))
    base_locator: dict[str, Any] = {
        "page": page_index,
        "block_id": block_id,
        "block_kind": kind,
        "section_path": section_path,
    }

    if kind == "paragraph":
        text = (block.get("text") or "").strip()
        if not text:
            return
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="paragraph")
        yield _make_atom(
            text=text,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            atom_type=atom_type,
            authority_class=authority,
            confidence=DEFAULT_BLOCK_CONFIDENCE,
            locator=base_locator,
            value={"kind": "paragraph"},
        )
        return

    if kind == "bullet_list":
        intro = (block.get("intro") or "").strip()
        if intro:
            intro_type, intro_auth = _classify_text_block(
                text=intro, section_path=section_path, kind="bullet_intro"
            )
            yield _make_atom(
                text=intro,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                atom_type=intro_type,
                authority_class=intro_auth,
                confidence=DEFAULT_BLOCK_CONFIDENCE,
                locator={**base_locator, "bullet_role": "intro"},
                value={"kind": "bullet_intro"},
            )
        # Bullets inherit the intro's classification context; for many docs
        # the intro line ("Partner(s) must:") is what colors every child.
        bullet_section_path = section_path + ([intro] if intro else [])
        for index, item in enumerate(block.get("items", []) or []):
            yield from _atoms_for_bullet(
                item=item,
                depth=1,
                path_indices=[index],
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                base_locator=base_locator,
                section_path=bullet_section_path,
            )
        return

    if kind == "table":
        columns = list(block.get("columns") or [])
        rows = list(block.get("rows") or [])
        sample_cells: list[str] = []
        for row in rows[:5]:
            if isinstance(row, dict):
                for value in row.values():
                    if value is None:
                        continue
                    s = str(value).strip()
                    if s:
                        sample_cells.append(s)
        atom_type, authority = _classify_table(
            section_path=section_path,
            columns=columns,
            sample_cells=sample_cells,
        )
        for row_index, row in enumerate(rows):
            row_text = _row_to_text(row)
            if not row_text:
                continue
            yield _make_atom(
                text=row_text,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                atom_type=atom_type,
                authority_class=authority,
                confidence=TABLE_ROW_CONFIDENCE,
                locator={**base_locator, "row_index": row_index},
                value={
                    "kind": "table_row",
                    "columns": columns,
                    "cells": dict(row),
                },
            )
        return

    if kind == "note":
        text = (block.get("text") or "").strip()
        if not text:
            return
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="note")
        yield _make_atom(
            text=text,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            atom_type=atom_type,
            authority_class=authority,
            confidence=DEFAULT_NOTE_CONFIDENCE,
            locator=base_locator,
            value={"kind": "note"},
        )
        return


def _atoms_for_bullet(
    *,
    item: dict[str, Any],
    depth: int,
    path_indices: list[int],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    base_locator: dict[str, Any],
    section_path: list[str],
) -> Iterator[EvidenceAtom]:
    text = (item.get("text") or "").strip()
    if text:
        atom_type, authority = _classify_text_block(text=text, section_path=section_path, kind="bullet")
        yield _make_atom(
            text=text,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            atom_type=atom_type,
            authority_class=authority,
            confidence=DEFAULT_BLOCK_CONFIDENCE,
            locator={
                **base_locator,
                "bullet_path": list(path_indices),
                "bullet_depth": depth,
            },
            value={"kind": "bullet", "depth": depth},
        )
    for child_index, child in enumerate(item.get("children", []) or []):
        yield from _atoms_for_bullet(
            item=child,
            depth=depth + 1,
            path_indices=path_indices + [child_index],
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
            parser_version=parser_version,
            base_locator=base_locator,
            section_path=section_path,
        )


def _row_to_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for col, val in row.items():
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        parts.append(f"{col}: {s}")
    return " | ".join(parts)


def _make_atom(
    *,
    text: str,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    atom_type: AtomType,
    authority_class: AuthorityClass,
    confidence: float,
    locator: dict[str, Any],
    value: dict[str, Any],
) -> EvidenceAtom:
    src_id = stable_id(
        "src",
        artifact_id,
        locator.get("page"),
        locator.get("block_id"),
        locator.get("bullet_path"),
        locator.get("row_index"),
    )
    source_ref = SourceRef(
        id=src_id,
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator=dict(locator),
        extraction_method=EXTRACTION_METHOD,
        parser_version=parser_version,
    )
    atom_id = stable_id(
        "atm",
        project_id,
        artifact_id,
        atom_type.value,
        text,
        locator.get("page"),
        locator.get("block_id"),
        locator.get("bullet_path"),
        locator.get("row_index"),
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=normalize_text(text),
        value=value,
        entity_keys=[],
        source_refs=[source_ref],
        receipts=[],
        authority_class=authority_class,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version=parser_version,
    )


# ──────────────────────── classification ─────────────────────────────────

# Each rule is a (regex, AtomType, AuthorityClass) tuple.  First match wins.
# Section-path matching is done case-insensitively against the joined path.
_SECTION_RULES: list[tuple[re.Pattern[str], AtomType, AuthorityClass]] = [
    (
        re.compile(r"\b(out\s*of\s*scope|exclusion(s)?|excluded|not\s+included)\b", re.I),
        AtomType.exclusion,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\bassumption(s)?\b", re.I),
        AtomType.assumption,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(constraint(s)?|requirement(s)?|prerequisite(s)?)\b", re.I),
        AtomType.constraint,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(open\s+question(s)?|tbd|to\s+be\s+determined|outstanding)\b", re.I),
        AtomType.open_question,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(decision(s)?|approved|approval(s)?)\b", re.I),
        AtomType.decision,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(
            r"\b(action\s*item(s)?|task(s)?|deliverable(s)?|to[-\s]?do(s)?|next\s+steps?)\b",
            re.I,
        ),
        AtomType.action_item,
        AuthorityClass.contractual_scope,
    ),
    (
        re.compile(r"\b(pricing|price|cost|quote(d)?|line\s+items?|sow\s+pricing|fees?)\b", re.I),
        AtomType.vendor_line_item,
        AuthorityClass.vendor_quote,
    ),
    (
        re.compile(
            r"\b(customer\s+(instruction|request|requirement)|client\s+(instruction|request|requirement))\b",
            re.I,
        ),
        AtomType.customer_instruction,
        AuthorityClass.customer_current_authored,
    ),
    (
        re.compile(
            r"\b(scope\s+of\s+work|sow|scope|kitting\s+requirement|partner\s+requirement|operational\s+expectation)\b",
            re.I,
        ),
        AtomType.scope_item,
        AuthorityClass.contractual_scope,
    ),
]

# Block-text overrides — applied after section rules, only when the text
# itself is unambiguous (modal verbs / question marks).  These let a
# constraint sentence in a "scope" section still be tagged as a constraint.
_TEXT_OVERRIDES: list[tuple[re.Pattern[str], AtomType]] = [
    (re.compile(r"\?\s*$"), AtomType.open_question),
    (re.compile(r"^\s*(must|shall|required to|is required to|will be required to)\b", re.I), AtomType.constraint),
    (re.compile(r"^\s*(do not|may not|cannot|must not|shall not)\b", re.I), AtomType.exclusion),
    (re.compile(r"^\s*(assume(s|d)?|assuming)\b", re.I), AtomType.assumption),
]

# Tight column-header regex: only fires on unambiguously-pricing words.
# We deliberately do NOT match bare "unit" / "total" / "amount" because
# those appear in scope-checklist tables ("Unit AP Installation",
# "Total Devices") that have nothing to do with pricing.  When in doubt,
# fall back to the section classifier.
_PRICING_COLUMN_HINTS = re.compile(
    r"\b("
    r"unit\s+(price|cost|rate)"
    r"|line\s+item(s)?"
    r"|extended\s+(price|cost|amount)"
    r"|subtotal"
    r"|hourly\s+rate"
    r"|price"
    r"|cost"
    r"|invoice"
    r"|fee(s)?"
    r"|rate\s+card"
    r")\b",
    re.I,
)
_CURRENCY_PATTERN = re.compile(r"(?:\$|£|€|usd|gbp|eur)\s*\d", re.I)


def _classify_text_block(
    *,
    text: str,
    section_path: list[str],
    kind: str,
) -> tuple[AtomType, AuthorityClass]:
    """Pick (AtomType, AuthorityClass) from section context + the block text.

    ``kind`` is the structural kind (``paragraph`` / ``bullet`` / ``note``).
    Notes always default to ``assumption / meeting_note`` unless the
    section path screams something different (e.g. a red callout under a
    pricing section is still a vendor signal, not a meeting note).
    """
    section_blob = " ".join(section_path or [])

    section_atom: AtomType | None = None
    section_auth: AuthorityClass | None = None
    for pattern, atom_type, auth in _SECTION_RULES:
        if pattern.search(section_blob):
            section_atom = atom_type
            section_auth = auth
            break

    text_atom: AtomType | None = None
    for pattern, atom_type in _TEXT_OVERRIDES:
        if pattern.search(text):
            text_atom = atom_type
            break

    if kind == "note":
        if section_atom is not None:
            # A red callout under a typed section keeps the section's authority
            # but stays an assumption (it's a callout, not a primary clause).
            return AtomType.assumption, section_auth or AuthorityClass.meeting_note
        return AtomType.assumption, AuthorityClass.meeting_note

    if text_atom is not None:
        # Text override wins over section default for definite signals.
        authority = section_auth or AuthorityClass.contractual_scope
        return text_atom, authority

    if section_atom is not None:
        return section_atom, section_auth or AuthorityClass.contractual_scope

    return AtomType.scope_item, AuthorityClass.contractual_scope


def _classify_table(
    *,
    section_path: list[str],
    columns: list[str],
    sample_cells: list[str] | None = None,
) -> tuple[AtomType, AuthorityClass]:
    """Pick (AtomType, AuthorityClass) for every row in a table.

    Strategy (in priority order):
      1. Section path screams a non-default role (exclusion / pricing /
         decision / etc) → use that.
      2. Column headers contain unambiguous pricing words ("unit price",
         "subtotal", "fee", etc.) → vendor_line_item / vendor_quote.
      3. Sample cell values contain currency markers ($, £, €, USD …) →
         vendor_line_item / vendor_quote.
      4. Default to scope_item / contractual_scope.

    We deliberately put the section classifier FIRST so a "SCOPE OF
    WORK" matrix doesn't get hijacked by a column whose name happens to
    contain the substring "unit".
    """
    section_blob = " ".join(section_path or [])
    for pattern, atom_type, auth in _SECTION_RULES:
        if pattern.search(section_blob):
            return atom_type, auth

    column_blob = " ".join(str(c) for c in columns)
    if _PRICING_COLUMN_HINTS.search(column_blob):
        return AtomType.vendor_line_item, AuthorityClass.vendor_quote

    if sample_cells:
        joined = " ".join(sample_cells[:20])
        if _CURRENCY_PATTERN.search(joined):
            return AtomType.vendor_line_item, AuthorityClass.vendor_quote

    return AtomType.scope_item, AuthorityClass.contractual_scope


# ──────────────────────── markdown projection ────────────────────────────


def structured_doc_to_markdown(structured_doc: dict[str, Any]) -> str:
    """Render the structured doc as LLM-friendly markdown with anchors.

    Output shape::

        ---
        schema: orbitbrief.pdf.structured.v1
        filename: <name>
        page_count: N
        ---

        # <document title>

        > **Metadata**
        > - line 1
        > - line 2

        <!-- page 0 -->

        ## <section>  <a id="sec_..."></a>

        <a id="blk_..."></a>
        body paragraph text

        <a id="blk_..."></a>
        **Intro:** intro line for bullet list
        - bullet
          - sub-bullet

        <a id="blk_..."></a>
        | col a | col b |
        |-------|-------|
        | v1    | v2    |

        > **Note:** note text  <a id="blk_..."></a>
    """
    lines: list[str] = []
    source = structured_doc.get("source") or {}
    document = structured_doc.get("document") or {}

    lines.append("---")
    lines.append(f"schema: {structured_doc.get('schema_version', STRUCTURED_SCHEMA_VERSION)}")
    if source.get("filename"):
        lines.append(f"filename: {source['filename']}")
    if source.get("page_count") is not None:
        lines.append(f"page_count: {source['page_count']}")
    lines.append("---")
    lines.append("")

    title = document.get("title")
    if title:
        lines.append(f"# {title}")
        lines.append("")

    metadata = document.get("metadata") or []
    if metadata:
        lines.append("> **Metadata**")
        for entry in metadata:
            lines.append(f"> - {entry}")
        lines.append("")

    for page in structured_doc.get("pages", []) or []:
        page_index = page.get("page", 0)
        lines.append(f"<!-- page {page_index} -->")
        lines.append("")
        page_meta = [m for m in (page.get("metadata") or []) if m and m not in metadata]
        if page_meta:
            for entry in page_meta:
                lines.append(f"_{entry}_")
            lines.append("")
        for section in page.get("sections", []) or []:
            _render_section_md(lines, section, depth=2)

    text = "\n".join(lines).rstrip() + "\n"
    return text


def _render_section_md(lines: list[str], section: dict[str, Any], *, depth: int) -> None:
    heading = (section.get("heading") or "").strip()
    section_id = section.get("id")
    if heading:
        prefix = "#" * min(max(depth, 1), 6)
        anchor = f'  <a id="{section_id}"></a>' if section_id else ""
        lines.append(f"{prefix} {heading}{anchor}")
        lines.append("")

    for block in section.get("blocks", []) or []:
        _render_block_md(lines, block)

    for child in section.get("subsections", []) or []:
        _render_section_md(lines, child, depth=depth + 1)


def _render_block_md(lines: list[str], block: dict[str, Any]) -> None:
    kind = block.get("kind")
    block_id = block.get("id")
    anchor = f'<a id="{block_id}"></a>' if block_id else ""

    if kind == "paragraph":
        text = (block.get("text") or "").strip()
        if not text:
            return
        if anchor:
            lines.append(anchor)
        lines.append(text)
        lines.append("")
        return

    if kind == "bullet_list":
        if anchor:
            lines.append(anchor)
        intro = (block.get("intro") or "").strip()
        if intro:
            lines.append(f"**Intro:** {intro}")
        for item in block.get("items", []) or []:
            _render_bullet_md(lines, item, depth=0)
        lines.append("")
        return

    if kind == "table":
        if anchor:
            lines.append(anchor)
        columns = list(block.get("columns") or [])
        rows = list(block.get("rows") or [])
        if not columns and rows:
            # Synthesize column names from the first row's keys to keep
            # markdown valid.
            columns = list(rows[0].keys())
        if not columns:
            raw = (block.get("raw_text") or "").strip()
            if raw:
                lines.append(raw)
                lines.append("")
            return
        lines.append("| " + " | ".join(_md_cell(c) for c in columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(_md_cell(row.get(col, "")) for col in columns)
                + " |"
            )
        lines.append("")
        return

    if kind == "note":
        text = (block.get("text") or "").strip()
        if not text:
            return
        suffix = f"  {anchor}" if anchor else ""
        lines.append(f"> **Note:** {text}{suffix}")
        lines.append("")
        return


def _render_bullet_md(lines: list[str], item: dict[str, Any], *, depth: int) -> None:
    text = (item.get("text") or "").strip()
    indent = "  " * depth
    if text:
        lines.append(f"{indent}- {text}")
    for child in item.get("children", []) or []:
        _render_bullet_md(lines, child, depth=depth + 1)


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Escape pipes so we never break the markdown table.
    return text.replace("|", "\\|").replace("\n", " ")


# ──────────────────────── internals ──────────────────────────────────────


def _stamp_section_and_block_ids(sections: list[dict[str, Any]], page_index: int) -> None:
    """Stamp every section and block in ``sections`` with a stable ``id``.

    IDs are deterministic strings derived from page + walk counter + kind
    (``sec_<digest>`` for sections, ``blk_<digest>`` for blocks), so a
    re-run on the same PDF produces the same ids without depending on
    object identity.
    """
    section_counter = [0]
    block_counter = [0]

    def visit(nodes: list[dict[str, Any]]) -> None:
        for section in nodes:
            section["id"] = stable_id(
                "sec", page_index, section_counter[0], section.get("level") or 1
            )
            section_counter[0] += 1
            for block in section.get("blocks", []) or []:
                block["id"] = stable_id(
                    "blk", page_index, block_counter[0], block.get("kind") or "?"
                )
                block_counter[0] += 1
            visit(section.get("subsections", []) or [])

    visit(sections)


__all__ = [
    "OrbitBriefPdfParser",
    "PARSER_NAME",
    "PARSER_VERSION",
    "STRUCTURED_SCHEMA_VERSION",
    "STRUCTURED_FILENAME",
    "STRUCTURED_MARKDOWN_FILENAME",
    "DERIVED_DIR_SUFFIX",
    "build_structured_document",
    "write_structured_doc",
    "write_structured_markdown",
    "structured_doc_to_markdown",
    "derived_dir_for",
    "overlay_payload_and_extraction",
    "atoms_from_structured_doc",
]
