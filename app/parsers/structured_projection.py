"""Shared structured-projection helpers for non-PDF parsers.

Every Parser OS parser ought to produce, on top of its ``EvidenceAtom``
stream, a small **structured document** describing the source's
hierarchy (sections, paragraphs, bullet lists, tables, notes). The
``OrbitBriefPdfParser`` already does this for PDFs; this module gives
the rest of the parsers (XLSX, CSV, DOCX, Email, Transcript) a uniform
way to do the same.

Why bother for non-PDFs?

* The OrbitBrief envelope renders one self-contained markdown document
  per artifact.  When non-PDF parsers don't emit a structured doc, the
  envelope synthesizes a poor "atom_projection" view — useful, but not
  great.  A real structured doc lets a single LLM prompt include
  every artifact in a project at full fidelity.
* Cache replay: parsers attach derived files via
  ``ParserOutput.derived_files`` and the compiler materializes them on
  cache hit AND miss, so the structured doc is always on disk next to
  the source artifact (under ``<stem>.derived/``).

The schema mirrors ``orbitbrief.pdf.structured.v1`` so the envelope
markdown renderer can handle every artifact uniformly:

    {
      "schema_version": "orbitbrief.<kind>.structured.v1",
      "source": {"filename": str, "artifact_type": str, "page_count": int},
      "document": {"title": str | None, "metadata": [str, ...]},
      "pages": [
          {
            "page": int,
            "title": str | None,
            "metadata": [str, ...],
            "outline": [{"level": int, "heading": str, "block_count": int}],
            "sections": [
                {
                  "id": "sec_<digest>",
                  "level": int,
                  "heading": str,
                  "blocks": [
                      {"id": "blk_<digest>", "kind": "paragraph", "text": str},
                      {"id": "blk_<digest>", "kind": "bullet_list", "intro": str | None,
                       "items": [{"text": str, "children": [...]}]},
                      {"id": "blk_<digest>", "kind": "table",
                       "columns": [str, ...], "rows": [{col: value}]},
                      {"id": "blk_<digest>", "kind": "note", "text": str},
                  ],
                  "subsections": [...]
                }
            ]
          }
      ]
    }

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.core.ids import stable_id
from app.core.schemas import ParserDerivedFile

DERIVED_DIR_SUFFIX = ".derived"
STRUCTURED_FILENAME = "structured.json"
STRUCTURED_MARKDOWN_FILENAME = "structured.md"


# ─────────────────────── builder API ─────────────────────────────────────


def make_paragraph(text: str) -> dict[str, Any]:
    return {"kind": "paragraph", "text": (text or "").strip()}


def make_note(text: str) -> dict[str, Any]:
    return {"kind": "note", "text": (text or "").strip()}


def make_bullet_list(
    items: Iterable[dict[str, Any] | str],
    *,
    intro: str | None = None,
) -> dict[str, Any]:
    """Build a bullet_list block.  ``items`` may be strings or dicts with
    ``{"text": str, "children": [...]}`` shape.
    """
    normalized: list[dict[str, Any]] = []
    for raw in items:
        if isinstance(raw, str):
            normalized.append({"text": raw.strip(), "children": []})
        elif isinstance(raw, dict):
            children = raw.get("children") or []
            normalized.append(
                {
                    "text": str(raw.get("text", "")).strip(),
                    "children": list(children),
                }
            )
    return {
        "kind": "bullet_list",
        "intro": intro.strip() if intro else None,
        "items": normalized,
    }


def make_table(columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    cols = [str(c) for c in columns]
    return {
        "kind": "table",
        "columns": cols,
        "rows": [dict(r) for r in rows],
    }


def make_section(
    *,
    heading: str,
    level: int = 2,
    blocks: list[dict[str, Any]] | None = None,
    subsections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "level": int(level),
        "heading": (heading or "").strip(),
        "blocks": list(blocks or []),
        "subsections": list(subsections or []),
    }


def make_page(
    *,
    page: int = 0,
    title: str | None = None,
    metadata: list[str] | None = None,
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sections = list(sections or [])
    outline = [
        {
            "level": s.get("level", 1),
            "heading": s.get("heading", ""),
            "block_count": len(s.get("blocks", []) or []),
        }
        for s in sections
    ]
    return {
        "page": int(page),
        "title": title,
        "metadata": list(metadata or []),
        "outline": outline,
        "sections": sections,
    }


def make_structured_document(
    *,
    schema_version: str,
    filename: str,
    artifact_type: str,
    title: str | None = None,
    metadata: list[str] | None = None,
    pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pages = list(pages or [])
    return {
        "schema_version": schema_version,
        "source": {
            "filename": filename,
            "artifact_type": artifact_type,
            "page_count": len(pages),
        },
        "document": {
            "title": title,
            "metadata": list(metadata or []),
        },
        "pages": pages,
    }


# ─────────────────────── id stamping ──────────────────────────────────────


def stamp_section_and_block_ids(
    structured_doc: dict[str, Any],
    *,
    artifact_seed: str,
) -> None:
    """Stamp every section & block with a deterministic ``sec_*`` / ``blk_*`` id.

    ``artifact_seed`` should be the artifact id (or any other stable
    identifier per artifact) so two artifacts can never collide on the
    same anchor.
    """
    section_counter = [0]
    block_counter = [0]

    def visit(sections: list[dict[str, Any]], page_index: int) -> None:
        for section in sections:
            section["id"] = stable_id(
                "sec",
                artifact_seed,
                page_index,
                section_counter[0],
                section.get("level") or 1,
            )
            section_counter[0] += 1
            for block in section.get("blocks", []) or []:
                block["id"] = stable_id(
                    "blk",
                    artifact_seed,
                    page_index,
                    block_counter[0],
                    block.get("kind") or "?",
                )
                block_counter[0] += 1
            visit(section.get("subsections", []) or [], page_index)

    for page in structured_doc.get("pages", []) or []:
        page_index = int(page.get("page", 0))
        visit(page.get("sections", []) or [], page_index)


# ─────────────────────── markdown renderer ───────────────────────────────


def structured_doc_to_markdown(structured_doc: dict[str, Any]) -> str:
    """Render any structured doc (PDF, XLSX, DOCX, email, transcript, …)
    as LLM-friendly markdown.

    The output is identical in shape to the OrbitBrief PDF markdown
    projection so the envelope renderer doesn't need parser-specific
    code paths.
    """
    lines: list[str] = []
    source = structured_doc.get("source") or {}
    document = structured_doc.get("document") or {}

    lines.append("---")
    lines.append(f"schema: {structured_doc.get('schema_version', '')}")
    if source.get("filename"):
        lines.append(f"filename: {source['filename']}")
    if source.get("artifact_type"):
        lines.append(f"artifact_type: {source['artifact_type']}")
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
        page_title = page.get("title")
        if page_title and page_title != title:
            lines.append(f"## {page_title}")
            lines.append("")
        for section in page.get("sections", []) or []:
            _render_section_md(lines, section, depth=2 if not page_title else 3)

    return "\n".join(lines).rstrip() + "\n"


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
    return text.replace("|", "\\|").replace("\n", " ")


# ─────────────────────── derived-file helpers ────────────────────────────


def derived_dir_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.stem}{DERIVED_DIR_SUFFIX}")


def parser_output_derived_files(
    *,
    structured_doc: dict[str, Any],
) -> list[ParserDerivedFile]:
    """Build the standard ``[structured.json, structured.md]`` pair."""
    derived_name = "{stem}.derived"  # placeholder — real path resolved by compiler.
    del derived_name
    return [
        ParserDerivedFile(
            relative_path=f"{{stem}}{DERIVED_DIR_SUFFIX}/{STRUCTURED_FILENAME}",
            content_kind="json",
            content_json=structured_doc,
        ),
        ParserDerivedFile(
            relative_path=f"{{stem}}{DERIVED_DIR_SUFFIX}/{STRUCTURED_MARKDOWN_FILENAME}",
            content_kind="markdown",
            content_text=structured_doc_to_markdown(structured_doc),
        ),
    ]


def derived_files_for(*, artifact_path: Path, structured_doc: dict[str, Any]) -> list[ParserDerivedFile]:
    """Build derived files with their *real* relative paths resolved against
    the artifact path.

    This is the function parsers should call from ``parse_artifact``.
    """
    derived = derived_dir_for(artifact_path).name  # e.g. "report.derived"
    return [
        ParserDerivedFile(
            relative_path=f"{derived}/{STRUCTURED_FILENAME}",
            content_kind="json",
            content_json=structured_doc,
        ),
        ParserDerivedFile(
            relative_path=f"{derived}/{STRUCTURED_MARKDOWN_FILENAME}",
            content_kind="markdown",
            content_text=structured_doc_to_markdown(structured_doc),
        ),
    ]


__all__ = [
    "DERIVED_DIR_SUFFIX",
    "STRUCTURED_FILENAME",
    "STRUCTURED_MARKDOWN_FILENAME",
    "derived_dir_for",
    "derived_files_for",
    "make_bullet_list",
    "make_note",
    "make_page",
    "make_paragraph",
    "make_section",
    "make_structured_document",
    "make_table",
    "stamp_section_and_block_ids",
    "structured_doc_to_markdown",
]
