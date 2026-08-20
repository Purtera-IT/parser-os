"""Azure Document Intelligence OCR — high-quality replacement for tesseract.

v53: Tesseract OCR (the current fallback) is unreliable on scanned bid
PDFs — column misalignment, character noise, missed signatures. Azure
Document Intelligence (formerly Form Recognizer) returns clean text +
layout + per-cell tables + per-field confidence in one call.

Behaviour:
  * No-op when ``AZURE_DOC_INTEL_ENDPOINT`` + ``AZURE_DOC_INTEL_KEY``
    env vars are missing → falls through to the existing tesseract /
    easyocr / ollama_vision chain.
  * Single PDF page → ``prebuilt-read`` model → text content with
    page-level structure preserved.
  * Returns ``str`` of extracted text or ``""`` on any failure.

The SDK call is wrapped in try/except so any Azure-side issue (auth
miss, throttling, network) downgrades to the legacy chain rather
than crashing the parse.

Pricing reference (F0 free tier): 500 pages/month free. S0: ~$1.50/1k pages.
"""
from __future__ import annotations

import os
from typing import Any


def doc_intel_available() -> bool:
    """True iff endpoint + key are configured."""
    return bool(
        os.environ.get("AZURE_DOC_INTEL_ENDPOINT")
        and os.environ.get("AZURE_DOC_INTEL_KEY")
    )


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Run Azure Doc Intel ``prebuilt-read`` on image bytes (PNG/JPG/PDF page).

    Returns extracted text content or empty string on failure.
    """
    if not doc_intel_available() or not image_bytes:
        return ""
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        return ""

    endpoint = os.environ["AZURE_DOC_INTEL_ENDPOINT"].rstrip("/")
    key = os.environ["AZURE_DOC_INTEL_KEY"]

    try:
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=image_bytes,
            content_type="application/octet-stream",
        )
        result = poller.result()
        if hasattr(result, "content") and result.content:
            return str(result.content)
    except Exception:
        return ""
    return ""


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Run Azure Doc Intel ``prebuilt-layout`` on a full PDF.

    Returns list of {page_number, text, tables} dicts. Tables are
    structured: {row_count, column_count, cells: [{row, col, text}]}.
    Empty list on failure / when not configured.
    """
    if not doc_intel_available() or not pdf_bytes:
        return []
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        return []

    endpoint = os.environ["AZURE_DOC_INTEL_ENDPOINT"].rstrip("/")
    key = os.environ["AZURE_DOC_INTEL_KEY"]

    try:
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=pdf_bytes,
            content_type="application/pdf",
        )
        result = poller.result()
    except Exception:
        return []

    pages_out: list[dict[str, Any]] = []
    pages = getattr(result, "pages", None) or []
    tables = getattr(result, "tables", None) or []

    # Build page → tables index
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    for tbl in tables:
        bounding = getattr(tbl, "bounding_regions", None) or []
        if not bounding:
            continue
        page_num = bounding[0].page_number if bounding else 1
        cells = getattr(tbl, "cells", None) or []
        cells_out: list[dict[str, Any]] = []
        for c in cells:
            cells_out.append({
                "row": getattr(c, "row_index", 0),
                "col": getattr(c, "column_index", 0),
                "text": (getattr(c, "content", "") or "").strip(),
            })
        tables_by_page.setdefault(page_num, []).append({
            "row_count": getattr(tbl, "row_count", 0),
            "column_count": getattr(tbl, "column_count", 0),
            "cells": cells_out,
        })

    # Paragraph roles, indexed by page. prebuilt-layout labels TITLE,
    # SECTION_HEADING, PAGE_HEADER, PAGE_FOOTER and PAGE_NUMBER natively --
    # which is exactly what a pile of regexes in orbitbrief_pdf.py exists to
    # GUESS, less accurately. Measured on the Phillips Connect install spec:
    # 21 titles, 22 section headings, 54 footers, 27 page numbers, including a
    # letterhead address ("5231 California Avenue, Suite 110 | Irvine, CA
    # 92617") that the bare-URL footer heuristic does not match at all -- while
    # that same heuristic strips a line reading "report.docx" as furniture.
    #
    # Reading these and then inferring them from line shape anyway is paying
    # for a good reader and using a worse one.
    paras_by_page: dict[int, list[dict[str, Any]]] = {}
    for para in getattr(result, "paragraphs", None) or []:
        regions = getattr(para, "bounding_regions", None) or []
        pnum = regions[0].page_number if regions else 1
        role = getattr(para, "role", None)
        paras_by_page.setdefault(pnum, []).append({
            "text": getattr(para, "content", "") or "",
            # role is an SDK enum; normalise to a plain lowercase string so no
            # caller has to import azure.ai to compare against it.
            "role": (str(role).rsplit(".", 1)[-1].lower() if role else ""),
        })

    for page in pages:
        pn = getattr(page, "page_number", 0)
        lines = getattr(page, "lines", None) or []
        text = "\n".join((getattr(ln, "content", "") or "") for ln in lines)
        pages_out.append({
            "page_number": pn,
            "text": text,
            "tables": tables_by_page.get(pn, []),
            "paragraphs": paras_by_page.get(pn, []),
        })
    return pages_out


__all__ = [
    "doc_intel_available",
    "extract_text_from_image_bytes",
    "extract_pdf_pages",
]
