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


def _lines_from_doc_intel_result(result: Any) -> list[str]:
    """Prefer per-line OCR (keeps HubSpot name/qty columns separable)."""
    lines: list[str] = []
    for page in getattr(result, "pages", None) or []:
        for line in getattr(page, "lines", None) or []:
            text = str(getattr(line, "content", "") or "").strip()
            if text:
                lines.append(text)
    return lines


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Run Azure Doc Intel ``prebuilt-read`` on image bytes (PNG/JPG/PDF page).

    Returns extracted text content or empty string on failure.
    Prefer page lines over flattened ``content`` so order-table quantities
    that OCR onto their own line stay recoverable by the email parser.
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
        lines = _lines_from_doc_intel_result(result)
        if lines:
            return "\n".join(lines)
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

    for page in pages:
        pn = getattr(page, "page_number", 0)
        lines = getattr(page, "lines", None) or []
        text = "\n".join((getattr(ln, "content", "") or "") for ln in lines)
        pages_out.append({
            "page_number": pn,
            "text": text,
            "tables": tables_by_page.get(pn, []),
        })
    return pages_out


__all__ = [
    "doc_intel_available",
    "extract_text_from_image_bytes",
    "extract_pdf_pages",
]
