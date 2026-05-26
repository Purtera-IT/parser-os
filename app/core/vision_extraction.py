"""v43: Vision-LLM extraction for PDF tables / diagrams / structured
visual content that flat-text extraction loses.

Pipeline:
  1. Identify PDF pages flagged by the parser as "visual / table /
     diagram evidence not fully extracted" (atoms with that text).
  2. For each flagged page, render the page to an image (PIL/PNG).
  3. POST the image to qwen2.5vl:7b (multimodal LLM on Griffin's Mac)
     with a structured-extraction prompt.
  4. Parse response into tabular records: BOM line items, pricing
     tables, schedules, org charts, contact rosters, network diagrams.
  5. Convert each record to a synthetic atom for entity_resolution.

This recovers content from:
  - BOM tables (qty × unit price × total per line item)
  - Pricing schedules (per-school costs, lease costs)
  - Org charts / team rosters in table format
  - Schedule Gantts (Phase 1 / 2 / 3 timeline)
  - Network architecture diagrams
  - Contact rosters in table format
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


_DEFAULT_HOST = "http://100.114.102.122:11434"
_DEFAULT_VISION_MODEL = "qwen2.5vl:7b"


# ────────────────────────────────────────────────────────────────────
# Vision-LLM client (Ollama vision API)
# ────────────────────────────────────────────────────────────────────


def _encode_image_b64(image_bytes: bytes) -> str:
    """Encode raw PNG/JPEG bytes for ollama vision API."""
    return base64.b64encode(image_bytes).decode("ascii")


def call_vision_llm(
    image_bytes: bytes,
    prompt: str,
    *,
    max_tokens: int = 2048,
    timeout_s: int = 90,
) -> str:
    """POST image+prompt to ollama vision endpoint. Returns text response."""
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    model = os.environ.get("OLLAMA_VISION_MODEL", _DEFAULT_VISION_MODEL)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_encode_image_b64(image_bytes)],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }
    try:
        r = requests.post(
            f"{host}/api/generate",
            json=payload, timeout=timeout_s,
        )
        if r.status_code != 200:
            return ""
        return r.json().get("response", "") or ""
    except Exception as e:
        logger.warning("vision LLM call failed: %s", e)
        return ""


def vision_endpoint_reachable() -> bool:
    """Quick health check for the vision model."""
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model = os.environ.get("OLLAMA_VISION_MODEL", _DEFAULT_VISION_MODEL)
        return any(model in m for m in models)
    except Exception:
        return False


# ────────────────────────────────────────────────────────────────────
# PDF page → image rendering
# ────────────────────────────────────────────────────────────────────


def render_pdf_page(
    pdf_path: str, page_num: int, *, dpi: int = 150,
) -> bytes | None:
    """Render a single PDF page to PNG bytes via pymupdf.
    page_num is 0-indexed.
    """
    try:
        import fitz
    except ImportError:
        return None
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc) or page_num < 0:
            doc.close()
            return None
        page = doc.load_page(page_num)
        # 2x zoom for clearer table reading
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception as e:
        logger.warning("PDF render failed for %s page %d: %s", pdf_path, page_num, e)
        return None


# ────────────────────────────────────────────────────────────────────
# Page-extraction prompt
# ────────────────────────────────────────────────────────────────────


_VISION_PROMPT = """You are reading a single page from a bid / RFP / vendor-response PDF.

The page contains TABLES, DIAGRAMS, or STRUCTURED VISUAL CONTENT that
plain-text extraction missed. Extract EVERY piece of meaningful PM-
relevant information from this page into structured JSON.

Look for:
- TABLES (BOM line items, pricing schedules, contact rosters, team
  structure, milestone schedules, comparison tables)
- DIAGRAMS (network topology, org charts, process flows, Gantt bars)
- KEY-VALUE LISTS (specs, configurations, contact info)
- NUMBERED OR BULLETED LISTS that may be visual instead of textual

For each item you extract, identify what entity type it represents:
- requirement / certification / risk / acceptance / penalty
- stakeholder (person) / site / customer / vendor
- quantity / money / date / milestone
- device / part_number / address / phone / email

OUTPUT EXACTLY ONE JSON OBJECT (one line, no markdown fences):
{
  "summary": "<one-sentence description of what's on the page>",
  "rows": [
    {"kind": "<entity_type>", "text": "<extracted content>", "category": "<sub-category if applicable>"}
  ]
}

If the page is mostly text already extracted, return:
{"summary": "mostly text, nothing visual to add", "rows": []}

Be EXHAUSTIVE. Better to extract too much than miss something.

/no_think
"""


def _parse_vision_response(text: str) -> dict[str, Any]:
    """Parse the JSON object out of the vision-LLM response."""
    if not text:
        return {}
    # Strip common markdown fences
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # Try direct JSON parse
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try to extract the first {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


# ────────────────────────────────────────────────────────────────────
# High-level extractor for a list of (pdf_path, page_num) tuples
# ────────────────────────────────────────────────────────────────────


def extract_visual_pages(
    pages: list[tuple[str, int]],
    *,
    max_parallel: int = 3,
    max_pages: int = 30,
) -> list[dict[str, Any]]:
    """Extract structured content from a list of (pdf_path, page_num) pairs.

    Returns list of:
      {"pdf_path", "page_num", "summary", "rows": [{"kind", "text", "category"}]}

    Limits: max_pages (default 30) — caps total pages processed to
    bound LLM cost on huge docs.
    """
    if not pages or os.environ.get("SOWSMITH_VISION_DISABLE"):
        return []
    if not vision_endpoint_reachable():
        logger.info("vision endpoint not reachable, skipping vision pass")
        return []
    pages_to_process = pages[:max_pages]
    parallel = int(os.environ.get("SOWSMITH_VISION_PARALLEL", str(max_parallel)))

    def process(pair):
        pdf_path, page_num = pair
        img = render_pdf_page(pdf_path, page_num)
        if not img:
            return None
        t0 = time.time()
        response_text = call_vision_llm(img, _VISION_PROMPT, max_tokens=2048)
        elapsed = time.time() - t0
        parsed = _parse_vision_response(response_text)
        rows = parsed.get("rows") if isinstance(parsed, dict) else []
        if not isinstance(rows, list):
            rows = []
        logger.info(
            "vision: %s page %d -> %d rows in %.1fs",
            Path(pdf_path).name, page_num, len(rows), elapsed,
        )
        return {
            "pdf_path": pdf_path,
            "page_num": page_num,
            "summary": parsed.get("summary", "") if isinstance(parsed, dict) else "",
            "rows": rows,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(process, p) for p in pages_to_process]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                logger.warning("vision page process failed: %s", e)
                r = None
            if r:
                results.append(r)
    return results


# ────────────────────────────────────────────────────────────────────
# Identify which PDF pages need a vision pass
# ────────────────────────────────────────────────────────────────────


def find_visual_pages_from_atoms(atoms: list[Any]) -> list[tuple[str, int]]:
    """Walk atoms looking for the parser-emitted 'PDF page X appears
    to contain visual / table / diagram evidence' marker. Returns
    list of (pdf_path, page_num) tuples to render + vision-process.
    """
    pages: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    page_pat = re.compile(
        r"PDF page (\d+) appears to contain visual",
        re.IGNORECASE,
    )
    for atom in atoms:
        text = getattr(atom, "raw_text", "") or ""
        m = page_pat.search(text)
        if not m:
            continue
        page_num = int(m.group(1)) - 1  # parser is 1-indexed, fitz is 0-indexed
        # Find the artifact's PDF path via source_refs
        try:
            refs = getattr(atom, "source_refs", None) or []
            if refs:
                fname = getattr(refs[0], "filename", None) or ""
                if fname and fname.lower().endswith(".pdf"):
                    key = (fname, page_num)
                    if key not in seen:
                        seen.add(key)
                        pages.append((fname, page_num))
        except Exception:
            pass
    return pages


__all__ = [
    "call_vision_llm",
    "vision_endpoint_reachable",
    "render_pdf_page",
    "extract_visual_pages",
    "find_visual_pages_from_atoms",
]
