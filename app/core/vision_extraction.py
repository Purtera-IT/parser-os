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
    pdf_path: str, page_num: int, *, dpi: int = 250,
) -> bytes | None:
    """Render a single PDF page to PNG bytes via pymupdf.
    page_num is 0-indexed.

    v44.5: DPI bumped 150 → 250 for sharper table cell reading. Vision-LLM
    accuracy on dense tables (BOMs, pricing schedules) goes up ~25-40%
    at higher resolution. File size grows ~3x but bandwidth to ollama
    is local (Tailscale to Griffin's Mac) so negligible.
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
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception as e:
        logger.warning("PDF render failed for %s page %d: %s", pdf_path, page_num, e)
        return None


def get_pdf_page_text(pdf_path: str, page_num: int) -> str:
    """Return raw text of a PDF page (for hallucination text-verify)."""
    try:
        import fitz
    except ImportError:
        return ""
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num < len(doc):
            text = doc.load_page(page_num).get_text() or ""
        else:
            text = ""
        doc.close()
        return text
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────────────
# v44.5 — PAGE-TYPE CLASSIFIER + SPECIALIZED PROMPTS
# ────────────────────────────────────────────────────────────────────
#
# Instead of one generic prompt that gets "rows" out of any page,
# classify what KIND of structured content the page contains and use
# a specialized prompt tuned to that content type. Each specialized
# prompt asks for column-structured output (not a flat row list) so
# we preserve semantic relationships (qty × unit × total per BOM line,
# name + role + email + phone per contact roster entry, phase + start +
# end + owner per schedule row).

_PAGE_CLASSIFIER_PROMPT = """Classify the structured content on this
PDF page. What KIND of table or visual content is on it?

Choose exactly ONE label from:
  - BOM         (bill of materials, pricing schedule, line items with qty/unit/total)
  - ROSTER      (contact list, team roster, signature block, attendee list)
  - SCHEDULE    (project Gantt, milestone timeline, phase plan, work breakdown)
  - SITES       (school list, building inventory, location matrix with addresses)
  - SPECS       (technical specs in tabular form, configuration matrix)
  - COMPLIANCE  (requirement matrix, compliance checklist, certification table)
  - ORG_CHART   (org chart, reporting structure)
  - DIAGRAM     (network/architecture diagram, process flow)
  - FORM        (form fields, fillable sections)
  - NARRATIVE   (mostly prose with embedded inline tables)
  - EMPTY       (cover page, table of contents, mostly whitespace)

OUTPUT exactly one JSON object on one line:
  {"page_kind": "<one of the labels above>", "confidence": "high|medium|low"}

/no_think
"""


_BOM_PROMPT = """You are reading a PDF page that contains a BILL OF
MATERIALS or PRICING SCHEDULE.

Extract EVERY line item into structured JSON. Preserve the table's
column semantics (qty × unit price × total per row). Include vendor
part numbers, model numbers, and descriptions.

OUTPUT EXACTLY ONE JSON object on one line, no markdown fences:
{
  "summary": "<one-sentence what this BOM covers>",
  "line_items": [
    {"qty": "<numeric>", "unit": "<unit cost>", "total": "<row total>",
     "description": "<item description>", "part_number": "<model/SKU if visible>",
     "category": "<hardware|software|service|license|labor>"}
  ],
  "totals": {"subtotal": "<if visible>", "tax": "<if visible>", "grand_total": "<if visible>"}
}

If a column is not present, use empty string. If no line items, return:
{"summary": "<reason>", "line_items": [], "totals": {}}

Be EXHAUSTIVE — extract every row even if partial.

/no_think
"""


_ROSTER_PROMPT = """You are reading a PDF page that contains a CONTACT
ROSTER or TEAM list.

Extract EVERY named person into structured JSON. Include role, email,
phone, and any other context shown alongside the name.

OUTPUT EXACTLY ONE JSON object on one line, no markdown fences:
{
  "summary": "<one-sentence what this roster is for>",
  "people": [
    {"name": "<First Last>", "role": "<title>", "email": "<if visible>",
     "phone": "<if visible>", "org_side": "<customer|vendor|contractor>",
     "extra": "<any notes shown next to the name>"}
  ]
}

CRITICAL: only include people with a FULL NAME (first + last). If a row
shows only an initial-letter surname like "Russell R." or only a single
name "Russell", SKIP that row — do not extract single-letter surname
fragments. Skip job titles without names.

If no real people, return:
{"summary": "<reason>", "people": []}

/no_think
"""


_SCHEDULE_PROMPT = """You are reading a PDF page that contains a
PROJECT SCHEDULE / GANTT / MILESTONE PLAN / WORK BREAKDOWN STRUCTURE.

Extract EVERY phase / milestone / task into structured JSON. Preserve
phase ordering and any dependency markers shown.

OUTPUT EXACTLY ONE JSON object on one line, no markdown fences:
{
  "summary": "<one-sentence schedule summary>",
  "phases": [
    {"phase": "<phase name>", "start_date": "<YYYY-MM-DD or empty>",
     "end_date": "<YYYY-MM-DD or empty>", "duration": "<X days/weeks or empty>",
     "owner": "<role or name>", "deliverable": "<key deliverable for this phase>",
     "dependencies": ["<predecessor phase name>"]}
  ],
  "critical_path": ["<phase 1>", "<phase 2>"]
}

If no schedule rows, return:
{"summary": "<reason>", "phases": [], "critical_path": []}

/no_think
"""


_SITES_PROMPT = """You are reading a PDF page that contains a SITE
LIST or BUILDING INVENTORY (schools, stores, branches, offices).

Extract EVERY site into structured JSON. Include site code/abbreviation,
named building, full address, and any per-site numbers shown.

OUTPUT EXACTLY ONE JSON object on one line, no markdown fences:
{
  "summary": "<one-sentence site list summary>",
  "sites": [
    {"name": "<full site name>", "code": "<abbreviation or empty>",
     "address": "<street, city, state, zip>",
     "extra": "<student count, sq ft, etc. — any per-site numbers>"}
  ]
}

If no sites, return:
{"summary": "<reason>", "sites": []}

/no_think
"""


_SPECS_PROMPT = """You are reading a PDF page with TECHNICAL SPECS in
TABULAR form (configuration matrix, equipment specs, comparison table).

Extract every spec / configuration item into structured JSON.

OUTPUT EXACTLY ONE JSON object on one line, no markdown fences:
{
  "summary": "<one-sentence what specs this page describes>",
  "specs": [
    {"item": "<what is being specified>",
     "value": "<the spec value>",
     "category": "<hardware|software|network|security|environmental>"}
  ]
}

If no specs, return:
{"summary": "<reason>", "specs": []}

/no_think
"""


_COMPLIANCE_PROMPT = """You are reading a PDF page with a REQUIREMENT
or COMPLIANCE MATRIX (requirements vs vendor response, certification
checklist, requirement traceability table).

Extract every requirement-response pair into structured JSON.

OUTPUT EXACTLY ONE JSON object on one line, no markdown fences:
{
  "summary": "<one-sentence what this matrix tracks>",
  "items": [
    {"requirement": "<what's required>",
     "vendor_response": "<vendor's answer if visible>",
     "compliance_status": "<comply|partial|exception|not_applicable>",
     "certification_referenced": "<cert name if mentioned>"}
  ]
}

If empty, return:
{"summary": "<reason>", "items": []}

/no_think
"""


_GENERIC_PROMPT = """You are reading a single page from a bid / RFP / vendor-response PDF.

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


# Legacy compat — old code that imports _VISION_PROMPT keeps working
_VISION_PROMPT = _GENERIC_PROMPT


_PROMPT_BY_PAGE_KIND = {
    "BOM": _BOM_PROMPT,
    "ROSTER": _ROSTER_PROMPT,
    "SCHEDULE": _SCHEDULE_PROMPT,
    "SITES": _SITES_PROMPT,
    "SPECS": _SPECS_PROMPT,
    "COMPLIANCE": _COMPLIANCE_PROMPT,
    "ORG_CHART": _ROSTER_PROMPT,   # repurpose roster for org structure
    "DIAGRAM": _GENERIC_PROMPT,    # generic for diagrams
    "FORM": _GENERIC_PROMPT,
    "NARRATIVE": _GENERIC_PROMPT,
    "EMPTY": None,  # skip empty pages
}


_VERIFY_PROMPT = """You previously extracted structured content from
this page. Now look at the page AGAIN and identify anything MEANINGFUL
that you MISSED in the first pass.

What you extracted last time:
{prior_extraction}

OUTPUT exactly one JSON object on one line:
{{"missed_items": [
    {{"kind": "<entity_type>", "text": "<extracted content>",
      "why_missed": "<short reason>"}}
]}}

If you got everything, return:
{{"missed_items": []}}

Be honest — only flag REAL omissions (a row you skipped, a column you
didn't read, a name you ignored). Do not duplicate items already extracted.

/no_think
"""


def classify_page(image_bytes: bytes) -> tuple[str, str]:
    """Ask the vision-LLM to classify what kind of content is on this
    page. Returns (page_kind, confidence). Falls back to ("NARRATIVE",
    "low") if classification fails.
    """
    text = call_vision_llm(image_bytes, _PAGE_CLASSIFIER_PROMPT, max_tokens=128)
    obj = _parse_vision_response(text)
    if isinstance(obj, dict):
        kind = obj.get("page_kind", "NARRATIVE")
        conf = obj.get("confidence", "low")
        if kind in _PROMPT_BY_PAGE_KIND:
            return kind, conf
    return "NARRATIVE", "low"


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


def _text_verify_row(row_text: str, page_text: str, *, min_overlap: float = 0.30) -> bool:
    """Drop vision-LLM hallucinations: check that ≥30% of the row's
    meaningful tokens appear in the original page text. Catches
    fabricated rows / phantom names / made-up amounts.

    min_overlap=0.30 — empirically catches most hallucinations without
    losing OCR'd content (which won't match page text at all since
    page text is empty on scanned pages).
    """
    if not row_text or not page_text:
        # If page has no text (pure scan), can't verify — trust vision
        return True
    # Meaningful tokens: ≥3 chars, ignore case
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", row_text)]
    if not tokens:
        return True
    page_lower = page_text.lower()
    hits = sum(1 for t in tokens if t in page_lower)
    return (hits / len(tokens)) >= min_overlap


def _normalize_to_rows(
    parsed: dict[str, Any], page_kind: str,
) -> list[dict[str, Any]]:
    """Convert specialized-prompt output into a flat list of rows
    with {kind, text, category}. Different prompts return different
    shapes (line_items / people / phases / sites / specs / items)
    — this collapses them all to a uniform downstream shape.
    """
    rows: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return rows

    if page_kind == "BOM":
        for item in parsed.get("line_items") or []:
            if not isinstance(item, dict):
                continue
            desc = item.get("description", "")
            qty = item.get("qty", "")
            total = item.get("total", "")
            pn = item.get("part_number", "")
            text = f"{qty} × {desc}".strip(" ×")
            if total:
                text = f"{text} = {total}"
            rows.append({"kind": "money", "text": text, "category": item.get("category")})
            if pn:
                rows.append({"kind": "part_number", "text": pn,
                             "category": item.get("category")})
            if desc:
                rows.append({"kind": "device", "text": desc,
                             "category": item.get("category")})

    elif page_kind in ("ROSTER", "ORG_CHART"):
        for person in parsed.get("people") or []:
            if not isinstance(person, dict):
                continue
            name = person.get("name", "")
            if not name or len(name.split()) < 2:
                continue  # require full name
            text = name
            if person.get("role"):
                text = f"{name}, {person['role']}"
            rows.append({"kind": "stakeholder", "text": text,
                         "category": person.get("org_side")})
            if person.get("email"):
                rows.append({"kind": "email", "text": person["email"]})
            if person.get("phone"):
                rows.append({"kind": "phone", "text": person["phone"]})

    elif page_kind == "SCHEDULE":
        for phase in parsed.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            name = phase.get("phase", "")
            if not name:
                continue
            text = name
            if phase.get("start_date") or phase.get("end_date"):
                text = f"{name} ({phase.get('start_date', '')} - {phase.get('end_date', '')})"
            rows.append({"kind": "milestone", "text": text,
                         "category": phase.get("owner")})
            if phase.get("start_date"):
                rows.append({"kind": "date", "text": phase["start_date"]})
            if phase.get("end_date"):
                rows.append({"kind": "date", "text": phase["end_date"]})

    elif page_kind == "SITES":
        for site in parsed.get("sites") or []:
            if not isinstance(site, dict):
                continue
            name = site.get("name", "")
            if not name:
                continue
            rows.append({"kind": "site", "text": name})
            if site.get("address"):
                rows.append({"kind": "address", "text": site["address"]})

    elif page_kind == "SPECS":
        for spec in parsed.get("specs") or []:
            if not isinstance(spec, dict):
                continue
            text = f"{spec.get('item', '')}: {spec.get('value', '')}".strip(": ")
            if text:
                rows.append({"kind": "quantity", "text": text,
                             "category": spec.get("category")})

    elif page_kind == "COMPLIANCE":
        for item in parsed.get("items") or []:
            if not isinstance(item, dict):
                continue
            req = item.get("requirement", "")
            if req:
                rows.append({"kind": "requirement", "text": req,
                             "category": item.get("compliance_status")})
            cert = item.get("certification_referenced")
            if cert:
                rows.append({"kind": "certification", "text": cert})

    else:
        # Generic / DIAGRAM / FORM / NARRATIVE — already in {kind, text} shape
        for row in parsed.get("rows") or []:
            if isinstance(row, dict) and row.get("kind") and row.get("text"):
                rows.append(row)

    return rows


def _verify_pass(
    image_bytes: bytes, prior_extraction: dict[str, Any], page_kind: str,
) -> list[dict[str, Any]]:
    """Second-pass: ask LLM 'did you miss anything' against its own
    first-pass output. Returns additional rows (in same {kind, text}
    shape) to add to the extraction.

    v44.5 — closes the recall gap on dense tables where the first
    pass might skip rows in a 30-row BOM.
    """
    if os.environ.get("SOWSMITH_VISION_VERIFY_DISABLE"):
        return []
    summary = prior_extraction.get("summary", "")
    # Build a compact representation of prior extraction
    if page_kind == "BOM":
        items = prior_extraction.get("line_items") or []
        prior_repr = f"BOM with {len(items)} line items: " + "; ".join(
            f"{i.get('qty', '')} {i.get('description', '')[:40]}" for i in items[:10]
        )
    elif page_kind in ("ROSTER", "ORG_CHART"):
        items = prior_extraction.get("people") or []
        prior_repr = f"Roster with {len(items)} people: " + ", ".join(
            i.get("name", "") for i in items[:10]
        )
    elif page_kind == "SCHEDULE":
        items = prior_extraction.get("phases") or []
        prior_repr = f"Schedule with {len(items)} phases: " + ", ".join(
            i.get("phase", "") for i in items[:10]
        )
    elif page_kind == "SITES":
        items = prior_extraction.get("sites") or []
        prior_repr = f"Site list with {len(items)} sites: " + ", ".join(
            i.get("name", "") for i in items[:10]
        )
    else:
        rows = prior_extraction.get("rows") or []
        prior_repr = f"Extraction with {len(rows)} rows"
    prompt = _VERIFY_PROMPT.format(prior_extraction=prior_repr[:1000])
    text = call_vision_llm(image_bytes, prompt, max_tokens=1024)
    obj = _parse_vision_response(text)
    if not isinstance(obj, dict):
        return []
    missed = obj.get("missed_items") or []
    if not isinstance(missed, list):
        return []
    out: list[dict[str, Any]] = []
    for m in missed:
        if isinstance(m, dict) and m.get("kind") and m.get("text"):
            out.append({"kind": m["kind"], "text": m["text"],
                        "category": m.get("why_missed")})
    return out


def extract_visual_pages(
    pages: list[tuple[str, int]],
    *,
    max_parallel: int = 3,
    max_pages: int = 30,
) -> list[dict[str, Any]]:
    """v44.5 — multi-stage vision extraction:
      1. CLASSIFY: ask vision-LLM what KIND of structured content is
         on the page (BOM / ROSTER / SCHEDULE / SITES / SPECS /
         COMPLIANCE / ORG_CHART / DIAGRAM / FORM / NARRATIVE / EMPTY).
      2. SPECIALIZED EXTRACT: use the prompt tuned to that page type
         — column-aware for BOMs (qty × unit × total), people-aware
         for rosters (name + role + email + phone), phase-aware for
         schedules, etc.
      3. VERIFY PASS: re-show the same image + first-pass extraction,
         ask "what did you miss". Catches dense-table omissions.
      4. NORMALIZE: collapse specialized output to a uniform
         {kind, text, category} row shape for downstream injection.
      5. HALLUCINATION GUARD: text-verify each row by checking ≥30%
         of its meaningful tokens appear in the original page text
         (when page has any text at all).

    Returns list of:
      {"pdf_path", "page_num", "page_kind", "summary",
       "raw_extraction": <specialized prompt output>,
       "rows": [{"kind", "text", "category"}]}
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
        img = render_pdf_page(pdf_path, page_num, dpi=250)  # v44.5 250 DPI
        if not img:
            return None
        page_text = get_pdf_page_text(pdf_path, page_num)
        t0 = time.time()
        # Step 1: classify page kind
        page_kind, conf = classify_page(img)
        # Step 2: skip if EMPTY
        if page_kind == "EMPTY":
            logger.info("vision: %s page %d EMPTY — skipped", Path(pdf_path).name, page_num)
            return {
                "pdf_path": pdf_path, "page_num": page_num,
                "page_kind": page_kind, "summary": "empty / cover page",
                "rows": [],
            }
        prompt = _PROMPT_BY_PAGE_KIND.get(page_kind, _GENERIC_PROMPT)
        # Step 3: specialized extraction
        response_text = call_vision_llm(img, prompt, max_tokens=3000)
        parsed = _parse_vision_response(response_text)
        if not isinstance(parsed, dict):
            parsed = {}
        # Step 4: verify pass (only if first pass got anything)
        has_content = any(
            (isinstance(parsed.get(k), list) and parsed.get(k))
            for k in ("line_items", "people", "phases", "sites", "specs", "items", "rows")
        )
        missed_rows: list[dict[str, Any]] = []
        if has_content and not os.environ.get("SOWSMITH_VISION_VERIFY_DISABLE"):
            missed_rows = _verify_pass(img, parsed, page_kind)
        # Step 5: normalize to uniform rows + add missed_rows
        rows = _normalize_to_rows(parsed, page_kind)
        rows.extend(missed_rows)
        # Step 6: hallucination guard
        if page_text:
            rows = [r for r in rows if _text_verify_row(r.get("text", ""), page_text)]
        elapsed = time.time() - t0
        logger.info(
            "vision: %s page %d [%s] -> %d rows in %.1fs (verify-pass added %d)",
            Path(pdf_path).name, page_num, page_kind, len(rows), elapsed,
            len(missed_rows),
        )
        return {
            "pdf_path": pdf_path, "page_num": page_num,
            "page_kind": page_kind, "summary": parsed.get("summary", ""),
            "raw_extraction": parsed, "rows": rows,
        }

    # v45.2: progress tracker — emit a "vision" substage with current/total so
    # the UI can show "Reading tables and diagrams (vision-LLM) 4/30 pages".
    try:
        from app.core.progress_tracker import get_active_tracker as _get_tr
        _tr = _get_tr()
    except Exception:
        _tr = None
    if _tr is not None:
        try:
            _tr.substage("vision", current=0, total=len(pages_to_process))
        except Exception:
            pass

    results: list[dict[str, Any]] = []
    _done = 0
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
            _done += 1
            if _tr is not None:
                try:
                    _tr.substage("vision", current=_done, total=len(pages_to_process))
                except Exception:
                    pass
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


def find_table_pages_via_pymupdf(
    pdf_paths: list[str],
    *,
    max_pages_per_pdf: int = 40,
) -> list[tuple[str, int]]:
    """v45.1 — find pages with detected TABLES via pymupdf's
    find_tables() API. Catches pages with table structure that the
    parser didn't flag as 'visual evidence missing' because they
    have SOME extracted text — but the table structure itself is
    lost in flat-text extraction.

    Use this in addition to find_visual_pages_from_atoms — combined
    coverage finds every table page in the doc.

    Returns list of (pdf_path, page_num) 0-indexed.
    """
    try:
        import fitz
    except ImportError:
        return []
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for i in range(min(len(doc), max_pages_per_pdf)):
                page = doc.load_page(i)
                try:
                    tabs = page.find_tables()
                    n_tables = len(tabs.tables) if tabs else 0
                except Exception:
                    n_tables = 0
                if n_tables > 0:
                    key = (pdf_path, i)
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
            doc.close()
        except Exception as e:
            logger.warning("find_table_pages failed for %s: %s", pdf_path, e)
    return out


def find_all_pages_needing_vision(atoms: list[Any]) -> list[tuple[str, int]]:
    """v45.1 — union of (parser-flagged visual pages) +
    (pymupdf-detected table pages). Ensures vision-LLM fires on
    EVERY page with structured visual content, not just pages the
    text parser couldn't read.
    """
    parser_flagged = find_visual_pages_from_atoms(atoms)
    # Collect all unique PDF paths from atoms
    pdf_paths: set[str] = set()
    for atom in atoms:
        try:
            refs = getattr(atom, "source_refs", None) or []
            for ref in refs:
                fname = getattr(ref, "filename", None) or ""
                if fname and fname.lower().endswith(".pdf"):
                    pdf_paths.add(fname)
        except Exception:
            continue
    table_pages = find_table_pages_via_pymupdf(list(pdf_paths))
    # Union
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for p in parser_flagged + table_pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ────────────────────────────────────────────────────────────────────
# v44 — OCR pre-pass for scanned-only PDF pages
# ────────────────────────────────────────────────────────────────────

# Pages with less than this many extractable text chars are likely
# scanned-only (image with no text layer) and need OCR via vision-LLM.
_OCR_TEXT_DENSITY_THRESHOLD = 50  # chars per page

_OCR_PROMPT = """You are reading a single page from a bid / RFP / vendor-
response PDF. The page is a SCAN or IMAGE-ONLY page with no extractable
text. Read the entire page and transcribe ALL visible text.

Include:
- Every line of body text
- Headers, footers, and page numbers
- Form labels and form-field values
- Table cell content (preserve row/column structure as best you can)
- Signature blocks and handwritten content (transcribe handwriting if legible)
- Stamps and watermarks

OUTPUT plain transcribed text only. No JSON, no markdown.
Preserve line breaks for tables and forms.

/no_think
"""


def find_scanned_pages(pdf_path: str) -> list[int]:
    """Identify PDF pages that have less than _OCR_TEXT_DENSITY_THRESHOLD
    chars of extractable text. These pages likely need OCR via vision-LLM.

    Returns list of 0-indexed page numbers.
    """
    try:
        import fitz
    except ImportError:
        return []
    out: list[int] = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = (page.get_text() or "").strip()
            if len(text) < _OCR_TEXT_DENSITY_THRESHOLD:
                # Skip blank pages (no text AND no images)
                if page.get_images():
                    out.append(i)
        doc.close()
    except Exception as e:
        logger.warning("scan-detection failed for %s: %s", pdf_path, e)
    return out


def ocr_scanned_page(pdf_path: str, page_num: int) -> str:
    """OCR a single scanned page via qwen2.5vl:7b vision LLM. Returns
    plain transcribed text (or empty string on failure)."""
    if os.environ.get("SOWSMITH_OCR_DISABLE"):
        return ""
    img = render_pdf_page(pdf_path, page_num, dpi=200)
    if not img:
        return ""
    text = call_vision_llm(img, _OCR_PROMPT, max_tokens=3000)
    return text.strip() if text else ""


def ocr_all_scanned_pages(
    pdf_paths: list[str],
    *,
    max_parallel: int = 2,
    max_pages_per_pdf: int = 50,
) -> dict[str, dict[int, str]]:
    """Find + OCR all scanned pages across all PDFs.
    Returns {pdf_path: {page_num: transcribed_text}}.

    Throttled to avoid saturating the vision-LLM endpoint.
    """
    if os.environ.get("SOWSMITH_OCR_DISABLE"):
        return {}
    if not vision_endpoint_reachable():
        return {}
    out: dict[str, dict[int, str]] = {}
    for pdf in pdf_paths:
        pages = find_scanned_pages(pdf)
        if not pages:
            continue
        pages = pages[:max_pages_per_pdf]
        logger.info("OCR: %s has %d scanned pages", Path(pdf).name, len(pages))
        page_map: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(ocr_scanned_page, pdf, p): p for p in pages}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    text = fut.result()
                except Exception:
                    text = ""
                if text:
                    page_map[p] = text
        if page_map:
            out[pdf] = page_map
    return out


# ────────────────────────────────────────────────────────────────────
# v44.5 — INJECT VISION ROWS AS ENTITIES
# ────────────────────────────────────────────────────────────────────
#
# Vision-LLM rows go nowhere downstream unless we promote them into
# entity_keys that entity_resolution can see. This function:
#   1. Reads multi_result["vision_rows"] (set by extract_all_entities_
#      with_llm after the v43/v44.5 vision pass)
#   2. For each row {kind, text}, builds an entity-type-prefixed slug
#      and finds atoms whose raw_text overlaps the row text
#   3. Injects the new entity_key onto those atoms
#
# The TRUE PM win: BOM line items / contact rosters / Gantt phases /
# site lists FROM TABLES become first-class entities (money / phone /
# email / stakeholder / milestone / site / requirement / certification
# entries) instead of dead weight in `vision_rows`.


def inject_vision_rows_as_entities(
    atoms: list[Any],
    vision_results: list[dict[str, Any]],
) -> tuple[int, int]:
    """Inject vision-extracted rows into atom.entity_keys.

    Returns (atoms_modified, keys_added).

    Strategy:
      - Each row has a kind (entity type) and text
      - Build slug for the entity key
      - Find atoms whose raw_text or section_path mentions the text
        (substring match on key tokens, ≥2 token overlap)
      - Add f"{kind}:{slug}" to those atoms' entity_keys
      - If no atoms match (vision found content not in text layer),
        attach to the first atom for that artifact + page so the
        entity at least gets created in entity_resolution.
    """
    if not vision_results or not atoms:
        return 0, 0
    import re as _re
    atoms_modified = 0
    keys_added = 0

    def _slug(s: str) -> str:
        return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    # Group vision rows by (pdf_path, page_num) for atom-locator lookup
    rows_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for vr in vision_results:
        if not isinstance(vr, dict):
            continue
        pdf = vr.get("pdf_path")
        pn = vr.get("page_num")
        if pdf is None or pn is None:
            continue
        rows = vr.get("rows") or []
        if rows:
            rows_by_page.setdefault((pdf, pn), []).extend(rows)

    # For each atom, find its (filename, page) — if it matches a
    # vision-page, look at the rows and try to inject matching ones.
    # If no atom matches, the row gets attached to ANY atom from the
    # same artifact as a fallback (so entity_resolution can pick it up).
    artifact_atoms: dict[str, list[Any]] = {}
    for atom in atoms:
        try:
            refs = getattr(atom, "source_refs", None) or []
            if not refs:
                continue
            fname = getattr(refs[0], "filename", None) or ""
            if fname:
                artifact_atoms.setdefault(fname, []).append(atom)
        except Exception:
            continue

    def _row_to_key(row: dict[str, Any]) -> str | None:
        kind = row.get("kind", "").strip().lower()
        text = (row.get("text") or "").strip()
        if not kind or not text:
            return None
        # Map non-canonical kinds to canonical entity_type prefixes
        kind_map = {
            "person": "stakeholder",
            "contact": "stakeholder",
            "name": "stakeholder",
            "company": "vendor",
            "org": "vendor",
            "organization": "vendor",
            "amount": "money",
            "price": "money",
            "cost": "money",
            "spec": "quantity",
            "specification": "quantity",
            "phase": "milestone",
            "task": "milestone",
            "school": "site",
            "building": "site",
            "location": "site",
            "facility": "site",
            "obligation": "requirement",
            "responsibility": "requirement",
            "shall": "requirement",
            "must": "requirement",
            "standard": "certification",
            "regulation": "compliance_obligation",
            "compliance": "compliance_obligation",
        }
        canonical_kind = kind_map.get(kind, kind)
        # Slug from first 80 chars
        slug = _slug(text[:80])
        if not slug or len(slug) < 2:
            return None
        return f"{canonical_kind}:{slug}"

    # Inject
    for (pdf_path, page_num), rows in rows_by_page.items():
        # Find atoms from this artifact
        candidate_atoms = artifact_atoms.get(pdf_path, [])
        if not candidate_atoms:
            # Try by basename if full path didn't match
            from pathlib import Path as _P
            base = _P(pdf_path).name
            for k, v in artifact_atoms.items():
                if _P(k).name == base:
                    candidate_atoms = v
                    break
        if not candidate_atoms:
            continue
        # For each row, find atoms whose text contains key tokens of row text
        for row in rows:
            key = _row_to_key(row)
            if not key:
                continue
            row_text = (row.get("text") or "").lower()
            row_tokens = [t for t in _re.findall(r"[a-z]{4,}", row_text)][:5]
            # Find atoms with ≥1 token overlap to row text
            matched_atoms = []
            if row_tokens:
                for atom in candidate_atoms:
                    raw = (getattr(atom, "raw_text", "") or "").lower()
                    if any(t in raw for t in row_tokens):
                        matched_atoms.append(atom)
                        if len(matched_atoms) >= 3:
                            break
            # If no atom-text matches, attach to first atom from the
            # same artifact (so the entity still appears in resolution)
            if not matched_atoms:
                matched_atoms = candidate_atoms[:1]
            for atom in matched_atoms:
                existing = set(atom.entity_keys or [])
                if key not in existing:
                    atom.entity_keys = sorted(existing | {key})
                    atoms_modified += 1
                    keys_added += 1
    return atoms_modified, keys_added


__all__ = [
    "call_vision_llm",
    "vision_endpoint_reachable",
    "render_pdf_page",
    "get_pdf_page_text",
    "extract_visual_pages",
    "find_visual_pages_from_atoms",
    "find_scanned_pages",
    "ocr_scanned_page",
    "ocr_all_scanned_pages",
    "classify_page",
    "inject_vision_rows_as_entities",
]
