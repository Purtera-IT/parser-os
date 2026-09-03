from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from app.domain import get_active_domain_pack
from app.core.address_parse import US_STATES, find_us_addresses_in_text
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
from app.parsers.email_body import _extract_email_text
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

_CID_REF_RE = re.compile(r"\[cid:([^\]]+)\]", re.I)
_EQUIPMENT_LINE_RE = re.compile(
    r"(?:"
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*|×\s*)?"
    r"(?:e7|u7)\s*aps?\b"
    r"|"
    r"(?<![\w/])(\d+)\s*(?:x\s*|×\s*)?e7\s*aps?\b"
    r"|"
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*|×\s*)?"
    r"(udm(?:\s*beast)?|dream\s+machine(?:\s*beast)?|enterprise\s+nvr|uni\s*nvr|unvr|nvr|"
    r"g6\s+(?:pro(?:\s+(?:turret|360))?|turret|instant|entry)|"
    r"(?:access\s+)?g3\s*reader|badge\s*reader|card\s*reader|access\s*reader(?:\s*pro)?|"
    r"access\s*(?:point(?:\s+e7)?|card|hub)|ap\b|switch(?:\s*pro)?|"
    r"camera(?:\s+g6|\s+ai)?|doorbell|sensor|mount)\b"
    r"|"
    # Order-screenshot rows: "Access Point E7 ..... 6" (qty right-aligned, no ×).
    r"(?:access\s+point(?:\s+e7)?|switch\s+pro(?:\s+max)?(?:\s+\d+)?(?:\s+poe)?|"
    r"enterprise\s+nvr|nvr|dream\s+machine(?:\s*beast)?|udm(?:\s*beast)?|"
    r"g6(?:\s+pro)?(?:\s+(?:turret|360))?|camera\s+g6(?:\s+pro)?(?:\s+(?:turret|360))?|"
    r"(?:access\s+)?g3\s*reader|access\s+reader(?:\s*pro)?|badge\s*reader|card\s*reader|"
    r"access\s*card|enterprise\s+access\s+hub|"
    r"protect(?:\s+all[- ]in[- ]one)?\s+sensor|g6\s+ptz\s+mount|"
    r"reader\s+g6\s+entry)[^\n]{0,80}?(?:[×x]\s*|(?:\s{2,}|\t))\s*(\d+)\s*$"
    r"|"
  # HubSpot order rows with middle-dot or × glyph: "Access Card × 10".
    r"(?:access\s*card|protect(?:\s+all[- ]in[- ]one)?\s+sensor|"
    r"switch\s+pro(?:\s+max)?(?:\s+\d+)?(?:\s+poe)?|access\s+point(?:\s+e7)?|"
    r"enterprise\s+nvr|(?:access\s+)?g3\s*reader|g6\s+ptz\s+mount)"
    r"[^\n]{0,40}?\s*[×x]\s*(\d+)\b"
    r")",
    re.I | re.M,
)
# Prefer digital PDF text when a page already has enough selectable chars.
_PDF_DIGITAL_TEXT_MIN_CHARS = 40
_ORDER_DETAILS_HDR_RE = re.compile(r"\border\s+details\b", re.I)
_TRANSCRIPT_DOC_RE = re.compile(
    r"\bmeeting\s+summary\s+and\s+full\s+transcript\b|\bfull\s+transcript\b",
    re.I,
)
_TRANSCRIPT_SPEECH_LINE_RE = re.compile(
    r"\[[0-9]{1,2}:[0-9]{2}\]|"
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z'-]+)+\s+\[[0-9]",
    re.I,
)
_WORD_QTY: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# HubSpot order rows: qty is usually the last number, not embedded model nums (Max 48).
_MODEL_QTY_IN_NAME_RE = re.compile(
    r"\b(?:max|pro|series)\s+(\d{2,3})\b"
    r"|\b(\d{2,3})\s+poe\b"
    r"|\bmulti\s+sensor\s+(\d)\b"
    r"|\b(\d{1,2})g\s+direct\b",
    re.I,
)


def _glued_trailing_order_qty(line: str) -> int | None:
    """OCR sometimes glues order qty to the product name (e.g. ``... PoE 2``)."""
    cleaned = (line or "").strip()
    if not cleaned:
        return None
    # ``Camera AI Multi Sensor 4 1`` — model variant then order qty.
    multi = re.search(r"\bmulti\s+sensor\s+(\d+)\s+(\d{1,2})\s*$", cleaned, re.I)
    if multi:
        return int(multi.group(2))
    # Bare ``… Multi Sensor 4`` — trailing digit is the model, not qty.
    if re.search(r"\bmulti\s+sensor\s+\d+\s*$", cleaned, re.I):
        return None
    m = re.search(r"(\d{1,2})\s*$", cleaned)
    if not m:
        return None
    qty = int(m.group(1))
    if qty <= 0:
        return None
    stem = cleaned[: m.start(1)].rstrip()
    if not stem:
        return None
    for mx in _MODEL_QTY_IN_NAME_RE.finditer(stem):
        model_num = int(next((g for g in mx.groups() if g), 0) or 0)
        if model_num == qty:
            return None
    if qty <= 10:
        return qty
    return None


def _trailing_order_qty(line: str) -> int | None:
    """Parse right-aligned / × qty from an order-table OCR line."""
    cleaned = (line or "").strip()
    if not cleaned:
        return None
    m = re.search(r"[×x]\s*(\d{1,3})\s*$", cleaned, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:\s{2,}|\t)(\d{1,3})\s*$", cleaned)
    if m:
        qty = int(m.group(1))
        name = cleaned[: m.start(1)].rstrip()
        sane = _sanity_order_qty(name, qty)
        if sane is not None:
            return sane
    glued = _glued_trailing_order_qty(cleaned)
    if glued is not None:
        return glued
    return None


def _sanity_order_qty(name: str, qty: int) -> int | None:
    """Reject OCR grabbing model numbers (e.g. Switch Pro Max 48) as order qty."""
    if qty <= 0 or qty > 99:
        return None
    for m in _MODEL_QTY_IN_NAME_RE.finditer(name or ""):
        model_num = int(next((g for g in m.groups() if g), 0) or 0)
        if model_num == qty and qty >= 10:
            return None
    return qty


def _order_row_name(line: str, qty: int) -> str:
    cleaned = (line or "").strip()
    cleaned = re.sub(r"[×x]\s*\d{1,3}\s*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"(?:\s{2,}|\t)\d{1,3}\s*$", "", cleaned).strip()
    if qty and re.search(rf"\s{qty}\s*$", cleaned):
        cleaned = re.sub(rf"\s{qty}\s*$", "", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip() or line.strip()


def _parse_email_message(path: Path):
    if path.suffix.lower() != ".eml":
        return None
    try:
        return BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    except Exception:
        return None


def _normalize_cid(cid: str) -> str:
    raw = (cid or "").strip().lower()
    raw = raw.strip("<>").strip()
    return raw.split("@")[0]


def _iter_cid_inline_parts(msg) -> dict[str, dict[str, Any]]:
    """Map Content-ID -> inline MIME part payload (text, html, or image bytes)."""
    out: dict[str, dict[str, Any]] = {}
    if msg is None:
        return out
    for part in msg.walk():
        cid = part.get("Content-ID") or part.get("Content-Id")
        if not cid:
            continue
        key = _normalize_cid(str(cid))
        if not key:
            continue
        payload = part.get_payload(decode=True) or b""
        ctype = (part.get_content_type() or "").lower()
        if payload[:5] == b"%PDF-":
            ctype = "application/pdf"
        if ctype.startswith("image/"):
            out[key] = {
                "content_id": key,
                "content_type": ctype,
                "text": "",
                "payload": payload,
                "size": len(payload),
                "is_image": True,
            }
            continue
        if ctype == "application/pdf":
            out[key] = {
                "content_id": key,
                "content_type": ctype,
                "text": "",
                "payload": payload,
                "size": len(payload),
                "is_pdf": True,
            }
            continue
        try:
            text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        except Exception:
            text = payload.decode("utf-8", errors="ignore")
        if ctype == "text/html":
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        out[key] = {
            "content_id": key,
            "content_type": ctype,
            "text": text.strip(),
            "size": len(payload),
            "is_image": False,
        }
    return out


def _digital_text_from_pdf_page(page) -> str:
    """Extract selectable text (and table cells) before OCR fallback."""
    chunks: list[str] = []
    try:
        plain = (page.get_text("text") or "").strip()
        if plain:
            chunks.append(plain)
    except Exception:
        plain = ""
    # Tables often have usable cell text even when page.get_text("text") is sparse.
    try:
        for table in page.find_tables().tables:  # type: ignore[attr-defined]
            for row in table.extract() or []:
                cells = [str(c).strip() for c in row if c and str(c).strip()]
                if cells:
                    chunks.append("  ".join(cells))
    except Exception:
        pass
    if not chunks:
        try:
            dict_text = page.get_text("dict") or {}
            lines: list[str] = []
            for block in dict_text.get("blocks", []) or []:
                for line in block.get("lines", []) or []:
                    spans = "".join(str(s.get("text") or "") for s in line.get("spans", []) or [])
                    if spans.strip():
                        lines.append(spans.strip())
            if lines:
                chunks.append("\n".join(lines))
        except Exception:
            pass
    text = "\n".join(chunks).strip()
    return text


def _slice_order_details_text(text: str) -> str:
    """Keep HubSpot order-table rows when an Order Details header is present."""
    raw = (text or "").strip()
    if not raw:
        return raw
    match = _ORDER_DETAILS_HDR_RE.search(raw)
    if not match:
        return raw
    chunk = raw[match.start() :]
    stop = re.search(
        r"\n(?:Meeting Summary|Full Transcript|Executive Summary|Action Items)\b",
        chunk,
        re.I,
    )
    if stop:
        chunk = chunk[: stop.start()]
    return chunk.strip()


def _rejoin_split_order_qty_lines(text: str) -> str:
    """Join HubSpot OCR rows where quantity lands on the next line.

    Doc Intel / tesseract often emit::

        Access Point E7
        6
        Enterprise NVR
        1

    Universal rule: a bare ``1–99`` line after a non-qty product stem is the
    order quantity for that stem — not a standalone atom.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (
            cur
            and nxt
            and re.fullmatch(r"\d{1,2}", nxt)
            and not re.search(r"(?:\s{2,}|\t|[×x]\s*)\d{1,3}\s*$", cur, re.I)
            and not re.fullmatch(r"\d{1,2}", cur)
            and not _ORDER_TABLE_HDR_LINE_RE.match(cur)
            and not _ORDER_DETAILS_HDR_RE.fullmatch(cur)
            and any(ch.isalpha() for ch in cur)
        ):
            qty = int(nxt)
            # Avoid gluing model-only stems that already end in the same digit.
            if _sanity_order_qty(_order_row_name(f"{cur} {qty}", qty) or cur, qty):
                out.append(f"{cur} {qty}")
                i += 2
                continue
        if cur:
            out.append(cur)
        i += 1
    return "\n".join(out)


def _focus_cid_equipment_text(text: str) -> str:
    """Prefer order-table text; ignore spoken transcript counts from wrong embeds."""
    raw = (text or "").strip()
    if not raw:
        return raw
    order = _slice_order_details_text(raw)
    focused = order if order != raw else ("" if _TRANSCRIPT_DOC_RE.search(raw) else raw)
    if not focused:
        return ""
    return _rejoin_split_order_qty_lines(focused)


def _ocr_text_from_cid_image(payload: bytes) -> str:
    if not payload:
        return ""
    try:
        from app.parsers._ocr_chain import ocr_image_bytes

        result = ocr_image_bytes(payload)
        return (result.get("text") or "").strip()
    except Exception:
        return ""


def _ocr_text_from_cid_pdf(payload: bytes) -> str:
    """Prefer PyMuPDF digital text; OCR only pages lacking a usable text layer."""
    if not payload:
        return ""
    try:
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(stream=payload, filetype="pdf")
        from app.parsers._ocr_chain import ocr_pdf_page

        parts: list[str] = []
        for page in doc:
            digital = _digital_text_from_pdf_page(page)
            if len(digital) >= _PDF_DIGITAL_TEXT_MIN_CHARS:
                parts.append(digital)
                continue
            res = ocr_pdf_page(page)
            ocr_text = (res.get("text") or "").strip()
            # Keep whichever path yielded more usable text.
            if len(ocr_text) > len(digital):
                parts.append(ocr_text)
            elif digital:
                parts.append(digital)
        return "\n".join(parts)
    except Exception:
        return ""


def _ocr_text_from_cid_inline(payload: bytes, *, content_type: str) -> str:
    if not payload:
        return ""
    if payload[:5] == b"%PDF-":
        return _ocr_text_from_cid_pdf(payload)
    ctype = (content_type or "").lower()
    if ctype.startswith("image/"):
        return _ocr_text_from_cid_image(payload)
    if ctype == "application/pdf":
        return _ocr_text_from_cid_pdf(payload)
    return ""


def _score_cid_ocr_text(text: str) -> int:
    """Rank inline MIME OCR — prefer HubSpot order-table screenshots."""
    raw = (text or "").strip()
    if not raw:
        return 0
    score = min(len(raw), 2500) // 30
    if _ORDER_DETAILS_HDR_RE.search(raw):
        score += 250
    product_hits = len(
        re.findall(
            r"\b(?:access point|switch|enterprise nvr|nvr|reader|access card|protect|g6|udm)\b",
            raw,
            re.I,
        )
    )
    score += product_hits * 20
    if _TRANSCRIPT_DOC_RE.search(raw):
        score -= 400
    if re.search(r"\bmeeting summary\b", raw, re.I):
        score -= 200
    # Spoken transcript lines mentioning products should not beat order tables.
    speech_hits = len(_TRANSCRIPT_SPEECH_LINE_RE.findall(raw))
    if speech_hits:
        score -= min(speech_hits, 20) * 25
    orderish = len(re.findall(r"(?:\s{2,}|\t|[×x]\s*)\d{1,3}\s*$", raw, re.I | re.M))
    score += min(orderish, 12) * 15
    return score


def _pick_best_cid_ocr(ocr_by_cid: dict[str, str]) -> tuple[str, str]:
    """Choose inline screenshot OCR — prefer HubSpot order tables over transcript embeds."""
    if not ocr_by_cid:
        return "", ""
    ranked = sorted(
        ocr_by_cid.items(),
        key=lambda kv: _score_cid_ocr_text(kv[1]),
        reverse=True,
    )
    for cid, blob in ranked:
        if _ORDER_DETAILS_HDR_RE.search(blob or ""):
            return cid, blob
    cid, blob = ranked[0]
    return cid, blob or ""


def _ocr_cid_part(part: dict[str, Any]) -> str:
    text = str(part.get("text") or "")
    payload = part.get("payload")
    if payload and (part.get("is_image") or part.get("is_pdf")):
        ocr_text = _ocr_text_from_cid_inline(
            bytes(payload),
            content_type=str(part.get("content_type") or ""),
        )
        if ocr_text:
            return ocr_text
    return text


def _parse_qty_token(token: str) -> int | None:
    raw = (token or "").strip().lower()
    if raw.isdigit():
        n = int(raw)
        return n if n > 0 else None
    return _WORD_QTY.get(raw)


def _cid_equipment_source_ref(
    *,
    artifact_id: str,
    filename: str,
    content_id: str,
    parser_version: str,
    message_index: int,
    line_start: int,
    row_index: int,
    lead_in: list[str] | None = None,
) -> SourceRef:
    """Pin CID equipment rows into email document order (after body prose).

    All OCR rows share the CID body's ``line_start`` (the inline ``[cid:…]``
    anchor). Intra-image order is ``row_index`` only — never expand
    ``line_start`` by row, or post-image body notes (same message, later
    line numbers) interleave mid-BOM in document-order sorts.
    """
    section_path = _list_section_path(None, lead_in=lead_in) + ["Equipment list"]
    # Deduplicate while preserving order (lead-in may already say "equipment list").
    seen: set[str] = set()
    path: list[str] = []
    for part in section_path:
        key = part.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        path.append(part)
    locator: dict[str, Any] = {
        "kind": "email_cid_inline",
        "content_id": content_id,
        "message_index": message_index,
        "line_start": line_start,
        "line_end": line_start,
        "row_index": row_index,
        "section_path": path or ["Equipment list"],
    }
    if lead_in:
        locator["lead_in"] = list(lead_in)
    return SourceRef(
        id=stable_id("src", artifact_id, "cid", content_id, line_start, row_index),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.email,
        filename=filename,
        locator=locator,
        extraction_method="email_cid_inline",
        parser_version=parser_version,
    )


def _equipment_list_clause(raw: str) -> str:
    """Prefer the short equipment-list framing clause as connective tissue."""
    text = (raw or "").strip()
    if not text:
        return ""
    m = re.search(
        r"[^.!?\n]*\b(?:full\s+)?equipment\s+list\b[^.!?\n]*[.!]?",
        text,
        re.I,
    )
    if m:
        clause = re.sub(r"\s+", " ", m.group(0)).strip()
        if clause:
            return clause[:120]
    if len(text) > 120:
        return text[:117].rstrip() + "…"
    return text


def _equipment_list_lead_in(body_text: str, blocks: list[dict[str, Any]] | None = None) -> list[str]:
    """Collect body intro lines that frame the following equipment CID image."""
    leads: list[str] = []
    for block in blocks or []:
        for line in block.get("lines") or []:
            raw = str(line or "").strip()
            if raw and _is_equipment_list_intro_line(raw):
                clipped = _equipment_list_clause(raw)
                if clipped and clipped not in leads:
                    leads.append(clipped)
    if leads:
        return leads[:2]
    body = (body_text or "").strip()
    for line in body.splitlines():
        raw = line.strip()
        if raw and _is_equipment_list_intro_line(raw):
            return [_equipment_list_clause(raw)]
    return []


def _hardware_atoms_from_equipment_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    text: str,
    content_id: str,
    parser_version: str,
    message_index: int = 0,
    anchor_line: int = 1,
    lead_in: list[str] | None = None,
) -> list[EvidenceAtom]:
    atoms: list[EvidenceAtom] = []
    text = _focus_cid_equipment_text(text)
    if not text.strip():
        return atoms
    row_index = 0
    for line in text.splitlines():
        cleaned = _repair_ocr_equipment_line(line)
        if not cleaned or _TRANSCRIPT_SPEECH_LINE_RE.search(cleaned):
            continue
        if _is_ocr_junk_equipment_line(cleaned):
            continue
        trail_qty = _trailing_order_qty(cleaned)
        if trail_qty is not None:
            name = _order_row_name(cleaned, trail_qty)
            trail_qty = _sanity_order_qty(name, trail_qty)
            if trail_qty:
                line_start = int(anchor_line)
                src = _cid_equipment_source_ref(
                    artifact_id=artifact_id,
                    filename=filename,
                    content_id=content_id,
                    parser_version=parser_version,
                    message_index=message_index,
                    line_start=line_start,
                    row_index=row_index,
                    lead_in=lead_in,
                )
                value: dict[str, Any] = {
                    "text": cleaned,
                    "kind": "email_cid_equipment_line",
                    "quantity": trail_qty,
                    "qty": trail_qty,
                    "item": name,
                    "content_id": content_id,
                    "line": line_start,
                    "row_index": row_index,
                    "list_section": "equipment",
                    "section_header": "Equipment list",
                }
                if lead_in:
                    value["lead_in"] = list(lead_in)
                    value["intro"] = lead_in[0]
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            "cid_hw",
                            content_id,
                            cleaned,
                            str(trail_qty),
                            row_index,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.scope_item,
                        raw_text=cleaned,
                        normalized_text=normalize_text(cleaned),
                        value=value,
                        entity_keys=[f"quantity:{trail_qty}"],
                        source_refs=[src],
                        authority_class=AuthorityClass.customer_current_authored,
                        confidence=0.8,
                        review_status=ReviewStatus.needs_review,
                        review_flags=["email_cid_equipment_line"],
                        parser_version=parser_version,
                    )
                )
                row_index += 1
                continue
        matched = False
        for match in _EQUIPMENT_LINE_RE.finditer(cleaned):
            matched = True
            qty = None
            item = match.group(0)
            for g in reversed([x for x in match.groups() if x]):
                qty = _parse_qty_token(str(g))
                if qty:
                    break
            if match.group(2):
                item = str(match.group(2))
            if not qty:
                continue
            item = _order_row_name(cleaned, qty)
            line_start = int(anchor_line)
            src = _cid_equipment_source_ref(
                artifact_id=artifact_id,
                filename=filename,
                content_id=content_id,
                parser_version=parser_version,
                message_index=message_index,
                line_start=line_start,
                row_index=row_index,
                lead_in=lead_in,
            )
            value = {
                "text": cleaned,
                "kind": "email_cid_equipment_line",
                "quantity": qty,
                "qty": qty,
                "item": item,
                "content_id": content_id,
                "line": line_start,
                "row_index": row_index,
                "list_section": "equipment",
                "section_header": "Equipment list",
            }
            if lead_in:
                value["lead_in"] = list(lead_in)
                value["intro"] = lead_in[0]
            atoms.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm",
                        project_id,
                        artifact_id,
                        "cid_hw",
                        content_id,
                        cleaned,
                        str(qty),
                        row_index,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.scope_item,
                    raw_text=cleaned,
                    normalized_text=normalize_text(cleaned),
                    value=value,
                    entity_keys=[f"quantity:{qty}"],
                    source_refs=[src],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.78,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["email_cid_equipment_line"],
                    parser_version=parser_version,
                )
            )
            row_index += 1
        if matched:
            continue
    # Whole-image OCR fallback. Skipped for brand chrome: a signature logo is
    # not scope, and emitting it as a scope_item puts unfalsifiable filler in the
    # evidence set that can never verify.
    if not atoms and any(ch.isalnum() for ch in text) and not _is_brand_chrome_ocr(text):
        src = _cid_equipment_source_ref(
            artifact_id=artifact_id,
            filename=filename,
            content_id=content_id,
            parser_version=parser_version,
            message_index=message_index,
            line_start=int(anchor_line),
            row_index=0,
            lead_in=lead_in,
        )
        value = {
            "text": text[:4000],
            "kind": "email_cid_inline_body",
            "content_id": content_id,
            "line": int(anchor_line),
        }
        if lead_in:
            value["lead_in"] = list(lead_in)
            value["intro"] = lead_in[0]
        atoms.append(
            EvidenceAtom(
                id=stable_id("atm", project_id, artifact_id, "cid_body", content_id, text[:120]),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.scope_item,
                raw_text=text[:4000],
                normalized_text=normalize_text(text),
                value=value,
                entity_keys=[],
                source_refs=[src],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.7,
                review_status=ReviewStatus.needs_review,
                review_flags=["email_cid_inline_body"],
                parser_version=parser_version,
            )
        )
    return atoms



#: Chrome that appears in an email SIGNATURE image, not in anyone's scope.
#
# The CID fallback below emits whole-image OCR when no equipment rows parse out
# of it. On deal 010215 that produced 91 atoms -- 17% of all email atoms -- typed
# scope_item at confidence 0.7, and every one was a logo:
#
#     "CDW) OFFICIAL PROVIDER"                  x22
#     "Tech & Services Support Request"         x22
#     "sodexo Ik ol setorts eth the eversdoy"   x22   (tagline, OCR garbled)
#     "PurTeraIT Intelligent Field Execution"          (our own signature)
#
# A logo read as scope is worse than no atom: it is unfalsifiable filler in the
# evidence set, and it can never verify because the text is not in the body.
_BRAND_CHROME_RE = re.compile(
    r"official\s+provider|intelligent\s+field\s+execution|"
    r"support\s+request|tech\s*&\s*services|"
    r"^\s*(cdw|sodexo|purtera\s*it)\b",
    re.IGNORECASE,
)


def _is_brand_chrome_ocr(text: str) -> bool:
    """True when whole-image OCR is a logo or signature block rather than scope.

    Deliberately conservative: it fires on a recognised brand phrase, not on a
    heuristic like "short" or "no digits". Dropping a real inline table would
    lose evidence, which is the expensive direction of this error.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return True
    if _BRAND_CHROME_RE.search(cleaned):
        return True
    # OCR of a logo is mostly non-words: no sentence, no quantities, very short.
    if len(cleaned) < 24 and not any(ch.isdigit() for ch in cleaned):
        return True
    return False


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


# ── Email body hygiene (universal + structural — NO name/vocabulary lists) ──
#
# These guards remove *email chrome* (salutations, signature blocks, bullet
# markers, list headers) so the atoms we emit are the customer's actual scope,
# not the envelope around it. Every rule keys off STRUCTURE (word count,
# trailing comma, closing-phrase, bullet glyph, "label:" line) — never a
# specific person, deal, or domain term — so it generalises to any email.

# A leading list-bullet glyph / ordinal. Stripped so the atom is the ITEM
# ("Okta integration"), not the marker ("*   Okta integration").
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[*•·▪◦‣o]|[-–—]|\(?\d{1,2}[.)])\s+")

# A greeting/salutation opener led by a greeting word ("Hi", "Dear", …).
_GREETING_LEAD_RE = re.compile(
    r"^(?:hi|hey|hiya|hello|dear|greetings|good\s+(?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)

_IDENTITY_NAME_RE = re.compile(r"^[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){1,3}$")


def _is_identity_only_line(text: str) -> bool:
    """True for a line that is only a person's name, an email, a phone, or a
    punctuation fragment around one. Shape only -- no names, no domains."""
    from app.parsers.value_shapes import classify_value

    core = (text or "").strip().strip(" ;,<>:|-")
    if not core or len(core) > 60:
        return False
    return bool(_IDENTITY_NAME_RE.match(core)) or classify_value(core) in ("email", "phone")


# A sign-off phrase that opens the trailing signature block. Everything after
# it in an AUTHORED message is name/title/contact chrome — the sender identity
# is already captured as structured email-header metadata, so it is not scope.
_SIGNOFF_RE = re.compile(
    r"^(?:thanks|thank\s+you|thanks\s+(?:so\s+much|again|a\s+lot|much)|many\s+thanks|"
    r"regards|best|best\s+regards|kind\s+regards|warm\s+regards|warmest\s+regards|"
    r"sincerely|cheers|respectfully|talk\s+soon|appreciate\s+it|much\s+appreciated|"
    r"all\s+the\s+best|take\s+care|yours(?:\s+(?:truly|sincerely))?)\s*[,.!]*\s*$",
    re.IGNORECASE,
)

# A standalone list-section HEADER. The label is not itself an atom; the ITEMS
# beneath it are, and they inherit its polarity (include → scope, exclude →
# exclusion). Anchored to the whole line so only a bare label matches — a real
# sentence that merely contains the word ("please exclude the buildout") still
# flows through the normal pattern extractor.
_INCLUDE_LABEL_RE = re.compile(
    r"^(?:include[ds]?|inclusions?|included\s+items?|in\s+scope|"
    r"scope|scope\s+of\s+work|in-?scope)\s*:?\s*$",
    re.IGNORECASE,
)
_EXCLUDE_LABEL_RE = re.compile(
    r"^(?:exclude[ds]?|exclusions?|excluded\s+items?|out\s+of\s+scope|"
    r"not\s+included|not\s+in\s+scope|out-?of-?scope)\s*:?\s*$",
    re.IGNORECASE,
)


def _list_section_label(section: str | None) -> str | None:
    """Human-readable list header for Include/Exclude polarity (PDF ``section_path`` parity)."""
    if section == "include":
        return "Include"
    if section == "exclude":
        return "Exclude"
    return None


def _list_section_path(section: str | None, *, lead_in: list[str] | None = None) -> list[str]:
    """Section breadcrumb: framing lead-in + Include/Exclude header.

    Framing prose ("By the end of the meeting customer clarified:") prefixes
    the polarity label so Atom Quality can render
    ``…clarified › Include: Okta integration`` instead of a bare ``Include:``.
    The same text also rides on ``locator.lead_in`` / ``value.intro``.
    """
    path: list[str] = []
    for lead in lead_in or []:
        cleaned = (lead or "").strip().rstrip(":")
        if cleaned and cleaned not in path:
            path.append(cleaned)
    label = _list_section_label(section)
    if label:
        path.append(label)
    return path


_EQUIPMENT_LIST_INTRO_RE = re.compile(
    r"\b(?:below\s+is|attached\s+is|here\s+is|see)\b.{0,40}\b(?:full\s+)?equipment\s+list\b"
    r"|\b(?:full\s+)?equipment\s+list\b.{0,20}\b(?:below|attached|follows|here)\b"
    r"|\border\s+details\b",
    re.I,
)


def _is_equipment_list_intro_line(cleaned: str) -> bool:
    """True when body prose introduces a following inline equipment image/list.

    Universal connective rule: "Below is the full equipment list…" is framing
    for the CID/order screenshot that follows — not a standalone scope atom.
    Extra clauses on the same line (e.g. Okta requirement) still extract via
    typed patterns below; only the fail-open scope_item path is suppressed.
    """
    text = (cleaned or "").strip()
    if not text or len(text) > 280:
        return False
    if _BULLET_PREFIX_RE.match(text):
        return False
    return bool(_EQUIPMENT_LIST_INTRO_RE.search(text))


def _is_email_list_framing_lead_in(cleaned: str) -> bool:
    """True for a short framing sentence that governs an Include/Exclude list.

    Universal structural rule (not deal-specific): a non-bullet line ending in
    ``:`` that announces clarification / inclusion / exclusion, kept as
    ``lead_in`` on the bullets beneath rather than as its own atom.
    """
    text = (cleaned or "").strip()
    if not text or len(text) > 200:
        return False
    if _BULLET_PREFIX_RE.match(text):
        return False
    if _INCLUDE_LABEL_RE.match(text) or _EXCLUDE_LABEL_RE.match(text):
        return False
    if not text.endswith(":"):
        return False
    # Label-only attribution ("Customer specifically said:") is framing too.
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    if not (2 <= len(words) <= 25):
        return False
    lowered = normalize_text(text)
    cues = (
        "clarif", "includ", "exclud", "following", "scope", "customer",
        "meeting", "said", "noted", "confirmed", "agreed", "require",
    )
    return any(c in lowered for c in cues)


_QUOTE_OPEN = ('"', "\u201c")
_QUOTE_CLOSE = ('"', "\u201d")


def _is_customer_quote_line(cleaned: str) -> bool:
    """True when the line is a quoted customer utterance (not a requirement).

    A closing mark used to be required. But lines are split into sentences
    before typing, so a quote spanning two sentences arrives as:

        "Network build out does not need to be built into this.
        We will handle that separately."

    The first piece opens a quote and never closes it, so it failed the test and
    was typed ``scope_item`` -- a paraphrase-shaped requirement -- when it is the
    customer telling us what NOT to build. That is the exact confusion the
    customer_instruction class exists to prevent.

    An opening mark is therefore enough. A closing mark with no opening one is
    not: that is the tail of a quote whose opening piece already claimed it, and
    typing it too would double-count one utterance.
    """
    text = (cleaned or "").strip()
    if len(text) < 8:
        return False
    return text.startswith(_QUOTE_OPEN)


# OCR equipment rows that are truncated / ellipsis-garbled must not become atoms.
_OCR_JUNK_ELLIPSIS_RE = re.compile(r"\.{3,}|…")
_OCR_JUNK_LEADING_NOISE_RE = re.compile(r"^[IiLl1]\s+[A-Z]")
_ORDER_TABLE_HDR_LINE_RE = re.compile(
    r"^(?:order\s+details|delivered|qty|quantity|item|product)\s*$",
    re.I,
)


def _repair_ocr_equipment_line(cleaned: str) -> str:
    """Normalize common HubSpot order-screenshot OCR defects (universal).

    - Strip a leading single-letter noise glyph before a product name
      (``I Access Reader…`` / ``m Enterprise NVR`` → product stem).
    - Repair ``Camera Al`` → ``Camera AI`` (OCR of AI).
    - Collapse ellipsis truncations so the stem + trailing qty survive
      (``Access Reader Pro Juncti ... 5`` → ``Access Reader Pro Juncti 5``).
    """
    text = re.sub(r"\s+", " ", (cleaned or "").strip())
    if not text:
        return ""
    text = re.sub(r"^[IiLl1mM]\s+(?=[A-Z])", "", text)
    text = re.sub(r"\b(camera)\s+al\b", r"\1 AI", text, flags=re.I)
    # Keep product stem + trailing qty when OCR truncates with ellipsis.
    text = re.sub(r"\s*(?:\.{3,}|…)\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_ocr_junk_equipment_line(cleaned: str) -> bool:
    """Drop unrecognizable OCR debris — not recoverable order-table rows.

    Prefer ``_repair_ocr_equipment_line`` first. Only drop chrome / empty /
    non-product debris. Do NOT require a product-family vocabulary hit —
    that falsely drops real HubSpot rows like ``Power Distribution Pro 2``.
    """
    text = _repair_ocr_equipment_line(cleaned)
    if not text:
        return True
    if _ORDER_TABLE_HDR_LINE_RE.match(text):
        return True
    if _ORDER_DETAILS_HDR_RE.fullmatch(text):
        return True
    # Still truncated with no trailing qty and almost no letters → junk.
    if _OCR_JUNK_ELLIPSIS_RE.search(cleaned or "") and _trailing_order_qty(text) is None:
        return True
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 3:
        return True
    return False


def _is_greeting_line(cleaned: str) -> bool:
    """True when a line is a salutation opener ("Eddie,", "Hi John,", "Dear
    all,"). Structural: a short line (≤4 words) ending in a comma that is
    either led by a greeting word or is purely name-shaped tokens.

    These are NOT first-class atoms — the addressee is stamped as metadata
    (``value.addressee`` / ``to_greeting``) on sibling body atoms and the
    email header so Atom Quality can show a "To: Eddie" tag without a
    standalone reviewable ``Eddie,`` atom.
    """
    if not cleaned.endswith(","):
        return False
    words = cleaned.rstrip(",").split()
    if not (1 <= len(words) <= 4):
        return False
    if _GREETING_LEAD_RE.match(cleaned):
        return True
    # Pure name-shaped salutation: every token is alphabetic (allowing an
    # initial's period / hyphen / apostrophe) — "Eddie", "Mr. Smith", "Jean-Luc".
    return all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words)


def _greeting_addressee_name(cleaned: str) -> str:
    """Extract the display name from a salutation line ("Eddie," → "Eddie")."""
    text = (cleaned or "").rstrip(",").strip()
    if not text:
        return ""
    if _GREETING_LEAD_RE.match(cleaned or ""):
        rest = _GREETING_LEAD_RE.sub("", text).strip(" ,")
        return rest or text
    return text


_ADDRESSEE_STAMP_KINDS: frozenset[str] = frozenset(
    {
        "email_body_line",
        "email_body_context",
        "email_cid_equipment_line",
        "email_cid_inline_body",
        "email_header",
        "email_attachment",
    }
)


def _stamp_email_addressee(atoms: list[EvidenceAtom], addressee: str) -> None:
    """Attach body-greeting addressee as structured metadata on email atoms."""
    name = (addressee or "").strip()
    if not name:
        return
    for atom in atoms:
        val = atom.value if isinstance(atom.value, dict) else None
        if val is None:
            continue
        kind = str(val.get("kind") or "")
        if kind in _ADDRESSEE_STAMP_KINDS:
            val["addressee"] = name
            val["to_greeting"] = name
            continue
        # BOM / backfill rows from CID OCR: stamp when locator is email-sourced.
        refs = list(getattr(atom, "source_refs", None) or [])
        loc = refs[0].locator if refs and isinstance(getattr(refs[0], "locator", None), dict) else {}
        if loc.get("kind") == "email_cid_inline" or "message_index" in loc:
            val["addressee"] = name
            val["to_greeting"] = name


def _body_greeting_addressee(blocks: list[dict[str, Any]]) -> str | None:
    """First authored-message salutation name, if any."""
    for block in blocks or []:
        if block.get("quoted"):
            continue
        for line in block.get("lines") or []:
            raw = str(line or "").lstrip("> ").strip()
            cleaned = _BULLET_PREFIX_RE.sub("", raw).strip()
            if cleaned and _is_greeting_line(cleaned):
                name = _greeting_addressee_name(cleaned)
                return name or None
    return None


_COURTESY_PROSE_RE = re.compile(
    r"^(?:appreciate\b|thanks?\b|thank you\b|looking forward\b|"
    r"hope (?:you(?:'re| are)?|this)\b|just (?:wanted|checking|following)\b|"
    r"wanted to (?:follow|check|touch)\b|as (?:discussed|mentioned)\b|"
    r"please (?:see|find|let me know)\b|attached (?:is|please find)\b|"
    r"below is\b|see (?:below|attached)\b|"
    r"let (?:me|us) know\b|feel free\b|"
    r"this is one that\b|if we(?:'re| are) fast\b)",
    re.IGNORECASE,
)


def _is_courtesy_prose_line(cleaned: str) -> bool:
    """True for framing/courtesy prose that must not become baseline scope_item.

    Universal structural rule: long conversational openers ("Appreciate you
    hopping on…", "Attached is a summary…") are email *communication context*,
    not contractual scope. They are emitted as ``deal_metadata`` /
    ``email_body_context`` (see body loop). Typed extractors (requirement /
    exclusion / …) still run; only the fail-open ``scope_item`` gate is skipped.

    Equipment-list intros are also courtesy/framing here so they do not mint
    orphan scope — they stay connective tissue via CID ``lead_in``.
    """
    text = (cleaned or "").strip()
    if not text or len(text) < 24:
        return False
    if _BULLET_PREFIX_RE.match(text):
        return False
    lowered = normalize_text(text)
    # Keep lines that already look like actionable scope / constraints.
    if any(
        needle in lowered
        for needle in (
            "include",
            "exclude",
            "requirement",
            "must ",
            "need to",
            "required",
            "after hours",
            "badge",
            "escort",
            "install",
            "configure",
            "deploy",
        )
    ):
        # Equipment-list intros often share a line with a hard requirement
        # ("…equipment list. One hard requirement… Otka…"). Still framing for
        # the following CID image — do not mint a bare scope_item for the intro.
        if _is_equipment_list_intro_line(text):
            return True
        return False
    if _is_equipment_list_intro_line(text):
        return True
    if _COURTESY_PROSE_RE.match(text):
        return True
    return False


def _is_email_body_context_line(cleaned: str) -> bool:
    """True when courtesy prose should be kept as an ``email_body_context`` atom.

    Universal: authored intro / logistics paragraphs that frame the ask without
    being Include/Exclude bullets or equipment-list CID lead-ins. Distinct from
    ``_is_courtesy_prose_line`` so equipment-list intros stay connective-only.
    """
    if not _is_courtesy_prose_line(cleaned):
        return False
    if _is_equipment_list_intro_line(cleaned):
        return False
    return True


def _cid_reading_anchor(
    *,
    body_text: str,
    content_id: str,
    blocks: list[dict[str, Any]],
) -> tuple[int, int]:
    """Return ``(message_index, line_start)`` where an inline CID belongs in reading order.

    Prefer the body line that references ``cid:…``. Fall back to the line that
    introduces the equipment/order list. Last resort: immediately after the
    last authored body line so CID rows sort *after* Include/Exclude prose.
    """
    cid_norm = _normalize_cid(content_id)
    intro_re = re.compile(
        r"\b(?:full\s+)?equipment\s+list\b|\border\s+details\b|\bsee\s+(?:the\s+)?(?:list|image|screenshot)\b",
        re.I,
    )
    fallback_msg = 0
    fallback_line = 1
    for block in blocks or []:
        msg_i = int(block.get("message_index") or 0)
        base = int(block.get("line_start") or 1)
        lines = list(block.get("lines") or [])
        if lines:
            fallback_msg = msg_i
            fallback_line = base + len(lines)
        for idx, line in enumerate(lines):
            raw = str(line or "")
            line_num = base + idx
            for m in _CID_REF_RE.finditer(raw):
                if _normalize_cid(m.group(1)) == cid_norm:
                    return msg_i, line_num
            if intro_re.search(raw):
                # Equipment image follows the intro sentence in natural reading.
                fallback_msg = msg_i
                fallback_line = line_num + 1
    return fallback_msg, fallback_line


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



#: A line is only split when it plainly holds more than one real sentence.
#: Short fragments ("Hi.", "Thanks.") and anything table- or header-shaped are
#: left whole, so the common case is byte-identical to the previous behaviour.
_MIN_SENTENCE_CHARS = 12


def _expand_lines_to_sentences(
    lines: list[str], line_start: int
) -> list[tuple[int, str]]:
    """Yield ``(source_line_number, text)`` with prose lines split by sentence.

    Returns the line unchanged unless every resulting piece is a substantial
    sentence, so a signature, a table row or a greeting never fragments. The
    line number is the ORIGINAL one for every piece: splitting changes what a
    single atom covers, never where it came from.
    """
    from app.core.sentences import split_sentences

    out: list[tuple[int, str]] = []
    for line_idx, line in enumerate(lines):
        line_num = line_start + line_idx
        stripped = (line or "").strip()
        # Table rows, header lines and quoted-only markers are not prose.
        if not stripped or "|" in stripped or stripped.count(".") < 2:
            out.append((line_num, line))
            continue
        try:
            pieces = [p.strip() for p in split_sentences(stripped) if p.strip()]
        except Exception:  # pragma: no cover - never fail a parse over this
            out.append((line_num, line))
            continue
        if len(pieces) < 2 or any(len(p) < _MIN_SENTENCE_CHARS for p in pieces):
            out.append((line_num, line))
            continue
        prefix = line[: len(line) - len(line.lstrip("> "))]
        for piece in pieces:
            out.append((line_num, prefix + piece))
    return out


_PSEUDO_HEADER_RE = re.compile(
    r"^(from|sent|date|to|cc|bcc|subject|reply-to|importance|attachments)"
    r"\s*:\s*(.*)$",
    re.IGNORECASE,
)


def _split_leading_pseudo_headers(text: str) -> tuple[dict[str, str], str]:
    """Peel an Outlook "save as text" header block off the top of a body.

    Returns ``(values, remaining_text)``. Only the LEADING run is taken, so a
    quoted reply further down keeps its own headers inside its quoted block --
    those already carry ``quoted_old_email`` authority and are the quoted
    message, not this one.

    Conservative on purpose, because the alternative is eating real content:
    at least two header lines, and one of them must be ``from`` or ``subject``,
    so an ordinary body opening "Note: ..." over "Owner: ..." is left alone.
    """
    lines = text.splitlines()
    values: dict[str, str] = {}
    consumed = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # A blank line ends the block, but only once something matched;
            # leading blank lines are just leading blank lines.
            if values:
                consumed = index + 1
                break
            consumed = index + 1
            continue
        match = _PSEUDO_HEADER_RE.match(stripped)
        if not match:
            break
        key = match.group(1).lower()
        if key not in values:
            values[key] = match.group(2).strip()
        consumed = index + 1
    if len(values) < 2 or not ({"from", "subject"} & set(values)):
        return {}, text
    # Blank the header lines rather than dropping them: every locator
    # downstream records a line number against the ORIGINAL file, and deleting
    # the block shifts the whole body up by ``consumed``. Receipt replay then
    # verifies each atom against the wrong line. Exactly the failure the
    # transcript speaker-fold hit, so it is worth stating twice: a rewrite that
    # changes line COUNT is a rewrite that breaks provenance.
    return values, "\n".join([""] * consumed + lines[consumed:])


def _normalize_sender(value: str) -> str:
    """Repair a display name whose address was wrapped across lines.

    Rejoining "Trent Torrence <" with the address on the next line produced
    ``Trent Torrence <t@purtera-it.com`` -- an unbalanced bracket, because the
    closing ">" had wrapped too, or sat inside a `<mailto:...>` decoration that
    the text conversion left behind.

    A malformed sender is not merely untidy. ``_originating_sender`` takes the
    HIGHEST message index in a forward chain -- the oldest message, the person
    a document actually came from -- so it lands precisely on the deepest,
    most-wrapped blocks. On deal 010215 the well-formed senders sat at index 1
    and every index from 2 to 16 carried a truncated one, so a forward whose
    chain started with the customer was attributed to us instead.
    """
    text = " ".join(str(value or "").split())
    if not text:
        return text
    # "name@host<mailto:NAME@HOST>" -- the anchor text repeated as a link.
    text = re.sub(r"<mailto:[^>]*>?", "", text, flags=re.IGNORECASE).strip()
    if "<" in text and ">" not in text:
        # Close it only when what follows the bracket is actually an address;
        # otherwise leave the text alone rather than inventing structure.
        head, _, tail = text.partition("<")
        if "@" in tail:
            text = f"{head.strip()} <{tail.strip()}>".strip()
    return text


class EmailParser(BaseParser):
    parser_name = "email"
    parser_version = "email_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".eml", ".txt", ".md"],
        supported_artifact_types=[ArtifactType.email, ArtifactType.txt],
        emitted_atom_types=[AtomType.exclusion, AtomType.customer_instruction, AtomType.constraint, AtomType.open_question, AtomType.deal_metadata],
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
                    # This existed so a keyword-bearing text file "isn't
                    # silently dropped" -- a real concern when NOTHING claimed
                    # .txt. MarkdownParser now floors that format, so the
                    # fallback is redundant, and at 0.55 it outranked the floor
                    # and took files it had no evidence for: a pipe-delimited
                    # site table containing "escort required" thirty times was
                    # claimed as an email.
                    #
                    # One scope phrase is not evidence of correspondence.
                    # Below threshold: the reason stays in the trace, the claim
                    # does not stand. Real email still claims at 0.91 on
                    # RFC-5322 headers and 0.83 on thread markers.
                    confidence = max(confidence, 0.45)
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
        # An Outlook "save as text" .txt carries the SAME From/Sent/To/Subject
        # block a .eml does, just as body text rather than as MIME headers. It
        # therefore reached _extract_atoms_from_block and each header line
        # became its own scope_item at customer_current_authored -- which is
        # exactly the defect _header_atom's comment below describes having
        # fixed, surviving in the other container.
        #
        # Measured on one message written both ways:
        #
        #   .eml   6 atoms   headers -> one deal_metadata atom
        #   .txt   9 atoms   headers -> four scope_item atoms at rank 90
        #                               "From: jane.customer@acme.example"
        #                               "Sent: Wednesday, January 15, 2026..."
        #                               "To: pm@purtera.example"
        #                               "Subject: Scope update"
        #
        # Same content, different container, four phantom units of work in the
        # customer-authored tier. Split the leading header run off the body so
        # both containers agree.
        pseudo_headers: dict[str, str] = {}
        if path.suffix.lower() != ".eml":
            pseudo_headers, text = _split_leading_pseudo_headers(text)
        # For .eml the RFC822 headers never reach the body, so the first
        # message has no From/Sent line to find. Read them off the envelope.
        envelope_sender = ""
        envelope_sent_at = ""
        if path.suffix.lower() == ".eml":
            try:
                hdrs = parse_email_thread_headers(path)
                envelope_sender = str(hdrs.get("sender") or "")
                envelope_sent_at = str(hdrs.get("date_raw") or "")
            except Exception:
                # A malformed .eml must not lose its body; an empty locator is
                # worse than no locator only when it is silent, and the atoms
                # still carry their text either way.
                pass
        blocks = self._split_blocks(
            text, envelope_sender=envelope_sender, envelope_sent_at=envelope_sent_at,
        )
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
        # Header atom — the From/To/Cc/Subject/Date line is deal/routing
        # metadata (and the content census inventories it). Emit it as
        # ``deal_metadata`` so the header is never silently absent from the atom
        # stream but is NEVER mistaken for contractual scope. .eml only;
        # headerless .txt/.md bodies have no structured headers to surface.
        header_atom = self._header_atom(
            project_id=project_id, artifact_id=artifact_id, path=path
        )
        if header_atom is None and pseudo_headers:
            # Same treatment for the .txt shape: surfaced, never scope.
            header_atom = self._pseudo_header_atom(
                project_id=project_id, artifact_id=artifact_id, path=path,
                values=pseudo_headers,
            )
        if header_atom is not None:
            atoms.append(header_atom)
        # Every quoted message's own routing line, so the chain survives
        # whatever body hygiene removes.
        atoms.extend(
            self._quoted_chain_atoms(
                project_id=project_id, artifact_id=artifact_id,
                filename=path.name, blocks=blocks,
            )
        )
        # Attachments are the real deal docs more often than the body — mark
        # each one so it can't silently vanish (the file content is a separate
        # artifact; this is a located pointer the PM/census can see).
        atoms.extend(
            self._attachment_markers(
                project_id=project_id, artifact_id=artifact_id, path=path
            )
        )
        atoms.extend(
            self._cid_inline_atoms(
                project_id=project_id,
                artifact_id=artifact_id,
                path=path,
                body_text=text,
                blocks=blocks,
            )
        )
        # Body greeting ("Eddie,") is metadata/tag — not a reviewable atom.
        greeting = _body_greeting_addressee(blocks)
        if greeting:
            _stamp_email_addressee(atoms, greeting)
        structured_doc = self._build_structured_doc(filename=path.name, blocks=blocks)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _pseudo_header_atom(
        self, *, project_id: str, artifact_id: str, path: Path, values: dict[str, str]
    ) -> EvidenceAtom | None:
        """The .eml header atom, for a message saved as plain text.

        Deliberately mirrors ``_header_atom``: same ``deal_metadata`` type,
        same ``email_header`` locator kind, so a message carries the same
        evidence whichever way the PM exported it. ``artifact_type`` follows
        the real file so source replay still verifies against a .txt.
        """
        parts = [
            f"{field.capitalize()}: {values[field]}"
            for field in ("from", "sent", "date", "to", "cc", "subject")
            if values.get(field)
        ]
        if not parts:
            return None
        text = " | ".join(parts)
        artifact_type = ArtifactType.txt
        src = SourceRef(
            id=stable_id("src", artifact_id, "email_header"),
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=path.name,
            locator={"kind": "email_header"},
            extraction_method="email_headers_plaintext",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "email_header", text),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.deal_metadata,
            raw_text=text,
            normalized_text=normalize_text(text),
            value={"kind": "email_header", **values},
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.86,
            review_status=ReviewStatus.auto_accepted,
            parser_version=self.parser_version,
            source_refs=[src],
        )

    def _quoted_chain_atoms(
        self, *, project_id: str, artifact_id: str, filename: str, blocks: list[dict]
    ) -> list[EvidenceAtom]:
        """One routing atom per QUOTED message, mirroring ``_header_atom``.

        The outer message's From/To/Subject/Date has always been surfaced as a
        ``deal_metadata`` atom, so it can never silently vanish and can never be
        mistaken for scope. Quoted messages had no equivalent: their sender and
        date lived only on the LOCATOR of whatever body atoms happened to
        survive.

        That is a real loss of information -- a forward's chain is who asked
        whom for what, and it is exactly the context a reader needs -- and it is
        also fragile. Cleaning header and signature chrome out of quoted blocks
        left one of deal 010215's messages with no atoms at all, so its sender
        went with them and ``_originating_sender`` credited the customer's own
        documents to somebody else.

        Attribution must not depend on which body lines happened to be worth
        keeping. Emitted as ``deal_metadata`` at low confidence: routing
        context, never a claim about the work.
        """
        atoms: list[EvidenceAtom] = []
        for block in blocks:
            if not block.get("quoted"):
                continue
            sender = str(block.get("sender") or "").strip()
            sent_at = str(block.get("sent_at") or "").strip()
            if not sender or sender.lower() == "unknown":
                continue
            index = int(block.get("message_index") or 0)
            parts = [f"From: {sender}"]
            if sent_at:
                parts.append(f"Sent: {sent_at}")
            text = " | ".join(parts)
            src = SourceRef(
                id=stable_id("src", artifact_id, "quoted_header", index),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.email,
                filename=filename,
                locator={
                    "kind": "email_quoted_header",
                    "message_index": index,
                    "sender": sender,
                    "sent_at": sent_at,
                    "line_start": int(block.get("line_start") or 0),
                    "line_end": int(block.get("line_start") or 0),
                    "quoted": True,
                },
                extraction_method="email_quoted_headers",
                parser_version=self.parser_version,
            )
            atoms.append(
                EvidenceAtom(
                    id=stable_id("atm", project_id, artifact_id, "quoted_header", index),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.deal_metadata,
                    raw_text=text,
                    normalized_text=normalize_text(text),
                    value={"kind": "quoted_message_header", "sender": sender,
                           "sent_at": sent_at, "message_index": index},
                    source_refs=[src],
                    authority_class=AuthorityClass.quoted_old_email,
                    confidence=0.45,
                    review_status=ReviewStatus.auto_accepted,
                    parser_version=self.parser_version,
                )
            )
        return atoms

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
            # A From/To/Subject line is not a unit of work. Typed as scope_item
            # it entered the SOW pipeline as customer-authored scope at 0.86,
            # auto-accepted, one per .eml -- and downstream had to grow text
            # heuristics to claw it back out (graph_builder._looks_like_email_
            # header suppresses it from exclusion fan-out by regex). It is
            # correspondence metadata, which is what deal_metadata is for.
            # Typing it deal_metadata keeps it out of the contractual-scope
            # surface a quote/scope head reads, while still surfacing it as a
            # first-class atom the census can reconcile.
            atom_type=AtomType.deal_metadata,
            raw_text=text,
            normalized_text=normalize_text(text),
            value={
                "kind": "email_header",
                "field_name": "email_metadata",
                **values,
                "email_thread_meta": thread_meta,
            },
            entity_keys=[],
            source_refs=[src],
            # Machine-extracted envelope metadata — lowest authority band so it
            # never governs a scope packet.
            authority_class=AuthorityClass.machine_extractor,
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

    def _cid_inline_atoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        path: Path,
        body_text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> list[EvidenceAtom]:
        if path.suffix.lower() != ".eml":
            return []
        msg = _parse_email_message(path)
        if msg is None:
            return []
        inline_parts = _iter_cid_inline_parts(msg)
        referenced = {_normalize_cid(m.group(1)) for m in _CID_REF_RE.finditer(body_text or "")}
        body_l = (body_text or "").lower()
        equipment_list_hint = (
            "equipment list" in body_l
            or "full equipment list" in body_l
            or "order details" in body_l
        )
        if not inline_parts and not referenced:
            return []
        atoms: list[EvidenceAtom] = []
        # When the email references an equipment screenshot, OCR every inline image —
        # not only CIDs mentioned in the plain-text body (HTML-only cid: refs).
        if equipment_list_hint and inline_parts:
            targets = set(inline_parts.keys())
        else:
            targets = referenced or set(inline_parts.keys())
        ocr_by_cid: dict[str, str] = {}
        for cid in targets:
            part = inline_parts.get(cid)
            if not part:
                if cid in referenced:
                    atoms.append(
                        self._unresolved_cid_atom(
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            content_id=cid,
                            referenced=sorted(referenced),
                        )
                    )
                continue
            ocr_by_cid[cid] = _ocr_cid_part(part)

        equipment_lead_in = _equipment_list_lead_in(body_text, blocks)

        def _hw_from(cid: str, blob: str) -> list[EvidenceAtom]:
            msg_i, line_i = _cid_reading_anchor(
                body_text=body_text, content_id=cid, blocks=blocks or []
            )
            return _hardware_atoms_from_equipment_text(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                text=blob,
                content_id=cid,
                parser_version=self.parser_version,
                message_index=msg_i,
                anchor_line=line_i,
                lead_in=equipment_lead_in or None,
            )

        equipment_lines: list[EvidenceAtom] = []
        if ocr_by_cid:
            if equipment_list_hint and len(ocr_by_cid) > 1:
                # Prefer HubSpot order-table OCR (score + payload size), not
                # whichever CID happens to emit the most regex hits — transcript
                # screenshots can mention products and win a naive count race.
                ranked: list[tuple[int, int, int, str, list[EvidenceAtom]]] = []
                for cid, text in ocr_by_cid.items():
                    batch = _hw_from(cid, text)
                    eq = [a for a in batch if a.value.get("kind") == "email_cid_equipment_line"]
                    part = inline_parts.get(cid) or {}
                    payload_size = int(part.get("size") or 0)
                    ranked.append(
                        (
                            _score_cid_ocr_text(text),
                            len(eq),
                            payload_size,
                            cid,
                            eq,
                        )
                    )
                ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
                if ranked and ranked[0][1] > 0:
                    equipment_lines = ranked[0][4]
                else:
                    pick_cid, pick_text = _pick_best_cid_ocr(ocr_by_cid)
                    if _score_cid_ocr_text(pick_text) > 0:
                        equipment_lines = [
                            a
                            for a in _hw_from(pick_cid, pick_text)
                            if a.value.get("kind") == "email_cid_equipment_line"
                        ]
            else:
                pick_cid, pick_text = _pick_best_cid_ocr(ocr_by_cid)
                if _score_cid_ocr_text(pick_text) > 0:
                    equipment_lines = [
                        a
                        for a in _hw_from(pick_cid, pick_text)
                        if a.value.get("kind") == "email_cid_equipment_line"
                    ]
        if not equipment_lines:
            for cid, text in ocr_by_cid.items():
                equipment_lines.extend(
                    _hw_from(cid, text)
                )
        atoms.extend(equipment_lines)
        has_equipment = any(a.value.get("kind") == "email_cid_equipment_line" for a in atoms)
        has_inline_body = any(a.value.get("kind") == "email_cid_inline_body" for a in atoms)
        if referenced:
            resolved = {
                cid
                for cid in referenced
                if any(
                    a.value.get("content_id") == cid
                    and a.value.get("kind") in {"email_cid_equipment_line", "email_cid_inline_body"}
                    for a in atoms
                )
            }
            for cid in sorted(referenced - resolved):
                # Image exists in MIME → not an unresolved CID. Empty OCR is a
                # backend gap, not a missing-part story; suppress open_question
                # when any equipment/body was extracted from sibling CIDs, or
                # when the part itself is present (connective tissue already
                # points body "equipment list" prose at these embeds).
                if cid in inline_parts:
                    continue
                if has_equipment or has_inline_body:
                    continue
                if not any(
                    a.value.get("kind") == "email_cid_unresolved"
                    and cid in (a.value.get("content_ids") or [])
                    for a in atoms
                ):
                    atoms.append(
                        self._unresolved_cid_atom(
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            content_id=cid,
                            referenced=sorted(referenced),
                        )
                    )

        return atoms

    def _unresolved_cid_atom(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        content_id: str,
        referenced: list[str],
    ) -> EvidenceAtom:
        src = SourceRef(
            id=stable_id("src", artifact_id, "cid_missing", content_id),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.email,
            filename=filename,
            locator={"kind": "email_cid_reference", "content_ids": referenced, "content_id": content_id},
            extraction_method="email_cid_reference",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "cid_unresolved", content_id),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.open_question,
            raw_text=(
                f"Referenced inline equipment image (cid:{content_id}) could not be resolved or OCR'd from MIME parts."
            ),
            normalized_text=(
                f"referenced inline equipment image (cid:{content_id}) could not be resolved or ocr'd from mime parts."
            ),
            value={
                "text": (
                    f"Referenced inline equipment image (cid:{content_id}) could not be resolved or OCR'd from MIME parts."
                ),
                "kind": "email_cid_unresolved",
                "content_id": content_id,
                "content_ids": referenced,
            },
            entity_keys=[],
            source_refs=[src],
            authority_class=AuthorityClass.customer_current_authored,
            confidence=0.6,
            review_status=ReviewStatus.needs_review,
            review_flags=["email_cid_unresolved"],
            parser_version=self.parser_version,
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

    def _split_blocks(
        self,
        text: str,
        *,
        envelope_sender: str = "",
        envelope_sent_at: str = "",
    ) -> list[dict[str, Any]]:
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
                blocks.append(
                    self._build_block(
                        blocks, current,
                        envelope_sender=envelope_sender,
                        envelope_sent_at=envelope_sent_at,
                    )
                )
                current = []
            current.append((idx, line))
        if current:
            blocks.append(
                self._build_block(
                    blocks, current,
                    envelope_sender=envelope_sender,
                    envelope_sent_at=envelope_sent_at,
                )
            )
        return blocks

    def _build_block(
        self,
        existing: list[dict[str, Any]],
        lines: list[tuple[int, str]],
        *,
        envelope_sender: str = "",
        envelope_sent_at: str = "",
    ) -> dict[str, Any]:
        """One message in a thread, with who sent it and when.

        Sender and date are read out of the BODY, because in a quoted reply
        chain that is where they live -- "From: ... Sent: ..." above each older
        message. The top-level message is the exception: in an .eml its headers
        are in the RFC822 envelope and were stripped before the body ever got
        here, so it found nothing and recorded sender "unknown" with no date.

        That is not cosmetic. Receipt verification anchors a quote through the
        locator, and an empty locator cannot be anchored: measured on deal
        010215, 82% of email atoms failed verification against 0% for
        documents, meetings and notes -- 257 of them with exactly this
        signature. Email is the largest evidence source in the corpus, so most
        of what the system reads was carrying claims it could not prove.

        The envelope values are a fallback for the FIRST block only. Later
        blocks are quoted messages whose own headers are in the text; letting
        the envelope override those would attribute every message in a thread
        to whoever sent the last reply.
        """
        stripped_lines = [line.strip() for _, line in lines]
        sender = self._find_header_value(stripped_lines, "from")
        sent_at = self._find_header_value(stripped_lines, "sent") or self._find_header_value(stripped_lines, "date")
        # Deliberately NOT folded into `sender`. _authority_for_block reads that
        # field, and giving the first block a real sender flips the top-level
        # message of every .eml we wrote from customer_current_authored to
        # machine_extractor. That is almost certainly the more correct answer --
        # an email we sent is not customer-authored -- but it re-ranks the
        # authority lattice across the whole corpus, which is a much larger
        # change than fixing a locator and needs its own review. Kept separate
        # so the receipt fix does not smuggle it in.
        locator_sender = sender or (envelope_sender if not existing else "")
        locator_sent_at = sent_at or (envelope_sent_at if not existing else "")
        quoted = any(line.startswith(">") for line in stripped_lines) or len(existing) > 0
        return {
            "message_index": len(existing),
            "line_start": lines[0][0],
            "line_end": lines[-1][0],
            "sender": sender or "unknown",
            "sent_at": sent_at or "",
            # What the LOCATOR should carry: the same values, plus the envelope
            # fallback for the first message. This is what receipt verification
            # anchors a quote through.
            "locator_sender": locator_sender or "unknown",
            "locator_sent_at": locator_sent_at or "",
            "quoted": quoted,
            "lines": stripped_lines,
        }

    def _find_header_value(self, lines: list[str], key: str) -> str | None:
        """The value of a quoted header, whether it is on the label's line or the next.

        HTML mail puts each header label in its own element, so converting to
        text yields:

            From:
            Patrick Kelly <patrick@purtera-it.com>
            Sent:
            Wednesday, August 12, 2026 1:12 PM

        Requiring the value on the SAME line matched nothing there, so every
        quoted message in a forward read sender "unknown". That is not cosmetic:
        the ten Marion County SOWs arrived inside a forward, and with no sender
        on the inner messages the only name available was whoever forwarded it
        last -- attributing the customer's own documents to us.
        """
        pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*)$", flags=re.IGNORECASE)
        for idx, line in enumerate(lines):
            match = pattern.match(line)
            if match and not (match.group(1) or "").strip():
                # Label alone on its line: the value is the next non-empty line,
                # unless that line is itself another header label.
                for offset, nxt in enumerate(lines[idx + 1 : idx + 3], start=1):
                    candidate = (nxt or "").strip()
                    if not candidate:
                        continue
                    if re.match(r"^(from|sent|date|to|cc|bcc|subject)\s*:", candidate, re.IGNORECASE):
                        break
                    # HTML mail wraps a long display name away from its address,
                    # leaving "Trent Torrence <" with "t@purtera-it.com>" on the
                    # next line. Rejoin them, or the sender is a name with no
                    # address and cannot be matched to anyone.
                    #
                    # The tail was searched from a FIXED lines[idx+2:idx+4],
                    # while the candidate itself may have come from idx+2 when
                    # idx+1 was blank -- so the join could swallow the candidate
                    # or reach past the address entirely. Search from wherever
                    # the candidate actually was.
                    if candidate.endswith("<"):
                        after = idx + 1 + offset
                        tail = next(
                            (x.strip() for x in lines[after : after + 3] if (x or "").strip()), ""
                        )
                        if "@" in tail:
                            candidate = f"{candidate}{tail}"
                    return _normalize_sender(candidate)
                continue
            if match:
                return _normalize_sender(match.group(1).strip())
        return None

    def _authority_for_block(self, block: dict[str, Any]) -> AuthorityClass:
        if block["quoted"]:
            return AuthorityClass.quoted_old_email
        sender = normalize_text(str(block.get("sender", "")))
        from app.core.internal_author import is_internal_author

        if is_internal_author(sender):
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

    def _site_atoms_from_line(
        self,
        *,
        project_id: str,
        artifact_id: str,
        cleaned: str,
        entity_keys: list[str],
        source_ref: SourceRef,
        authority: AuthorityClass,
        confidence: float,
    ) -> list[EvidenceAtom]:
        out: list[EvidenceAtom] = []
        try:
            from app.core.vendor_site_ban import is_purtera_vendor_address

            if is_purtera_vendor_address(text=cleaned):
                return []
        except Exception:
            pass
        for parsed in find_us_addresses_in_text(cleaned):
            if not parsed.city or not parsed.state or parsed.state not in US_STATES:
                continue
            if not parsed.street_address:
                continue
            slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                f"{parsed.city}_{parsed.state}_{parsed.zip or parsed.street_address}".lower(),
            ).strip("_")
            display = f"{parsed.street_address}, {parsed.city}, {parsed.state} {parsed.zip or ''}".strip()
            keys = list(dict.fromkeys([*entity_keys, f"site:{slug}"]))
            aliases = list(dict.fromkeys(parsed.aliases))
            names = list(dict.fromkeys([display, parsed.city, *aliases]))
            out.append(
                EvidenceAtom(
                    id=stable_id("atm", project_id, artifact_id, "email_note_physical_site", slug),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.physical_site,
                    raw_text=display,
                    normalized_text=normalize_text(display),
                    value={
                        "kind": "physical_site",
                        "id": slug,
                        "site_id": slug,
                        "name": display,
                        "names": names,
                        "aliases": aliases,
                        "street_address": parsed.street_address,
                        "address": parsed.street_address,
                        "city": parsed.city,
                        "state": parsed.state,
                        "zip": parsed.zip,
                        "inferred": True,
                        "source_context": cleaned[:600],
                    },
                    entity_keys=keys,
                    source_refs=[source_ref],
                    authority_class=authority,
                    confidence=max(confidence, 0.72),
                    review_status=ReviewStatus.needs_review,
                    review_flags=["email_note_physical_site"],
                    parser_version=self.parser_version,
                )
            )
        return out

    def _build_source_ref(
        self,
        artifact_id: str,
        filename: str,
        block: dict[str, Any],
        *,
        line_num: int | None = None,
        section_path: list[str] | None = None,
        lead_in: list[str] | None = None,
    ) -> SourceRef:
        """Build a source ref pinned to one body line when ``line_num`` is set.

        Block-level refs (whole message) use ``line_start``/``line_end``; per-atom
        refs pin a single line so replay/verification and document-order sort work.
        """
        start = int(line_num if line_num is not None else block["line_start"])
        end = int(line_num if line_num is not None else block["line_end"])
        locator: dict[str, Any] = {
            "message_index": block["message_index"],
            "line_start": start,
            "line_end": end,
            "sender": block.get("locator_sender") or block["sender"],
            "sent_at": block.get("locator_sent_at") or block["sent_at"],
            "quoted": block["quoted"],
        }
        if section_path:
            locator["section_path"] = list(section_path)
        if lead_in:
            locator["lead_in"] = list(lead_in)
        return SourceRef(
            id=stable_id("src", artifact_id, block["message_index"], start, end),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.email,
            filename=filename,
            locator=locator,
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
        confidence = 0.45 if authority == AuthorityClass.quoted_old_email else 0.86

        # Body-hygiene state, per message block. ``in_signature`` latches once
        # an authored message reaches its sign-off. ``current_section`` carries
        # an Include/Exclude list header down onto the bullet items beneath it.
        # ``pending_lead_in`` holds framing prose ("By the end of the meeting
        # customer clarified:") until the next Include/Exclude list consumes it.
        in_signature = False
        # A top-posted Outlook signature -- name line, then address line --
        # opens some authored messages. The sign-off latch below only catches
        # the TRAILING block, so "Nick Robateau" / "Nick.Robateau@CDW.com"
        # walked through as content and the classifier typed them `exclusion`.
        # Until an authored block says something substantive, an identity-only
        # line is chrome.
        seen_substantive = False
        # HTML mail puts each header label in its own element, so converting to
        # text yields the label and its value on SEPARATE lines:
        #
        #     From:
        #     Quinton James <quinton.james@cdw.com>
        #     Sent:
        #     Wednesday, August 12, 2026 10:20 AM
        #
        # `_PSEUDO_HEADER_RE` needs "Label:" on the same line, so the value line
        # walked straight through and became a scope_item. This latches on a
        # bare label and consumes the one line that follows it.
        after_bare_header_label = False
        current_section: str | None = None  # "include" | "exclude" | None
        pending_lead_in: list[str] = []
        active_lead_in: list[str] = []

        # One physical line can hold several sentences with different speech
        # acts, and typing the line as a whole makes them fight. The customer
        # email "Please remove West Wing from scope. Main Campus requires
        # escort access after 5pm." is one line and two facts: an exclusion and
        # a constraint. Typed together the constraint cue wins, so the
        # customer's own exclusion is never emitted as an exclusion -- and
        # ``prefer_customer_exclusion`` looks for exactly that, so the West
        # Wing packet ended up governed by a PM's transcript note (meeting_note,
        # rank 55) instead of the customer's written instruction (rank 90).
        #
        # The line number is preserved on every piece, so locators, replay and
        # document order are unchanged; only the granularity of typing moves.
        for line_num, line in _expand_lines_to_sentences(
            block["lines"], int(block["line_start"])
        ):
            raw_cleaned = line.lstrip("> ").strip()
            if not raw_cleaned:
                continue
            is_bullet = bool(_BULLET_PREFIX_RE.match(raw_cleaned))
            cleaned = _BULLET_PREFIX_RE.sub("", raw_cleaned).strip()
            if not cleaned:
                continue
            if not block.get("quoted") and not seen_substantive:
                if _is_identity_only_line(cleaned):
                    continue
                seen_substantive = True
            # Bullets inherit the active Include/Exclude header; compute before
            # hygiene continues so per-line locators carry section_path.
            section_for_line = current_section if is_bullet else None
            lead_for_line = list(active_lead_in) if section_for_line else []
            section_path = _list_section_path(section_for_line, lead_in=lead_for_line or None)
            source_ref = self._build_source_ref(
                artifact_id=artifact_id,
                filename=filename,
                block=block,
                line_num=line_num,
                section_path=section_path or None,
                lead_in=lead_for_line or None,
            )
            lowered = normalize_text(cleaned)
            entity_keys = self._extract_entity_keys(cleaned)
            # Site atoms are attempted on EVERY line (an address can appear in a
            # signature or under any label) — the helper guards vendor
            # letterhead — so hygiene skips below never lose a real site.
            atoms.extend(
                self._site_atoms_from_line(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    cleaned=cleaned,
                    entity_keys=entity_keys,
                    source_ref=source_ref,
                    authority=authority,
                    confidence=confidence,
                )
            )

            # 0) Bare inline-attachment reference ("[cid:…]") — MIME chrome. The
            #    referenced part is handled by ``_cid_inline_atoms``; the marker
            #    line itself is not deal content.
            if _CID_REF_RE.sub("", cleaned).strip() == "":
                continue
            # 1) Signature block: once an authored message signs off, the rest
            #    is name/title/phone/URL chrome, not deal content.
            if in_signature:
                continue
            # Quoted messages sign off too, and this was gated to the authored
            # one only -- so every quoted signature was atomised in full. A
            # forward chain carries one signature per message, and deal 010215's
            # runs sixteen deep: 64 of its 75 chrome atoms sat at depth >= 1,
            # emitting `t`, `Q`, `404.771.3490` and `M: 404-918-0783` as
            # scope_items. `t` and `Q` are wrapped initials of Trent and
            # Quinton.
            #
            # `in_signature` is per-block state, so latching it inside a quoted
            # message ends at that message's boundary and never bleeds into the
            # next. Site extraction runs above this and is untouched, so an
            # address in a signature is still recovered.
            if _SIGNOFF_RE.match(cleaned):
                in_signature = True
                continue
            # 1b) A quoted message's own header block ("To: …", "Sent: …",
            #     "Cc: …"). The sender and date are already captured
            #     structurally on the locator, so these lines carry nothing a
            #     reader needs and arrive as address fragments like
            #     "mike.stephens <" once the line wraps.
            if block["quoted"]:
                header_match = _PSEUDO_HEADER_RE.match(cleaned)
                if header_match:
                    # A bare label ("From:") means its value is the next line.
                    after_bare_header_label = not (header_match.group(2) or "").strip()
                    continue
                if after_bare_header_label:
                    after_bare_header_label = False
                    continue
            # 2) Salutation opener — not an atom. Captured later as
            #    ``value.addressee`` / ``to_greeting`` on sibling body atoms
            #    (and the email header) so Atom Quality can show "To: Eddie"
            #    without a standalone reviewable ``Eddie,`` card.
            if _is_greeting_line(cleaned):
                continue
            # 2b) Framing lead-in above Include/Exclude — connective tissue,
            #     not a standalone atom. Hold until the list header arrives.
            if not is_bullet and _is_email_list_framing_lead_in(cleaned):
                pending_lead_in = [cleaned.rstrip()]
                continue
            # 3) List-section header ("Include:"/"Exclude:") — not an atom; the
            #    items beneath inherit its polarity + any pending lead-in.
            if _INCLUDE_LABEL_RE.match(cleaned):
                current_section = "include"
                if pending_lead_in:
                    active_lead_in = list(pending_lead_in)
                    pending_lead_in = []
                continue
            if _EXCLUDE_LABEL_RE.match(cleaned):
                current_section = "exclude"
                if pending_lead_in:
                    active_lead_in = list(pending_lead_in)
                    pending_lead_in = []
                continue
            # Non-bullet content ends the active list section for following lines.
            if not is_bullet:
                current_section = None
                active_lead_in = []
                # Keep pending lead-in only if the next line may still open a list.
                if not _is_email_list_framing_lead_in(cleaned):
                    pending_lead_in = []

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

            # List polarity wins: Include/Exclude bullets are one typed atom
            # each (verbatim evidence). Do not also emit requirement/task/
            # open_question twins for the same bullet.
            if section_for_line == "exclude":
                atom_types = [AtomType.exclusion]
            elif section_for_line == "include":
                atom_types = [AtomType.scope_item]
            elif _is_customer_quote_line(cleaned):
                # Quoted customer utterance — never a requirement paraphrase.
                atom_types = [AtomType.customer_instruction]
            else:
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

            # Deduplicate while preserving order (include may already set scope_item).
            seen_types: set[AtomType] = set()
            unique_types: list[AtomType] = []
            for atom_type in atom_types:
                if atom_type not in seen_types:
                    seen_types.add(atom_type)
                    unique_types.append(atom_type)
            atom_types = unique_types

            # Equipment-list framing is connective tissue for the CID image —
            # never mint a typed body atom for the intro line itself.
            if _is_equipment_list_intro_line(cleaned):
                atom_types = []

            for atom_type in atom_types:
                review_status = ReviewStatus.auto_accepted
                if atom_type == AtomType.open_question:
                    review_status = ReviewStatus.needs_review
                atom_value: dict[str, Any] = {
                    "text": cleaned,
                    "message_index": block["message_index"],
                    "quoted": block["quoted"],
                    "kind": "email_body_line",
                    "line": line_num,
                }
                if section_for_line:
                    atom_value["list_section"] = section_for_line
                    atom_value["section_header"] = _list_section_label(section_for_line)
                if lead_for_line:
                    atom_value["lead_in"] = list(lead_for_line)
                    atom_value["intro"] = lead_for_line[0]
                if delta_payload and atom_type == AtomType.customer_instruction:
                    atom_value["change_delta"] = delta_payload
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            block["message_index"],
                            line_num,
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

            # Courtesy / intro body prose — legitimate email communication
            # context (not fake scope). Emit as deal_metadata so Atom Quality
            # can show "Appreciate you hopping on…" without inventing scope.
            # Equipment-list intros stay connective-only (CID lead_in).
            if (
                not atom_types
                and any(ch.isalnum() for ch in cleaned)
                and _is_email_body_context_line(cleaned)
            ):
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            block["message_index"],
                            line_num,
                            "email_body_context",
                            cleaned,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.deal_metadata,
                        raw_text=cleaned,
                        normalized_text=normalize_text(cleaned),
                        value={
                            "text": cleaned,
                            "message_index": block["message_index"],
                            "quoted": block["quoted"],
                            "kind": "email_body_context",
                            "role": "intro",
                            "line": line_num,
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
                continue

            # Baseline body coverage: a body line that matched no typed
            # pattern is still real content — emit it as a scope_item so it
            # is never silently absent from the atom stream (the content
            # census inventories every body line). This mirrors the docx
            # fail-open prose gate and the MboxParser per-paragraph behavior:
            # keep + let the downstream learnable seam decide, never drop.
            # Courtesy/framing prose is handled above (context atom or CID
            # connective) — never fail-open as scope_item.
            if (
                not atom_types
                and any(ch.isalnum() for ch in cleaned)
                and not _is_courtesy_prose_line(cleaned)
            ):
                baseline_value: dict[str, Any] = {
                    "text": cleaned,
                    "message_index": block["message_index"],
                    "quoted": block["quoted"],
                    "kind": "email_body_line",
                    "line": line_num,
                }
                if section_for_line:
                    baseline_value["list_section"] = section_for_line
                    baseline_value["section_header"] = _list_section_label(section_for_line)
                if lead_for_line:
                    baseline_value["lead_in"] = list(lead_for_line)
                    baseline_value["intro"] = lead_for_line[0]
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            block["message_index"],
                            line_num,
                            "scope_item",
                            cleaned,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.scope_item,
                        raw_text=cleaned,
                        normalized_text=normalize_text(cleaned),
                        value=baseline_value,
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
