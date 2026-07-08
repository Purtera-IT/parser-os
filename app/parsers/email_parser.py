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


def _focus_cid_equipment_text(text: str) -> str:
    """Prefer order-table text; ignore spoken transcript counts from wrong embeds."""
    raw = (text or "").strip()
    if not raw:
        return raw
    order = _slice_order_details_text(raw)
    if order != raw:
        return order
    if _TRANSCRIPT_DOC_RE.search(raw):
        return ""
    return raw


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


def _parse_qty_token(token: str) -> int | None:
    raw = (token or "").strip().lower()
    if raw.isdigit():
        n = int(raw)
        return n if n > 0 else None
    return _WORD_QTY.get(raw)


def _hardware_atoms_from_equipment_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    text: str,
    content_id: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    atoms: list[EvidenceAtom] = []
    text = _focus_cid_equipment_text(text)
    if not text.strip():
        return atoms
    src = SourceRef(
        id=stable_id("src", artifact_id, "cid", content_id),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.email,
        filename=filename,
        locator={"kind": "email_cid_inline", "content_id": content_id},
        extraction_method="email_cid_inline",
        parser_version=parser_version,
    )
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned or _TRANSCRIPT_SPEECH_LINE_RE.search(cleaned):
            continue
        for match in _EQUIPMENT_LINE_RE.finditer(cleaned):
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
            atoms.append(
                EvidenceAtom(
                    id=stable_id("atm", project_id, artifact_id, "cid_hw", content_id, cleaned, str(qty)),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.scope_item,
                    raw_text=cleaned,
                    normalized_text=normalize_text(cleaned),
                    value={
                        "text": cleaned,
                        "kind": "email_cid_equipment_line",
                        "quantity": qty,
                        "item": item,
                        "content_id": content_id,
                    },
                    entity_keys=[],
                    source_refs=[src],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.78,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["email_cid_equipment_line"],
                    parser_version=parser_version,
                )
            )
    if not atoms and any(ch.isalnum() for ch in text):
        atoms.append(
            EvidenceAtom(
                id=stable_id("atm", project_id, artifact_id, "cid_body", content_id, text[:120]),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.scope_item,
                raw_text=text[:4000],
                normalized_text=normalize_text(text),
                value={
                    "text": text[:4000],
                    "kind": "email_cid_inline_body",
                    "content_id": content_id,
                },
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


def _list_section_path(section: str | None) -> list[str]:
    label = _list_section_label(section)
    return [label] if label else []


def _is_greeting_line(cleaned: str) -> bool:
    """True when a line is a salutation opener ("Eddie,", "Hi John,", "Dear
    all,"). Structural: a short line (≤4 words) ending in a comma that is
    either led by a greeting word or is purely name-shaped tokens. Carries no
    scope, role, or instruction, so it should never become an atom."""
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
        # Header atom — the From/To/Cc/Subject/Date line is deal/routing
        # metadata (and the content census inventories it). Emit it as
        # ``deal_metadata`` so the header is never silently absent from the atom
        # stream but is NEVER mistaken for contractual scope. .eml only;
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
        atoms.extend(
            self._cid_inline_atoms(
                project_id=project_id, artifact_id=artifact_id, path=path, body_text=text
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
            # The header block is routing/threading metadata, not scope. Typing
            # it as deal_metadata (not scope_item) keeps it out of the
            # contractual-scope surface a quote/scope head reads, while still
            # surfacing it as a first-class atom the census can reconcile.
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
    ) -> list[EvidenceAtom]:
        if path.suffix.lower() != ".eml":
            return []
        msg = _parse_email_message(path)
        if msg is None:
            return []
        inline_parts = _iter_cid_inline_parts(msg)
        referenced = {_normalize_cid(m.group(1)) for m in _CID_REF_RE.finditer(body_text or "")}
        if not inline_parts and not referenced:
            return []
        atoms: list[EvidenceAtom] = []
        targets = referenced or set(inline_parts.keys())
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
            text = str(part.get("text") or "")
            payload = part.get("payload")
            if payload and (part.get("is_image") or part.get("is_pdf")):
                ocr_text = _ocr_text_from_cid_inline(
                    bytes(payload),
                    content_type=str(part.get("content_type") or ""),
                )
                if ocr_text:
                    text = ocr_text
            atoms.extend(
                _hardware_atoms_from_equipment_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    text=text,
                    content_id=cid,
                    parser_version=self.parser_version,
                )
            )
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
            "sender": block["sender"],
            "sent_at": block["sent_at"],
            "quoted": block["quoted"],
        }
        if section_path:
            locator["section_path"] = list(section_path)
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
        in_signature = False
        current_section: str | None = None  # "include" | "exclude" | None

        for line_idx, line in enumerate(block["lines"]):
            line_num = int(block["line_start"]) + line_idx
            raw_cleaned = line.lstrip("> ").strip()
            if not raw_cleaned:
                continue
            is_bullet = bool(_BULLET_PREFIX_RE.match(raw_cleaned))
            cleaned = _BULLET_PREFIX_RE.sub("", raw_cleaned).strip()
            if not cleaned:
                continue
            # Bullets inherit the active Include/Exclude header; compute before
            # hygiene continues so per-line locators carry section_path.
            section_for_line = current_section if is_bullet else None
            section_path = _list_section_path(section_for_line)
            source_ref = self._build_source_ref(
                artifact_id=artifact_id,
                filename=filename,
                block=block,
                line_num=line_num,
                section_path=section_path or None,
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
            if not block["quoted"] and _SIGNOFF_RE.match(cleaned):
                in_signature = True
                continue
            # 2) Salutation opener — no scope/role/instruction.
            if _is_greeting_line(cleaned):
                continue
            # 3) List-section header ("Include:"/"Exclude:") — not an atom; the
            #    items beneath inherit its polarity.
            if _INCLUDE_LABEL_RE.match(cleaned):
                current_section = "include"
                continue
            if _EXCLUDE_LABEL_RE.match(cleaned):
                current_section = "exclude"
                continue
            # Non-bullet content ends the active list section for following lines.
            if not is_bullet:
                current_section = None

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

            # A bullet under an "Exclude:" header IS an exclusion even though
            # the item text itself ("Network buildout") carries no exclusion
            # keyword — the header supplied the polarity. Likewise an item
            # under "Include:" is scope (handled by the baseline gate below).
            if section_for_line == "exclude":
                atom_types.append(AtomType.exclusion)
            if any(re.search(pattern, lowered) for pattern in exclusion_patterns):
                if AtomType.exclusion not in atom_types:
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
                    "kind": "email_body_line",
                    "line": line_num,
                }
                if section_for_line:
                    atom_value["list_section"] = section_for_line
                    atom_value["section_header"] = _list_section_label(section_for_line)
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
                            line_num,
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
                            "line": line_num,
                            **(
                                {
                                    "list_section": section_for_line,
                                    "section_header": _list_section_label(section_for_line),
                                }
                                if section_for_line
                                else {}
                            ),
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
