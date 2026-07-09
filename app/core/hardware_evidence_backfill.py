"""Mint bom_line atoms from scope evidence when hardware counts appear in prose.

Deal Kit Hardware should reflect parser ``bom_line`` atoms — not frontend regex.
Cold start extracts grounded counts and logs training rows for the
``hardware_evidence_line`` relation so a promoted head can own extraction later.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.core.training_log import TEACHER_STORE, TrainingRow, log_rows
from app.parsers.email_parser import _glued_trailing_order_qty

HARDWARE_EVIDENCE_RELATION = "hardware_evidence_line"

_SOURCE_TYPES = frozenset({"scope_item", "requirement", "customer_instruction", "open_question"})
_EMAIL_CID_KIND = "email_cid_equipment_line"
_EMAIL_CID_SOURCE = "email_cid_equipment_line"

_WORD_QTY: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# Manifest JSON flatten emits ``artifacts[N].blob_url`` scope rows — not prose evidence.
_JSON_MANIFEST_KEY_RE = re.compile(
    r"^artifacts\[\d+\]\.(?:attachment_id|blob_url|content_sha256|filename|content_type|size_bytes|mime_type)\b",
    re.I,
)

_QTY = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
# Order-list lines: "Access Point E7 … × 6" or right-aligned "Access Point E7   6".
_NAME_THEN_QTY = r"(?:[×x]\s*|(?:\s{2,}|\t))\s*(\d+)\s*$"


_GLUED_QTY = r"(?:\s+(\d{1,2})\s*$)"

_GLUED_STEM_SKU: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"access\s+point\s+e7(?:\s+enterprise)?\b", re.I), "UBNT-E7-AP", "Ubiquiti E7 Access Point"),
    (re.compile(r"(?:udm(?:[-\s]*beast)?|dream\s+machine(?:\s*beast)?)\b", re.I), "UBNT-UDM-BEAST", "Ubiquiti Dream Machine Beast"),
    (re.compile(r"enterprise\s+nvr\b", re.I), "UBNT-NVR", "Ubiquiti NVR"),
    (re.compile(r"access\s*cards?\b", re.I), "UBNT-ACCESS-CARD", "Ubiquiti Access Card"),
    (re.compile(r"protect(?:\s+all[- ]in[- ]one)?\s*sensors?\b", re.I), "UBNT-PROTECT-SENSOR", "Ubiquiti Protect All-In-One Sensor"),
    (re.compile(r"switch\s+pro(?:\s+max)?(?:\s+\d+)?(?:\s+poe)?\b", re.I), "UBNT-SW-PRO", "Ubiquiti Pro Switch"),
    (re.compile(r"(?:camera\s+)?g6(?:\s+pro)?\s*turrets?\b", re.I), "UBNT-G6-TURRET", "Ubiquiti G6 Turret"),
    (re.compile(r"(?:camera\s+)?g6(?:\s+pro)?\s*360\b", re.I), "UBNT-G6-PRO-360", "Ubiquiti G6 Pro 360"),
    (re.compile(r"(?:g6(?:/g5)?\s+ptz\s+pendant\s+mount|g6\s+ptz\s+mounts?)\b", re.I), "UBNT-G6-PTZ-MOUNT", "Ubiquiti G6 PTZ Mount"),
    (re.compile(r"access\s+g3\s*readers?(?:\s*pro)?\b", re.I), "UBNT-ACCESS-G3-READER", "Ubiquiti Access G3 Reader"),
    (re.compile(r"reader\s+g6\s+entry\b", re.I), "UBNT-READER-G6-ENTRY", "Ubiquiti Reader G6 Entry"),
    (re.compile(r"access\s+reader(?:\s*pro)?(?:\s+juncti\w*)?\b", re.I), "UBNT-ACCESS-READER-PRO", "Ubiquiti Access Reader Pro"),
    (re.compile(r"access\s+rescue\s+key\s*switch\b", re.I), "UBNT-ACCESS-RESCUE-KEYSWITCH", "Ubiquiti Access Rescue KeySwitch"),
    (re.compile(r"power\s+distribution\s+pro\b", re.I), "UBNT-POWER-DIST-PRO", "Ubiquiti Power Distribution Pro"),
    (re.compile(r"enterprise\s+access\s+hubs?\b", re.I), "UBNT-ACCESS-HUB", "Ubiquiti Enterprise Access Hub"),
    (re.compile(r"access\s+intercom\s+viewer\b", re.I), "UBNT-ACCESS-INTERCOM-VIEWER", "Ubiquiti Access Intercom Viewer"),
    (re.compile(r"camera\s+ai\s+multi\s+sensor(?:\s+\d+)?\b", re.I), "UBNT-AI-MULTI-SENSOR", "Ubiquiti Camera AI Multi Sensor"),
    (re.compile(r"(?<!\d)\d{1,2}g\s+direct\s+attach\s+cables?\b", re.I), "UBNT-25G-DAC", "Ubiquiti 25G Direct Attach Cable"),
    (re.compile(r"g6\s+entry\s+wedge\s+mounts?\b", re.I), "UBNT-G6-ENTRY-WEDGE", "Ubiquiti G6 Entry Wedge Mount"),
]


_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "UBNT-E7-AP",
        "Ubiquiti E7 Access Point",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*|×\s*)?(?:e7|u7)\s*aps?\b"
            r"|(?<![\w/])(\d+)\s*(?:x\s*|×\s*)?e7\s*aps?\b"
            rf"|access\s+point\s+e7(?:\s+enterprise)?[^\n]{{0,60}}?{_NAME_THEN_QTY}"
            rf"|(?:access\s+point\s+)?e7(?:\s+enterprise)?[^\n]{{0,40}}?\s*[×x]\s*(\d+)\b",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-UDM-BEAST",
        "Ubiquiti Dream Machine Beast",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:udm(?:[-\s]*beast)?|dream\s+machine(?:\s*beast)?)\b"
            rf"|(?:udm(?:[-\s]*beast)?|dream\s+machine(?:\s*beast)?)[^\n]{{0,40}}?{_NAME_THEN_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-SW-PRO",
        "Ubiquiti Pro Switch",
        re.compile(
            rf"\b{_QTY}\s+(?:\d+\s*)?port\s*switches?\b"
            rf"|switch\s+pro(?:\s+max)?(?:\s+\d+)?(?:\s+poe)?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            r"|switch\s+pro(?:\s+\w+){0,6}\s*[×x]\s*(\d+)\b",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-UNVR",
        "Ubiquiti UNVR",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:uni\s*)?unvr\b"
            rf"|(?:uni\s*)?unvr[^\n]{{0,40}}?{_NAME_THEN_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-NVR",
        "Ubiquiti NVR",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:\d+\s*)?nvr\b"
            rf"|enterprise\s+nvr[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            r"|enterprise\s+nvr[^\n]{0,40}?\s*[×x]\s*(\d+)\b",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-G6-TURRET",
        "Ubiquiti G6 Turret",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:camera\s+)?g6(?:\s+pro)?\s*turrets?\b"
            rf"|(?:camera\s+)?g6(?:\s+pro)?\s*turrets?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|(?:camera\s+)?g6(?:\s+pro)?\s*turrets?\s+(\d{{1,2}})\s*$"
            r"|(?:camera\s+)?g6(?:\s+pro)?\s*turrets?\b",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-G6-PRO-360",
        "Ubiquiti G6 Pro 360",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:camera\s+)?g6(?:\s+pro)?\s*360\b"
            rf"|(?:camera\s+)?g6(?:\s+pro)?\s*360[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|(?:camera\s+)?g6(?:\s+pro)?\s*360{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-G6-PRO-DB",
        "Ubiquiti G6 Pro Doorbell",
        re.compile(
            r"g6\s+pro(?:\s+doorbell)?\s*[×x]\s*(\d+)\b"
            rf"|\b{_QTY}[ \t]*(?:x[ \t]*)?g6\s+pro(?:\s+doorbell)?\b"
            rf"|g6\s+pro(?:\s+doorbell)?[^\n]{{0,40}}?{_NAME_THEN_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-G6-INSTANT",
        "Ubiquiti G6 Instant",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?g6\s*instant\b"
            rf"|g6\s*instant[^\n]{{0,40}}?{_NAME_THEN_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-ACCESS-G3-READER",
        "Ubiquiti Access G3 Reader",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:access\s+)?g3\s*readers?(?:\s*pro)?\b"
            rf"|(?:access\s+)?g3\s*readers?(?:\s*pro)?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|(?:access\s+)?g3\s*readers?(?:\s*pro)?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-READER-G6-ENTRY",
        "Ubiquiti Reader G6 Entry",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?reader\s+g6\s+entry\b"
            rf"|reader\s+g6\s+entry[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|reader\s+g6\s+entry{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-ACCESS-READER-PRO",
        "Ubiquiti Access Reader Pro",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?access\s+reader(?:\s*pro)?(?:\s+juncti\w*)?\b"
            rf"|access\s+reader(?:\s*pro)?(?:\s+juncti\w*)?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|access\s+reader(?:\s*pro)?(?:\s+juncti\w*)?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-BADGE-READER",
        "Ubiquiti Card / Badge Reader",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:badge\s*readers?|card\s*readers?)\b"
            rf"|(?:badge|card)\s*readers?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|(?:badge|card)\s*readers?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-ACCESS-CARD",
        "Ubiquiti Access Card",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?access\s*cards?\b"
            rf"|access\s*cards?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|access\s*cards?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-ACCESS-HUB",
        "Ubiquiti Enterprise Access Hub",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:enterprise\s+)?access\s+hubs?\b"
            rf"|enterprise\s+access\s+hubs?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|enterprise\s+access\s+hubs?{_GLUED_QTY}"
            rf"|(?:enterprise\s+)?access\s+hubs?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-PROTECT-SENSOR",
        "Ubiquiti Protect All-In-One Sensor",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?protect(?:\s+all[- ]in[- ]one)?\s+sensors?\b"
            rf"|protect(?:\s+all[- ]in[- ]one)?\s+sensors?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            r"|protect(?:\s+all[- ]in[- ]one)?\s+sensors?[^\n]{0,40}?\s*[×x]\s*(\d+)\b"
            rf"|protect(?:\s+all[- ]in[- ]one)?\s+sensors?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-G6-PTZ-MOUNT",
        "Ubiquiti G6 PTZ Mount",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?(?:g6(?:/g5)?\s+ptz\s+pendant\s+)?g6\s+ptz\s+mounts?\b"
            rf"|(?:g6(?:/g5)?\s+ptz\s+pendant\s+mount|g6\s+ptz\s+mounts?)[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|(?:g6(?:/g5)?\s+ptz\s+pendant\s+mount|g6\s+ptz\s+mounts?)\s+(\d{{1,2}})\s*$"
            r"|(?:g6(?:/g5)?\s+ptz\s+pendant\s+mount|g6\s+ptz\s+mounts?)\b",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-ACCESS-RESCUE-KEYSWITCH",
        "Ubiquiti Access Rescue KeySwitch",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?access\s+rescue\s+key\s*switch\b"
            rf"|access\s+rescue\s+key\s*switch[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|access\s+rescue\s+key\s*switch{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-POWER-DIST-PRO",
        "Ubiquiti Power Distribution Pro",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?power\s+distribution\s+pro\b"
            rf"|power\s+distribution\s+pro[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|power\s+distribution\s+pro{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-ACCESS-INTERCOM-VIEWER",
        "Ubiquiti Access Intercom Viewer",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?access\s+intercom\s+viewer\b"
            rf"|access\s+intercom\s+viewer[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|access\s+intercom\s+viewer{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-AI-MULTI-SENSOR",
        "Ubiquiti Camera AI Multi Sensor",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?camera\s+ai\s+multi\s+sensor(?:\s+\d+)?\b"
            rf"|camera\s+ai\s+multi\s+sensor(?:\s+\d+)?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|camera\s+ai\s+multi\s+sensor(?:\s+\d+)?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-25G-DAC",
        "Ubiquiti 25G Direct Attach Cable",
        re.compile(
            # Do not let leading ``_QTY`` steal the ``2`` from ``25G``.
            rf"\b{_QTY}\s*(?:x\s*|×\s+)\d{{1,2}}g\s+direct\s+attach\s+cables?\b"
            rf"|\d{{1,2}}g\s+direct\s+attach\s+cables?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|\d{{1,2}}g\s+direct\s+attach\s+cables?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-G6-ENTRY-WEDGE",
        "Ubiquiti G6 Entry Wedge Mount",
        re.compile(
            rf"\b{_QTY}\s*(?:x\s*)?g6\s+entry\s+wedge\s+mounts?\b"
            rf"|g6\s+entry\s+wedge\s+mounts?[^\n]{{0,40}}?{_NAME_THEN_QTY}"
            rf"|g6\s+entry\s+wedge\s+mounts?{_GLUED_QTY}",
            re.I | re.M,
        ),
    ),
    (
        "UBNT-AP-GENERIC",
        "Ubiquiti Access Point",
        re.compile(r"(?<![%/\w-])(\d+)\s+(?:x\s*)?(?:access points?|aps?)\b", re.I),
    ),
]


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _text(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    if str(raw).strip():
        return str(raw).strip()
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("text") or val.get("description") or "").strip()
    return ""


def _locator_dict(atom: Any) -> dict[str, Any]:
    locator = getattr(atom, "locator", None)
    if isinstance(locator, dict):
        return locator
    for ref in getattr(atom, "source_refs", None) or []:
        loc = getattr(ref, "locator", None)
        if isinstance(loc, dict):
            return loc
    return {}


def _is_prose_evidence(atom: Any) -> bool:
    """Skip manifest JSON flatten rows and URL blobs — they poison hardware regex."""
    text = _text(atom)
    if not text:
        return False
    if "blob.core.windows.net" in text or text.startswith("artifacts["):
        label = text.split(":", 1)[0].strip()
        if _JSON_MANIFEST_KEY_RE.match(label):
            return False
    if "%20" in text.lower() and re.search(r"%20\s*\d*\s*aps?\b", text, re.I):
        return False
    locator = _locator_dict(atom)
    if locator.get("kind") == "json_value":
        key_path = str(locator.get("key_path") or "")
        if key_path.startswith("artifacts[") or key_path in {"org_id", "deal_id"}:
            return False
    return True


def _parse_qty(token: str) -> int | None:
    raw = (token or "").strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        n = int(raw)
        return n if n > 0 else None
    return _WORD_QTY.get(raw)


def _existing_bom_skus(atoms: list[Any]) -> set[str]:
    out: set[str] = set()
    for atom in atoms:
        if _atom_type_str(atom) != "bom_line":
            continue
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        for key in ("sku", "item_id", "part_number"):
            sku = str(val.get(key) or "").strip().lower()
            if sku:
                out.add(sku)
    return out


def _existing_bom_row_keys(atoms: list[Any]) -> set[str]:
    """Per-row keys so duplicate SKUs (two wedge mounts) stay distinct lines."""
    out: set[str] = set()
    for atom in atoms:
        if _atom_type_str(atom) != "bom_line":
            continue
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        sku = str(val.get("sku") or val.get("item_id") or "").strip().lower()
        item = str(val.get("item") or val.get("description") or "").strip().lower()
        qty = str(val.get("quantity") or val.get("qty") or "").strip()
        row = str(val.get("row_index") if val.get("row_index") is not None else "")
        if sku:
            out.add(f"{sku}|{item}|{qty}|{row}")
    return out


def _value_kind(atom: Any) -> str:
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("kind") or "").strip()
    return ""


def _is_email_cid_equipment_atom(atom: Any) -> bool:
    return _value_kind(atom) == _EMAIL_CID_KIND


def _slug_sku_from_item(item: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", (item or "").strip().upper()).strip("-")
    stem = re.sub(r"-{2,}", "-", stem)
    if not stem:
        return "UBNT-UNKNOWN"
    if not stem.startswith("UBNT-"):
        stem = f"UBNT-{stem}"
    return stem[:64]


def _sku_from_equipment_text(text: str) -> tuple[str, str] | None:
    line = (text or "").strip()
    if not line:
        return None
    candidates = [line]
    glued = _glued_trailing_order_qty(line)
    if glued is not None:
        stem = re.sub(rf"\s{re.escape(str(glued))}\s*$", "", line).strip()
        if stem and stem not in candidates:
            candidates.append(stem)
    for candidate in candidates:
        # Prefer specific glued stems before broad prose patterns so
        # Reader G6 Entry / Access G3 Reader do not collapse together.
        for pattern, sku, description in _GLUED_STEM_SKU:
            if pattern.search(candidate):
                return sku, description
        for sku, description, pattern in _PATTERNS:
            if pattern.search(candidate):
                return sku, description
    return None


def _mint_bom_line(
    *,
    project_id: str,
    sku: str,
    description: str,
    qty: int,
    source_atom: Any,
    notes: str,
    source: str = "hardware_evidence_backfill",
    item: str | None = None,
    row_index: int | None = None,
) -> Any:
    from app.core.schemas import ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

    artifact_id = getattr(source_atom, "artifact_id", "") or "hardware_evidence_backfill"
    item_name = (item or description or "").strip()
    atom_id = stable_id(
        "bom_line",
        project_id,
        sku,
        str(qty),
        item_name[:120],
        str(row_index if row_index is not None else ""),
        _text(source_atom)[:80],
    )
    refs = list(getattr(source_atom, "source_refs", None) or [])
    if not refs:
        refs = [
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.txt,
                filename=artifact_id,
                locator={"extraction": "hardware_evidence_backfill"},
                extraction_method="hardware_evidence_backfill",
                parser_version="hardware_evidence_backfill_v1",
            )
        ]
    value: dict[str, Any] = {
        "sku": sku,
        "item_id": sku,
        "description": description,
        "item": item_name or description,
        "quantity": qty,
        "qty": qty,
        "vendor": "Ubiquiti",
        "source": source,
        "notes": notes,
    }
    if row_index is not None:
        value["row_index"] = row_index
    src_val = getattr(source_atom, "value", None) or {}
    if isinstance(src_val, dict):
        if src_val.get("lead_in") and "lead_in" not in value:
            value["lead_in"] = list(src_val.get("lead_in") or [])
            value["intro"] = src_val.get("intro") or (value["lead_in"][0] if value["lead_in"] else None)
        if src_val.get("list_section"):
            value["list_section"] = src_val.get("list_section")
        if src_val.get("section_header"):
            value["section_header"] = src_val.get("section_header")
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.bom_line,
        raw_text=_text(source_atom)[:2000],
        normalized_text=description.lower(),
        value=value,
        entity_keys=[f"quantity:{qty}"],
        source_refs=refs[:1],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.78 if source == _EMAIL_CID_SOURCE else 0.72,
        confidence_raw=0.78 if source == _EMAIL_CID_SOURCE else 0.72,
        calibrated_confidence=0.78 if source == _EMAIL_CID_SOURCE else 0.72,
        review_status=ReviewStatus.needs_review,
        review_flags=[source, "hardware_evidence_training_row"],
        parser_version="hardware_evidence_backfill_v1",
    )


def _mint_bom_from_email_cid_equipment_lines(
    atoms: list[Any],
    *,
    project_id: str,
    existing: set[str],
) -> tuple[list[Any], int]:
    """Mint one BOM row per CID equipment line; drop the source scope twins.

    Universal rules:
    - Every recoverable OCR order row becomes a ``bom_line`` with quantity.
    - Distinct products keep distinct SKUs (G3 Reader ≠ Reader G6 Entry).
    - Duplicate product rows (two wedge mounts) stay separate BOM lines.
    - Unmapped but qty-bearing product names mint a slug SKU rather than
      being dropped — completeness over vocabulary coverage.
    - BOM inherits CID locator / lead_in / section_path connective tissue.
    """
    from app.parsers.email_parser import (
        _is_ocr_junk_equipment_line,
        _order_row_name,
        _repair_ocr_equipment_line,
    )

    minted = 0
    drop_ids: set[str] = set()
    seen_rows = _existing_bom_row_keys(atoms)
    for atom in list(atoms):
        if not _is_email_cid_equipment_atom(atom):
            continue
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        lines = [
            _repair_ocr_equipment_line(line)
            for line in str(val.get("text") or _text(atom) or "").splitlines()
            if line.strip()
        ]
        if not lines:
            item_fallback = _repair_ocr_equipment_line(str(val.get("item") or ""))
            lines = [item_fallback] if item_fallback else []
        atom_minted = False
        all_junk = True
        for line in lines:
            if not line:
                continue
            if _is_ocr_junk_equipment_line(line):
                continue
            all_junk = False
            # Universal: trailing / glued order qty wins over mid-name digits
            # (``25G``, ``Max 48``, ``Multi Sensor 4``).
            qty_n = 0
            try:
                qty_n = int(val.get("quantity") or val.get("qty") or 0)
            except (TypeError, ValueError):
                qty_n = 0
            glued_qty = _glued_trailing_order_qty(line)
            if glued_qty:
                qty_n = glued_qty
            if qty_n <= 0:
                for _sku, _description, pattern in _PATTERNS:
                    match = pattern.search(line)
                    if not match:
                        continue
                    qty_n = _parse_qty_from_match(match) or 0
                    if qty_n > 0:
                        break
            qty_n = _sanity_cid_line_qty(line, qty_n)
            if qty_n <= 0:
                continue
            item_name = _order_row_name(line, qty_n) or str(val.get("item") or line)
            mapped = _sku_from_equipment_text(line) or _sku_from_equipment_text(item_name)
            if mapped:
                sku, description = mapped
            else:
                sku = _slug_sku_from_item(item_name)
                description = item_name
            try:
                row_index = int(val.get("row_index")) if val.get("row_index") is not None else None
            except (TypeError, ValueError):
                row_index = None
            row_key = f"{sku.lower()}|{item_name.strip().lower()}|{qty_n}|{row_index if row_index is not None else ''}"
            if row_key in seen_rows:
                atom_minted = True
                continue
            atoms.append(
                _mint_bom_line(
                    project_id=project_id,
                    sku=sku,
                    description=description,
                    qty=qty_n,
                    source_atom=atom,
                    notes=_EMAIL_CID_SOURCE,
                    source=_EMAIL_CID_SOURCE,
                    item=item_name,
                    row_index=row_index,
                )
            )
            seen_rows.add(row_key)
            existing.add(sku.lower())
            minted += 1
            atom_minted = True
        if atom_minted or all_junk:
            aid = str(getattr(atom, "id", "") or "")
            if aid:
                drop_ids.add(aid)
    if drop_ids:
        atoms[:] = [a for a in atoms if str(getattr(a, "id", "") or "") not in drop_ids]
    return atoms, minted


def _parse_qty_from_match(match: re.Match[str]) -> int | None:
    last = match.lastindex or 0
    for idx in range(last, 0, -1):
        qty = _parse_qty(str(match.group(idx) or ""))
        if qty:
            return qty
    return None


def _sanity_cid_line_qty(line: str, qty: int) -> int:
    """Prefer trailing order qty when OCR embeds model numbers in the product name."""
    cleaned = (line or "").strip()
    glued_qty = _glued_trailing_order_qty(cleaned)
    if qty <= 0:
        return glued_qty or 0
    if glued_qty is None:
        return qty
    if glued_qty == qty:
        return qty
    # Mid-name model digits that regex mistook for order qty.
    if re.search(
        rf"(?:max|pro|series)\s+{qty}\b|\b{qty}\s+poe\b|\b{qty}g\b|"
        rf"\bmulti\s+sensor\s+{qty}\b",
        cleaned,
        re.I,
    ):
        return glued_qty
    trail = re.search(r"(?:\s{2,}|\t|[×x])\s*(\d{1,2})\s*$", cleaned, re.I)
    if trail:
        return int(trail.group(1))
    if glued_qty <= 10:
        return glued_qty
    return qty



def backfill_hardware_bom_lines(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], int]:
    """Add bom_line atoms from grounded equipment counts in scope prose."""
    has_email_cid_equipment = any(_is_email_cid_equipment_atom(a) for a in atoms)
    existing = _existing_bom_skus(atoms)
    atoms, email_minted = _mint_bom_from_email_cid_equipment_lines(
        atoms, project_id=project_id, existing=existing
    )
    if has_email_cid_equipment:
        return atoms, email_minted
    prose_atoms = [
        a for a in atoms if _atom_type_str(a) in _SOURCE_TYPES and _is_prose_evidence(a)
    ]
    corpus_parts = [_text(a) for a in prose_atoms]
    corpus = "\n".join(x for x in corpus_parts if x)
    if not corpus.strip():
        return atoms, email_minted

    minted = email_minted
    rows: list[TrainingRow] = []
    for sku, description, pattern in _PATTERNS:
        if sku.lower() in existing:
            continue
        match = None
        for line in corpus.splitlines():
            line = line.strip()
            if not line:
                continue
            m = pattern.search(line)
            if m:
                match = m
                break
        if not match:
            continue
        qty = _parse_qty_from_match(match)
        if not qty:
            continue
        source_atom = next(
            (a for a in prose_atoms if any(pattern.search(line) for line in _text(a).splitlines())),
            prose_atoms[0] if prose_atoms else atoms[0],
        )
        atoms.append(
            _mint_bom_line(
                project_id=project_id,
                sku=sku,
                description=description,
                qty=qty,
                source_atom=source_atom,
                notes="hardware_evidence_backfill",
            )
        )
        existing.add(sku.lower())
        minted += 1
        rows.append(
            TrainingRow(
                relation=HARDWARE_EVIDENCE_RELATION,
                label=f"{sku}|{qty}",
                raw_text=corpus[:4000],
                label_kind="judgment",
                teacher=TEACHER_STORE,
                confidence=0.72,
                deal_id=project_id,
                project_id=project_id,
                provenance={"sku": sku, "qty": qty, "match": match.group(0)},
            )
        )

    if rows:
        log_rows(rows)
    return atoms, minted


__all__ = [
    "HARDWARE_EVIDENCE_RELATION",
    "backfill_hardware_bom_lines",
]
