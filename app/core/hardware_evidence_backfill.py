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

_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "UBNT-E7-AP",
        "Ubiquiti E7 Access Point",
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*|×\s*)?(?:e7|u7)\s*aps?\b"
            r"|(?<![\w/])(\d+)\s*(?:x\s*|×\s*)?e7\s*aps?\b"
            r"|(?:access\s+point\s+)?e7(?:\s+enterprise)?[^\n]{0,40}?\s×\s*(\d+)\b",
            re.I,
        ),
    ),
    (
        "UBNT-UDM-BEAST",
        "Ubiquiti Dream Machine Beast",
        re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*)?udm(?:[-\s]*beast)?\b", re.I),
    ),
    (
        "UBNT-SW-PRO",
        "Ubiquiti Pro Switch",
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\d+\s*)?port\s*switches?\b"
            r"|switch\s+pro(?:\s+\w+){0,6}\s*×\s*(\d+)\b",
            re.I,
        ),
    ),
    (
        "UBNT-UNVR",
        "Ubiquiti UNVR",
        re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*)?(?:uni\s*)?unvr\b", re.I),
    ),
    ("UBNT-NVR", "Ubiquiti NVR", re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*)?(?:\d+\s*)?nvr\b"
        r"|enterprise\s+nvr[^\n]{0,40}?\s×\s*(\d+)\b",
        re.I,
    )),
    (
        "UBNT-G6-PRO-DB",
        "Ubiquiti G6 Pro Doorbell",
        re.compile(
            r"g6\s+pro(?:\s+turret)?\s*×\s*(\d+)\b"
            r"|\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[ \t]*(?:x[ \t]*)?g6\s+pro(?:\s+(?:turret|doorbell))?\b",
            re.I,
        ),
    ),
    (
        "UBNT-BADGE-READER",
        "Ubiquiti Card / Badge Reader",
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s*)?"
            r"(?:badge\s*readers?|card\s*readers?|access\s*readers?)\b"
            r"|(?:badge|card)\s*reader[^\n]{0,40}?\s×\s*(\d+)\b",
            re.I,
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


def _value_kind(atom: Any) -> str:
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("kind") or "").strip()
    return ""


def _is_email_cid_equipment_atom(atom: Any) -> bool:
    return _value_kind(atom) == _EMAIL_CID_KIND


def _sku_from_equipment_text(text: str) -> tuple[str, str] | None:
    line = (text or "").strip()
    if not line:
        return None
    for sku, description, pattern in _PATTERNS:
        if pattern.search(line):
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
) -> Any:
    from app.core.schemas import ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

    artifact_id = getattr(source_atom, "artifact_id", "") or "hardware_evidence_backfill"
    atom_id = stable_id("bom_line", project_id, sku, str(qty), _text(source_atom)[:120])
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
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.bom_line,
        raw_text=_text(source_atom)[:2000],
        normalized_text=description.lower(),
        value={
            "sku": sku,
            "item_id": sku,
            "description": description,
            "quantity": qty,
            "qty": qty,
            "vendor": "Ubiquiti",
            "source": source,
            "notes": notes,
        },
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
    minted = 0
    for atom in atoms:
        if not _is_email_cid_equipment_atom(atom):
            continue
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        lines = [
            line.strip()
            for line in str(val.get("text") or _text(atom) or "").splitlines()
            if line.strip()
        ]
        if not lines:
            lines = [str(val.get("item") or "").strip()]
        for line in lines:
            if not line:
                continue
            qty_n = 0
            for _sku, _description, pattern in _PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                qty_n = _parse_qty_from_match(match) or 0
                if qty_n > 0:
                    break
            if qty_n <= 0:
                try:
                    qty_n = int(val.get("quantity") or 0)
                except (TypeError, ValueError):
                    qty_n = 0
            if qty_n <= 0:
                continue
            mapped = _sku_from_equipment_text(line) or _sku_from_equipment_text(str(val.get("item") or ""))
            if not mapped:
                continue
            sku, description = mapped
            if sku.lower() in existing:
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
                )
            )
            existing.add(sku.lower())
            minted += 1
    return atoms, minted


def _parse_qty_from_match(match: re.Match[str]) -> int | None:
    last = match.lastindex or 0
    for idx in range(last, 0, -1):
        qty = _parse_qty(str(match.group(idx) or ""))
        if qty:
            return qty
    return None


def backfill_hardware_bom_lines(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], int]:
    """Add bom_line atoms from grounded equipment counts in scope prose."""
    existing = _existing_bom_skus(atoms)
    atoms, email_minted = _mint_bom_from_email_cid_equipment_lines(
        atoms, project_id=project_id, existing=existing
    )
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
