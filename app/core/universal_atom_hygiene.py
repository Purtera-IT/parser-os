"""Universal publish hygiene for atoms (P1–P12).

These rules are deal-agnostic. They run at emit / substance-gate time so
OrbitBrief never sees OCR stubs, email chrome, JSON-wrapped vision, shred
tokens, or soft aesthetic “risks” as publishable evidence.

Rule map
--------
P1  No publishable OCR / vision stubs
P2  Strip email / marketing chrome
P3  Unwrap JSON-wrapped vision text
P5  No speculative aesthetic risk atoms
P7  Atom-type discipline for chrome / headers
P8  No shred / empty atoms
P9  Near-dedupe vision facts
P11 Drop stubs once vision succeeds on the same region
P12 Install image_facts typed as scope_item (not deal_metadata)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

# ── patterns ────────────────────────────────────────────────────────

_VISION_STUB_RE = re.compile(
    r"(?i)\b(?:awaiting\s+ocr(?:\s*/\s*vision)?|image\s+vision\s+abstain|"
    r"\[image\s+extracted\b|image_vision_abstained:)"
)

_EMAIL_SECURITY_RE = re.compile(
    r"(?i)\b(?:urldefense|proofpoint|mimecast|safelinks\.protection|"
    r"mimecastcybergraph|cgbannerindicator|mark\s+safe|powered\s+by\s+mimecast)\b"
)

_MARKETING_CHROME_RE = re.compile(
    r"(?i)(?:"
    r"quotes\s+in\s+24|"
    r"ai[\-\s]?driven\s+pmo|"
    r"^account\s+executive$|"
    r"global\s+field\s+services|"
    r"wifi,\s+and\s+cabling|"
    r"proven\s+execution\s+across|"
    r"how\s+you\s+doing|"
    r"^www\.purtera|"
    r"^purtera\-it\.com\b"
    r")"
)

_EMAIL_HEADER_LINE_RE = re.compile(
    r"(?i)^(?:from|sent|to|cc|bcc|subject|date)\s*:|"
    r"mailto:|"
    r"similar\s+name\s+as\s+someone"
)

_SPECULATIVE_RISK_RE = re.compile(
    r"(?i)(?:"
    r"may\s+pose|"
    r"may\s+impact|"
    r"potentially\s+affecting|"
    r"slight\s+trip|"
    r"patterned\s+carpet|"
    r"field\s+of\s+view|"
    r"aesthetically\s+unappealing|"
    r"trip\s+hazard\s+if\s+cables\s+are\s+not|"
    r"pose\s+a\s+(?:potential\s+)?trip\s+hazard|"
    r"posing\s+a\s+(?:potential\s+)?trip\s+hazard|"
    r"non[\-\s]?standard\s+tile\s+layout"
    r")"
)

_GROUNDED_RISK_HINT_RE = re.compile(
    r"(?i)(?:"
    r"annotation|"
    r"behind\s+the\s+wall|"
    r"drywall|"
    r"raceway|"
    r"noted\s+for|"
    r"must\s+be|"
    r"should\s+be\s+(?:moved|rerouted|hidden|removed)|"
    r"hard\s+to\s+get"
    r")"
)

_SHRED_RE = re.compile(
    r"(?i)^(?:ss|ph|&nbsp;|nbsp|;|&amp;|\u00b0shi|shi°?|\.|\-|–|—)+$"
)

_INSTALL_FACT_KINDS = frozenset(
    {
        "equipment",
        "mount",
        "cable",
        "placement",
        "connection",
        "power_data",
        "annotation",
        "site_condition",
    }
)


def _atom_text(atom: Any) -> str:
    if isinstance(atom, dict):
        for key in ("raw_text", "text", "normalized_text"):
            val = atom.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    for attr in ("raw_text", "text", "normalized_text"):
        val = getattr(atom, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _set_atom_text(atom: Any, text: str) -> None:
    text = (text or "").strip()
    if isinstance(atom, dict):
        if "raw_text" in atom or "text" in atom:
            if "raw_text" in atom:
                atom["raw_text"] = text
            if "text" in atom:
                atom["text"] = text
        else:
            atom["raw_text"] = text
        return
    if hasattr(atom, "raw_text"):
        atom.raw_text = text
    if hasattr(atom, "normalized_text"):
        try:
            from app.core.textnorm import normalize_text

            atom.normalized_text = normalize_text(text)
        except Exception:
            atom.normalized_text = text


def _atom_type_str(atom: Any) -> str:
    if isinstance(atom, dict):
        return str(atom.get("atom_type") or "").lower()
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "").lower()


def _atom_value(atom: Any) -> dict:
    if isinstance(atom, dict):
        val = atom.get("value")
        return dict(val) if isinstance(val, dict) else {}
    val = getattr(atom, "value", None)
    return dict(val) if isinstance(val, dict) else {}


def _region_ref(atom: Any) -> str:
    val = _atom_value(atom)
    rr = str(val.get("region_ref") or "").strip()
    if rr:
        return rr
    if isinstance(atom, dict):
        loc = atom.get("locator")
        if isinstance(loc, dict):
            return str(loc.get("region_ref") or "").strip()
    refs = getattr(atom, "source_refs", None) or []
    if refs:
        loc = getattr(refs[0], "locator", None) or {}
        if isinstance(loc, dict):
            return str(loc.get("region_ref") or "").strip()
    return ""


def unwrap_vision_text(text: str) -> str:
    """P3 — coerce JSON / Python string-list wrappers to plain prose."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) < 4:
        return text
    if text[0] in "[{\"'":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, str) and first.strip():
                    text = first.strip()
            elif isinstance(parsed, str) and parsed.strip():
                text = parsed.strip()
        except Exception:
            pass
    # Mismatched wrappers: ["…'] or ['…"]
    if (
        len(text) > 4
        and text[0] == "["
        and text[-1] == "]"
        and text[1] in "'\""
        and text[-2] in "'\""
    ):
        text = text[2:-2].replace('\\"', '"').replace("\\'", "'").strip()
    # Python repr list leftover
    if text.startswith("['") and text.endswith("']") and text.count("']") == 1:
        text = text[2:-2].strip()
    return re.sub(r"\s+", " ", text).strip()


def is_vision_stub(text: str) -> bool:
    return bool(_VISION_STUB_RE.search(text or ""))


def is_shred_atom(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 6 and not re.search(r"[a-zA-Z]{3,}", t):
        return True
    if _SHRED_RE.fullmatch(t):
        return True
    if t in {"&nbsp;", "nbsp;", "SS", "Ph", "ss", "ph"}:
        return True
    return False


def is_email_or_marketing_chrome(text: str) -> bool:
    t = text or ""
    if _EMAIL_SECURITY_RE.search(t):
        return True
    if _EMAIL_HEADER_LINE_RE.search(t):
        return True
    if _MARKETING_CHROME_RE.search(t):
        return True
    return False


def is_speculative_risk_text(text: str) -> bool:
    """P5 — soft aesthetic observations are not install risks."""
    t = text or ""
    if not _SPECULATIVE_RISK_RE.search(t):
        return False
    # Keep if clearly annotation / SOW grounded.
    if _GROUNDED_RISK_HINT_RE.search(t):
        return False
    return True


def is_vision_success_atom(atom: Any) -> bool:
    val = _atom_value(atom)
    via = str(val.get("via") or "")
    fk = str(val.get("fact_kind") or "")
    if "pdf_image_vision" in via:
        return not is_vision_stub(_atom_text(atom))
    if fk.startswith("image_fact") or fk in {"image_description", "image_instructions_summary"}:
        return not is_vision_stub(_atom_text(atom))
    return False


def drop_resolved_vision_stubs(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """P1 + P11 — drop awaiting-OCR stubs when any vision fact exists for region.

    Also drop *all* publishable stubs (even stub-only images): markers stay
    internal until vision succeeds; stub-only regions must not pollute packs.
    """
    success_regions: set[str] = set()
    for atom in atoms:
        if is_vision_success_atom(atom):
            rr = _region_ref(atom)
            if rr:
                success_regions.add(rr)

    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        text = _atom_text(atom)
        if not is_vision_stub(text):
            kept.append(atom)
            continue
        # Always drop publishable stubs (P1). Success pairing is audited via
        # success_regions for telemetry but does not keep orphan stubs.
        dropped.append(atom)
    return kept, dropped


def drop_chrome_and_shred(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """P2 / P7 / P8 — chrome, headers, shred."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        text = _atom_text(atom)
        at = _atom_type_str(atom)
        if is_shred_atom(text):
            dropped.append(atom)
            continue
        if is_email_or_marketing_chrome(text):
            # Never keep chrome as scope / exclusion / stakeholder / constraint.
            if at in {
                "scope_item",
                "exclusion",
                "stakeholder",
                "constraint",
                "assumption",
                "deal_metadata",
                "bom_line",
            }:
                dropped.append(atom)
                continue
        # Header lines typed as stakeholder/scope even without marketing keywords
        if at in {"stakeholder", "scope_item"} and _EMAIL_HEADER_LINE_RE.search(text):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def drop_speculative_risks(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """P5 — drop soft aesthetic risks (any type that carries speculative text)."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        text = _atom_text(atom)
        if is_speculative_risk_text(text):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def unwrap_vision_atom_texts(atoms: list[Any]) -> int:
    """P3 — mutate atom texts in place; return count unwrapped."""
    n = 0
    for atom in atoms:
        val = _atom_value(atom)
        via = str(val.get("via") or "")
        fk = str(val.get("fact_kind") or "")
        text = _atom_text(atom)
        if "pdf_image_vision" not in via and not fk.startswith("image"):
            # Still unwrap if looks wrapped
            if not (text.startswith("[") or text.startswith("['") or text.startswith('["')):
                continue
        cleaned = unwrap_vision_text(text)
        if cleaned != text:
            _set_atom_text(atom, cleaned)
            n += 1
    return n


def retag_install_vision_types(atoms: list[Any]) -> int:
    """P12 — image_fact install kinds → scope_item; grounded risk stays risk."""
    n = 0
    try:
        from app.core.schemas import AtomType
    except Exception:
        return 0
    for atom in atoms:
        val = _atom_value(atom)
        fk = str(val.get("fact_kind") or "")
        if not fk.startswith("image_fact:"):
            continue
        kind = fk.split(":", 1)[1].strip().lower()
        at = _atom_type_str(atom)
        if kind == "risk":
            target = AtomType.risk
        elif kind in _INSTALL_FACT_KINDS:
            target = AtomType.scope_item
        else:
            continue
        target_s = str(getattr(target, "value", target) or "").lower()
        if at == target_s:
            continue
        if isinstance(atom, dict):
            atom["atom_type"] = target_s
        else:
            atom.atom_type = target
        n += 1
    return n


def dedupe_near_vision_facts(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """P9 — keep highest-confidence atom per near-identical vision text key."""
    groups: dict[str, list[Any]] = defaultdict(list)
    non_vision: list[Any] = []
    for atom in atoms:
        val = _atom_value(atom)
        via = str(val.get("via") or "")
        fk = str(val.get("fact_kind") or "")
        if "pdf_image_vision" not in via and not fk.startswith("image"):
            non_vision.append(atom)
            continue
        key = re.sub(r"\W+", " ", unwrap_vision_text(_atom_text(atom)).lower())[:90].strip()
        if len(key) < 24:
            non_vision.append(atom)
            continue
        groups[key].append(atom)

    kept_v: list[Any] = []
    dropped: list[Any] = []
    for _key, group in groups.items():
        if len(group) == 1:
            kept_v.append(group[0])
            continue

        def _score(a: Any) -> float:
            conf = getattr(a, "confidence", None)
            if conf is None and isinstance(a, dict):
                conf = a.get("confidence")
            try:
                return float(conf or 0.0)
            except (TypeError, ValueError):
                return 0.0

        winner = max(group, key=_score)
        kept_v.append(winner)
        for a in group:
            if a is not winner:
                dropped.append(a)
    return non_vision + kept_v, dropped


def apply_universal_atom_hygiene(atoms: list[Any]) -> tuple[list[Any], list[Any], dict[str, int]]:
    """Run P1–P12 publish hygiene. Returns (kept, dropped, stats)."""
    stats = {
        "unwrapped": 0,
        "retagged": 0,
        "dropped_stubs": 0,
        "dropped_chrome": 0,
        "dropped_spec_risk": 0,
        "dropped_dedupe": 0,
    }
    stats["unwrapped"] = unwrap_vision_atom_texts(atoms)
    stats["retagged"] = retag_install_vision_types(atoms)

    all_dropped: list[Any] = []
    kept, d = drop_resolved_vision_stubs(atoms)
    stats["dropped_stubs"] = len(d)
    all_dropped.extend(d)

    kept, d = drop_chrome_and_shred(kept)
    stats["dropped_chrome"] = len(d)
    all_dropped.extend(d)

    kept, d = drop_speculative_risks(kept)
    stats["dropped_spec_risk"] = len(d)
    all_dropped.extend(d)

    kept, d = dedupe_near_vision_facts(kept)
    stats["dropped_dedupe"] = len(d)
    all_dropped.extend(d)

    return kept, all_dropped, stats


__all__ = [
    "apply_universal_atom_hygiene",
    "dedupe_near_vision_facts",
    "drop_chrome_and_shred",
    "drop_resolved_vision_stubs",
    "drop_speculative_risks",
    "is_email_or_marketing_chrome",
    "is_shred_atom",
    "is_speculative_risk_text",
    "is_vision_stub",
    "retag_install_vision_types",
    "unwrap_vision_atom_texts",
    "unwrap_vision_text",
]
