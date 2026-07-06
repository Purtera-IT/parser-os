"""Trainable site facility label head for physical_site atoms.

Deal Kit assigns technicians against a human site name (``Pittsburgh Office``),
not a raw street line. Parser atoms should carry that label on ``facility_name``
before prefill — the same seam pattern as :mod:`app.core.quote_context_head`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.training_log import TEACHER_STORE, TrainingRow, log_rows

SITE_FACILITY_RELATION = "site_facility_label"
CITY_OFFICE = "city_office"
KEEP_FACILITY = "keep_facility"
KEEP_NAME = "keep_name"
_CANDIDATES = [CITY_OFFICE, KEEP_FACILITY, KEEP_NAME]

_STREET_RE = re.compile(
    r"^\d{1,6}\s+\S|"
    r"\b(st|street|ave|avenue|blvd|boulevard|dr|drive|rd|road|ln|lane|way|hwy|ct|court|ste|suite)\b",
    re.I,
)


@dataclass(frozen=True)
class SiteFacilityDecision:
    label: str
    facility_name: str
    source: str
    confidence: float
    relation: str = SITE_FACILITY_RELATION
    route_trainable: bool = False


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_value(atom: Any) -> dict[str, Any]:
    val = getattr(atom, "value", None)
    return val if isinstance(val, dict) else {}


def _title_city(city: str) -> str:
    city = (city or "").strip()
    if not city:
        return ""
    if re.fullmatch(r"[A-Z]{4,}", city):
        return city[0] + city[1:].lower()
    return " ".join(w[:1].upper() + w[1:].lower() for w in city.split())


def _looks_like_street_label(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if _STREET_RE.search(text):
        return True
    return bool(re.search(r"\d{5}", text) and "," in text)


def _facility_corpus(atom: Any) -> str:
    val = _atom_value(atom)
    return "\n".join(
        str(x or "")
        for x in (
            val.get("facility_name"),
            val.get("name"),
            val.get("street_address"),
            val.get("address"),
            val.get("city"),
            val.get("state"),
            getattr(atom, "raw_text", ""),
        )
        if x
    ).strip()


def _rule_facility_label(atom: Any) -> SiteFacilityDecision:
    val = _atom_value(atom)
    raw_name = str(val.get("facility_name") or val.get("name") or "").strip()
    city = _title_city(str(val.get("city") or ""))
    aliases = [str(a).strip() for a in (val.get("aliases") or val.get("names") or []) if str(a).strip()]

    if (raw_name and not _looks_like_street_label(raw_name)):
        return SiteFacilityDecision(KEEP_FACILITY, raw_name, "deterministic_fallback", 0.82)

    if city:
        return SiteFacilityDecision(
            CITY_OFFICE,
            f"{city} Office",
            "deterministic_fallback",
            0.78,
            route_trainable=True,
        )

    for alias in aliases:
        if alias and not _looks_like_street_label(alias) and not alias.isdigit():
            if re.search(r"\b(office|workshop|campus|facility|hq|store|plant)\b", alias, re.I):
                label = " ".join(w[:1].upper() + w[1:] for w in alias.split())
                return SiteFacilityDecision(KEEP_FACILITY, label, "deterministic_fallback", 0.74, route_trainable=True)

    return SiteFacilityDecision(KEEP_NAME, raw_name or "Site 1", "deterministic_fallback", 0.5, route_trainable=True)


def decide_site_facility_label(atom: Any) -> SiteFacilityDecision:
    corpus = _facility_corpus(atom)
    if not corpus:
        return SiteFacilityDecision(KEEP_NAME, "Site 1", "empty", 0.0, route_trainable=True)
    try:
        from app.core.embedding_retrieval import embed_texts
        from app.learning.head_registry import get_head_registry

        registry = get_head_registry()
        if registry is not None:
            champ = registry.champion(SITE_FACILITY_RELATION)
            if champ is not None:
                head, _meta = champ
                vec = embed_texts([corpus])[0]
                hd = head.classify(vec, _CANDIDATES)
                if hd.verdict and not hd.route_llm:
                    val = _atom_value(atom)
                    city = _title_city(str(val.get("city") or ""))
                    if hd.verdict == CITY_OFFICE and city:
                        facility = f"{city} Office"
                    else:
                        facility = str(val.get("facility_name") or val.get("name") or "").strip() or "Site 1"
                    return SiteFacilityDecision(
                        str(hd.verdict),
                        facility,
                        "neural_head",
                        float(hd.confidence),
                        route_trainable=False,
                    )
    except Exception:
        pass
    return _rule_facility_label(atom)


def annotate_site_facility_labels(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], int]:
    """Stamp ``facility_name`` on physical_site atoms from the trainable head seam."""
    n = 0
    for atom in atoms:
        if _atom_type_str(atom) != "physical_site":
            continue
        decision = decide_site_facility_label(atom)
        val = dict(_atom_value(atom))
        corpus = _facility_corpus(atom)
        if decision.route_trainable and corpus:
            log_rows([
                TrainingRow(
                    relation=SITE_FACILITY_RELATION,
                    label=decision.label,
                    raw_text=corpus[:4000],
                    label_kind="judgment",
                    teacher=TEACHER_STORE,
                    confidence=decision.confidence,
                    deal_id=project_id,
                    project_id=project_id,
                    provenance={"source": decision.source, "relation": SITE_FACILITY_RELATION},
                )
            ])
        val["facility_name"] = decision.facility_name
        val["name"] = decision.facility_name
        val["display_name"] = decision.facility_name
        val["facility_label"] = {
            "label": decision.label,
            "source": decision.source,
            "confidence": decision.confidence,
            "relation": decision.relation,
        }
        atom.value = val
        flags = list(getattr(atom, "review_flags", None) or [])
        flag = f"facility_label:{decision.label}"
        if flag not in flags:
            flags.append(flag)
        if decision.source == "neural_head" and "facility_label_neural_head" not in flags:
            flags.append("facility_label_neural_head")
        elif decision.route_trainable and "facility_label_training_row" not in flags:
            flags.append("facility_label_training_row")
        atom.review_flags = flags
        n += 1
    return atoms, n


__all__ = [
    "SITE_FACILITY_RELATION",
    "CITY_OFFICE",
    "KEEP_FACILITY",
    "KEEP_NAME",
    "SiteFacilityDecision",
    "annotate_site_facility_labels",
    "decide_site_facility_label",
]
