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
SITE_NAME_UNKNOWN_FLAG = "site_name_unknown"
CITY_OFFICE = "city_office"
KEEP_FACILITY = "keep_facility"
KEEP_NAME = "keep_name"
_CANDIDATES = [CITY_OFFICE, KEEP_FACILITY, KEEP_NAME]

_STREET_RE = re.compile(
    r"^\d{1,6}\s+\S|"
    r"\b(st|street|ave|avenue|blvd|boulevard|dr|drive|rd|road|ln|lane|way|hwy|ct|court|ste|suite)\b",
    re.I,
)


#: Why a site carries no name. The site still exists; only its name is unknown.
ABSTAIN_NO_DOCUMENT_NAME = "no_document_names_site"
ABSTAIN_EMPTY = "no_site_evidence"
ABSTAIN_HEAD_NAME_NOT_IN_SOURCE = "head_verdict_name_not_in_source"

#: Code paths that can put a name on a site (PUR-49 attribution).
RULE_FACILITY_FIELD = "rule:facility_name_field"
RULE_ALIAS_VERBATIM = "rule:alias_verbatim"
RULE_ABSTAIN = "rule:abstain"
HEAD_KEEP = "head:keep_name"
HEAD_CITY_OFFICE = "head:city_office_verbatim"
HEAD_ABSTAIN = "head:abstain"


@dataclass(frozen=True)
class SiteFacilityDecision:
    label: str
    #: None when the head/rule abstains: no document names this site.
    facility_name: str | None
    source: str
    confidence: float
    relation: str = SITE_FACILITY_RELATION
    route_trainable: bool = False
    #: The code path that produced ``facility_name`` (or abstained).
    name_rule: str = ""
    abstain_reason: str | None = None


def _normalize(text: Any) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"[\-_/.]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_is_verbatim(name: str | None, source_text: str) -> bool | None:
    """Is ``name`` present (dress-blind) in ``source_text``? None = undecidable."""
    if not name:
        return None
    src = _normalize(source_text)
    if not src:
        return None
    probe = _normalize(name)
    return bool(probe) and f" {probe} " in f" {src} "


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_value(atom: Any) -> dict[str, Any]:
    val = getattr(atom, "value", None)
    return val if isinstance(val, dict) else {}


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
    """Fallback when no trained champion exists. Never composes a name.

    It may only copy a name a document wrote. Removed (PUR-22/PUR-50):
    composing "<City> Office", using the bare city as the site's name (a real
    string read from the wrong field), re-casing aliases, and the "Site 1"
    placeholder. When nothing names the site it abstains; the site survives
    with ``facility_name=None`` and downstream renders "site N of M, name
    unknown".
    """
    val = _atom_value(atom)
    raw_name = str(val.get("facility_name") or val.get("name") or "").strip()
    aliases = [str(a).strip() for a in (val.get("aliases") or val.get("names") or []) if str(a).strip()]

    if raw_name and not _looks_like_street_label(raw_name):
        return SiteFacilityDecision(KEEP_FACILITY, raw_name, "deterministic_fallback", 0.82,
                                    name_rule=RULE_FACILITY_FIELD)

    for alias in aliases:
        if alias and not _looks_like_street_label(alias) and not alias.isdigit():
            if re.search(r"\b(office|workshop|campus|facility|hq|store|plant)\b", alias, re.I):
                return SiteFacilityDecision(KEEP_FACILITY, alias, "deterministic_fallback", 0.74,
                                            route_trainable=True, name_rule=RULE_ALIAS_VERBATIM)

    return SiteFacilityDecision(KEEP_NAME, None, "deterministic_fallback", 0.5, route_trainable=True,
                                name_rule=RULE_ABSTAIN, abstain_reason=ABSTAIN_NO_DOCUMENT_NAME)


def decide_site_facility_label(atom: Any) -> SiteFacilityDecision:
    corpus = _facility_corpus(atom)
    if not corpus:
        return SiteFacilityDecision(KEEP_NAME, None, "empty", 0.0, route_trainable=True,
                                    name_rule=RULE_ABSTAIN, abstain_reason=ABSTAIN_EMPTY)
    try:
        from app.core.embedding_retrieval import embed_texts
        from app.learning.head_registry import get_head_registry

        registry = get_head_registry()
        champ = registry.champion(SITE_FACILITY_RELATION) if registry is not None else None
    except Exception:
        champ = None
    if champ is not None:
        try:
            head, _meta = champ
            vec = embed_texts([corpus])[0]
            hd = head.classify(vec, _CANDIDATES)
        except Exception:
            hd = None
        if hd is not None and hd.verdict and not hd.route_llm:
            return _head_decision(atom, str(hd.verdict), float(hd.confidence))
    return _rule_facility_label(atom)


def _head_decision(atom: Any, verdict: str, confidence: float) -> SiteFacilityDecision:
    """Turn a head verdict into a name WITHOUT inventing text.

    The head picks which evidence names the site; it cannot mint a string.
    ``city_office`` is honoured only when "<City> Office" is literally in the
    atom's source text; otherwise the head abstains.
    """
    val = _atom_value(atom)
    source = str(getattr(atom, "raw_text", "") or "")
    if verdict == CITY_OFFICE:
        city = str(val.get("city") or "").strip()
        m = re.search(rf"\b{re.escape(city)}\s+office\b", source, re.I) if city else None
        if m:
            return SiteFacilityDecision(verdict, m.group(0), "neural_head", confidence,
                                        name_rule=HEAD_CITY_OFFICE)
        return SiteFacilityDecision(verdict, None, "neural_head", confidence, name_rule=HEAD_ABSTAIN,
                                    abstain_reason=ABSTAIN_HEAD_NAME_NOT_IN_SOURCE)
    raw_name = str(val.get("facility_name") or val.get("name") or "").strip()
    if raw_name and not _looks_like_street_label(raw_name):
        return SiteFacilityDecision(verdict, raw_name, "neural_head", confidence, name_rule=HEAD_KEEP)
    return SiteFacilityDecision(verdict, None, "neural_head", confidence, name_rule=HEAD_ABSTAIN,
                                abstain_reason=ABSTAIN_NO_DOCUMENT_NAME)


def annotate_site_facility_labels(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], int]:
    """Stamp ``facility_name`` on physical_site atoms from the trainable head seam.

    The label head decides *which* string names the site; it does not
    decide whether that string reads. ``BUILDING-EIGHT-HUNDRED-800``
    round-trips into "building eight hundred 800" and the head approves it
    as ``keep_facility`` — correctly, because it is the facility's name;
    it is just unreadable. :mod:`app.core.site_naming` does the reading
    pass afterwards, over the whole deal at once so that names colliding
    across two rows are visible, and stamps only names its evidence
    supports.
    """
    from app.core.site_naming import (
        SITE_NAME_BARE_IDENTIFIER,
        SITE_NAME_DUPLICATE,
        resolve_site_names,
    )

    site_atoms = [a for a in atoms if _atom_type_str(a) == "physical_site"]
    decisions = {id(a): decide_site_facility_label(a) for a in site_atoms}

    # Deal-wide naming pass. Keyed on the atom's identity rather than its
    # site id, because two atoms can carry the same id and each still
    # needs its own stamp.
    named: dict[int, Any] = {}
    entries = []
    for atom in site_atoms:
        val = _atom_value(atom)
        if not decisions[id(atom)].facility_name:
            continue
        entries.append({
            "site_id": str(val.get("id") or val.get("site_id") or id(atom)),
            "name": decisions[id(atom)].facility_name,
            "source_text": str(getattr(atom, "raw_text", "") or ""),
            "city": str(val.get("city") or ""),
        })
    try:
        resolved = resolve_site_names(entries)
        by_sid = {e["site_id"]: e for e in entries}
        for atom in site_atoms:
            val = _atom_value(atom)
            sid = str(val.get("id") or val.get("site_id") or id(atom))
            if sid not in by_sid:
                continue
            got = resolved.get(sid)
            if got is not None and got.name:
                named[id(atom)] = got
    except Exception:
        named = {}

    n = 0
    for atom in site_atoms:
        decision = decisions[id(atom)]
        readable = named.get(id(atom))
        if readable is not None and decision.facility_name:
            decision = SiteFacilityDecision(
                decision.label, readable.name, decision.source,
                decision.confidence, decision.relation, decision.route_trainable,
                decision.name_rule, decision.abstain_reason,
            )
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
        # PUR-49: every emitted name says which code path produced it and
        # whether its text is verbatim in the atom's own source.
        val["name_source"] = {
            "rule": decision.name_rule,
            "verbatim": name_is_verbatim(decision.facility_name, str(getattr(atom, "raw_text", "") or "")),
            "abstain_reason": decision.abstain_reason,
        }
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
        if decision.facility_name is None and SITE_NAME_UNKNOWN_FLAG not in flags:
            flags.append(SITE_NAME_UNKNOWN_FLAG)
        if decision.source == "neural_head" and "facility_label_neural_head" not in flags:
            flags.append("facility_label_neural_head")
        elif decision.route_trainable and "facility_label_training_row" not in flags:
            flags.append("facility_label_training_row")
        # A site the evidence could not name, and a name two rows share,
        # both stay visible to review instead of shipping as a number or
        # as a silent collision.
        for naming_flag in (readable.flags if readable is not None else ()):
            if naming_flag in (SITE_NAME_BARE_IDENTIFIER, SITE_NAME_DUPLICATE):
                if naming_flag not in flags:
                    flags.append(naming_flag)
        atom.review_flags = flags
        n += 1
    return atoms, n


__all__ = [
    "SITE_FACILITY_RELATION",
    "CITY_OFFICE",
    "KEEP_FACILITY",
    "KEEP_NAME",
    "SiteFacilityDecision",
    "SITE_NAME_UNKNOWN_FLAG",
    "name_is_verbatim",
    "annotate_site_facility_labels",
    "decide_site_facility_label",
]
