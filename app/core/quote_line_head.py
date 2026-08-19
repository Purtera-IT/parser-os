"""Trainable quote-line head — umbrella labor tasks for technician assignment.

Parser step-level tasks (camera config, badge setup, Okta) are runbook granularity.
Deal Kit quote lines should be umbrella tasks (``Ubiquiti configuration / install support``) with an
optional technician skill stamp for Labor/PMO. Cold start logs training rows;
promoted heads own the behavior once enough PM labels exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.training_log import TEACHER_STORE, TrainingRow, log_rows

QUOTE_LABOR_LINE_RELATION = "quote_labor_line"
TASK_TECHNICIAN_SKILL_RELATION = "task_technician_skill"

CONFIG_UMBRELLA = "Ubiquiti configuration / install support"
KNOWLEDGE_HANDOFF = "Knowledge transfer / guided handoff"
INSTALL_UMBRELLA = "Equipment installation"
SURVEY_UMBRELLA = "Wireless site survey"
CONFIG_INSTALL_MODELS = frozenset({"config_only", "install_buildout"})

_LABOR_CANDIDATES = [
    CONFIG_UMBRELLA,
    KNOWLEDGE_HANDOFF,
    INSTALL_UMBRELLA,
    SURVEY_UMBRELLA,
    "Security camera configuration",
    "Badge and access control setup",
    "Identity integration (Okta)",
]

_CONFIG_INSTALL_MICRO_LABELS = frozenset({
    CONFIG_UMBRELLA,
    "Ubiquiti configuration",
    "Security camera configuration",
    "Badge and access control setup",
    "Identity integration (Okta)",
    "UID Enterprise setup",
})

_SKILL_CANDIDATES = [
    "Network / Wireless L2",
    "Security / AV L2",
    "Security / Access L2",
    "Integration / IAM L2",
    "Senior engineer",
    "Wireless survey tech",
    "Field technician L2",
]

_CONFIG_MICRO_RE = re.compile(
    r"\b(configur(?:ation|e)|/ setup|enterprise setup|install support)\b",
    re.I,
)
_CONFIG_INSTALL_LINE_RE = re.compile(
    r"\b("
    r"camera|nvr|surveillance|cctv|"
    r"badge|access control|reader|"
    r"uid\s+enterprise|"
    r"okta|otka|sso|idp|"
    r"ubiquiti|unifi|udm|vlan|firewall|wifi|"
    r"configur(?:ation|e)|/ setup|enterprise setup|install support"
    r")\b",
    re.I,
)
_UID_ENTERPRISE_RE = re.compile(r"\buid\s+enterprise\b", re.I)
_PMO_ADMIN_RE = re.compile(
    r"^(complete billing|weekly:|attend debriefing|coordinate work in a manner|develop schedule based on stakeholder)",
    re.I,
)


@dataclass(frozen=True)
class QuoteLineDecision:
    quote_line: str
    technician_skill: str
    source: str
    confidence: float
    route_trainable: bool = False


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_value(atom: Any) -> dict[str, Any]:
    val = getattr(atom, "value", None)
    return val if isinstance(val, dict) else {}


def _task_text(atom: Any) -> str:
    val = _atom_value(atom)
    return str(
        val.get("text")
        or val.get("name")
        or getattr(atom, "raw_text", "")
        or getattr(atom, "text", "")
        or ""
    ).strip()


def _delivery_model(atoms: list[Any]) -> str:
    for atom in atoms:
        if _atom_type_str(atom) != "task":
            continue
        ctx = _atom_value(atom).get("quote_context") or {}
        if isinstance(ctx, dict) and ctx.get("delivery_model"):
            return str(ctx["delivery_model"]).strip().lower()
    return ""


def _is_config_install_deal(delivery_model: str) -> bool:
    return delivery_model.strip().lower() in CONFIG_INSTALL_MODELS


def _is_knowledge_handoff_line(label: str) -> bool:
    low = (label or "").strip().lower()
    if not low:
        return False
    if low == KNOWLEDGE_HANDOFF.lower():
        return True
    return bool(re.search(r"\b(knowledge\s+transfer|white\s+glove|walk(?:ing)?\b)", low))


def _coalesce_config_install_decision(decision: QuoteLineDecision, *, config_install: bool) -> QuoteLineDecision:
    if not config_install or not decision.quote_line:
        return decision
    if _is_knowledge_handoff_line(decision.quote_line):
        return decision
    if decision.quote_line in _CONFIG_INSTALL_MICRO_LABELS:
        return QuoteLineDecision(
            CONFIG_UMBRELLA,
            "Network / Wireless L2",
            decision.source,
            decision.confidence,
            route_trainable=decision.route_trainable,
        )
    return decision


def _rule_quote_line(text: str, *, config_install: bool) -> QuoteLineDecision:
    low = text.lower()
    if _PMO_ADMIN_RE.match(text.strip()):
        return QuoteLineDecision("", "", "pmo_filtered", 1.0)

    if re.search(r"\b(knowledge transfer|white glove|walk(?:ing)?\b)", low):
        # Keep source wording ("walking him through the setup") — never invent
        # "guided handoff" when the email already named the work unit.
        label = text.strip()[:120] if text.strip() else KNOWLEDGE_HANDOFF
        return QuoteLineDecision(label, "Senior engineer", "deterministic_fallback", 0.76, route_trainable=True)
    if re.search(r"\b(site survey|heatmap|passive survey|gap analysis)\b", low):
        return QuoteLineDecision(SURVEY_UMBRELLA, "Wireless survey tech", "deterministic_fallback", 0.72, route_trainable=True)
    if config_install and _CONFIG_INSTALL_LINE_RE.search(low):
        return QuoteLineDecision(CONFIG_UMBRELLA, "Network / Wireless L2", "deterministic_fallback", 0.82, route_trainable=True)
    if re.search(r"\b(camera|nvr|surveillance|cctv)\b", low) and re.search(r"\bconfigur", low):
        return QuoteLineDecision("Security camera configuration", "Security / AV L2", "deterministic_fallback", 0.72, route_trainable=True)
    if _UID_ENTERPRISE_RE.search(low):
        return QuoteLineDecision("UID Enterprise setup", "Security / Access L2", "deterministic_fallback", 0.74, route_trainable=True)
    if re.search(r"\b(okta|otka|sso|idp)\b", low):
        return QuoteLineDecision("Identity integration (Okta)", "Integration / IAM L2", "deterministic_fallback", 0.72, route_trainable=True)
    if re.search(r"\b(badge|access control|reader)\b", low) and not _UID_ENTERPRISE_RE.search(low):
        return QuoteLineDecision("Badge and access control setup", "Security / Access L2", "deterministic_fallback", 0.72, route_trainable=True)
    if re.search(r"\b(ubiquiti|unifi|udm|vlan|firewall|wifi)\b", low):
        return QuoteLineDecision(CONFIG_UMBRELLA, "Network / Wireless L2", "deterministic_fallback", 0.74, route_trainable=True)
    if config_install and _CONFIG_MICRO_RE.search(low):
        return QuoteLineDecision(CONFIG_UMBRELLA, "Network / Wireless L2", "deterministic_fallback", 0.8, route_trainable=True)
    if re.search(r"\binstall(?:ation)?\b", low) and not re.search(r"\bconfigur", low):
        return QuoteLineDecision(text[:120], "Field technician L2", "deterministic_fallback", 0.62, route_trainable=True)

    return QuoteLineDecision(text[:120], "Field technician L2", "deterministic_fallback", 0.5, route_trainable=True)


def _head_classify(relation: str, text: str, candidates: list[str]) -> tuple[str | None, float, str]:
    try:
        from app.core.embedding_retrieval import embed_texts
        from app.learning.head_registry import get_head_registry

        registry = get_head_registry()
        if registry is None:
            return None, 0.0, "no_registry"
        champ = registry.champion(relation)
        if champ is None:
            return None, 0.0, "no_champion"
        head, _meta = champ
        vec = embed_texts([text])[0]
        hd = head.classify(vec, candidates)
        if hd.verdict and not hd.route_llm:
            return str(hd.verdict), float(hd.confidence), "neural_head"
    except Exception:
        pass
    return None, 0.0, "head_miss"


def decide_quote_line(text: str, *, config_install: bool) -> QuoteLineDecision:
    if not text.strip():
        return QuoteLineDecision("", "", "empty", 0.0, route_trainable=True)
    source_text = text.strip()
    quote, conf, source = _head_classify(QUOTE_LABOR_LINE_RELATION, text, _LABOR_CANDIDATES)
    if quote:
        # Neural/embedding heads pick from canned candidates — never let them
        # replace a knowledge-transfer line with the "guided handoff" paraphrase
        # when the source already named the work unit.
        if _is_knowledge_handoff_line(quote) and _is_knowledge_handoff_line(source_text):
            quote = source_text[:120]
        skill, skill_conf, skill_source = _head_classify(TASK_TECHNICIAN_SKILL_RELATION, text, _SKILL_CANDIDATES)
        if not skill:
            fallback = _rule_quote_line(text, config_install=config_install)
            skill = fallback.technician_skill
        decision = QuoteLineDecision(quote, skill or "Field technician L2", source, conf)
    else:
        decision = _rule_quote_line(text, config_install=config_install)
    return _coalesce_config_install_decision(decision, config_install=config_install)


def _is_quote_line_task(atom: Any) -> bool:
    val = _atom_value(atom)
    if val.get("is_quote_line") is True or val.get("task_tier") == "parent":
        return True
    return False


def _quote_line_bucket_key(
    atom: Any,
    decision: QuoteLineDecision,
    text: str,
    *,
    config_install: bool,
) -> tuple[str, ...]:
    site_key = ""
    for key in getattr(atom, "entity_keys", None) or []:
        if str(key).startswith("site:"):
            site_key = str(key)
            break
    line_key = decision.quote_line.strip().lower()
    if _is_knowledge_handoff_line(decision.quote_line):
        # Bucket all knowledge-transfer variants together without rewriting text.
        return (site_key, "knowledge_handoff")
    if config_install and (
        line_key == CONFIG_UMBRELLA.lower()
        or decision.quote_line in _CONFIG_INSTALL_MICRO_LABELS
        or _CONFIG_INSTALL_LINE_RE.search(text)
    ):
        return (site_key, CONFIG_UMBRELLA.lower())
    if line_key == CONFIG_UMBRELLA.lower():
        return (site_key, line_key)
    return (site_key, line_key, text.strip().lower()[:80])


def consolidate_quote_line_tasks(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], int]:
    """Rewrite quote-level task atoms to umbrella lines; drop PMO/admin tasks."""
    delivery_model = _delivery_model(atoms)
    config_install = _is_config_install_deal(delivery_model)
    kept: list[Any] = []
    umbrellas: dict[tuple[str, ...], Any] = {}
    changed = 0

    for atom in atoms:
        if _atom_type_str(atom) != "task":
            kept.append(atom)
            continue
        if not _is_quote_line_task(atom):
            kept.append(atom)
            continue

        text = _task_text(atom)
        decision = decide_quote_line(text, config_install=config_install)
        if not decision.quote_line:
            changed += 1
            continue

        if decision.route_trainable and text:
            log_rows([
                TrainingRow(
                    relation=QUOTE_LABOR_LINE_RELATION,
                    label=decision.quote_line,
                    raw_text=text[:4000],
                    label_kind="judgment",
                    teacher=TEACHER_STORE,
                    confidence=decision.confidence,
                    deal_id=project_id,
                    project_id=project_id,
                    provenance={"source": decision.source, "delivery_model": delivery_model},
                ),
                TrainingRow(
                    relation=TASK_TECHNICIAN_SKILL_RELATION,
                    label=decision.technician_skill,
                    raw_text=text[:4000],
                    label_kind="judgment",
                    teacher=TEACHER_STORE,
                    confidence=decision.confidence,
                    deal_id=project_id,
                    project_id=project_id,
                    provenance={"source": decision.source},
                ),
            ])

        bucket = _quote_line_bucket_key(atom, decision, text, config_install=config_install)

        val = dict(_atom_value(atom))
        val["quote_line"] = {
            "label": decision.quote_line,
            "technician_skill": decision.technician_skill,
            "source": decision.source,
            "confidence": decision.confidence,
            "original_text": text,
        }
        val["text"] = decision.quote_line
        val["name"] = decision.quote_line
        val["technician_skill"] = decision.technician_skill
        atom.value = val

        existing = umbrellas.get(bucket)
        if existing is None:
            umbrellas[bucket] = atom
            changed += 1
            continue

        ev = dict(_atom_value(existing).get("quote_line") or {})
        originals = [str(ev.get("original_text") or "")]
        originals.append(text)
        ev["original_text"] = "; ".join(x for x in originals if x)
        ex_val = dict(_atom_value(existing))
        ex_val["quote_line"] = ev
        existing.value = ex_val
        changed += 1

    out = kept + list(umbrellas.values())
    return out, changed


__all__ = [
    "CONFIG_INSTALL_MODELS",
    "CONFIG_UMBRELLA",
    "QUOTE_LABOR_LINE_RELATION",
    "TASK_TECHNICIAN_SKILL_RELATION",
    "QuoteLineDecision",
    "consolidate_quote_line_tasks",
    "decide_quote_line",
]
