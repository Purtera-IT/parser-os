"""Trainable quote-context head seam for Deal Kit / PM handoff decisions.

This module does not hardcode a customer. It creates a universal decision
surface for "what kind of work are we quoting?" so a promoted neural head can
own the behavior once enough PM/teacher rows exist. Cold start falls back to a
small, source-grounded rule and logs the row into TrainingLog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.training_log import TEACHER_STORE, TrainingRow, log_rows

QUOTE_DELIVERY_RELATION = "quote_delivery_model"
CONFIG_ONLY = "config_only"
INSTALL_BUILDOUT = "install_buildout"
SURVEY_DESIGN = "survey_design"
UNKNOWN = "unknown"
_CANDIDATES = [CONFIG_ONLY, INSTALL_BUILDOUT, SURVEY_DESIGN, UNKNOWN]


@dataclass(frozen=True)
class QuoteContextDecision:
    delivery_model: str
    source: str
    confidence: float
    relation: str = QUOTE_DELIVERY_RELATION
    route_trainable: bool = False


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_value(atom: Any) -> dict[str, Any]:
    val = getattr(atom, "value", None)
    return val if isinstance(val, dict) else {}


def _atom_text(atom: Any) -> str:
    val = _atom_value(atom)
    return " ".join(
        str(x or "")
        for x in (
            getattr(atom, "raw_text", ""),
            getattr(atom, "text", ""),
            val.get("text"),
            val.get("source_context"),
            val.get("description"),
        )
        if x
    ).strip()


def _corpus_text(atoms: list[Any]) -> str:
    return "\n".join(_atom_text(a) for a in atoms if _atom_text(a)).strip()


def _rule_delivery_model(text: str) -> QuoteContextDecision:
    low = text.lower()
    installed = bool(
        re.search(r"\b(?:everything|it|that)\s+(?:is\s+)?(?:already\s+)?(?:physically\s+)?installed\b", low)
        or re.search(r"\bphysically\s+installed\b", low)
    )
    config_only = bool(
        re.search(r"\bjust\s+(?:needs?\s+to\s+be\s+)?configur(?:ed|ation)\b", low)
        or re.search(r"\bconfiguration\s+part\b", low)
    )
    excluded_buildout = bool(
        re.search(r"\bnetwork\s+build\s*out\s+does\s+not\s+need\b", low)
        or re.search(r"\bexclude:\s*(?:.|\n){0,120}(?:network\s+buildout|general\s+firewall)", low)
    )
    if (installed and config_only) or excluded_buildout:
        return QuoteContextDecision(CONFIG_ONLY, "deterministic_fallback", 0.78, route_trainable=True)
    if re.search(r"\b(?:install|mount|pull|run|terminate|cable|cabling|survey|heatmap)\b", low):
        if re.search(r"\b(?:survey|heatmap|rf)\b", low):
            return QuoteContextDecision(SURVEY_DESIGN, "deterministic_fallback", 0.62, route_trainable=True)
        return QuoteContextDecision(INSTALL_BUILDOUT, "deterministic_fallback", 0.62, route_trainable=True)
    return QuoteContextDecision(UNKNOWN, "deterministic_fallback", 0.5, route_trainable=True)


def decide_quote_delivery_model(atoms: list[Any]) -> QuoteContextDecision:
    text = _corpus_text(atoms)
    if not text:
        return QuoteContextDecision(UNKNOWN, "empty", 0.0, route_trainable=True)
    try:
        from app.core.embedding_retrieval import embed_texts
        from app.learning.head_registry import get_head_registry

        registry = get_head_registry()
        if registry is not None:
            champ = registry.champion(QUOTE_DELIVERY_RELATION)
            if champ is not None:
                head, _meta = champ
                vec = embed_texts([text])[0]
                hd = head.classify(vec, _CANDIDATES)
                if hd.verdict and not hd.route_llm:
                    return QuoteContextDecision(
                        str(hd.verdict),
                        "neural_head",
                        float(hd.confidence),
                        route_trainable=False,
                    )
    except Exception:
        pass
    return _rule_delivery_model(text)


def annotate_quote_context(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], int]:
    """Attach quote_context metadata to quote-level task atoms.

    Mutates atoms in place. Returns (atoms, annotated_count).
    """
    decision = decide_quote_delivery_model(atoms)
    text = _corpus_text(atoms)
    if decision.route_trainable and text:
        log_rows([
            TrainingRow(
                relation=QUOTE_DELIVERY_RELATION,
                label=decision.delivery_model,
                raw_text=text[:4000],
                label_kind="judgment",
                teacher=TEACHER_STORE,
                confidence=decision.confidence,
                deal_id=project_id,
                project_id=project_id,
                provenance={"source": decision.source, "relation": QUOTE_DELIVERY_RELATION},
            )
        ])

    n = 0
    for atom in atoms:
        if _atom_type_str(atom) != "task":
            continue
        val = dict(_atom_value(atom))
        if val.get("is_quote_line") is not True and val.get("task_tier") != "parent":
            continue
        val["quote_context"] = {
            "delivery_model": decision.delivery_model,
            "source": decision.source,
            "confidence": decision.confidence,
            "relation": decision.relation,
        }
        atom.value = val
        flags = list(getattr(atom, "review_flags", None) or [])
        flag = f"quote_context:{decision.delivery_model}"
        if flag not in flags:
            flags.append(flag)
        if decision.source == "neural_head" and "quote_context_neural_head" not in flags:
            flags.append("quote_context_neural_head")
        elif decision.route_trainable and "quote_context_training_row" not in flags:
            flags.append("quote_context_training_row")
        atom.review_flags = flags
        n += 1
    return atoms, n


__all__ = [
    "QUOTE_DELIVERY_RELATION",
    "CONFIG_ONLY",
    "INSTALL_BUILDOUT",
    "SURVEY_DESIGN",
    "UNKNOWN",
    "QuoteContextDecision",
    "annotate_quote_context",
    "decide_quote_delivery_model",
]
