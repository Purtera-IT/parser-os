from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RuleSuggestionType = Literal[
    "domain_alias",
    "exclusion_pattern",
    "constraint_pattern",
    "parser_header_alias",
    "risk_default",
    "authority_override_candidate",
    "entity_normalization_rule",
]


class RuleSuggestion(BaseModel):
    suggestion_id: str
    suggestion_type: RuleSuggestionType
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = Field(ge=0)
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_approval: bool = True
    target_file: str | None = None


class RuleSuggestionFile(BaseModel):
    suggestions: list[RuleSuggestion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
