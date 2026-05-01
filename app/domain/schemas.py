from __future__ import annotations

from pydantic import BaseModel, Field


class DomainEntityType(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class DomainPack(BaseModel):
    pack_id: str
    name: str
    version: str
    service_lines: list[str] = Field(default_factory=list)
    entity_types: list[DomainEntityType] = Field(default_factory=list)
    device_aliases: dict[str, list[str]] = Field(default_factory=dict)
    site_alias_patterns: list[str] = Field(default_factory=list)
    action_aliases: dict[str, list[str]] = Field(default_factory=dict)
    constraint_patterns: dict[str, list[str]] = Field(default_factory=dict)
    exclusion_patterns: list[str] = Field(default_factory=list)
    customer_instruction_patterns: list[str] = Field(default_factory=list)
    quantity_units: dict[str, list[str]] = Field(default_factory=dict)
    artifact_role_patterns: dict[str, list[str]] = Field(default_factory=dict)
    risk_defaults: dict[str, float] = Field(default_factory=dict)
    packet_family_hints: dict[str, list[str]] = Field(default_factory=dict)
    # Bundled ontology YAML (relative to app/domain/) when the on-disk pack is a wide reference file
    # not fully modeled by DomainPack; see loader._adapt_reference_pack_to_domain_pack.
    reference_ontology_path: str | None = None
