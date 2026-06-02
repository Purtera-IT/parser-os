from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DomainEntityType(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class DetectionTargetSpec(BaseModel):
    """A single symbol type a domain pack expects to find on its drawings.

    Detection targets are pack-declared and parser-agnostic. The PDF
    parser intersects the parsed legend with this list to decide what
    counts as ``parse complete`` for a drawing page in this service
    line. ``completeness="load_bearing"`` targets that appear in the
    parsed legend but produce zero detections become ``legend_orphan``
    warnings; targets the pack declares as load-bearing but the
    legend omits become ``legend_gap`` warnings.

    ``modalities`` tells the detector which matchers to try
    (text_tag for legend tokens like ``WN``, ``CR``, ``TV``; glyph_template
    for crop-based matching of swatches/icons; vector_shape for
    PDF drawing primitives; zone for filled polygons such as a
    wireless heat-map area; line_run for cable trunk segments).

    ``parent_entity_keys`` lets a subtype target roll up to broader
    entity buckets so the cross-artifact conflict detector can pair
    a schematic ``device:ptz_camera`` count with a BOM line item
    keyed on the broader ``device:ip_camera`` or ``device:camera``.
    Without these, the schematic upgrade improves subtype recall at
    the cost of cross-artifact conflict recall.
    """

    key: str
    entity_key: str
    ontology_key: str | None = None
    aliases: list[str] = Field(default_factory=list)
    aliases_from: list[str] = Field(default_factory=list)
    completeness: Literal["load_bearing", "informational"] = "informational"
    modalities: list[
        Literal["text_tag", "glyph_template", "vector_shape", "zone", "line_run"]
    ] = Field(default_factory=lambda: ["text_tag", "glyph_template"])
    count_semantics: Literal["each_instance", "each_zone", "each_run", "presence_only"] = (
        "each_instance"
    )
    parent_entity_keys: list[str] = Field(default_factory=list)

    @field_validator("key", "entity_key")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("detection target key/entity_key cannot be empty")
        return v.strip()

    @field_validator("modalities")
    @classmethod
    def _at_least_one_modality(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("detection target needs at least one modality")
        return v


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
    # v57: optional allow-list of multi_entity_llm extractor keys this
    # pack should run (e.g. ["requirements", "site_clusters", "quantities"]).
    # EMPTY = run ALL extractors (backward-compatible default for every
    # existing pack). When non-empty, enrich_entities runs ONLY the named
    # extractors (plus the always-needed "customer" anchor), cutting the
    # per-document extractor fan-out from ~28 down to the handful relevant
    # to the domain. This is the dominant cost of enrich_entities, so a
    # tight list is the biggest single speed lever — but it trades away
    # recall for any entity type omitted, so narrow it only against a
    # measured baseline.
    llm_extractors: list[str] = Field(default_factory=list)
    # Bundled ontology YAML (relative to app/domain/) when the on-disk pack is a wide reference file
    # not fully modeled by DomainPack; see loader._adapt_reference_pack_to_domain_pack.
    reference_ontology_path: str | None = None
    # Schematic detection targets (PR2 of the schematic upgrade).
    # Packs with no schematic story leave this empty; the parser then
    # treats any drawing pages as ``missing_detection_targets`` and
    # emits warnings rather than silent target sets.
    detection_targets: list[DetectionTargetSpec] = Field(default_factory=list)

    def resolved_target_aliases(self, target: DetectionTargetSpec) -> list[str]:
        """Return the deduplicated alias list for ``target`` honoring ``aliases_from``.

        ``aliases_from`` entries point at ``device_aliases`` keys
        (e.g. ``device_aliases.ip_camera``). Strings that do not match
        the pattern are ignored.  Explicit ``aliases`` are appended
        after the resolved aliases so callers can extend without
        editing the underlying device_aliases table.
        """

        out: list[str] = []
        seen: set[str] = set()
        for ref in target.aliases_from:
            if not isinstance(ref, str):
                continue
            head, sep, tail = ref.partition(".")
            if head != "device_aliases" or not sep or not tail:
                continue
            for alias in self.device_aliases.get(tail, []) or []:
                norm = alias.strip().lower()
                if norm and norm not in seen:
                    seen.add(norm)
                    out.append(alias)
        for alias in target.aliases:
            norm = alias.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(alias)
        return out
