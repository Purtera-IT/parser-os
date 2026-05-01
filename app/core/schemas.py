from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "0.2.0"
COMPILER_VERSION = "0.2.0"
PACKETIZER_VERSION = "0.2.0"
AUTHORITY_POLICY_VERSION = "0.2.0"


class ArtifactType(str, Enum):
    xlsx = "xlsx"
    csv = "csv"
    email = "email"
    docx = "docx"
    transcript = "transcript"
    vendor_quote = "vendor_quote"
    po = "po"
    txt = "txt"


class AtomType(str, Enum):
    quantity = "quantity"
    entity = "entity"
    constraint = "constraint"
    exclusion = "exclusion"
    scope_item = "scope_item"
    customer_instruction = "customer_instruction"
    vendor_line_item = "vendor_line_item"
    assumption = "assumption"
    open_question = "open_question"
    decision = "decision"
    action_item = "action_item"
    meeting_commitment = "meeting_commitment"


class AuthorityClass(str, Enum):
    contractual_scope = "contractual_scope"
    customer_current_authored = "customer_current_authored"
    approved_site_roster = "approved_site_roster"
    vendor_quote = "vendor_quote"
    meeting_note = "meeting_note"
    machine_extractor = "machine_extractor"
    quoted_old_email = "quoted_old_email"
    deleted_text = "deleted_text"


class ReviewStatus(str, Enum):
    auto_accepted = "auto_accepted"
    needs_review = "needs_review"
    rejected = "rejected"
    approved = "approved"


class EdgeType(str, Enum):
    same_as = "same_as"
    supports = "supports"
    contradicts = "contradicts"
    excludes = "excludes"
    requires = "requires"
    located_in = "located_in"
    derived_from = "derived_from"
    quoted_from = "quoted_from"


class PacketStatus(str, Enum):
    active = "active"
    accepted = "active"
    needs_review = "needs_review"
    rejected = "rejected"
    invalidated = "invalidated"


class PacketFamily(str, Enum):
    scope_inclusion = "scope_inclusion"
    scope_exclusion = "scope_exclusion"
    quantity_claim = "quantity_claim"
    quantity_conflict = "quantity_conflict"
    site_access = "site_access"
    missing_info = "missing_info"
    customer_override = "customer_override"
    vendor_mismatch = "vendor_mismatch"
    meeting_decision = "meeting_decision"
    action_item = "action_item"


class ParserCapability(BaseModel):
    parser_name: str
    parser_version: str
    supported_extensions: list[str] = Field(default_factory=list)
    supported_artifact_types: list[ArtifactType] = Field(default_factory=list)
    emitted_atom_types: list[AtomType] = Field(default_factory=list)
    supported_domain_packs: list[str] = Field(default_factory=list)
    requires_binary: bool = False
    supports_source_replay: bool = True


class ParserMatch(BaseModel):
    parser_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    artifact_type: ArtifactType


class CandidateAtom(BaseModel):
    id: str
    project_id: str
    artifact_id: str
    candidate_type: AtomType
    raw_text: str
    proposed_normalized_text: str
    proposed_value: dict[str, Any] = Field(default_factory=dict)
    proposed_entity_keys: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    proposed_authority_class: AuthorityClass
    extractor_name: str
    extractor_version: str
    extraction_method: Literal[
        "deterministic_rule",
        "domain_pack_rule",
        "semantic_candidate",
        "llm_candidate",
        "human_label",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_span: str | None = None
    validation_status: Literal["pending", "accepted", "rejected", "needs_review"] = "pending"
    validation_reasons: list[str] = Field(default_factory=list)


class ParserOutput(BaseModel):
    candidates: list[CandidateAtom] = Field(default_factory=list)
    atoms: list[EvidenceAtom] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    candidate_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    needs_review_count: int = Field(ge=0)
    by_extractor: dict[str, int] = Field(default_factory=dict)


class SemanticLinkCandidate(BaseModel):
    id: str
    from_atom_id: str
    to_atom_id: str
    proposed_edge_type: EdgeType
    similarity_score: float = Field(ge=0.0, le=1.0)
    method: Literal["tfidf_char_ngram", "sentence_transformer"]
    reason: str
    status: Literal["accepted", "needs_review", "rejected"]


class SourceRef(BaseModel):
    id: str
    artifact_id: str
    artifact_type: ArtifactType
    filename: str
    locator: dict[str, Any]
    extraction_method: str
    parser_version: str
    # Compatibility fields used by current MVP pipeline/tests.
    parser: str | None = None
    path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def from_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "parser" in data or "path" in data:
            filename = data.get("filename") or data.get("path", "")
            parser_value = str(data.get("parser", "txt")).lower()
            try:
                artifact_type = ArtifactType(parser_value)
            except ValueError:
                artifact_type = ArtifactType.txt
            return {
                "id": data.get("id") or f"src_{data.get('artifact_id', 'unknown')}",
                "artifact_id": data.get("artifact_id", "unknown_artifact"),
                "artifact_type": data.get("artifact_type", artifact_type),
                "filename": filename,
                "locator": data.get("locator") or {"location": data.get("location")},
                "extraction_method": data.get("extraction_method", "legacy"),
                "parser_version": data.get("parser_version", "legacy"),
                "parser": data.get("parser"),
                "path": data.get("path"),
            }
        return data

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> "SourceRef":
        if not self.parser:
            self.parser = self.artifact_type.value
        if not self.path:
            self.path = self.filename
        return self


class EvidenceReceipt(BaseModel):
    atom_id: str
    artifact_id: str
    filename: str
    source_ref_id: str
    replay_status: Literal["verified", "failed", "unsupported"]
    extracted_snippet: str | None = None
    locator: dict[str, Any] = Field(default_factory=dict)
    reason: str
    verifier_version: str


class EvidenceAtom(BaseModel):
    id: str
    project_id: str
    artifact_id: str
    atom_type: AtomType
    raw_text: str
    normalized_text: str
    value: dict[str, Any] = Field(default_factory=dict)
    entity_keys: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    receipts: list[EvidenceReceipt] = Field(default_factory=list)
    authority_class: AuthorityClass
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review_status: ReviewStatus
    review_flags: list[str] = Field(default_factory=list)
    parser_version: str
    # Compatibility fields used by current MVP pipeline/tests.
    atom_id: str | None = None
    claim: str | None = None
    normalized_claim: str | None = None
    entity: str | None = None
    predicate: str | None = None
    authority_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def from_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "atom_id" in data or "claim" in data or "entity" in data:
            raw_text = data.get("raw_text") or data.get("claim", "")
            normalized_text = data.get("normalized_text") or data.get("normalized_claim", raw_text)
            legacy_value = data.get("value", {})
            if not isinstance(legacy_value, dict):
                legacy_value = {"value": legacy_value}
            entity = data.get("entity", "")
            source_refs = data.get("source_refs", [])
            artifact_id = data.get("artifact_id")
            if artifact_id is None and source_refs:
                first_ref = source_refs[0]
                artifact_id = first_ref.get("artifact_id") if isinstance(first_ref, dict) else first_ref.artifact_id
            return {
                "id": data.get("id") or data.get("atom_id", ""),
                "project_id": data.get("project_id", "unknown_project"),
                "artifact_id": artifact_id or "unknown_artifact",
                "atom_type": data.get("atom_type", AtomType.entity),
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "value": legacy_value,
                "entity_keys": data.get("entity_keys") or ([entity] if entity else []),
                "source_refs": source_refs,
                "receipts": data.get("receipts", []),
                "authority_class": data.get("authority_class", AuthorityClass.machine_extractor),
                "confidence": data.get("confidence", 0.5),
                "review_status": data.get("review_status", ReviewStatus.auto_accepted),
                "review_flags": data.get("review_flags", []),
                "parser_version": data.get("parser_version", "legacy"),
                "atom_id": data.get("atom_id"),
                "claim": data.get("claim"),
                "normalized_claim": data.get("normalized_claim"),
                "entity": data.get("entity"),
                "predicate": data.get("predicate"),
                "authority_score": data.get("authority_score", 0.0),
            }
        return data

    @field_validator("source_refs")
    @classmethod
    def ensure_source_refs(cls, value: list[SourceRef]) -> list[SourceRef]:
        if not value:
            raise ValueError("Every EvidenceAtom must have at least one SourceRef")
        return value

    @field_validator("raw_text", "normalized_text")
    @classmethod
    def ensure_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_text and normalized_text cannot be empty")
        return value

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> "EvidenceAtom":
        if not self.atom_id:
            self.atom_id = self.id
        if not self.claim:
            self.claim = self.raw_text
        if not self.normalized_claim:
            self.normalized_claim = self.normalized_text
        if not self.entity and self.entity_keys:
            self.entity = self.entity_keys[0]
        if not self.predicate and isinstance(self.value, dict):
            self.predicate = str(self.value.get("predicate", "value"))
        return self


class EvidenceEdge(BaseModel):
    id: str
    project_id: str
    from_atom_id: str
    to_atom_id: str
    edge_type: EdgeType
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuthorityRankedAtom(BaseModel):
    atom_id: str
    rank: int = Field(ge=1)
    authority_score: float = Field(ge=0.0, le=1.0)


class AuthorityScore(BaseModel):
    atom_id: str
    base_rank: int
    recency_score: float
    authorship_score: float
    artifact_role_score: float
    evidence_state_penalty: float
    review_penalty: float
    final_score: float
    dimensions: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class EntityEdge(BaseModel):
    edge_id: str
    src_entity: str
    dst_entity: str
    relation: str
    weight: float = Field(ge=0.0, le=1.0)
    supporting_atom_ids: list[str] = Field(default_factory=list)

    @field_validator("dst_entity", mode="before")
    @classmethod
    def coerce_dst_entity(cls, value: Any) -> str:
        if isinstance(value, dict):
            if "value" in value:
                return str(value["value"])
            return str(value)
        return str(value)


class EntityRecord(BaseModel):
    id: str
    project_id: str
    entity_type: str
    canonical_key: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    source_atom_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: ReviewStatus


class PacketCertificate(BaseModel):
    packet_id: str
    certificate_version: str
    domain_pack_id: str | None = None
    domain_pack_version: str | None = None
    existence_reason: str
    governing_rationale: str
    minimal_sufficient_atom_ids: list[str] = Field(default_factory=list)
    contradiction_summary: str | None = None
    authority_path: list[dict[str, Any]] = Field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = Field(default_factory=list)
    blast_radius: list[str] = Field(default_factory=list)
    evidence_completeness_score: float = Field(ge=0.0, le=1.0)
    ambiguity_score: float = Field(ge=0.0, le=1.0)


class PacketRisk(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high", "critical"]
    risk_reasons: list[str] = Field(default_factory=list)
    estimated_cost_exposure: float | None = None
    operational_impact: list[str] = Field(default_factory=list)
    review_priority: int = Field(ge=1, le=5)
    # Lower = earlier in PM review queue (0 = procurement conflicts first; 90+ = noise / deprioritized).
    queue_tier: int = Field(default=50, ge=0, le=99)


class AnchorSignature(BaseModel):
    anchor_type: str
    canonical_key: str
    entity_keys: list[str] = Field(default_factory=list)
    normalized_topic: str
    scope_dimension: str | None = None
    hash: str


class EvidencePacket(BaseModel):
    id: str
    project_id: str
    family: PacketFamily
    anchor_type: str
    anchor_key: str
    governing_atom_ids: list[str] = Field(default_factory=list)
    supporting_atom_ids: list[str] = Field(default_factory=list)
    contradicting_atom_ids: list[str] = Field(default_factory=list)
    related_edge_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: PacketStatus
    reason: str
    review_flags: list[str] = Field(default_factory=list)
    anchor_signature: AnchorSignature | None = None
    certificate: PacketCertificate | None = None
    risk: PacketRisk | None = None
    # Compatibility fields used by current MVP pipeline/tests.
    packet_id: str | None = None
    topic: str | None = None
    atom_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def from_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "packet_id" in data or "topic" in data or "atom_ids" in data:
            packet_status = data.get("status", PacketStatus.active)
            if packet_status == "accepted":
                packet_status = PacketStatus.active
            return {
                "id": data.get("id") or data.get("packet_id", ""),
                "project_id": data.get("project_id", "unknown_project"),
                "family": data.get("family", PacketFamily.missing_info),
                "anchor_type": data.get("anchor_type", "entity"),
                "anchor_key": data.get("anchor_key") or data.get("topic", "unknown"),
                "governing_atom_ids": data.get("governing_atom_ids", []),
                "supporting_atom_ids": data.get("supporting_atom_ids", []),
                "contradicting_atom_ids": data.get("contradicting_atom_ids", []),
                "related_edge_ids": data.get("related_edge_ids", []),
                "confidence": data.get("confidence", 0.5),
                "status": packet_status,
                "reason": data.get("reason", "legacy packet"),
                "review_flags": data.get("review_flags", []),
                "anchor_signature": data.get("anchor_signature"),
                "certificate": data.get("certificate"),
                "risk": data.get("risk"),
                "packet_id": data.get("packet_id"),
                "topic": data.get("topic"),
                "atom_ids": data.get("atom_ids", []),
            }
        return data

    @model_validator(mode="after")
    def validate_governing_atoms(self) -> "EvidencePacket":
        if self.status in {PacketStatus.active, PacketStatus.needs_review} and len(self.governing_atom_ids) < 1:
            allow_vendor_pollution_orphan = (
                self.family == PacketFamily.scope_exclusion
                and self.review_flags
                and "vendor_scope_pollution_candidate" in self.review_flags
                and "power_vendor_scope_mismatch" not in self.review_flags
            )
            if not allow_vendor_pollution_orphan:
                raise ValueError("Active or needs_review packets require governing_atom_ids")
        if not self.packet_id:
            self.packet_id = self.id
        if not self.topic:
            self.topic = self.anchor_key
        return self


class ParsedClaim(BaseModel):
    artifact_id: str
    parser: str
    path: str
    claim: str
    entity: str
    predicate: str
    value: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    location: str | None = None
    snippet: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class CompileResult(BaseModel):
    project_id: str
    schema_version: str = SCHEMA_VERSION
    compiler_version: str = COMPILER_VERSION
    compile_id: str = ""
    atoms: list[EvidenceAtom]
    entities: list[EntityRecord]
    edges: list[EvidenceEdge]
    packets: list[EvidencePacket]
    warnings: list[str] = Field(default_factory=list)
    manifest: "CompileManifest | None" = None
    trace: "CompileTrace | None" = None
    candidate_summary: CandidateSummary | None = None
    # Compatibility fields used by current MVP pipeline/tests.
    project_dir: str | None = None
    ranked_atoms: list[AuthorityRankedAtom] = Field(default_factory=list)
    entity_edges: list[EntityEdge] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def from_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "project_dir" in data or "ranked_atoms" in data or "entity_edges" in data:
            return {
                "project_id": data.get("project_id") or data.get("project_dir", "unknown_project"),
                "schema_version": data.get("schema_version", SCHEMA_VERSION),
                "compiler_version": data.get("compiler_version", COMPILER_VERSION),
                "compile_id": data.get("compile_id", ""),
                "atoms": data.get("atoms", []),
                "entities": data.get("entities", []),
                "edges": data.get("edges", []),
                "packets": data.get("packets", []),
                "warnings": data.get("warnings", []),
                "manifest": data.get("manifest"),
                "trace": data.get("trace"),
                "candidate_summary": data.get("candidate_summary"),
                "project_dir": data.get("project_dir"),
                "ranked_atoms": data.get("ranked_atoms", []),
                "entity_edges": data.get("entity_edges", []),
            }
        return data


class ArtifactFingerprint(BaseModel):
    artifact_id: str
    filename: str
    artifact_type: ArtifactType
    sha256: str
    size_bytes: int
    modified_time: str | None = None
    parser_name: str
    parser_version: str


class CompileManifest(BaseModel):
    compile_id: str
    project_id: str
    schema_version: str = SCHEMA_VERSION
    compiler_version: str = COMPILER_VERSION
    packetizer_version: str = PACKETIZER_VERSION
    authority_policy_version: str = AUTHORITY_POLICY_VERSION
    artifact_fingerprints: list[ArtifactFingerprint] = Field(default_factory=list)
    parser_versions: dict[str, str] = Field(default_factory=dict)
    started_at: str
    completed_at: str | None = None
    deterministic_seed: str
    input_signature: str
    output_signature: str | None = None
    domain_pack_id: str | None = None
    domain_pack_version: str | None = None
    parser_routing: list[dict[str, Any]] = Field(default_factory=list)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    reused_artifact_ids: list[str] = Field(default_factory=list)


class CompileStageTrace(BaseModel):
    stage_name: str
    started_at: str
    completed_at: str
    duration_ms: float = Field(ge=0.0)
    input_count: int | None = None
    output_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CompileTrace(BaseModel):
    compile_id: str
    project_id: str
    stages: list[CompileStageTrace] = Field(default_factory=list)
    total_duration_ms: float = Field(ge=0.0)
    artifact_count: int = Field(ge=0)
    atom_count: int = Field(ge=0)
    entity_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    packet_count: int = Field(ge=0)
    parser_atom_counts: dict[str, int] = Field(default_factory=dict)
    packet_family_counts: dict[str, int] = Field(default_factory=dict)
    parser_routing: list[dict[str, Any]] = Field(default_factory=list)


class AtomDiff(BaseModel):
    atom_id: str
    change_type: Literal["added", "removed", "changed", "unchanged"]
    before_hash: str | None = None
    after_hash: str | None = None
    reason: str


class PacketDiff(BaseModel):
    packet_id: str
    change_type: Literal["added", "removed", "changed", "unchanged", "invalidated"]
    before_status: str | None = None
    after_status: str | None = None
    affected_atom_ids: list[str] = Field(default_factory=list)
    reason: str


class CompileDiff(BaseModel):
    before_compile_id: str
    after_compile_id: str
    atom_diffs: list[AtomDiff] = Field(default_factory=list)
    packet_diffs: list[PacketDiff] = Field(default_factory=list)
    invalidated_packet_ids: list[str] = Field(default_factory=list)
    blast_radius_summary: dict[str, Any] = Field(default_factory=dict)


CompileResult.model_rebuild()
