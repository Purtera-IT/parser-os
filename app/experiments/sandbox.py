from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.cache import compute_artifact_sha256
from app.core.candidate_adjudicator import adjudicate_candidates
from app.core.diffing import diff_compile_results
from app.core.entity_resolution import extract_entity_records, resolve_aliases
from app.core.graph_builder import build_edges
from app.core.ids import stable_id
from app.core.packet_certificates import build_packet_certificate
from app.core.packetizer import build_packets
from app.core.risk import score_packet_risk
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CandidateAtom,
    CompileResult,
    ReviewStatus,
)
from app.core.source_replay import attach_receipts_to_atoms
from app.core.telemetry import CompileTelemetry
from app.domain import get_active_domain_pack, load_domain_pack, set_active_domain_pack
from app.parsers.segmenters import segment_docx, segment_email, segment_quote, segment_text, segment_transcript, segment_xlsx
from app.semantic.linker import propose_semantic_link_candidates
from app.core.compiler import compile_project


class ExperimentDelta(BaseModel):
    new_candidates: int = 0
    new_atoms_if_accepted: int = 0
    new_packets_if_accepted: int = 0
    changed_packets_if_accepted: int = 0


class ExperimentRun(BaseModel):
    experiment_id: str
    compile_id: str
    extractor_name: str
    extractor_version: str
    domain_pack_id: str
    candidate_count: int
    accepted_count_if_adjudicated: int
    rejected_count_if_adjudicated: int
    delta_vs_baseline: ExperimentDelta
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _iter_artifacts(project_dir: Path) -> list[Path]:
    return sorted([p for p in project_dir.rglob("*") if p.is_file()], key=lambda p: str(p).lower())


def _artifact_id(project_id: str, project_dir: Path, artifact_path: Path) -> str:
    relative_name = str(artifact_path.relative_to(project_dir)).replace("\\", "/")
    return stable_id("art", project_id, relative_name)


def _segments_for_artifact(project_id: str, artifact_id: str, path: Path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".csv"}:
        if any(token in name for token in ("quote", "vendor", "po")):
            return segment_quote(project_id=project_id, artifact_id=artifact_id, path=path)
        return segment_xlsx(project_id=project_id, artifact_id=artifact_id, path=path)
    if suffix == ".docx":
        return segment_docx(project_id=project_id, artifact_id=artifact_id, path=path)
    if suffix == ".eml":
        return segment_email(project_id=project_id, artifact_id=artifact_id, path=path)
    if suffix in {".txt", ".md", ".vtt", ".srt"}:
        if "transcript" in name or "meeting" in name or "kickoff" in name:
            return segment_transcript(project_id=project_id, artifact_id=artifact_id, path=path)
        if "email" in name:
            return segment_email(project_id=project_id, artifact_id=artifact_id, path=path)
        return segment_text(project_id=project_id, artifact_id=artifact_id, path=path)
    return []


def _build_experimental_candidates(
    *,
    extractor_name: str,
    extractor_version: str,
    baseline: CompileResult,
    segments_by_artifact: dict[str, list[Any]],
) -> list[CandidateAtom]:
    candidates: list[CandidateAtom] = []
    if extractor_name == "semantic_linker":
        semantic_links = propose_semantic_link_candidates(baseline.atoms)
        atom_by_id = {atom.id: atom for atom in baseline.atoms}
        for link in semantic_links:
            if link.status not in {"accepted", "needs_review"}:
                continue
            left = atom_by_id.get(link.from_atom_id)
            right = atom_by_id.get(link.to_atom_id)
            if left is None or right is None or not left.source_refs:
                continue
            source_ref = left.source_refs[0]
            text = f"semantic-link:{link.proposed_edge_type.value}:{left.raw_text} <-> {right.raw_text}"
            candidates.append(
                CandidateAtom(
                    id=stable_id("cand", baseline.project_id, left.artifact_id, link.id),
                    project_id=baseline.project_id,
                    artifact_id=left.artifact_id,
                    candidate_type=left.atom_type,
                    raw_text=text,
                    proposed_normalized_text=text.lower(),
                    proposed_value={"semantic_link_id": link.id, "edge_type": link.proposed_edge_type.value},
                    proposed_entity_keys=sorted(set(left.entity_keys + right.entity_keys)),
                    source_refs=[source_ref],
                    proposed_authority_class=AuthorityClass.machine_extractor,
                    extractor_name=extractor_name,
                    extractor_version=extractor_version,
                    extraction_method="semantic_candidate",
                    confidence=min(0.99, max(0.55, link.similarity_score)),
                    evidence_span=left.raw_text,
                    validation_status="pending",
                    validation_reasons=[],
                )
            )
    elif extractor_name == "llm_candidate_extractor":
        for artifact_id, segments in segments_by_artifact.items():
            for seg in segments:
                text = str(seg.text).strip()
                if len(text) < 12:
                    continue
                if "?" not in text and "open question" not in text.lower():
                    continue
                candidates.append(
                    CandidateAtom(
                        id=stable_id("cand", baseline.project_id, artifact_id, "llm", seg.id),
                        project_id=baseline.project_id,
                        artifact_id=artifact_id,
                        candidate_type=AtomType.open_question,
                        raw_text=text,
                        proposed_normalized_text=text.lower(),
                        proposed_value={"llm_stub": True},
                        proposed_entity_keys=[],
                        source_refs=[seg.source_ref],
                        proposed_authority_class=AuthorityClass.machine_extractor,
                        extractor_name=extractor_name,
                        extractor_version=extractor_version,
                        extraction_method="llm_candidate",
                        confidence=0.62,
                        evidence_span=text,
                        validation_status="pending",
                        validation_reasons=[],
                    )
                )
    elif extractor_name == "weak_supervision_rules":
        pack = get_active_domain_pack()
        patterns = [str(row).lower() for row in pack.exclusion_patterns]
        for artifact_id, segments in segments_by_artifact.items():
            for seg in segments:
                lowered = str(seg.text).lower()
                if not any(pattern in lowered for pattern in patterns):
                    continue
                candidates.append(
                    CandidateAtom(
                        id=stable_id("cand", baseline.project_id, artifact_id, "ws", seg.id),
                        project_id=baseline.project_id,
                        artifact_id=artifact_id,
                        candidate_type=AtomType.exclusion,
                        raw_text=str(seg.text),
                        proposed_normalized_text=str(seg.normalized_text),
                        proposed_value={"weak_supervision_rule": True},
                        proposed_entity_keys=[],
                        source_refs=[seg.source_ref],
                        proposed_authority_class=AuthorityClass.machine_extractor,
                        extractor_name=extractor_name,
                        extractor_version=extractor_version,
                        extraction_method="domain_pack_rule",
                        confidence=0.66,
                        evidence_span=str(seg.text),
                        validation_status="pending",
                        validation_reasons=[],
                    )
                )
    else:
        raise ValueError(f"Unsupported extractor: {extractor_name}")
    return sorted(candidates, key=lambda row: row.id)


def run_extraction_sandbox(
    *,
    project_dir: Path,
    extractor_name: Literal["semantic_linker", "llm_candidate_extractor", "weak_supervision_rules"],
    extractor_version: str = "exp_v1",
    domain_pack: str | Path | None = None,
) -> tuple[ExperimentRun, dict[str, Any]]:
    project_dir = project_dir.resolve()
    resolved_pack = load_domain_pack(domain_pack)
    set_active_domain_pack(resolved_pack)

    baseline = compile_project(
        project_dir=project_dir,
        project_id=project_dir.name,
        allow_errors=True,
        allow_unverified_receipts=True,
    )
    compile_id = baseline.compile_id

    segments_by_artifact: dict[str, list[Any]] = {}
    artifact_paths: dict[str, Path] = {}
    for artifact in _iter_artifacts(project_dir):
        artifact_id = _artifact_id(baseline.project_id, project_dir, artifact)
        artifact_paths[artifact_id] = artifact
        segments_by_artifact[artifact_id] = _segments_for_artifact(baseline.project_id, artifact_id, artifact)

    candidates = _build_experimental_candidates(
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        baseline=baseline,
        segments_by_artifact=segments_by_artifact,
    )
    adjudication = adjudicate_candidates(candidates, artifact_paths)

    hypothetical_atoms = sorted(
        list(baseline.atoms) + list(adjudication.accepted_atoms),
        key=lambda row: row.id,
    )
    hypothetical_atoms = attach_receipts_to_atoms(hypothetical_atoms, artifact_paths)
    entities = resolve_aliases(extract_entity_records(baseline.project_id, hypothetical_atoms))
    edges = build_edges(project_id=baseline.project_id, atoms=hypothetical_atoms, entities=entities)
    packets = build_packets(project_id=baseline.project_id, atoms=hypothetical_atoms, entities=entities, edges=edges, attach_metadata=False)
    atom_by_id = {atom.id: atom for atom in hypothetical_atoms}
    edge_by_id = {e.id: e for e in edges}
    for packet in packets:
        packet.certificate = build_packet_certificate(packet, atom_by_id, edge_by_id=edge_by_id)
        packet_atoms = [
            atom_by_id[atom_id]
            for atom_id in (packet.supporting_atom_ids + packet.contradicting_atom_ids)
            if atom_id in atom_by_id
        ]
        packet.risk = score_packet_risk(packet, packet_atoms, edges)
    hypothetical = CompileResult(
        project_id=baseline.project_id,
        schema_version=baseline.schema_version,
        compiler_version=baseline.compiler_version,
        compile_id=baseline.compile_id,
        atoms=hypothetical_atoms,
        entities=entities,
        edges=edges,
        packets=sorted(packets, key=lambda row: row.id),
        warnings=sorted(set(baseline.warnings + adjudication.warnings)),
        manifest=baseline.manifest,
        trace=baseline.trace,
        candidate_summary=baseline.candidate_summary,
        project_dir=baseline.project_dir,
        ranked_atoms=[],
        entity_edges=[],
    )
    delta = diff_compile_results(baseline, hypothetical)
    new_packet_count = len([row for row in delta.packet_diffs if row.change_type == "added"])
    changed_packet_count = len([row for row in delta.packet_diffs if row.change_type == "changed"])
    new_atom_ids = {atom.id for atom in adjudication.accepted_atoms if atom.id not in {row.id for row in baseline.atoms}}

    experiment_id = stable_id("exp", compile_id, extractor_name, extractor_version, resolved_pack.pack_id)
    run = ExperimentRun(
        experiment_id=experiment_id,
        compile_id=compile_id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        domain_pack_id=resolved_pack.pack_id,
        candidate_count=len(candidates),
        accepted_count_if_adjudicated=len(adjudication.accepted_atoms),
        rejected_count_if_adjudicated=len(adjudication.rejected_candidates),
        delta_vs_baseline=ExperimentDelta(
            new_candidates=len(candidates),
            new_atoms_if_accepted=len(new_atom_ids),
            new_packets_if_accepted=new_packet_count,
            changed_packets_if_accepted=changed_packet_count,
        ),
        metrics={
            "accept_rate": round(len(adjudication.accepted_atoms) / len(candidates), 6) if candidates else 0.0,
            "artifact_count": len(segments_by_artifact),
            "segment_count": sum(len(rows) for rows in segments_by_artifact.values()),
            "baseline_packet_count": len(baseline.packets),
            "hypothetical_packet_count": len(hypothetical.packets),
        },
        created_at=_now_iso(),
    )
    report_payload = {
        "experiment_run": run.model_dump(mode="json"),
        "baseline_compile_id": baseline.compile_id,
        "project_id": baseline.project_id,
        "domain_pack_id": resolved_pack.pack_id,
        "domain_pack_version": resolved_pack.version,
        "candidate_ids": [row.id for row in candidates],
        "accepted_candidate_atom_ids": [row.id for row in adjudication.accepted_atoms],
        "rejected_candidate_ids": [row.id for row in adjudication.rejected_candidates],
        "delta": delta.model_dump(mode="json"),
    }
    return run, report_payload
