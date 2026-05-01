from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from app.core.cache import (
    build_cached_artifact_result,
    compute_artifact_sha256,
    load_cached_artifact_result,
    save_cached_artifact_result,
)
from app.core.candidate_adjudicator import adjudicate_candidates
from app.core.candidates import summarize_candidate_outcomes
from app.core.entity_resolution import extract_entity_records, resolve_aliases
from app.core.graph_builder import build_edges
from app.core.ids import stable_id
from app.core.manifest import (
    build_artifact_fingerprint,
    compute_output_signature,
    create_manifest,
    finalize_manifest,
)
from app.core.packet_certificates import build_packet_certificate
from app.core.packetizer import build_packets
from app.core.risk import packet_pm_sort_key, score_packet_risk
from app.core.schemas import COMPILER_VERSION, SCHEMA_VERSION, CandidateAtom, CompileResult, ParserOutput
from app.core.source_replay import attach_receipts_to_atoms, summarize_receipts
from app.core.telemetry import CompileTelemetry
from app.core.validators import validate_compile_result
from app.domain import load_domain_pack, set_active_domain_pack
from app.domain.schemas import DomainPack
from app.learning.calibration import apply_calibration
from app.parsers.parser_router import choose_parser


def _iter_artifacts(project_dir: Path) -> list[Path]:
    return sorted([p for p in project_dir.rglob("*") if p.is_file()], key=lambda p: str(p).lower())


def compile_project(
    project_dir: Path,
    project_id: str | None = None,
    allow_errors: bool = False,
    allow_unverified_receipts: bool = False,
    persistence_hook: Callable[[CompileResult], None] | None = None,
    domain_pack: DomainPack | str | Path | None = None,
    calibrator_path: Path | None = None,
    abstain_threshold: float = 0.70,
    use_cache: bool = True,
) -> CompileResult:
    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_dir}")

    resolved_project_id = project_id or project_dir.name
    resolved_domain_pack = domain_pack if isinstance(domain_pack, DomainPack) else load_domain_pack(domain_pack)
    set_active_domain_pack(resolved_domain_pack)
    telemetry = CompileTelemetry(project_id=resolved_project_id)
    warnings: list[str] = []
    if resolved_domain_pack.reference_ontology_path:
        warnings.append(
            "WARNING: Domain pack uses reference-schema subset adapter (TODO: strict DomainPack mapper); "
            f"bundled ontology: {resolved_domain_pack.reference_ontology_path}"
        )
    atoms = []
    candidates: list[CandidateAtom] = []
    rejected_candidates: list[CandidateAtom] = []
    fingerprints = []
    parser_atom_counts: Counter[str] = Counter()
    parser_routing: list[dict] = []
    artifact_paths: dict[str, Path] = {}
    cache_hits = 0
    cache_misses = 0
    reused_artifact_ids: list[str] = []

    with telemetry.stage("discover_artifacts", input_count=1) as stage:
        artifacts = _iter_artifacts(project_dir)
        telemetry.end_stage(stage, output_count=len(artifacts))

    parse_warnings: list[str] = []
    parse_errors: list[str] = []
    with telemetry.stage("parse_artifacts", input_count=len(artifacts)) as stage:
        for artifact in artifacts:
            relative_name = str(artifact.relative_to(project_dir)).replace("\\", "/")
            artifact_id = stable_id("art", resolved_project_id, relative_name)
            artifact_paths[artifact_id] = artifact
            parsed_atoms = []
            parsed_candidates: list[CandidateAtom] = []
            per_artifact_warnings: list[str] = []
            parser_key = "none"
            parser_name = "none"
            parser_version = "unknown"
            cache_hit = False
            artifact_sha256 = compute_artifact_sha256(artifact)
            try:
                parser, match, all_matches = choose_parser(
                    path=artifact,
                    domain_pack=resolved_domain_pack,
                )
                parser_name = match.parser_name
                parser_version = parser.capability.parser_version if parser is not None else "unknown"
                parser_routing.append(
                    {
                        "artifact_id": artifact_id,
                        "filename": relative_name,
                        "chosen_parser": parser_name,
                        "parser_version": parser_version,
                        "confidence": match.confidence,
                        "reasons": match.reasons,
                        "cache_hit": False,
                        "matches": [row.model_dump(mode="json") for row in all_matches],
                    }
                )
                parser_key = parser_name
                if parser is None:
                    warning = f"WARNING: No parser matched artifact {relative_name}; skipping file"
                    parse_warnings.append(warning)
                else:
                    cached = None
                    if use_cache:
                        cached = load_cached_artifact_result(
                            artifact_id=artifact_id,
                            sha256=artifact_sha256,
                            parser_name=parser_name,
                            parser_version=parser_version,
                            domain_pack_id=resolved_domain_pack.pack_id,
                            domain_pack_version=resolved_domain_pack.version,
                        )
                    if cached is not None:
                        parsed_atoms = list(cached.atoms)
                        parsed_candidates = list(cached.candidates)
                        per_artifact_warnings.extend(cached.warnings)
                        cache_hits += 1
                        reused_artifact_ids.append(artifact_id)
                        cache_hit = True
                    else:
                        parser_result = parser.parse_artifact(
                            project_id=resolved_project_id,
                            artifact_id=artifact_id,
                            path=artifact,
                            domain_pack=resolved_domain_pack,
                        )
                        if isinstance(parser_result, ParserOutput):
                            parsed_atoms = list(parser_result.atoms)
                            parsed_candidates = list(parser_result.candidates)
                            per_artifact_warnings.extend(parser_result.warnings)
                        else:
                            parsed_atoms = list(parser_result)
                        if use_cache:
                            save_cached_artifact_result(
                                build_cached_artifact_result(
                                    artifact_id=artifact_id,
                                    sha256=artifact_sha256,
                                    parser_name=parser_name,
                                    parser_version=parser_version,
                                    domain_pack_id=resolved_domain_pack.pack_id,
                                    domain_pack_version=resolved_domain_pack.version,
                                    candidates=parsed_candidates,
                                    atoms=parsed_atoms,
                                    warnings=per_artifact_warnings,
                                )
                            )
                        cache_misses += 1
                    if parser_routing:
                        parser_routing[-1]["cache_hit"] = cache_hit
                    candidates.extend(parsed_candidates)
                    parse_warnings.extend(per_artifact_warnings)
                    atoms.extend(parsed_atoms)
                    parser_atom_counts[parser_key] += len(parsed_atoms)
            except Exception as exc:  # pragma: no cover
                message = f"Failed parsing {artifact.name} ({parser_key}): {exc}"
                parse_warnings.append(message)
                parse_errors.append(message)
                cache_misses += 1
            fingerprints.append(
                build_artifact_fingerprint(
                    artifact,
                    artifact_id,
                    parsed_atoms,
                    filename=relative_name,
                    parser_name=parser_name,
                    parser_version=parser_version,
                )
            )
        warnings.extend(parse_warnings)
        telemetry.end_stage(
            stage,
            output_count=len(atoms),
            warnings=parse_warnings,
            errors=parse_errors,
        )

    with telemetry.stage("candidate_adjudication", input_count=len(candidates)) as stage:
        adjudication = adjudicate_candidates(candidates, artifact_paths)
        atoms.extend(adjudication.accepted_atoms)
        rejected_candidates = adjudication.rejected_candidates
        warnings.extend(adjudication.warnings)
        telemetry.end_stage(
            stage,
            output_count=len(adjudication.accepted_atoms),
            warnings=adjudication.warnings,
            errors=[],
        )

    manifest = create_manifest(resolved_project_id, fingerprints, domain_pack=resolved_domain_pack)
    manifest.parser_routing = sorted(parser_routing, key=lambda row: row["artifact_id"])
    manifest.cache_hits = cache_hits if use_cache else 0
    manifest.cache_misses = cache_misses if use_cache else len(artifacts)
    manifest.reused_artifact_ids = sorted(set(reused_artifact_ids)) if use_cache else []
    telemetry.set_compile_id(manifest.compile_id)

    replay_warnings: list[str] = []
    with telemetry.stage("source_replay", input_count=len(atoms)) as stage:
        atoms = attach_receipts_to_atoms(atoms, artifact_paths)
        receipt_summary = summarize_receipts(atoms)
        if receipt_summary["unsupported"] > 0:
            replay_warnings.append(
                f"WARNING: {receipt_summary['unsupported']} source receipts are unsupported and require manual audit"
            )
            warnings.extend(replay_warnings)
        telemetry.end_stage(stage, output_count=len(atoms), warnings=replay_warnings)

    with telemetry.stage("entity_resolution", input_count=len(atoms)) as stage:
        entities = resolve_aliases(extract_entity_records(resolved_project_id, atoms))
        telemetry.end_stage(stage, output_count=len(entities))

    with telemetry.stage("graph_build", input_count=len(atoms)) as stage:
        edges = build_edges(project_id=resolved_project_id, atoms=atoms, entities=entities)
        telemetry.end_stage(stage, output_count=len(edges))

    with telemetry.stage("packetize", input_count=len(edges)) as stage:
        packets = build_packets(
            project_id=resolved_project_id,
            atoms=atoms,
            entities=entities,
            edges=edges,
            attach_metadata=False,
        )
        telemetry.end_stage(stage, output_count=len(packets))

    with telemetry.stage("packet_certificates", input_count=len(packets)) as stage:
        atom_by_id = {atom.id: atom for atom in atoms}
        edge_by_id = {edge.id: edge for edge in edges}
        for packet in packets:
            packet.certificate = build_packet_certificate(packet, atom_by_id, edge_by_id=edge_by_id)
            packet_atoms = [
                atom_by_id[atom_id]
                for atom_id in (packet.supporting_atom_ids + packet.contradicting_atom_ids)
                if atom_id in atom_by_id
            ]
            packet.risk = score_packet_risk(packet, packet_atoms, edges)
        telemetry.end_stage(stage, output_count=len(packets))

    result = CompileResult(
        project_id=resolved_project_id,
        atoms=atoms,
        entities=entities,
        edges=edges,
        packets=packets,
        warnings=sorted(warnings),
        schema_version=SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
        compile_id=manifest.compile_id,
        manifest=manifest,
        project_dir=str(project_dir),
        ranked_atoms=[],
        entity_edges=[],
        candidate_summary=summarize_candidate_outcomes(
            candidates=candidates,
            accepted_atoms=adjudication.accepted_atoms if candidates else [],
            rejected_candidates=rejected_candidates,
        ),
    )
    result.atoms = sorted(result.atoms, key=lambda x: x.id)
    result.entities = sorted(result.entities, key=lambda x: x.id)
    result.edges = sorted(result.edges, key=lambda x: x.id)
    result.packets = sorted(
        result.packets,
        key=lambda p: packet_pm_sort_key(p) if p.risk is not None else (50, 50, 0.0, p.anchor_key, p.id),
    )
    if calibrator_path is not None:
        with telemetry.stage("confidence_calibration", input_count=len(result.packets)) as stage:
            try:
                result = apply_calibration(result, calibrator_path, abstain_threshold=abstain_threshold)
                telemetry.end_stage(stage, output_count=len(result.packets))
            except Exception as exc:
                warning = f"WARNING: Failed to apply calibrator {calibrator_path}: {exc}"
                result.warnings = sorted(set(result.warnings + [warning]))
                telemetry.end_stage(stage, output_count=len(result.packets), warnings=[warning], errors=[])
    with telemetry.stage("quality_gates", input_count=len(result.packets)) as stage:
        validation_messages = validate_compile_result(result, source_files_available=True)
        hard_errors = [m for m in validation_messages if m.startswith("ERROR:")]
        validation_warnings = [m for m in validation_messages if m.startswith("WARNING:")]
        telemetry.end_stage(
            stage,
            output_count=len(validation_messages),
            warnings=validation_warnings,
            errors=hard_errors,
        )
    if allow_unverified_receipts:
        receipt_hard_errors = [m for m in hard_errors if "receipt" in m.lower()]
        if receipt_hard_errors:
            validation_warnings.extend(
                [f"WARNING: downgraded receipt validation under --allow-unverified-receipts: {m}" for m in receipt_hard_errors]
            )
        hard_errors = [m for m in hard_errors if "receipt" not in m.lower()]
    result.warnings = sorted(set(result.warnings + validation_warnings))
    output_signature = compute_output_signature(result)
    result.manifest = finalize_manifest(manifest, output_signature)

    if persistence_hook is not None:
        with telemetry.stage("persistence", input_count=1) as stage:
            persistence_hook(result)
            telemetry.end_stage(stage, output_count=1)

    packet_family_counts = Counter(packet.family.value for packet in result.packets)
    result.trace = telemetry.build_trace(
        artifact_count=len(artifacts),
        atom_count=len(result.atoms),
        entity_count=len(result.entities),
        edge_count=len(result.edges),
        packet_count=len(result.packets),
        parser_atom_counts=dict(parser_atom_counts),
        packet_family_counts=dict(packet_family_counts),
        parser_routing=manifest.parser_routing,
    )

    if hard_errors and not allow_errors:
        raise ValueError("Compile validation failed:\n" + "\n".join(hard_errors))
    if hard_errors and allow_errors:
        result.warnings = sorted(set(result.warnings + hard_errors))
    return result
