from __future__ import annotations

import json
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
from app.core.entity_extraction import enrich_atoms as enrich_entity_keys
from app.core.entity_resolution import (
    collect_site_alias_groups,
    extract_entity_records,
    fuse_alias_groups,
    resolve_aliases,
)
from app.core.quality_metrics import compute_quality
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
from app.core.schemas import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    CandidateAtom,
    CompileResult,
    ParserDerivedFile,
    ParserOutput,
)
from app.core.source_replay import attach_receipts_to_atoms, summarize_receipts
from app.core.telemetry import CompileTelemetry
from app.core.validators import validate_compile_result
from app.domain import load_domain_pack, set_active_domain_pack
from app.domain.pack_router import auto_route_pack
from app.domain.schemas import DomainPack
from app.learning.calibration import apply_calibration
from app.parsers.parser_router import choose_parser

# Atoms below this confidence are forced to needs_review with a stable flag.
# Anything below the floor is too uncertain to govern a packet without a human
# look — keep them in the result for transparency, but never let them ride into
# active packets unchallenged.
LOW_CONFIDENCE_FLOOR = 0.50

_DERIVED_DIR_SUFFIXES = (".derived",)

# Directory names that should never be walked for input artifacts.
# These are project metadata / outputs from previous compiles, not
# scope content.  See PRODUCTION_GAPS.md P1.6.
_NON_ARTIFACT_DIRS = frozenset(
    {
        "labels",          # gold standards / human-curated review labels
        ".orbitbrief",     # OrbitBrief envelope outputs from prior compiles
        ".cache",          # generic cache dir
        ".git",            # vcs metadata
        ".github",
        ".vscode",
        ".idea",
        "node_modules",
        "__pycache__",
    }
)

# File names (case-insensitive) that should never be parsed as artifacts.
# These are project metadata or known output sentinels.
_NON_ARTIFACT_FILES = frozenset(
    {
        "source_notes.md",       # case-level provenance notes
        "readme.md",             # project README
        "license",
        "license.md",
        "license.txt",
        ".gitignore",
        ".gitattributes",
        "thumbs.db",
        ".ds_store",
        "project.yaml",          # parser-os project config (read separately)
        "project.yml",
        ".parserignore",         # ignore-pattern list
    }
)

# File-name patterns (case-insensitive substring) that mark gold/review
# files which must never be parsed as scope content.
_NON_ARTIFACT_PATTERNS = (
    "gold_standard",
    ".gold.",
    "_gold.",
    "_review.",
    ".review.",
)


def _materialize_derived_files(
    artifact: Path,
    derived_files: list[ParserDerivedFile],
) -> None:
    """Write parser-emitted derived files next to ``artifact``.

    ``relative_path`` is interpreted relative to the artifact's parent
    directory so a parser can write into ``<stem>.derived/...`` or any
    other sibling location.  Path traversal is rejected up-front to
    keep the cache safe (a malicious cache entry can't escape the
    project directory).
    """
    base = artifact.parent.resolve()
    for entry in derived_files:
        rel = (entry.relative_path or "").strip()
        if not rel:
            continue
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            # Path tried to escape the artifact directory — skip.
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.content_kind == "json":
            target.write_text(
                json.dumps(entry.content_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        elif entry.content_kind in {"markdown", "text"}:
            target.write_text(entry.content_text or "", encoding="utf-8")


def _is_derived_path(path: Path, project_dir: Path) -> bool:
    """Return True when ``path`` is inside a parser-managed derived dir.

    Parsers (e.g. orbitbrief_pdf) materialize structured artifacts into
    sibling ``<stem>.derived/`` directories next to their source.  These
    are downstream consumer outputs (OrbitBrief input format), not
    inputs to compile — skip them so they don't get re-routed as
    unknown ``.json`` artifacts on the next compile pass.
    """
    try:
        rel = path.relative_to(project_dir)
    except ValueError:
        return False
    return any(
        part.endswith(_DERIVED_DIR_SUFFIXES) for part in rel.parts[:-1]
    )


def _is_excluded_artifact(path: Path, project_dir: Path) -> bool:
    """Return True when ``path`` should never be parsed as an artifact.

    Excludes project metadata (`labels/`, `SOURCE_NOTES.md`,
    `project.yaml`), VCS / IDE dirs, and gold-standard files that
    accompany the corpus but aren't scope content.  See PRODUCTION_GAPS
    P1.6.
    """
    try:
        rel = path.relative_to(project_dir)
    except ValueError:
        return False
    parts = rel.parts
    # Any ancestor directory in the no-walk list?
    for part in parts[:-1]:
        if part.lower() in _NON_ARTIFACT_DIRS:
            return True
    name_lower = path.name.lower()
    if name_lower in _NON_ARTIFACT_FILES:
        return True
    for pattern in _NON_ARTIFACT_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def _read_parserignore(project_dir: Path) -> list[str]:
    """Read ``<project>/.parserignore`` glob patterns if present.

    Returns lowercased glob patterns; ``#`` comments and blank lines
    are skipped.  Honors ``project.yaml``'s ``parserignore_extra`` too
    so operators can keep ignore rules in one config file.
    """
    patterns: list[str] = []
    ignore_path = project_dir / ".parserignore"
    if ignore_path.is_file():
        try:
            for line in ignore_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                patterns.append(stripped.lower())
        except Exception:  # pragma: no cover — never fail compile on ignore read
            pass
    # project.yaml extras — silently merged so a missing /.parserignore
    # doesn't matter.
    try:
        from app.domain.project_config import load_project_config

        cfg = load_project_config(project_dir)
        if cfg is not None and cfg.parserignore_extra:
            patterns.extend(p.strip().lower() for p in cfg.parserignore_extra if p and p.strip())
    except Exception:  # pragma: no cover — config load errors shouldn't kill compile
        pass
    return patterns


def _matches_ignore_pattern(rel_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    from fnmatch import fnmatch

    rel_lower = rel_path.replace("\\", "/").lower()
    for pattern in patterns:
        if fnmatch(rel_lower, pattern) or fnmatch("/" + rel_lower, pattern):
            return True
    return False


def _iter_artifacts(project_dir: Path) -> list[Path]:
    ignore_patterns = _read_parserignore(project_dir)
    results: list[Path] = []
    # Prefer a dedicated `artifacts/` subdir if one exists — this is the
    # canonical "this is real scope content" convention used across the
    # STRESS_* corpus.  When present, walk only that subtree.
    artifacts_dir = project_dir / "artifacts"
    if artifacts_dir.is_dir():
        walk_root = artifacts_dir
    else:
        walk_root = project_dir
    for path in walk_root.rglob("*"):
        if not path.is_file():
            continue
        if _is_derived_path(path, project_dir):
            continue
        if _is_excluded_artifact(path, project_dir):
            continue
        try:
            rel = str(path.relative_to(project_dir))
        except ValueError:
            rel = path.name
        if _matches_ignore_pattern(rel, ignore_patterns):
            continue
        results.append(path)
    return sorted(results, key=lambda p: str(p).lower())


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
    if isinstance(domain_pack, DomainPack):
        # Pre-loaded pack from caller wins outright (e.g. tests)
        resolved_domain_pack = domain_pack
        pack_routing_decision = None
    else:
        # Pack auto-routing: explicit `--domain-pack` overrides; otherwise
        # we look at project.yaml → SOURCE_NOTES.md → content scoring.
        # See PRODUCTION_GAPS.md P0.1.
        resolved_domain_pack, pack_routing_decision = auto_route_pack(
            project_dir, explicit=domain_pack
        )
    set_active_domain_pack(resolved_domain_pack)
    telemetry = CompileTelemetry(project_id=resolved_project_id)
    warnings: list[str] = []
    if pack_routing_decision is not None:
        warnings.append(
            f"INFO: domain pack '{resolved_domain_pack.pack_id}' selected via "
            f"{pack_routing_decision.source} ({pack_routing_decision.rationale})"
        )
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
                    parsed_derived_files: list[ParserDerivedFile] = []
                    if cached is not None:
                        parsed_atoms = list(cached.atoms)
                        parsed_candidates = list(cached.candidates)
                        per_artifact_warnings.extend(cached.warnings)
                        parsed_derived_files = list(cached.derived_files)
                        cache_hits += 1
                        reused_artifact_ids.append(artifact_id)
                        cache_hit = True
                    else:
                        parser_result = parser.parse_artifact_full(
                            project_id=resolved_project_id,
                            artifact_id=artifact_id,
                            path=artifact,
                            domain_pack=resolved_domain_pack,
                        )
                        parsed_atoms = list(parser_result.atoms)
                        parsed_candidates = list(parser_result.candidates)
                        per_artifact_warnings.extend(parser_result.warnings)
                        parsed_derived_files = list(parser_result.derived_files)
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
                                    derived_files=parsed_derived_files,
                                )
                            )
                        cache_misses += 1
                    # Materialize parser-emitted derived files next to
                    # the source artifact on every pass — cache or no
                    # cache.  This is what keeps OrbitBrief PDF
                    # ``structured.json`` projections in lock-step with
                    # the cached atom set.
                    if parsed_derived_files:
                        _materialize_derived_files(artifact, parsed_derived_files)
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

    # Hardening: enforce a confidence floor so atoms whose extractor was very
    # uncertain can't quietly govern packets.  We intentionally don't drop the
    # atoms — OrbitBrief still benefits from seeing them — we just refuse to
    # trust them without a human in the loop.
    floor_warnings: list[str] = []
    with telemetry.stage("confidence_floor", input_count=len(atoms)) as stage:
        floored = 0
        from app.core.schemas import ReviewStatus  # local import keeps top of file tidy
        for atom in atoms:
            if atom.confidence < LOW_CONFIDENCE_FLOOR:
                if atom.review_status != ReviewStatus.needs_review:
                    atom.review_status = ReviewStatus.needs_review
                if "low_confidence_floor" not in atom.review_flags:
                    atom.review_flags = sorted(set(atom.review_flags + ["low_confidence_floor"]))
                floored += 1
        if floored:
            floor_warnings.append(
                f"WARNING: {floored} atoms below confidence floor {LOW_CONFIDENCE_FLOOR:.2f} forced to needs_review"
            )
            warnings.extend(floor_warnings)
        telemetry.end_stage(stage, output_count=floored, warnings=floor_warnings)

    enrich_warnings: list[str] = []
    with telemetry.stage("enrich_entities", input_count=len(atoms)) as stage:
        # Universal entity extraction — populates atom.entity_keys for any
        # atom whose parser hardcoded an empty list.  Without this, the
        # downstream graph_builder anchors land on `device:unknown` and
        # quantity_conflict edges never form.  See PRODUCTION_GAPS.md P0.2.
        atoms_enriched, keys_added = enrich_entity_keys(atoms, resolved_domain_pack)
        if atoms_enriched:
            enrich_warnings.append(
                f"INFO: enriched {atoms_enriched} atoms with {keys_added} entity keys "
                f"(parser-supplied entity_keys preserved)"
            )
        telemetry.end_stage(stage, output_count=atoms_enriched, warnings=enrich_warnings)
    warnings.extend(enrich_warnings)

    with telemetry.stage("entity_resolution", input_count=len(atoms)) as stage:
        entities = resolve_aliases(
            extract_entity_records(resolved_project_id, atoms, pack=resolved_domain_pack)
        )
        # Cross-mention alias fusion: collapse `site:atl_hq +
        # site:atlanta_headquarters + site:innovation_tower` (three
        # surface names for one physical place) into a single
        # EntityRecord whose `aliases` field carries all three keys.
        # Detected via co-mention patterns in atom raw_text (copular
        # "is the", em-dash, slash, parenthetical aliasing, ...).
        site_alias_groups = collect_site_alias_groups(atoms)
        entities = fuse_alias_groups(entities, site_alias_groups)
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
    # Finalize the manifest (including output_signature) BEFORE quality gates so
    # the validator doesn't fire a spurious "missing output_signature" warning
    # on every compile.  output_signature is content-addressed over the result
    # pre-validation; validation messages are excluded from the signature so a
    # warning later doesn't recursively change the signature.
    output_signature = compute_output_signature(result)
    result.manifest = finalize_manifest(manifest, output_signature)
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

    # PRODUCTION_GAPS P3.4 / P3.5: compute quality metrics + fail-loud
    # warnings.  Metrics are deterministic over the finalized result so
    # they're safe to surface in the JSON output and as telemetry.
    routing_source = "unknown"
    routing_confidence_value = 0.0
    if pack_routing_decision is not None:
        routing_source = pack_routing_decision.source
        routing_confidence_value = pack_routing_decision.confidence
    result.quality = compute_quality(
        result,
        pack_routing_source=routing_source,
        pack_routing_confidence=routing_confidence_value,
    )

    # Fail-loud: a parser that successfully routed a file but produced
    # zero atoms is a regression signal (XLSX header detection bug,
    # PDF table extraction failure, etc.).  We surface a clear ERROR-
    # adjacent warning so operators see it without diffing JSON.
    if result.quality.parsers_with_zero_atoms:
        names = ", ".join(result.quality.parsers_with_zero_atoms[:5])
        suffix = (
            f" (and {len(result.quality.parsers_with_zero_atoms) - 5} more)"
            if len(result.quality.parsers_with_zero_atoms) > 5 else ""
        )
        result.warnings = sorted(
            set(
                result.warnings
                + [f"WARNING: parser produced 0 atoms for: {names}{suffix}"]
            )
        )
    if result.quality.entity_resolution_rate < 0.30 and result.quality.atom_count >= 20:
        result.warnings = sorted(
            set(
                result.warnings
                + [
                    f"WARNING: low entity_resolution_rate "
                    f"({result.quality.entity_resolution_rate:.2f}); "
                    "atoms aren't getting entity_keys — review pack vocabulary"
                ]
            )
        )
    if result.quality.packet_specificity < 0.50 and result.quality.packet_count >= 5:
        result.warnings = sorted(
            set(
                result.warnings
                + [
                    f"WARNING: low packet_specificity "
                    f"({result.quality.packet_specificity:.2f}); "
                    "many packets anchor on `*:unknown` — review entity extraction"
                ]
            )
        )

    return result
