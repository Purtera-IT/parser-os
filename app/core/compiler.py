from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

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
    collect_stakeholder_alias_groups,
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
    EvidenceAtom,
    ParserDerivedFile,
    ParserOutput,
)
from app.core.source_replay import attach_receipts_to_atoms, replay_atom_receipts, summarize_receipts
from app.core.suppression_ledger import (
    SUPPRESSION_FLAG_PREFIX,
    capture_suppressed,
    merge_suppressed,
)
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
        # long_write_text: Windows >260-char paths (deep _rerun/<uuid>/.../<sha>/
        # <long name>.derived/...) otherwise throw WinError 206 here AFTER the
        # parser did all the work, silently dropping the whole file's output.
        from app.core.longpath import long_write_text
        if entry.content_kind == "json":
            long_write_text(
                target,
                json.dumps(entry.content_json, indent=2, ensure_ascii=False),
            )
        elif entry.content_kind in {"markdown", "text"}:
            long_write_text(target, entry.content_text or "")


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


def project_census(project_dir: Path | str, atoms):
    """Build the deal-level content census and reconcile it against ``atoms``.

    The census is an inventory of every region of every modality in the deal's
    source files, read *independently* of the production parser. Reconciling it
    against the emitted atoms yields the never-detected loss set
    (``census.uncovered()``) — exactly the denominator
    :func:`app.core.complaint_router.route` needs for its NEEDS_EXTRACTOR
    bucket.

    It deliberately reuses :func:`_iter_artifacts`, so the census denominator is
    drawn from the *same* file set the compile consumed — no drift between what
    was compiled and what we check coverage against. Pure read; no LLM, no
    network; safe to call after any compile from ``(project_dir, result.atoms)``.
    Never raises — returns an empty census on any failure.
    """
    from app.parsers.census import reconciled_census

    try:
        root = Path(project_dir).resolve()
        return reconciled_census(
            _iter_artifacts(root), atoms, artifact=root.name
        )
    except Exception:  # pragma: no cover - census must never break a caller
        from app.core.content_census import ContentCensus

        return ContentCensus(artifact=str(project_dir))


# Opt-in feedback store. The store stays OFF by default (decide() is a
# transparent pass-through to the LLM, byte-identical to Phase 2) so the whole
# test suite is unaffected. Operations turn it on by pointing
# SOWSMITH_FEEDBACK_STORE_DB at a SQLite path; we wire it once per process,
# seed the built-in global corrections (PurTera self-address), and register it
# via decide.set_store. Any failure is swallowed — a store that won't load
# must never break a compile.
_FEEDBACK_STORE_WIRED = False


def _maybe_wire_feedback_store() -> None:
    global _FEEDBACK_STORE_WIRED
    if _FEEDBACK_STORE_WIRED:
        return
    import os as _os

    db_path = _os.environ.get("SOWSMITH_FEEDBACK_STORE_DB", "").strip()
    if not db_path:
        return
    _FEEDBACK_STORE_WIRED = True  # one attempt per process, success or not
    try:
        from app.core.decide import get_store, set_store
        from app.core.feedback_store import (
            FeedbackStore,
            seed_default_corrections,
        )

        if get_store() is not None:  # already wired (e.g. by a test/host)
            return
        store = FeedbackStore(db_path)
        seed_default_corrections(store)
        set_store(store)
    except Exception:  # pragma: no cover - store must never break a compile
        pass


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
    stage_callback: Callable[..., None] | None = None,
) -> CompileResult:
    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_dir}")

    # Opt-in: activate the feedback store iff SOWSMITH_FEEDBACK_STORE_DB is set.
    # No-op otherwise, so default compiles (and the test suite) are unchanged.
    _maybe_wire_feedback_store()

    resolved_project_id = project_id or project_dir.name

    # v44: expose project_dir name to per-pack domain extractors via env.
    # Used by app.core.exemplars.detect_domain_extras() to add domain-
    # specific exemplars (POS / ITAD / cabling / wireless / etc.).
    import os as _os
    _os.environ["SOWSMITH_PROJECT_DIR_NAME"] = project_dir.name
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
    telemetry = CompileTelemetry(
        project_id=resolved_project_id,
        on_stage_end=stage_callback,
    )
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
    # Retained-suppression ledger: atoms removed by a drop-stage are captured
    # here (flagged + reason-stamped) instead of vanishing, so omission
    # complaints stay localizable and every drop stays auditable. Pure sidecar
    # — never re-enters the accepted atom set.
    suppressed_atoms: list[EvidenceAtom] = []
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
                        # A6 graceful degradation: per-file outcome
                        # status. Defaults to pending; overwritten below
                        # when the parse succeeds, is skipped, or fails.
                        # PM_HANDOFF readers (and the systems engineer
                        # diffing successful vs failed files) depend on
                        # this being present on every routing entry.
                        "outcome": {
                            "status": "pending",
                            "atom_count": 0,
                            "warning_count": 0,
                        },
                    }
                )
                parser_key = parser_name
                if parser is None:
                    warning = f"WARNING: No parser matched artifact {relative_name}; skipping file"
                    parse_warnings.append(warning)
                    parser_routing[-1]["outcome"] = {
                        "status": "skipped_no_parser",
                        "reason": (
                            f"no parser matched (best candidate: {parser_name} "
                            f"@ confidence={match.confidence:.2f})"
                        ),
                        "atom_count": 0,
                        "warning_count": 0,
                    }
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
                    if parser_routing:
                        # Successful parse — record concrete outcome.
                        # Use ``ok`` when the parser produced ≥1 atom;
                        # ``ok_empty`` when it ran but produced none
                        # (e.g. an image-only PDF the parser skipped
                        # without erroring). PM_HANDOFF distinguishes
                        # so reviewers know whether a 0-atom file means
                        # "parser is healthy, just no content" vs "parser
                        # silently failed."
                        status = "ok" if len(parsed_atoms) > 0 else "ok_empty"
                        parser_routing[-1]["outcome"] = {
                            "status": status,
                            "atom_count": len(parsed_atoms),
                            "warning_count": len(per_artifact_warnings),
                            "cache_hit": cache_hit,
                        }
                        # Parse-coverage gate: a parser that ran without
                        # error but extracted ZERO atoms is the most
                        # dangerous failure mode — silent data loss. A
                        # scanned / image-only PDF (e.g. Notes.pdf that
                        # yields 0 text), an empty sheet, or a parser that
                        # quietly bailed all land here. Surface a hard
                        # warning so the reviewer knows an input contributed
                        # nothing, instead of the file vanishing without a
                        # trace. Universal: keys off atom_count, not file type.
                        if len(parsed_atoms) == 0:
                            parse_warnings.append(
                                f"WARNING: artifact '{relative_name}' parsed cleanly "
                                f"with {parser_name} but yielded 0 atoms — no content "
                                f"extracted (possible scanned/image-only PDF, empty "
                                f"file, or unextractable layout). This input "
                                f"contributed nothing to the compile."
                            )
            except Exception as exc:  # pragma: no cover
                message = f"Failed parsing {artifact.name} ({parser_key}): {exc}"
                parse_warnings.append(message)
                parse_errors.append(message)
                if parser_routing:
                    parser_routing[-1]["outcome"] = {
                        "status": "failed_parse",
                        "reason": f"{type(exc).__name__}: {str(exc)[:280]}",
                        "atom_count": 0,
                        "warning_count": len(per_artifact_warnings),
                    }
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

    # Divert parse-time pre-suppressed atoms into the suppressed sidecar. A
    # parser can mark an atom suppressed at parse time (e.g. the xlsx sheet-role
    # router emits a ``dropped_sheet`` marker for a whole sheet it routes DROP,
    # stamped ``suppressed:sheet_router``). Such atoms must never reach scope,
    # but routing them through the ledger keeps them auditable and lets an
    # omission complaint ("you missed the Lookup tab") localize to the stage
    # that removed them — instead of the sheet vanishing without a trace.
    # Universal: keys off the suppression-flag prefix, not any file type.
    pre_suppressed = [
        a
        for a in atoms
        if any(
            str(f).startswith(SUPPRESSION_FLAG_PREFIX)
            for f in (getattr(a, "review_flags", None) or [])
        )
    ]
    if pre_suppressed:
        merge_suppressed(suppressed_atoms, pre_suppressed)
        _sup_ids = {str(getattr(a, "id", "")) for a in pre_suppressed}
        atoms = [a for a in atoms if str(getattr(a, "id", "")) not in _sup_ids]
        warnings.append(
            f"INFO: diverted {len(pre_suppressed)} parse-time suppressed "
            f"atom(s) (e.g. whole-sheet drops) to the suppressed sidecar"
        )

    # Register {artifact_id: Path} with the vision module so its leaf
    # fitz.open() calls — invoked from enrich_entities via atom
    # source_refs, which only carry basenames — can resolve to the
    # absolute on-disk path. Without this, find_table_pages and PDF
    # render fall through to "no such file" warnings and we lose
    # table-derived atoms.
    try:
        from app.core.vision_extraction import register_artifact_paths
        register_artifact_paths(artifact_paths)
    except Exception as _vp_exc:
        warnings.append(
            f"WARNING: vision artifact-path registration failed: "
            f"{type(_vp_exc).__name__}: {_vp_exc}"
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

    # v50 PROSE-LIST SPLITTER — atomize multi-fact paragraphs.
    # A single scope_item containing 6 stakeholders / 6 phases / 4
    # payment tiers becomes N child atoms. Each child inherits the
    # parent's source_ref + a sub_idx locator. Universal patterns
    # (pipe records, numbered prefixes, semicolon-parallel, bulleted
    # lines, label-prefix runs) — no customer-specific tuning.
    with telemetry.stage("prose_list_split", input_count=len(atoms)) as stage:
        split_count = 0
        try:
            from app.core.prose_list_splitter import split_prose_paragraph
            from app.core.schemas import (
                ArtifactType as _AT, AtomType as _AtomT, AuthorityClass as _Auth,
                EvidenceAtom as _EvAtom, ReviewStatus as _Rev, SourceRef as _SrcRef,
            )
            from app.core.ids import stable_id as _stable_id

            _splittable_types = {"scope_item", "entity", "raw_table_row"}
            _child_atoms: list = []
            for parent in atoms:
                _ptype = getattr(parent, "atom_type", None)
                _ptype_v = _ptype.value if hasattr(_ptype, "value") else str(_ptype or "")
                if _ptype_v not in _splittable_types:
                    continue
                _ptext = getattr(parent, "raw_text", "") or ""
                items = split_prose_paragraph(_ptext)
                if not items:
                    continue
                # Inherit source_ref from parent
                _parent_refs = list(getattr(parent, "source_refs", None) or [])
                _parent_aid = getattr(parent, "artifact_id", "") or ""
                _parent_pid = getattr(parent, "project_id", "") or ""
                # v52: detect section_path signal — if the parent atom
                # sits under a "Deliverables" / "Stakeholders" / etc.
                # heading, type the child atoms accordingly instead of
                # generic scope_item.
                _section_path = []
                if _parent_refs:
                    _loc0 = getattr(_parent_refs[0], "locator", None) or {}
                    if isinstance(_loc0, dict):
                        _section_path = _loc0.get("section_path") or []
                _section_blob = " ".join(str(s).lower() for s in _section_path)
                _child_type = _AtomT.scope_item
                _SECTION_TYPE_HINTS = {
                    "deliverable": _AtomT.deliverable,
                    "deliverables": _AtomT.deliverable,
                    "assumption": _AtomT.assumption,
                    "assumptions": _AtomT.assumption,
                    "exclusion": _AtomT.exclusion,
                    "out of scope": _AtomT.exclusion,
                    "exclusions": _AtomT.exclusion,
                    "signature": _AtomT.signatory,
                    "signatures": _AtomT.signatory,
                    "signatories": _AtomT.signatory,
                    "stakeholders": _AtomT.stakeholder,
                    "approver": _AtomT.approval_authority,
                    "approvers": _AtomT.approval_authority,
                    "approval matrix": _AtomT.approval_authority,
                    "payment schedule": _AtomT.payment_term,
                    "payment terms": _AtomT.payment_term,
                    "milestone": _AtomT.milestone_phase,
                    "milestones": _AtomT.milestone_phase,
                    "phase plan": _AtomT.milestone_phase,
                    "phase": _AtomT.milestone_phase,
                    "acceptance criteria": _AtomT.acceptance_criterion,
                    "acceptance": _AtomT.acceptance_criterion,
                    "lead time": _AtomT.lead_time_constraint,
                    "lead times": _AtomT.lead_time_constraint,
                    "cutover": _AtomT.cutover_step,
                    "cutover checklist": _AtomT.cutover_step,
                    "compliance": _AtomT.compliance_rule,
                    "data flow": _AtomT.data_flow_step,
                    "field mapping": _AtomT.system_mapping,
                    "system mapping": _AtomT.system_mapping,
                    "blackout": _AtomT.blackout_date_range,
                    "blackouts": _AtomT.blackout_date_range,
                }
                for hint, atype in _SECTION_TYPE_HINTS.items():
                    if hint in _section_blob:
                        _child_type = atype
                        break
                for sub_idx, item_text in enumerate(items):
                    _aid = _stable_id("atm", _parent_aid, "prose_split", parent.id, sub_idx)
                    _srcs: list = []
                    for r in _parent_refs[:1]:
                        # Build a fresh SourceRef carrying the parent's locator
                        # plus the sub_idx so provenance traces back to the
                        # original paragraph.
                        _loc = dict(getattr(r, "locator", None) or {})
                        _loc["prose_split_sub_idx"] = sub_idx
                        _loc["parent_atom_id"] = parent.id
                        _srcs.append(_SrcRef(
                            id=_stable_id("src", _aid),
                            artifact_id=_parent_aid,
                            artifact_type=getattr(r, "artifact_type", _AT.docx),
                            filename=getattr(r, "filename", ""),
                            locator=_loc,
                            extraction_method="prose_list_split_v50",
                            parser_version=getattr(r, "parser_version", "prose_split_v50"),
                        ))
                    _child_atoms.append(_EvAtom(
                        id=_aid,
                        project_id=_parent_pid,
                        artifact_id=_parent_aid,
                        atom_type=_child_type,
                        raw_text=item_text[:4000],
                        normalized_text=item_text.lower()[:4000],
                        value={"_prose_split": True, "_parent_atom_id": parent.id, "_sub_idx": sub_idx},
                        entity_keys=[],
                        source_refs=_srcs,
                        receipts=[],
                        authority_class=getattr(parent, "authority_class", _Auth.contractual_scope),
                        confidence=max(0.5, getattr(parent, "confidence", 0.8) - 0.05),
                        confidence_raw=max(0.5, getattr(parent, "confidence_raw", 0.8) - 0.05),
                        calibrated_confidence=max(0.5, getattr(parent, "calibrated_confidence", 0.8) - 0.05),
                        review_status=_Rev.auto_accepted,
                        review_flags=[],
                        parser_version="prose_split_v50",
                    ))
                split_count += 1
            if _child_atoms:
                atoms.extend(_child_atoms)
                warnings.append(f"INFO: prose-list splitter created {len(_child_atoms)} child atoms from {split_count} multi-fact paragraphs")
        except Exception as _split_exc:
            warnings.append(f"WARNING: prose_list_split failed: {type(_split_exc).__name__}: {_split_exc}")
        telemetry.end_stage(stage, output_count=split_count)

    # v57 PIPELINE REORDER (#10): prune BEFORE the expensive per-atom LLM
    # stages (enrich_entities, typed_atom_classification). Both pruners
    # below are deterministic and depend on neither entity_keys nor typed
    # classification, so running them up front shrinks the atom set the
    # LLM has to chew through WITHOUT changing the final atom set — the
    # same atoms are dropped, just before they burn an LLM call instead
    # of after.
    with telemetry.stage("duplicate_atom_collapse", input_count=len(atoms)) as stage:
        before_atoms = list(atoms)
        before = len(atoms)
        try:
            from app.core.entity_resolution import collapse_duplicate_atoms
            atoms = collapse_duplicate_atoms(atoms)
        except Exception as exc:
            warnings.append(f"WARNING: duplicate_atom_collapse failed: {type(exc).__name__}: {exc}")
        dropped = before - len(atoms)
        if dropped > 0:
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_atoms, atoms,
                    stage="duplicate_atom_collapse",
                    reason="collapsed as an intra-document duplicate of another atom",
                ),
            )
            warnings.append(f"INFO: collapsed {dropped} duplicate atoms (intra-doc)")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Execution-block hygiene: SOW signature pages ("Signature:",
    # "Name: ___ Date: ___", "Services By: PurTera | Agreed By:") get
    # swept into scope_item / raw_table_row atoms and masquerade as
    # scope. Drop them up front so they never reach scope_truth or burn
    # an LLM enrich / classification call.
    with telemetry.stage("execution_boilerplate_drop", input_count=len(atoms)) as stage:
        before_bp_atoms = list(atoms)
        before_bp = len(atoms)
        try:
            from app.core.entity_hygiene import drop_execution_boilerplate
            atoms = drop_execution_boilerplate(atoms)
        except Exception as exc:
            warnings.append(f"WARNING: execution_boilerplate_drop failed: {type(exc).__name__}: {exc}")
        dropped_bp = before_bp - len(atoms)
        if dropped_bp > 0:
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_bp_atoms, atoms,
                    stage="execution_boilerplate_drop",
                    reason="signature / execution-block boilerplate, not deal scope",
                ),
            )
            warnings.append(f"INFO: dropped {dropped_bp} signature/execution-block boilerplate atoms")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Pre-enrich table rollup — universal backstop for money-bearing
    # spreadsheet tables the parser's sheet-classifier missed. A 9 MB rate
    # card otherwise lands here as tens of thousands of per-row raw_table_row
    # atoms, each dragged through the per-atom LLM enrich pass (hours) and
    # flooding the training store with low-diversity catalog rows. Folding
    # each high-cardinality group losslessly into one pricing_assumption
    # summary atom (value.rows preserved) collapses cost and training noise
    # while leaving the deliverable unchanged (build_bill_of_materials and the
    # commercial_summary packet already read value.rows). Runs BEFORE enrich
    # so the expensive stages never see the exploded cardinality.
    with telemetry.stage("table_rollup", input_count=len(atoms)) as stage:
        before_tr_atoms = list(atoms)
        before_tr = len(atoms)
        try:
            from app.core.table_rollup import roll_up_table_rows
            atoms, tr_stats = roll_up_table_rows(atoms)
        except Exception as exc:
            tr_stats = {}
            warnings.append(f"WARNING: table_rollup failed: {type(exc).__name__}: {exc}")
        folded_tr = before_tr - len(atoms)
        if folded_tr > 0:
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_tr_atoms, atoms,
                    stage="table_rollup",
                    reason="high-cardinality money-bearing table folded into pricing rollup (value.rows preserved)",
                ),
            )
            warnings.append(
                f"INFO: table_rollup folded {tr_stats.get('rows_folded', folded_tr)} rows "
                f"across {tr_stats.get('groups_folded', 0)} table(s) into "
                f"{tr_stats.get('summary_atoms', 0)} pricing rollup atom(s)"
            )
        telemetry.end_stage(stage, output_count=len(atoms))

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

    # v47 typed-atom classification — promotes scope_item / entity
    # into the rich taxonomy (milestone_phase, stakeholder, bom_line,
    # commercial_total, payment_term, requirement, acceptance_criterion,
    # electrical_acceptance_test, compliance_*, ...). LLM-driven so
    # it generalises across customer terminology variations without
    # hardcoded regex column headers.
    with telemetry.stage("typed_atom_classification", input_count=len(atoms)) as stage:
        promoted = 0
        try:
            from app.core.typed_atom_classifier import classify_atoms
            promoted = classify_atoms(atoms)
        except Exception as exc:
            warnings.append(f"WARNING: typed_atom_classifier failed: {type(exc).__name__}: {exc}")
        if promoted:
            warnings.append(f"INFO: typed-atom classifier promoted {promoted} atoms from scope_item/entity")
        telemetry.end_stage(stage, output_count=promoted)

    # Geographic fallback: a deal whose only locational anchor is a bare
    # "City, ST ZIP" in a notes file produces zero physical_site atoms,
    # an empty site_readiness, and a RED "no confirmed site" brief. When
    # no real site exists, mine a City/State/ZIP anchor and emit one
    # low-confidence (needs_review) physical_site so the deal anchors
    # somewhere instead of going blank-RED. No-op when a real site exists.
    # Type-sanity guardrail: classification labels atoms in isolation, so
    # commercial meta ("28.57% margin", "99 pricing lines") leaks into the
    # quantity bucket and the deal's headline count ("110 displays") stays
    # buried in prose. This deterministic pass demotes non-deliverable
    # quantities to pricing_assumption and surfaces strong prose counts as
    # quantity atoms. No LLM, no customer tuning.
    with telemetry.stage("atom_type_sanity", input_count=len(atoms)) as stage:
        sanity_changed = 0
        try:
            from app.core.atom_type_sanity import apply_type_sanity
            atoms, _demoted, _surfaced = apply_type_sanity(atoms, project_id=resolved_project_id)
            sanity_changed = _demoted + _surfaced
            if _demoted:
                warnings.append(f"INFO: atom_type_sanity demoted {_demoted} non-deliverable quantity atom(s) to pricing_assumption")
            if _surfaced:
                warnings.append(f"INFO: atom_type_sanity surfaced {_surfaced} headline quantity atom(s) from prose")
        except Exception as exc:
            warnings.append(f"WARNING: atom_type_sanity failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=sanity_changed)

    # Span admission (#span_admission seam): for atoms still sitting in a
    # generic/retained type (scope_item/entity/deal_metadata/site_note), ask the
    # decide() STORE whether they should be re-typed into a recovered specific
    # type (milestone_phase, requirement, acceptance_criterion, quantity,
    # commercial category, ...). STORE-ONLY (no LLM), guess-free (abstain →
    # untouched). This makes RECALL text-ruleable: a PM teaches the system to
    # catch a missed class by adding a correction, no code change. Flag-gated so
    # it is a no-op in production until enabled AND a feedback store is wired.
    with telemetry.stage("span_admission", input_count=len(atoms)) as stage:
        readmitted = 0
        import os as _os_sa
        if _os_sa.environ.get("SOWSMITH_SPAN_ADMISSION", "") in ("1", "true", "yes", "on"):
            try:
                from app.core.span_admission import readmit_atom_types
                readmitted = readmit_atom_types(atoms)
                if readmitted:
                    warnings.append(
                        f"INFO: span_admission re-typed {readmitted} retained atom(s) "
                        f"into recovered types via the store"
                    )
            except Exception as exc:
                warnings.append(f"WARNING: span_admission failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=readmitted)

    # Open-question resolution: an open_question whose answer already
    # exists in the corpus (shares an answer-bearing entity key with a
    # fact atom) is flagged answered so it stops surfacing as a PM
    # blocker. Genuine gaps are added later from the SRL schema, not from
    # literal "?" detection. Deterministic, no LLM.
    with telemetry.stage("open_question_resolution", input_count=len(atoms)) as stage:
        resolved_q = 0
        try:
            from app.core.open_question_resolution import resolve_open_questions
            resolved_q = resolve_open_questions(atoms)
            if resolved_q:
                warnings.append(f"INFO: open_question_resolution flagged {resolved_q} already-answered question(s)")
        except Exception as exc:
            warnings.append(f"WARNING: open_question_resolution failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=resolved_q)

    with telemetry.stage("site_geo_fallback", input_count=len(atoms)) as stage:
        added_geo = 0
        import os as _os_geo
        if _os_geo.environ.get("SOWSMITH_DISABLE_GEO_FALLBACK"):
            # Opt-out for bulk / training runs. On site-rich deals the geo
            # inference short-circuits to 0 (real sites already exist), but the
            # LLM-backed vendor-suppression sub-step still runs one call per
            # site serially — pure wall-time for ~zero net atoms. Skip the
            # whole stage when this flag is set. Default-off: production keeps
            # the vendor gate (which is what stops PurTera's own address from
            # being mistaken for a job site).
            warnings.append("INFO: site_geo_fallback skipped (SOWSMITH_DISABLE_GEO_FALLBACK set)")
        else:
            try:
                from app.core.site_geo_fallback import (
                    geo_fallback_sites,
                    suppress_vendor_sites,
                )
                # 1) Infer fallback physical_site atoms from bare City/State/ZIP
                #    anchors. This must run FIRST: a vendor letterhead address
                #    ("PurTera LLC … Alpharetta, GA 30009") only becomes a
                #    physical_site atom HERE, so the suppression gate below has
                #    nothing to act on until these exist.
                geo_atoms = geo_fallback_sites(atoms, project_id=resolved_project_id)
                if geo_atoms:
                    atoms.extend(geo_atoms)
                    added_geo = len(geo_atoms)
                    warnings.append(
                        f"INFO: site_geo_fallback inferred {added_geo} physical_site atom(s) "
                        f"from City/State/ZIP (no confirmed site found)"
                    )
                # 2) Demote any physical_site that is actually the vendor's own
                #    letterhead / billing address (semantic role gate, LLM-backed;
                #    a no-op when the LLM is unreachable, only one site exists, or
                #    suppression would remove every site).
                before_vendor_atoms = list(atoms)
                atoms, dropped_vendor = suppress_vendor_sites(
                    atoms, project_id=resolved_project_id
                )
                if dropped_vendor:
                    merge_suppressed(
                        suppressed_atoms,
                        capture_suppressed(
                            before_vendor_atoms, atoms,
                            stage="site_geo_fallback",
                            reason="vendor / selling-party letterhead address, not a job site",
                        ),
                    )
                    warnings.append(
                        f"INFO: site_geo_fallback suppressed {dropped_vendor} "
                        f"vendor/letterhead address(es) misread as job sites"
                    )
            except Exception as exc:
                warnings.append(f"WARNING: site_geo_fallback failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=added_geo)

    # Receipt backfill: source_replay (stage 4) runs before the late
    # atom-creating stages (typed_atom_classification, site_geo_fallback),
    # so any atom *born* after replay carries source_refs but empty
    # receipts — which the quality gate rejects ("no receipts while source
    # files are available"). Re-replay only those stragglers so every atom
    # reaching the gate is provenance-complete. Idempotent: atoms that
    # already have receipts are untouched, and this generalises to any
    # future stage that mints atoms post-replay.
    with telemetry.stage("receipt_backfill", input_count=len(atoms)) as stage:
        backfilled = 0
        try:
            for atom in atoms:
                if getattr(atom, "source_refs", None) and not getattr(atom, "receipts", None):
                    atom.receipts = replay_atom_receipts(atom, artifact_paths)
                    backfilled += 1
        except Exception as exc:
            warnings.append(f"WARNING: receipt_backfill failed: {type(exc).__name__}: {exc}")
        if backfilled:
            warnings.append(f"INFO: receipt_backfill attached receipts to {backfilled} late-created atom(s)")
        telemetry.end_stage(stage, output_count=backfilled)

    # v52: semantic dedup by entity key. Catches the cases the text-based
    # v48 collapse misses — same fact extracted via 3 paths (schema /
    # prose / LLM bridge) with different text shapes but same phase_id /
    # req_id / sku / email. Drops milestone_phase from 23→6, requirement
    # from 19→5, etc., losslessly (loser fields merged into winner).
    with telemetry.stage("semantic_dedup", input_count=len(atoms)) as stage:
        before_sem_atoms = list(atoms)
        before_sem = len(atoms)
        try:
            from app.core.semantic_dedup import (
                cross_type_dedup_atoms,
                semantic_dedup_atoms,
            )
            atoms = semantic_dedup_atoms(atoms)
            # Cross-type pass: the same sentence emitted as raw_table_row +
            # scope_item + service_line + task collapses to the single most-
            # specific type. semantic_dedup keys with atom_type so it can't
            # catch these; without this, one table row inflates scope_truth
            # and the scorecards four-fold.
            before_xt = len(atoms)
            atoms = cross_type_dedup_atoms(atoms)
            dropped_xt = before_xt - len(atoms)
            if dropped_xt > 0:
                warnings.append(
                    f"INFO: cross_type_dedup collapsed {dropped_xt} same-text cross-type atoms"
                )
        except Exception as exc:
            warnings.append(f"WARNING: semantic_dedup failed: {type(exc).__name__}: {exc}")
        dropped_sem = before_sem - len(atoms)
        if dropped_sem > 0:
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_sem_atoms, atoms,
                    stage="semantic_dedup",
                    reason="semantic/cross-type duplicate collapsed into a canonical atom",
                ),
            )
            warnings.append(f"INFO: semantic_dedup collapsed {dropped_sem} duplicate-by-key atoms")
        telemetry.end_stage(stage, output_count=len(atoms))

    # v53 SMART CONFIDENCE — recalibrate every atom from hardcoded
    # provenance defaults (0.82/0.85) to content-aware scoring:
    # semantic-key anchored + value completeness + cross-doc
    # corroboration + source authority tier + receipts verified
    # + text-length quality. PMs get a confidence score that
    # actually correlates with truth.
    with telemetry.stage("confidence_recalibration", input_count=len(atoms)) as stage:
        recal_count = 0
        try:
            from app.core.confidence_recalibration import recalibrate_confidence
            from app.core.authority import classify_artifact_authority
            # Build artifact_id → tier from filenames
            _artifact_tier: dict[str, str] = {}
            for _a in atoms:
                _aid = getattr(_a, "artifact_id", None)
                if not _aid or _aid in _artifact_tier:
                    continue
                _refs = getattr(_a, "source_refs", None) or []
                _fname = ""
                if _refs:
                    _fname = getattr(_refs[0], "filename", "") or ""
                _artifact_tier[_aid] = (
                    classify_artifact_authority(_fname)
                    if _fname else "supporting_evidence"
                )
            recal_count = recalibrate_confidence(
                atoms, artifact_authority=_artifact_tier, edges=[],
            )
        except Exception as exc:
            warnings.append(f"WARNING: confidence_recalibration failed: {type(exc).__name__}: {exc}")
        if recal_count:
            warnings.append(f"INFO: recalibrated confidence on {recal_count} atoms")
        telemetry.end_stage(stage, output_count=recal_count)

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
        # D3: collapse multiple surface forms of the same person
        # (``stakeholder:watkins`` + ``stakeholder:r_watkins`` →
        # ``stakeholder:renee_watkins``) using the same fusion
        # mechanism. Key-shape based, so no false positives across
        # documents.
        stakeholder_alias_groups = collect_stakeholder_alias_groups(atoms)
        entities = fuse_alias_groups(
            entities, site_alias_groups + stakeholder_alias_groups
        )
        telemetry.end_stage(stage, output_count=len(entities))

    # v48 FIX 6: Cross-doc conflict detection.
    # Build artifact_id → authority_tier map from filenames, then scan
    # atoms for the same entity_key appearing with contradictory numeric
    # values across docs with different tiers.
    cross_doc_conflicts: list[dict[str, Any]] = []
    try:
        from app.core.authority import classify_artifact_authority
        from collections import defaultdict
        # Map artifact_id → filename → tier
        artifact_tier: dict[str, str] = {}
        for atom in atoms:
            aid = getattr(atom, "artifact_id", None)
            if not aid or aid in artifact_tier:
                continue
            refs = getattr(atom, "source_refs", None) or []
            fname = ""
            if refs:
                fname = getattr(refs[0], "filename", "") or ""
            artifact_tier[aid] = classify_artifact_authority(fname) if fname else "supporting_evidence"
        # Group (entity_key, first_int_in_atom) → list of (artifact_id, tier, raw_text)
        groups: dict[tuple, list] = defaultdict(list)
        for atom in atoms:
            ekeys = getattr(atom, "entity_keys", None) or []
            if not ekeys:
                continue
            rt = getattr(atom, "raw_text", "") or ""
            nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', rt)
            if not nums:
                continue
            aid = getattr(atom, "artifact_id", "unknown")
            tier = artifact_tier.get(aid, "supporting_evidence")
            for ekey in ekeys:
                groups[(ekey, nums[0])].append({
                    "value": nums[0],
                    "artifact_id": aid,
                    "authority_tier": tier,
                    "raw_text": rt[:200],
                })
        # Look for entity keys with multiple distinct values across multiple tiers
        by_entity: dict[str, list] = defaultdict(list)
        for (ekey, val), entries in groups.items():
            for entry in entries:
                by_entity[ekey].append({**entry, "value": val})
        for ekey, entries in by_entity.items():
            tiers = {e["authority_tier"] for e in entries}
            values = {e["value"] for e in entries}
            if len(values) > 1 and len(tiers) > 1:
                if "contractual_final" in tiers:
                    severity = "high"
                elif "approved_scope" in tiers:
                    severity = "medium"
                else:
                    severity = "low"
                cross_doc_conflicts.append({
                    "entity_key": ekey,
                    "values": entries,
                    "severity": severity,
                })
        if cross_doc_conflicts:
            warnings.append(f"INFO: detected {len(cross_doc_conflicts)} cross-doc conflicts")
    except Exception as exc:
        warnings.append(f"WARNING: cross_doc_conflicts failed: {type(exc).__name__}: {exc}")

    # v48 FIX 8: BOM arithmetic cross-check.
    bom_arithmetic_check: dict[str, Any] | None = None
    try:
        def _extract_dollars(text: str) -> float | None:
            m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', text)
            if m:
                return float(m.group(1).replace(",", ""))
            return None
        line_items: list[float] = []
        stated_total: float | None = None
        for atom in atoms:
            atype = str(getattr(atom, "atom_type", "") or "")
            if hasattr(atom.atom_type, "value"):
                atype = atom.atom_type.value
            rt = getattr(atom, "raw_text", "") or ""
            rt_lower = rt.lower()
            if "vendor_line_item" in atype or "bom_line" in atype:
                v = _extract_dollars(rt)
                if v and v > 0:
                    line_items.append(v)
            elif any(kw in rt_lower for kw in ("grand total", "total price", "contract total", "project total")):
                v = _extract_dollars(rt)
                if v and v > 0:
                    stated_total = v
        if len(line_items) >= 2 and stated_total is not None:
            line_sum = sum(line_items)
            discrepancy = abs(line_sum - stated_total)
            pct = (discrepancy / stated_total * 100) if stated_total else 0
            if pct >= 0.5:
                bom_arithmetic_check = {
                    "line_item_sum": round(line_sum, 2),
                    "stated_total": round(stated_total, 2),
                    "discrepancy": round(discrepancy, 2),
                    "discrepancy_pct": round(pct, 2),
                    "severity": "high" if pct > 5 else "medium" if pct > 1 else "low",
                }
                warnings.append(
                    f"INFO: BOM arithmetic discrepancy: line-item sum ${line_sum:,.2f} "
                    f"vs stated total ${stated_total:,.2f} ({pct:.1f}%)"
                )
    except Exception as exc:
        warnings.append(f"WARNING: bom_arithmetic_check failed: {type(exc).__name__}: {exc}")

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
        suppressed_atoms=sorted(suppressed_atoms, key=lambda x: x.id),
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
