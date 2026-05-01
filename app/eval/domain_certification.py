from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.compiler import compile_project
from app.domain.loader import load_domain_pack
from app.domain.schemas import DomainPack
from app.eval.benchmark import run_packetizer_benchmark, threshold_failures

SUPPORTED_ARTIFACT_EXTENSIONS = {".xlsx", ".csv", ".txt", ".docx"}


class CertificationCheckResult(BaseModel):
    check_id: str
    name: str
    passed: bool
    severity: str = "error"
    details: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    recommendation: str | None = None


class CertificationReport(BaseModel):
    pack_id: str
    pack_version: str
    passed: bool
    checks: list[CertificationCheckResult] = Field(default_factory=list)
    parser_coverage: dict[str, Any] = Field(default_factory=dict)
    alias_collision_count: int = 0
    required_fixture_count: int = 0
    benchmark_metrics: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


def _normalize_alias(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Domain pack YAML must contain a mapping object.")
    return payload


def _discover_projects(fixtures_dir: Path) -> tuple[list[Path], bool]:
    projects: list[Path] = []
    gold_files = sorted(fixtures_dir.rglob("gold.json"), key=lambda p: str(p).lower())
    if gold_files:
        for gold_file in gold_files:
            payload = yaml.safe_load(gold_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            project_dir = payload.get("project_dir")
            if isinstance(project_dir, str) and project_dir.strip():
                candidate = Path(project_dir).resolve()
            else:
                candidate = gold_file.parent / "project"
            if candidate.exists() and candidate.is_dir():
                projects.append(candidate)
        return sorted(set(projects), key=lambda p: str(p).lower()), True
    if fixtures_dir.exists() and fixtures_dir.is_dir():
        return [fixtures_dir.resolve()], False
    return [], False


def _parser_coverage(project_dirs: list[Path], domain_pack: DomainPack | str | Path) -> dict[str, Any]:
    supported_count = 0
    matched_count = 0
    parser_counts: dict[str, int] = {}
    unmatched_files: list[str] = []

    for project in project_dirs:
        result = compile_project(
            project_dir=project,
            project_id=f"cert_{project.name}",
            allow_errors=True,
            allow_unverified_receipts=True,
            domain_pack=domain_pack,
        )
        routes = result.manifest.parser_routing if result.manifest is not None else []
        route_by_name = {str(row.get("filename")): row for row in routes}
        for artifact in sorted([p for p in project.rglob("*") if p.is_file()], key=lambda p: str(p).lower()):
            rel = str(artifact.relative_to(project)).replace("\\", "/")
            if artifact.suffix.lower() not in SUPPORTED_ARTIFACT_EXTENSIONS:
                continue
            supported_count += 1
            route = route_by_name.get(rel)
            chosen = str((route or {}).get("chosen_parser", "none"))
            if route is not None and chosen != "none":
                matched_count += 1
                parser_counts[chosen] = parser_counts.get(chosen, 0) + 1
            else:
                unmatched_files.append(f"{project.name}:{rel}")
    match_rate = round(matched_count / supported_count, 6) if supported_count else 0.0
    return {
        "supported_artifact_count": supported_count,
        "matched_artifact_count": matched_count,
        "match_rate": match_rate,
        "parser_counts": dict(sorted(parser_counts.items())),
        "unmatched_files": sorted(unmatched_files),
    }


def certify_domain_pack(*, domain_pack_path: Path | str, fixtures_dir: Path) -> CertificationReport:
    path = Path(domain_pack_path).resolve()
    checks: list[CertificationCheckResult] = []
    recommendations: list[str] = []
    benchmark_metrics: dict[str, Any] = {}
    parser_coverage: dict[str, Any] = {}
    alias_collision_count = 0
    required_fixture_count = 0
    pack_id = path.stem
    pack_version = "unknown"

    raw_payload: dict[str, Any] | None = None
    pack: DomainPack | None = None

    try:
        raw_payload = _load_raw_yaml(path)
        pack = load_domain_pack(path)
        pack_id = pack.pack_id
        pack_version = pack.version
        checks.append(
            CertificationCheckResult(
                check_id="check_01_yaml_schema_valid",
                name="YAML schema valid",
                passed=True,
                details="Domain pack parses and validates against schema.",
            )
        )
    except Exception as exc:
        checks.append(
            CertificationCheckResult(
                check_id="check_01_yaml_schema_valid",
                name="YAML schema valid",
                passed=False,
                severity="error",
                details=str(exc),
                recommendation="Fix YAML shape and required schema fields before certification.",
            )
        )
        recommendations.append("Fix domain pack YAML/schema errors.")

    if pack is not None and raw_payload is not None:
        # Check 2: duplicate aliases across unrelated entity types.
        alias_to_types: dict[str, set[str]] = {}
        for entity_type in pack.entity_types:
            for alias in entity_type.aliases:
                token = _normalize_alias(alias)
                if not token:
                    continue
                alias_to_types.setdefault(token, set()).add(entity_type.name)
        duplicate_aliases = sorted(
            alias for alias, owners in alias_to_types.items() if len(owners) > 1
        )
        checks.append(
            CertificationCheckResult(
                check_id="check_02_cross_entity_alias_duplicates",
                name="No duplicate canonical aliases across unrelated entity types",
                passed=not duplicate_aliases,
                severity="error",
                details="No duplicate aliases found." if not duplicate_aliases else f"Duplicate aliases: {duplicate_aliases}",
                metrics={"duplicate_aliases": duplicate_aliases},
                recommendation=(
                    None
                    if not duplicate_aliases
                    else "Split overloaded aliases per entity type or add explicit disambiguation rules."
                ),
            )
        )
        if duplicate_aliases:
            recommendations.append("Resolve duplicate aliases across entity types.")

        # Check 3: site/device alias collisions.
        allowed = {
            _normalize_alias(row)
            for row in (raw_payload.get("allowed_site_device_alias_collisions") or [])
            if isinstance(row, str)
        }
        site_terms: set[str] = set()
        device_terms: set[str] = set()
        for entity_type in pack.entity_types:
            target = site_terms if entity_type.name.strip().lower() == "site" else None
            if entity_type.name.strip().lower() == "device":
                target = device_terms
            if target is not None:
                for alias in entity_type.aliases:
                    norm = _normalize_alias(alias)
                    if norm:
                        target.add(norm)
        for pattern in pack.site_alias_patterns:
            norm = _normalize_alias(pattern)
            if norm:
                site_terms.add(norm)
        for canonical, aliases in pack.device_aliases.items():
            norm_c = _normalize_alias(canonical)
            if norm_c:
                device_terms.add(norm_c)
            for alias in aliases:
                norm = _normalize_alias(alias)
                if norm:
                    device_terms.add(norm)
        collisions = sorted((site_terms & device_terms) - allowed)
        alias_collision_count = len(collisions)
        checks.append(
            CertificationCheckResult(
                check_id="check_03_site_device_alias_collision",
                name="No site/device alias collisions unless explicitly allowed",
                passed=not collisions,
                severity="error",
                details="No site/device collisions found." if not collisions else f"Collisions: {collisions}",
                metrics={"collisions": collisions, "allowed_collisions": sorted(allowed)},
                recommendation=(
                    None
                    if not collisions
                    else "Rename colliding aliases or declare intentional overlaps in allowed_site_device_alias_collisions."
                ),
            )
        )
        if collisions:
            recommendations.append("Resolve site/device alias collisions.")

        # Check 4: risk defaults coverage for key devices.
        missing_device_risk_keys: list[str] = []
        for device_key in sorted(pack.device_aliases.keys()):
            expected = f"{device_key}_unit_exposure"
            if expected not in pack.risk_defaults:
                missing_device_risk_keys.append(expected)
        missing_ratio = (
            len(missing_device_risk_keys) / len(pack.device_aliases)
            if pack.device_aliases
            else 0.0
        )
        severity = "warning" if missing_device_risk_keys and missing_ratio <= 0.5 else "error"
        passed = not missing_device_risk_keys
        checks.append(
            CertificationCheckResult(
                check_id="check_04_risk_defaults_for_key_devices",
                name="Risk defaults present for key device types",
                passed=passed,
                severity=severity,
                details=(
                    "All key device defaults present."
                    if passed
                    else f"Missing risk defaults: {missing_device_risk_keys}"
                ),
                metrics={"missing_risk_defaults": missing_device_risk_keys},
                recommendation=(
                    None
                    if passed
                    else "Add per-device *_unit_exposure defaults for each canonical device alias."
                ),
            )
        )
        if missing_device_risk_keys:
            recommendations.append("Add missing device risk defaults.")

        # Check 5: constraint patterns populated.
        constraint_count = sum(len(values) for values in pack.constraint_patterns.values())
        checks.append(
            CertificationCheckResult(
                check_id="check_05_constraint_patterns_present",
                name="Constraint patterns not empty",
                passed=constraint_count > 0,
                severity="error",
                details=f"constraint_pattern_count={constraint_count}",
                recommendation=(
                    None if constraint_count > 0 else "Add at least one constraint pattern for pack-specific extraction."
                ),
            )
        )
        if constraint_count == 0:
            recommendations.append("Add constraint patterns.")

        # Check 6: exclusion + customer instruction patterns.
        exclusion_count = len(pack.exclusion_patterns)
        instruction_count = len(pack.customer_instruction_patterns)
        has_both = exclusion_count > 0 and instruction_count > 0
        checks.append(
            CertificationCheckResult(
                check_id="check_06_exclusion_and_instruction_patterns_present",
                name="Exclusion/customer instruction patterns present",
                passed=has_both,
                severity="error",
                details=(
                    f"exclusion_count={exclusion_count}, customer_instruction_count={instruction_count}"
                ),
                recommendation=(
                    None if has_both else "Add exclusion and customer instruction patterns to avoid silent scope regressions."
                ),
            )
        )
        if not has_both:
            recommendations.append("Add exclusion/customer instruction patterns.")

    project_dirs, has_gold = _discover_projects(fixtures_dir.resolve())
    required_fixture_count = sum(
        1
        for project in project_dirs
        for artifact in project.rglob("*")
        if artifact.is_file() and artifact.suffix.lower() in SUPPORTED_ARTIFACT_EXTENSIONS
    )

    # Check 7: synthetic fixtures exist.
    checks.append(
        CertificationCheckResult(
            check_id="check_07_fixture_presence",
            name="At least one synthetic fixture exists for the pack",
            passed=required_fixture_count > 0,
            severity="error",
            details=f"required_fixture_count={required_fixture_count}",
            recommendation=(
                None if required_fixture_count > 0 else "Add at least one pack-specific synthetic fixture file."
            ),
        )
    )
    if required_fixture_count == 0:
        recommendations.append("Add synthetic fixtures for the domain pack.")

    # Check 8: parser benchmark-like coverage for pack fixtures.
    if project_dirs:
        parser_coverage = _parser_coverage(project_dirs, domain_pack=path)
    else:
        parser_coverage = {
            "supported_artifact_count": 0,
            "matched_artifact_count": 0,
            "match_rate": 0.0,
            "parser_counts": {},
            "unmatched_files": [],
        }
    parser_pass = (
        parser_coverage.get("supported_artifact_count", 0) > 0
        and parser_coverage.get("matched_artifact_count", 0) == parser_coverage.get("supported_artifact_count", 0)
    )
    checks.append(
        CertificationCheckResult(
            check_id="check_08_parser_coverage",
            name="Parser benchmark passes pack-specific fixtures",
            passed=bool(parser_pass),
            severity="error",
            details=(
                "All supported fixture artifacts were matched by a parser."
                if parser_pass
                else f"Unmatched fixture artifacts: {parser_coverage.get('unmatched_files', [])}"
            ),
            metrics=parser_coverage,
            recommendation=(
                None if parser_pass else "Adjust parser routing hints/capabilities or fixture naming/content markers."
            ),
        )
    )
    if not parser_pass:
        recommendations.append("Improve parser coverage for pack fixtures.")

    # Checks 9/10: packetizer benchmark + governance.
    if has_gold:
        bench_report = run_packetizer_benchmark(fixtures_dir.resolve())
        failures: list[str] = []
        aggregate = bench_report.aggregate_metrics
        if aggregate.get("compile_success_rate", 0.0) < 1.0:
            failures.append("compile_success_rate below 1.0")
        if aggregate.get("packet_family_recall", 0.0) < 0.95:
            failures.append("packet_family_recall below 0.95")
        if aggregate.get("governing_accuracy", 0.0) < 0.95:
            failures.append("governing_accuracy below 0.95")
        if aggregate.get("packet_anchor_recall", 0.0) < 0.95:
            failures.append("packet_anchor_recall below 0.95")
        if int(aggregate.get("invalid_governance_count", 1)) != 0:
            failures.append("invalid_governance_count is non-zero")
        # Keep visibility into generic benchmark threshold issues without making
        # contradiction-only misses a hard gate for initial domain-pack promotion.
        generic_failures = threshold_failures(bench_report)
        benchmark_metrics = dict(bench_report.aggregate_metrics)
        benchmark_metrics["threshold_failures"] = failures
        benchmark_metrics["generic_threshold_failures"] = generic_failures
        packetizer_pass = not failures
        checks.append(
            CertificationCheckResult(
                check_id="check_09_packetizer_benchmark",
                name="Packetizer benchmark passes pack-specific gold labels",
                passed=packetizer_pass,
                severity="error",
                details="Packetizer benchmark passed thresholds." if packetizer_pass else f"Threshold failures: {failures}",
                metrics=benchmark_metrics,
                recommendation=(
                    None if packetizer_pass else "Improve packet family recall/governing accuracy for this pack before promotion."
                ),
            )
        )
        if not packetizer_pass:
            recommendations.append("Fix packetizer benchmark threshold failures for this pack.")

        invalid_governance = int(benchmark_metrics.get("invalid_governance_count", 0))
        governance_pass = invalid_governance == 0
        checks.append(
            CertificationCheckResult(
                check_id="check_10_no_invalid_governance",
                name="No invalid governance in pack benchmark",
                passed=governance_pass,
                severity="error",
                details=f"invalid_governance_count={invalid_governance}",
                metrics={"invalid_governance_count": invalid_governance},
                recommendation=(
                    None if governance_pass else "Resolve governance violations (deleted/quoted/rejected evidence constraints)."
                ),
            )
        )
        if not governance_pass:
            recommendations.append("Resolve invalid governance in benchmark scenarios.")
    else:
        benchmark_metrics = {"threshold_failures": ["No gold scenarios discovered under fixtures directory."]}
        checks.append(
            CertificationCheckResult(
                check_id="check_09_packetizer_benchmark",
                name="Packetizer benchmark passes pack-specific gold labels",
                passed=False,
                severity="error",
                details="No gold scenarios discovered under fixtures directory.",
                metrics=benchmark_metrics,
                recommendation="Provide pack-specific gold.json scenarios for packetizer certification.",
            )
        )
        checks.append(
            CertificationCheckResult(
                check_id="check_10_no_invalid_governance",
                name="No invalid governance in pack benchmark",
                passed=False,
                severity="error",
                details="Cannot evaluate invalid governance without gold scenarios.",
                metrics={"invalid_governance_count": None},
                recommendation="Add gold scenarios and rerun certification.",
            )
        )
        recommendations.append("Add pack-specific gold scenarios.")

    passed = all(check.passed or check.severity == "warning" for check in checks)
    recommendations.extend([check.recommendation for check in checks if check.recommendation])
    recommendations = sorted(set([row for row in recommendations if row]))
    return CertificationReport(
        pack_id=pack_id,
        pack_version=pack_version,
        passed=passed,
        checks=checks,
        parser_coverage=parser_coverage,
        alias_collision_count=alias_collision_count,
        required_fixture_count=required_fixture_count,
        benchmark_metrics=benchmark_metrics,
        recommendations=recommendations,
    )
