from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.compiler import compile_project
from app.core.validators import validation_failure_records
from app.eval.failure_taxonomy import (
    FailureCategory,
    FailureRecord,
    failure_records_from_expected_label_mismatches,
    make_failure_record,
)
from app.eval.gold import GoldScenario, load_gold
from app.eval.metrics import ScenarioMetricValues, evaluate_scenario_metrics


class ScenarioBenchmarkResult(BaseModel):
    scenario_id: str
    scenario_path: str
    metrics: dict[str, Any]
    failed_invariants: list[str] = Field(default_factory=list)
    compile_error: str | None = None
    failure_records: list[FailureRecord] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    scenario_count: int
    aggregate_metrics: dict[str, Any]
    scenario_results: list[ScenarioBenchmarkResult]
    failed_invariants: list[str] = Field(default_factory=list)
    recommended_next_fixes: list[str] = Field(default_factory=list)
    failure_records: list[FailureRecord] = Field(default_factory=list)


def _discover_scenarios(fixtures_dir: Path) -> list[tuple[Path, GoldScenario]]:
    scenarios: list[tuple[Path, GoldScenario]] = []
    for gold_path in sorted(fixtures_dir.rglob("gold.json"), key=lambda p: str(p).lower()):
        scenario_root = gold_path.parent
        gold = load_gold(gold_path)
        if gold.project_dir:
            candidate = (scenario_root / gold.project_dir).resolve()
            project_dir = candidate
        else:
            project_dir = scenario_root / "project"
            if not project_dir.exists():
                project_dir = scenario_root
        scenarios.append((project_dir, gold))
    return scenarios


def _aggregate(results: list[ScenarioBenchmarkResult]) -> dict[str, Any]:
    if not results:
        return {}
    metric_keys = [
        "packet_family_recall",
        "packet_anchor_recall",
        "governing_accuracy",
        "contradiction_recall",
        "receipt_coverage",
        "verified_receipt_rate",
        "false_active_rate",
        "compile_latency_ms",
        "packet_count",
        "atom_count",
    ]
    aggregate: dict[str, Any] = {}
    for key in metric_keys:
        values = [float(result.metrics[key]) for result in results if key in result.metrics]
        aggregate[key] = round(sum(values) / len(values), 4) if values else 0.0
    aggregate["invalid_governance_count"] = int(sum(int(result.metrics.get("invalid_governance_count", 0)) for result in results))
    aggregate["determinism_pass"] = all(bool(result.metrics.get("determinism_pass", False)) for result in results)
    compile_successes = sum(1 for result in results if bool(result.metrics.get("compile_success", False)))
    aggregate["compile_success_rate"] = round(compile_successes / len(results), 4)
    return aggregate


def threshold_failures(report: BenchmarkReport) -> list[str]:
    aggregate = report.aggregate_metrics
    failures: list[str] = []
    if aggregate.get("packet_family_recall", 0.0) < 0.95:
        failures.append("packet_family_recall below 0.95")
    if aggregate.get("governing_accuracy", 0.0) < 0.95:
        failures.append("governing_accuracy below 0.95")
    if aggregate.get("contradiction_recall", 0.0) < 0.95:
        failures.append("contradiction_recall below 0.95")
    if aggregate.get("invalid_governance_count", 1) != 0:
        failures.append("invalid_governance_count is non-zero")
    if not bool(aggregate.get("determinism_pass", False)):
        failures.append("determinism_pass is false")
    if aggregate.get("false_active_rate", 1.0) != 0.0:
        failures.append("false_active_rate is non-zero")
    if aggregate.get("compile_success_rate", 0.0) != 1.0:
        failures.append("compile_success_rate is not 1.0")
    return failures


def _recommended_fixes(failures: list[str]) -> list[str]:
    fixes: list[str] = []
    for failure in failures:
        if "packet_family_recall" in failure:
            fixes.append("Increase parser robustness for missing packet families in adversarial inputs.")
        elif "governing_accuracy" in failure:
            fixes.append("Refine authority lattice penalties/tie-breaks for governing atom selection.")
        elif "contradiction_recall" in failure:
            fixes.append("Improve contradiction edge detection and conflict packet generation.")
        elif "invalid_governance_count" in failure:
            fixes.append("Tighten governance validators for deleted/rejected/quoted authority constraints.")
        elif "determinism_pass" in failure:
            fixes.append("Remove non-deterministic fields from signatures and enforce stable sorting.")
        elif "false_active_rate" in failure:
            fixes.append("Downgrade active contradictory packets to needs_review in packetizer.")
        elif "compile_success_rate" in failure:
            fixes.append("Investigate parser crashes and validation hard errors in failing scenarios.")
    return sorted(set(fixes))


def run_packetizer_benchmark(fixtures_dir: Path) -> BenchmarkReport:
    scenarios = _discover_scenarios(fixtures_dir)
    results: list[ScenarioBenchmarkResult] = []
    all_failures: list[str] = []
    all_failure_records: list[FailureRecord] = []
    for project_dir, gold in scenarios:
        start = time.perf_counter()
        failed_invariants: list[str] = []
        compile_error: str | None = None
        failure_records: list[FailureRecord] = []
        try:
            first = compile_project(project_dir, project_id=gold.scenario_id, allow_errors=True, allow_unverified_receipts=True)
            latency_ms = (time.perf_counter() - start) * 1000.0
            second = compile_project(project_dir, project_id=gold.scenario_id, allow_errors=True, allow_unverified_receipts=True)
            determinism = bool(
                first.manifest
                and second.manifest
                and first.manifest.output_signature == second.manifest.output_signature
            )
            metrics, forbidden_failures = evaluate_scenario_metrics(
                first,
                gold,
                compile_latency_ms=latency_ms,
                determinism_pass=determinism,
                compile_success=True,
            )
            failed_invariants.extend(forbidden_failures)
            failure_records.extend(
                failure_records_from_expected_label_mismatches(first, gold, scenario_id=gold.scenario_id)
            )
            failure_records.extend(validation_failure_records(first, source_files_available=False))
            for invariant in forbidden_failures:
                if invariant == "deleted_text_governs":
                    failure_records.append(
                        make_failure_record(
                            category=FailureCategory.INVALID_DELETED_TEXT_GOVERNANCE,
                            severity="critical",
                            scenario_id=gold.scenario_id,
                            message=f"Forbidden condition triggered: {invariant}",
                        )
                    )
                if invariant == "quoted_old_email_governs_current_conflict":
                    failure_records.append(
                        make_failure_record(
                            category=FailureCategory.INVALID_QUOTED_EMAIL_GOVERNANCE,
                            severity="critical",
                            scenario_id=gold.scenario_id,
                            message=f"Forbidden condition triggered: {invariant}",
                        )
                    )
            if not determinism:
                failed_invariants.append("determinism_failed")
                failure_records.append(
                    make_failure_record(
                        category=FailureCategory.NON_DETERMINISTIC_OUTPUT,
                        severity="critical",
                        scenario_id=gold.scenario_id,
                        message=f"Deterministic replay failed for scenario {gold.scenario_id}",
                    )
                )
            if latency_ms > 30000:
                failure_records.append(
                    make_failure_record(
                        category=FailureCategory.PERF_BUDGET_EXCEEDED,
                        severity="high",
                        scenario_id=gold.scenario_id,
                        message=f"Compile latency {latency_ms:.2f}ms exceeded 30000ms budget",
                    )
                )
            metrics_dict = metrics.as_dict()
        except Exception as exc:  # pragma: no cover
            compile_error = str(exc)
            metrics_dict = ScenarioMetricValues(
                packet_family_recall=0.0,
                packet_anchor_recall=0.0,
                governing_accuracy=0.0,
                contradiction_recall=0.0,
                receipt_coverage=0.0,
                verified_receipt_rate=0.0,
                false_active_rate=1.0,
                invalid_governance_count=1,
                determinism_pass=False,
                compile_latency_ms=(time.perf_counter() - start) * 1000.0,
                packet_count=0,
                atom_count=0,
                compile_success=False,
            ).as_dict()
            failed_invariants.append("compile_error")
            failure_records.append(
                make_failure_record(
                    category=FailureCategory.PARSER_CRASH,
                    severity="critical",
                    scenario_id=gold.scenario_id,
                    message=f"Compile crash for scenario {gold.scenario_id}: {compile_error}",
                )
            )

        scenario_result = ScenarioBenchmarkResult(
            scenario_id=gold.scenario_id,
            scenario_path=str(project_dir),
            metrics=metrics_dict,
            failed_invariants=sorted(set(failed_invariants)),
            compile_error=compile_error,
            failure_records=sorted(failure_records, key=lambda row: row.failure_id),
        )
        all_failure_records.extend(scenario_result.failure_records)
        if scenario_result.failed_invariants:
            all_failures.append(gold.scenario_id)
        results.append(scenario_result)

    aggregate = _aggregate(results)
    threshold_issues = threshold_failures(
        BenchmarkReport(
            scenario_count=len(results),
            aggregate_metrics=aggregate,
            scenario_results=results,
            failed_invariants=sorted(set(all_failures)),
            recommended_next_fixes=[],
        )
    )
    return BenchmarkReport(
        scenario_count=len(results),
        aggregate_metrics=aggregate,
        scenario_results=results,
        failed_invariants=sorted(set(all_failures + threshold_issues)),
        recommended_next_fixes=_recommended_fixes(threshold_issues),
        failure_records=sorted(all_failure_records, key=lambda row: row.failure_id),
    )
