from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ParserMetricReport(BaseModel):
    parser_name: str
    atom_recall_by_type: float
    source_ref_coverage: float
    entity_key_accuracy: float
    quantity_accuracy: float
    authority_class_accuracy: float
    review_flag_accuracy: float
    parse_crash_rate: float
    unsupported_feature_warnings: int
    case_count: int


class ParserBenchmarkReport(BaseModel):
    aggregate_metrics: dict[str, float]
    parser_reports: list[ParserMetricReport] = Field(default_factory=list)
    threshold_failures: list[str] = Field(default_factory=list)


@dataclass
class ParserCaseExpectation:
    expected_atom_types: set[str] = field(default_factory=set)
    expected_entity_keys: set[str] = field(default_factory=set)
    expected_quantities: set[float] = field(default_factory=set)
    expected_authorities: set[str] = field(default_factory=set)
    expected_review_flags: set[str] = field(default_factory=set)
    unsupported_warning_count: int = 0


def evaluate_case_metrics(parsed_atoms: list[Any], expected: ParserCaseExpectation, crashed: bool = False) -> dict[str, float | int]:
    atom_types = {atom.atom_type.value for atom in parsed_atoms}
    entity_keys = {key for atom in parsed_atoms for key in atom.entity_keys}
    quantities = {
        float(atom.value["quantity"])
        for atom in parsed_atoms
        if isinstance(atom.value, dict) and isinstance(atom.value.get("quantity"), (int, float))
    }
    authorities = {atom.authority_class.value for atom in parsed_atoms}
    flags = {flag for atom in parsed_atoms for flag in atom.review_flags}

    atom_recall = len(expected.expected_atom_types & atom_types) / len(expected.expected_atom_types) if expected.expected_atom_types else 1.0
    entity_accuracy = len(expected.expected_entity_keys & entity_keys) / len(expected.expected_entity_keys) if expected.expected_entity_keys else 1.0
    qty_accuracy = len(expected.expected_quantities & quantities) / len(expected.expected_quantities) if expected.expected_quantities else 1.0
    authority_accuracy = len(expected.expected_authorities & authorities) / len(expected.expected_authorities) if expected.expected_authorities else 1.0
    flag_accuracy = len(expected.expected_review_flags & flags) / len(expected.expected_review_flags) if expected.expected_review_flags else 1.0
    source_ref_coverage = (
        sum(1 for atom in parsed_atoms if atom.source_refs) / len(parsed_atoms)
        if parsed_atoms
        else 1.0
    )
    return {
        "atom_recall_by_type": atom_recall,
        "source_ref_coverage": source_ref_coverage,
        "entity_key_accuracy": entity_accuracy,
        "quantity_accuracy": qty_accuracy,
        "authority_class_accuracy": authority_accuracy,
        "review_flag_accuracy": flag_accuracy,
        "parse_crash_rate": 1.0 if crashed else 0.0,
        "unsupported_feature_warnings": expected.unsupported_warning_count,
    }


def aggregate_parser_metrics(parser_name: str, case_metrics: list[dict[str, float | int]]) -> ParserMetricReport:
    if not case_metrics:
        return ParserMetricReport(
            parser_name=parser_name,
            atom_recall_by_type=0.0,
            source_ref_coverage=0.0,
            entity_key_accuracy=0.0,
            quantity_accuracy=0.0,
            authority_class_accuracy=0.0,
            review_flag_accuracy=0.0,
            parse_crash_rate=1.0,
            unsupported_feature_warnings=0,
            case_count=0,
        )
    keys = [
        "atom_recall_by_type",
        "source_ref_coverage",
        "entity_key_accuracy",
        "quantity_accuracy",
        "authority_class_accuracy",
        "review_flag_accuracy",
        "parse_crash_rate",
    ]
    averaged = {
        key: float(sum(float(case[key]) for case in case_metrics) / len(case_metrics))
        for key in keys
    }
    warnings = int(sum(int(case["unsupported_feature_warnings"]) for case in case_metrics))
    return ParserMetricReport(
        parser_name=parser_name,
        atom_recall_by_type=round(averaged["atom_recall_by_type"], 4),
        source_ref_coverage=round(averaged["source_ref_coverage"], 4),
        entity_key_accuracy=round(averaged["entity_key_accuracy"], 4),
        quantity_accuracy=round(averaged["quantity_accuracy"], 4),
        authority_class_accuracy=round(averaged["authority_class_accuracy"], 4),
        review_flag_accuracy=round(averaged["review_flag_accuracy"], 4),
        parse_crash_rate=round(averaged["parse_crash_rate"], 4),
        unsupported_feature_warnings=warnings,
        case_count=len(case_metrics),
    )


def parser_threshold_failures(report: ParserBenchmarkReport) -> list[str]:
    failures: list[str] = []
    aggregate = report.aggregate_metrics
    if aggregate.get("source_ref_coverage", 0.0) != 1.0:
        failures.append("source_ref_coverage must be 1.0")
    if aggregate.get("parse_crash_rate", 1.0) != 0.0:
        failures.append("parse_crash_rate must be 0")
    if aggregate.get("quantity_accuracy", 0.0) < 0.98:
        failures.append("quantity_accuracy below 0.98")
    if aggregate.get("authority_class_accuracy", 0.0) < 0.95:
        failures.append("authority_class_accuracy below 0.95")
    if aggregate.get("entity_key_accuracy", 0.0) < 0.90:
        failures.append("entity_key_accuracy below 0.90")
    if aggregate.get("atom_recall_by_type", 0.0) < 0.90:
        failures.append("atom_recall_by_type below 0.90")
    return failures
