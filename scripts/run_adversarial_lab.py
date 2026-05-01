from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.compiler import compile_project
from app.core.schemas import AuthorityClass, PacketFamily, PacketStatus
from app.eval.failure_taxonomy import FailureCategory, make_failure_record
from app.testing.scenarios import default_mutations, generate_scenario

EXPECTED_FAMILIES = {
    PacketFamily.quantity_conflict.value,
    PacketFamily.scope_exclusion.value,
    PacketFamily.site_access.value,
    PacketFamily.missing_info.value,
}


def _check_invariants(result) -> list[str]:
    errors: list[str] = []
    atoms_by_id = {atom.id: atom for atom in result.atoms}
    if not all(atom.source_refs for atom in result.atoms):
        errors.append("missing_source_refs")
    if not all(
        packet.status in {PacketStatus.rejected, PacketStatus.invalidated}
        or packet.governing_atom_ids
        for packet in result.packets
    ):
        errors.append("missing_governing_for_active_or_needs_review")
    governed = [atoms_by_id[atom_id] for packet in result.packets for atom_id in packet.governing_atom_ids if atom_id in atoms_by_id]
    if any(atom.authority_class == AuthorityClass.deleted_text for atom in governed):
        errors.append("deleted_text_governing")
    if any(atom.authority_class == AuthorityClass.quoted_old_email for atom in governed):
        errors.append("quoted_old_email_governing")

    families = {packet.family.value for packet in result.packets}
    for family in EXPECTED_FAMILIES:
        if family not in families:
            errors.append(f"missing_family:{family}")
    if any(packet.family == PacketFamily.quantity_conflict and packet.status == PacketStatus.active for packet in result.packets):
        errors.append("false_active_conflict")
    return errors


def run_lab(count: int, out: Path, seed: int = 1000) -> dict:
    scenarios_root = Path(tempfile.mkdtemp(prefix="purtera_adversarial_"))
    metrics = {
        "total_scenarios": count,
        "compile_pass_count": 0,
        "hard_error_count": 0,
        "expected_packet_recall_by_family": {family: 0 for family in sorted(EXPECTED_FAMILIES)},
        "false_active_conflicts": 0,
        "invalid_receipt_count": 0,
        "determinism_failures": 0,
    }
    mutation_family_coverage: dict[str, set[str]] = defaultdict(set)
    scenario_results: list[dict] = []
    failure_records: list[dict] = []

    for i in range(count):
        scenario_seed = seed + i
        mutation_set = default_mutations(scenario_seed)
        scenario_dir = generate_scenario(scenario_seed, mutation_set, output_root=scenarios_root)
        for family, mutation_name in mutation_set.items():
            mutation_family_coverage[family].add(mutation_name)

        scenario_summary: dict[str, object] = {
            "seed": scenario_seed,
            "scenario_dir": str(scenario_dir),
            "mutations": mutation_set,
            "status": "pass",
            "invariant_errors": [],
            "failure_records": [],
        }
        try:
            first = compile_project(scenario_dir, project_id=f"adv_{scenario_seed}", allow_unverified_receipts=True)
            second = compile_project(scenario_dir, project_id=f"adv_{scenario_seed}", allow_unverified_receipts=True)
            metrics["compile_pass_count"] += 1

            if (first.manifest and second.manifest and first.manifest.output_signature != second.manifest.output_signature):
                metrics["determinism_failures"] += 1
                scenario_summary["status"] = "fail"
                scenario_summary["invariant_errors"] = ["determinism_failure"]
                record = make_failure_record(
                    category=FailureCategory.NON_DETERMINISTIC_OUTPUT,
                    severity="critical",
                    scenario_id=f"adv_{scenario_seed}",
                    message="Deterministic replay failed in adversarial lab",
                )
                scenario_summary["failure_records"].append(record.model_dump())
                failure_records.append(record.model_dump())

            invariant_errors = _check_invariants(first)
            if invariant_errors:
                scenario_summary["status"] = "fail"
                scenario_summary["invariant_errors"] = invariant_errors
                for invariant in invariant_errors:
                    category = None
                    if invariant.startswith("missing_family:"):
                        category = FailureCategory.PACKET_MISSING_EXPECTED
                    elif invariant == "missing_source_refs":
                        category = FailureCategory.SOURCE_REF_MISSING
                    elif invariant == "deleted_text_governing":
                        category = FailureCategory.INVALID_DELETED_TEXT_GOVERNANCE
                    elif invariant == "quoted_old_email_governing":
                        category = FailureCategory.INVALID_QUOTED_EMAIL_GOVERNANCE
                    elif invariant == "false_active_conflict":
                        category = FailureCategory.PACKET_BAD_STATUS
                    if category is not None:
                        record = make_failure_record(
                            category=category,
                            severity="high",
                            scenario_id=f"adv_{scenario_seed}",
                            message=f"Adversarial invariant failure: {invariant}",
                        )
                        scenario_summary["failure_records"].append(record.model_dump())
                        failure_records.append(record.model_dump())
                if "false_active_conflict" in invariant_errors:
                    metrics["false_active_conflicts"] += 1
            families = {packet.family.value for packet in first.packets}
            for family in EXPECTED_FAMILIES:
                if family in families:
                    metrics["expected_packet_recall_by_family"][family] += 1
            metrics["invalid_receipt_count"] += sum(
                1
                for atom in first.atoms
                for receipt in atom.receipts
                if receipt.replay_status == "failed"
            )
            scenario_summary["packet_count"] = len(first.packets)
            scenario_summary["warning_count"] = len(first.warnings)
        except Exception as exc:  # pragma: no cover
            metrics["hard_error_count"] += 1
            scenario_summary["status"] = "error"
            scenario_summary["error"] = str(exc)
            record = make_failure_record(
                category=FailureCategory.PARSER_CRASH,
                severity="critical",
                scenario_id=f"adv_{scenario_seed}",
                message=f"Adversarial compile crashed: {exc}",
            )
            scenario_summary["failure_records"].append(record.model_dump())
            failure_records.append(record.model_dump())
        scenario_results.append(scenario_summary)

    report = {
        "metrics": metrics,
        "mutation_family_coverage": {key: sorted(values) for key, values in mutation_family_coverage.items()},
        "scenarios": scenario_results,
        "failure_records": failure_records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic adversarial compile lab")
    parser.add_argument("--count", type=int, default=25, help="Number of adversarial scenarios")
    parser.add_argument("--seed", type=int, default=1000, help="Base deterministic seed")
    parser.add_argument("--out", type=Path, required=True, help="Report output path")
    args = parser.parse_args()
    report = run_lab(count=args.count, out=args.out, seed=args.seed)
    print(
        json.dumps(
            {
                "total_scenarios": report["metrics"]["total_scenarios"],
                "compile_pass_count": report["metrics"]["compile_pass_count"],
                "hard_error_count": report["metrics"]["hard_error_count"],
                "determinism_failures": report["metrics"]["determinism_failures"],
            }
        )
    )


if __name__ == "__main__":
    main()
