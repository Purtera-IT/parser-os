from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.ids import stable_id


class FreezeResult(BaseModel):
    freeze_id: str
    experiment_id: str
    status: str
    proposed_files: list[str] = Field(default_factory=list)
    created_at: str
    message: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment report payload must be a JSON object.")
    return payload


def freeze_experiment_output(
    *,
    experiment_path: Path,
    approve: bool,
    out_dir: Path | None = None,
) -> FreezeResult:
    payload = _load_json(experiment_path)
    run = payload.get("experiment_run")
    if not isinstance(run, dict):
        raise ValueError("Experiment report missing experiment_run object.")
    experiment_id = str(run.get("experiment_id", "unknown_experiment"))
    out = (out_dir or (Path("tests/fixtures/regression/frozen") / experiment_id)).resolve()
    freeze_id = stable_id("freeze", experiment_id, str(experiment_path.resolve()))

    domain_alias_path = out / "domain_pack_aliases.yaml"
    parser_rule_fixture_path = out / "parser_rule_fixture.json"
    gold_regression_path = out / "gold_regression.json"
    frozen_candidate_set_path = out / "frozen_candidate_set.json"
    metadata_path = out / "freeze_metadata.json"
    proposed_files = [
        str(domain_alias_path),
        str(parser_rule_fixture_path),
        str(gold_regression_path),
        str(frozen_candidate_set_path),
        str(metadata_path),
    ]

    if not approve:
        return FreezeResult(
            freeze_id=freeze_id,
            experiment_id=experiment_id,
            status="proposed",
            proposed_files=proposed_files,
            created_at=_now_iso(),
            message="Approval required: rerun with --approve to write frozen outputs.",
        )

    out.mkdir(parents=True, exist_ok=True)
    accepted_ids = payload.get("accepted_candidate_atom_ids") or []
    candidate_ids = payload.get("candidate_ids") or []
    run_metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}

    domain_alias_payload = {
        "experiment_id": experiment_id,
        "domain_pack_id": payload.get("domain_pack_id"),
        "proposed_alias_updates": {},
        "note": "Frozen deterministic aliases from approved experiment.",
    }
    domain_alias_path.write_text(yaml.safe_dump(domain_alias_payload, sort_keys=False), encoding="utf-8")

    parser_rule_payload = {
        "experiment_id": experiment_id,
        "extractor_name": run.get("extractor_name"),
        "extractor_version": run.get("extractor_version"),
        "fixture_rule_count": len(candidate_ids),
        "note": "Parser-facing deterministic fixture metadata derived from experiment.",
    }
    parser_rule_fixture_path.write_text(json.dumps(parser_rule_payload, indent=2), encoding="utf-8")

    gold_payload = {
        "scenario_id": f"frozen_{experiment_id}",
        "project_dir": "project",
        "expected_packets": [],
        "expected_governing": [],
        "forbidden": [
            {"condition": "deleted_text_governs"},
            {"condition": "quoted_old_email_governs_current_conflict"},
        ],
    }
    gold_regression_path.write_text(json.dumps(gold_payload, indent=2), encoding="utf-8")

    frozen_candidate_set_payload = {
        "experiment_id": experiment_id,
        "accepted_candidate_atom_ids": accepted_ids,
        "all_candidate_ids": candidate_ids,
        "metrics": run_metrics,
        "frozen_at": _now_iso(),
    }
    frozen_candidate_set_path.write_text(json.dumps(frozen_candidate_set_payload, indent=2), encoding="utf-8")

    metadata = {
        "freeze_id": freeze_id,
        "experiment_id": experiment_id,
        "status": "applied",
        "created_at": _now_iso(),
        "source_experiment_file": str(experiment_path.resolve()),
        "requires_human_approval": True,
        "normal_compile_impact": "none_unless_frozen_rules_are_explicitly_loaded",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return FreezeResult(
        freeze_id=freeze_id,
        experiment_id=experiment_id,
        status="applied",
        proposed_files=proposed_files,
        created_at=metadata["created_at"],
        message="Frozen deterministic artifacts written.",
    )
