from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app.core.ids import stable_id
from app.domain.suggestions import RuleSuggestion


PromotionType = Literal[
    "new_gold_fixture",
    "domain_pack_alias",
    "parser_rule",
    "authority_rule_candidate",
    "packetizer_regression",
]

PromotionStatus = Literal["proposed", "approved", "rejected", "applied"]


class PromotionArtifact(BaseModel):
    promotion_id: str
    source_review_id: str
    source_compile_id: str
    promotion_type: PromotionType
    proposed_files: list[str] = Field(default_factory=list)
    patch_preview: str
    status: PromotionStatus = "proposed"
    created_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_rows(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _extract_review_rows(review_payload: Any) -> list[dict[str, Any]]:
    return _as_rows(review_payload, "reviews")


def _extract_reviewed_packet_ids(review_rows: list[dict[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for row in review_rows:
        packet_id = row.get("packet_id")
        if not isinstance(packet_id, str) or not packet_id.strip():
            continue
        if any(
            row.get(key) is not None
            for key in (
                "correct_packet",
                "correct_governing_atom",
                "correct_severity",
                "should_be_status",
                "missing_evidence",
                "false_positive_reason",
            )
        ):
            ids.add(packet_id)
    return sorted(ids)


def _atom_snippets_for_packets(compile_payload: dict[str, Any], packet_ids: list[str]) -> list[str]:
    packets = {
        str(row.get("id")): row
        for row in _as_rows(compile_payload, "packets")
    }
    atoms = {
        str(row.get("id")): row
        for row in _as_rows(compile_payload, "atoms")
    }
    snippets: list[str] = []
    seen: set[str] = set()
    for packet_id in packet_ids:
        packet = packets.get(packet_id)
        if not packet:
            continue
        atom_ids = []
        atom_ids.extend(packet.get("governing_atom_ids") or [])
        atom_ids.extend(packet.get("supporting_atom_ids") or [])
        atom_ids.extend(packet.get("contradicting_atom_ids") or [])
        for atom_id in atom_ids:
            atom = atoms.get(str(atom_id))
            if not atom:
                continue
            raw = str(atom.get("raw_text", "")).strip()
            if not raw:
                continue
            if raw in seen:
                continue
            seen.add(raw)
            snippets.append(raw)
    return snippets


def promote_review_to_fixture(
    *,
    review_labels_path: Path,
    compile_result_path: Path,
    out_dir: Path,
) -> PromotionArtifact:
    review_payload = _load_json(review_labels_path)
    compile_payload = _load_json(compile_result_path)
    review_rows = _extract_review_rows(review_payload)
    reviewed_packet_ids = _extract_reviewed_packet_ids(review_rows)
    if not reviewed_packet_ids:
        reviewed_packet_ids = [
            str(row.get("packet_id"))
            for row in review_rows
            if isinstance(row.get("packet_id"), str) and str(row.get("packet_id")).strip()
        ]
    snippets = _atom_snippets_for_packets(compile_payload, reviewed_packet_ids)
    if not snippets:
        snippets = ["Please validate scope and quantities for this regression fixture."]

    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = out_dir / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = project_dir / "review_regression_email.txt"
    fixture_lines = [
        "From: promotion-review@purtera.local",
        "Sent: 2026-01-01 09:00",
        "Subject: Regression fixture from reviewed labels",
        "",
        "These lines are synthesized from reviewer-approved snippets.",
    ]
    fixture_lines.extend(f"- {line}" for line in snippets[:20])
    artifact_file.write_text("\n".join(fixture_lines) + "\n", encoding="utf-8")

    packets = {
        str(row.get("id")): row
        for row in _as_rows(compile_payload, "packets")
    }
    expected_packets: list[dict[str, Any]] = []
    for packet_id in reviewed_packet_ids:
        packet = packets.get(packet_id)
        if not packet:
            continue
        expected_packets.append(
            {
                "family": packet.get("family", "missing_info"),
                "anchor_key_contains": "",
                "must_contain_quantities": [],
                "expected_status": packet.get("status"),
                "forbidden_governing_authority": [],
            }
        )
    if not expected_packets:
        expected_packets = [
            {
                "family": "missing_info",
                "anchor_key_contains": "",
                "must_contain_quantities": [],
                "expected_status": "needs_review",
                "forbidden_governing_authority": [],
            }
        ]
    gold_payload = {
        "scenario_id": out_dir.name,
        "project_dir": "project",
        "expected_packets": expected_packets,
        "expected_governing": [],
        "forbidden": [
            {"condition": "deleted_text_governs"},
            {"condition": "quoted_old_email_governs_current_conflict"},
        ],
    }
    gold_file = out_dir / "gold.json"
    gold_file.write_text(json.dumps(gold_payload, indent=2), encoding="utf-8")

    source_review_id = stable_id("review", review_labels_path.resolve(), len(review_rows))
    source_compile_id = str(compile_payload.get("compile_id") or "unknown_compile")
    promotion_id = stable_id("promo", source_review_id, source_compile_id, out_dir.resolve())
    metadata = {
        "review_labels_path": str(review_labels_path),
        "compile_result_path": str(compile_result_path),
        "reviewed_packet_ids": reviewed_packet_ids,
        "snippet_count": len(snippets),
    }
    metadata_file = out_dir / "promotion_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    proposed_files = [str(path) for path in sorted([artifact_file, gold_file, metadata_file], key=lambda p: str(p))]
    patch_preview = (
        f"Create regression fixture at {artifact_file} with {min(len(snippets), 20)} snippet lines; "
        f"create gold scenario at {gold_file}."
    )
    artifact = PromotionArtifact(
        promotion_id=promotion_id,
        source_review_id=source_review_id,
        source_compile_id=source_compile_id,
        promotion_type="new_gold_fixture",
        proposed_files=proposed_files,
        patch_preview=patch_preview,
        status="proposed",
        created_at=_now_iso(),
    )
    (out_dir / "promotion_artifact.json").write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return artifact


def _apply_to_domain_pack(suggestion: RuleSuggestion, *, test_file: Path, domain_pack_file: Path) -> str:
    payload = yaml.safe_load(domain_pack_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Domain pack at {domain_pack_file} is not a mapping.")
    change = suggestion.proposed_change
    if suggestion.suggestion_type == "domain_alias":
        additions = change.get("device_aliases", {})
        if isinstance(additions, dict):
            current = payload.setdefault("device_aliases", {})
            if not isinstance(current, dict):
                current = {}
                payload["device_aliases"] = current
            for canonical, aliases in additions.items():
                if not isinstance(canonical, str):
                    continue
                existing = current.setdefault(canonical, [])
                if not isinstance(existing, list):
                    existing = []
                    current[canonical] = existing
                for alias in aliases if isinstance(aliases, list) else []:
                    if alias not in existing:
                        existing.append(alias)
    elif suggestion.suggestion_type == "parser_header_alias":
        additions = change.get("parser_header_aliases", {})
        current = payload.setdefault("parser_header_aliases", {})
        if not isinstance(current, dict):
            current = {}
            payload["parser_header_aliases"] = current
        if isinstance(additions, dict):
            for field, aliases in additions.items():
                existing = current.setdefault(field, [])
                if not isinstance(existing, list):
                    existing = []
                    current[field] = existing
                for alias in aliases if isinstance(aliases, list) else []:
                    if alias not in existing:
                        existing.append(alias)
    elif suggestion.suggestion_type == "exclusion_pattern":
        rows = change.get("exclusion_patterns", [])
        existing = payload.setdefault("exclusion_patterns", [])
        if not isinstance(existing, list):
            existing = []
            payload["exclusion_patterns"] = existing
        for row in rows if isinstance(rows, list) else []:
            if row not in existing:
                existing.append(row)
    elif suggestion.suggestion_type == "constraint_pattern":
        additions = change.get("constraint_patterns", {})
        current = payload.setdefault("constraint_patterns", {})
        if not isinstance(current, dict):
            current = {}
            payload["constraint_patterns"] = current
        if isinstance(additions, dict):
            for family, patterns in additions.items():
                existing = current.setdefault(family, [])
                if not isinstance(existing, list):
                    existing = []
                    current[family] = existing
                for pattern in patterns if isinstance(patterns, list) else []:
                    if pattern not in existing:
                        existing.append(pattern)

    domain_pack_file.parent.mkdir(parents=True, exist_ok=True)
    domain_pack_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # Always create/update regression test fixture note before/with pack edits.
    test_file.parent.mkdir(parents=True, exist_ok=True)
    if not test_file.exists():
        test_file.write_text(
            "Regression guard for approved suggestion.\n",
            encoding="utf-8",
        )
    else:
        previous = test_file.read_text(encoding="utf-8")
        if "Regression guard for approved suggestion." not in previous:
            test_file.write_text(previous + "\nRegression guard for approved suggestion.\n", encoding="utf-8")
    return (
        f"Update domain pack {domain_pack_file} with suggestion {suggestion.suggestion_id}; "
        f"ensure regression fixture/test marker at {test_file}."
    )


def apply_approved_suggestion(
    *,
    suggestion_payload: dict[str, Any],
    approve: bool,
) -> PromotionArtifact:
    suggestion = RuleSuggestion.model_validate(suggestion_payload)
    status = str(suggestion_payload.get("status", "proposed")).lower()
    source_review_id = str(suggestion_payload.get("source_review_id") or suggestion.suggestion_id)
    source_compile_id = str(suggestion_payload.get("source_compile_id") or "unknown_compile")
    promotion_id = stable_id("promo_apply", suggestion.suggestion_id, source_review_id, source_compile_id)
    created_at = _now_iso()
    domain_pack_file = Path(str(suggestion.target_file or "app/domain/default_pack.yaml"))
    requested_test_file = suggestion_payload.get("test_file")
    if isinstance(requested_test_file, str) and requested_test_file.strip():
        test_file = Path(requested_test_file)
    else:
        test_file = Path("tests/fixtures/regression/promotions") / f"{suggestion.suggestion_id}.txt"

    promotion_type: PromotionType
    if suggestion.suggestion_type in {"domain_alias", "entity_normalization_rule"}:
        promotion_type = "domain_pack_alias"
    elif suggestion.suggestion_type in {"parser_header_alias", "constraint_pattern", "exclusion_pattern"}:
        promotion_type = "parser_rule"
    elif suggestion.suggestion_type in {"authority_override_candidate", "risk_default"}:
        promotion_type = "authority_rule_candidate"
    else:
        promotion_type = "packetizer_regression"

    proposed_files = [str(test_file), str(domain_pack_file)]
    patch_preview = (
        f"Would update {domain_pack_file} and add/update regression test artifact {test_file}. "
        "Explicit --approve is required."
    )

    if status == "rejected":
        return PromotionArtifact(
            promotion_id=promotion_id,
            source_review_id=source_review_id,
            source_compile_id=source_compile_id,
            promotion_type=promotion_type,
            proposed_files=proposed_files,
            patch_preview=patch_preview,
            status="rejected",
            created_at=created_at,
        )

    if not approve:
        return PromotionArtifact(
            promotion_id=promotion_id,
            source_review_id=source_review_id,
            source_compile_id=source_compile_id,
            promotion_type=promotion_type,
            proposed_files=proposed_files,
            patch_preview=patch_preview,
            status="proposed",
            created_at=created_at,
        )

    patch_preview = _apply_to_domain_pack(suggestion, test_file=test_file, domain_pack_file=domain_pack_file)
    return PromotionArtifact(
        promotion_id=promotion_id,
        source_review_id=source_review_id,
        source_compile_id=source_compile_id,
        promotion_type=promotion_type,
        proposed_files=proposed_files,
        patch_preview=patch_preview,
        status="applied",
        created_at=created_at,
    )
