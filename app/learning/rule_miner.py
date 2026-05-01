from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.domain.suggestions import RuleSuggestion


def _bool_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "approved", "accept", "correct"}:
            return True
        if lowered in {"0", "false", "no", "rejected", "reject", "incorrect"}:
            return False
    return None


def _confidence(pos: int, neg: int) -> float:
    total = pos + neg
    if total <= 0:
        return 0.0
    return round(pos / total, 6)


def _row_example(row: dict[str, Any]) -> str:
    for key in ("example_id", "id", "packet_id", "candidate_id", "target_id", "raw_text", "notes"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            compact = " ".join(value.strip().split())
            return compact[:180]
    return "example"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_mining_inputs(root_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    compile_results: list[dict[str, Any]] = []
    packet_labels: list[dict[str, Any]] = []
    candidate_labels: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    for path in sorted(root_dir.rglob("*.json"), key=lambda p: str(p).lower()):
        payload = _load_json(path)
        if not isinstance(payload, (dict, list)):
            continue

        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            if all(key in payload for key in ("atoms", "packets", "project_id")):
                compile_results.append(payload)
            if isinstance(payload.get("reviews"), list):
                rows = [row for row in payload["reviews"] if isinstance(row, dict)]
                packet_labels.extend(rows)
            if isinstance(payload.get("packet_labels"), list):
                for row in payload["packet_labels"]:
                    if not isinstance(row, dict):
                        continue
                    merged = dict(row)
                    human = row.get("human_label")
                    if isinstance(human, dict):
                        merged.update(human)
                    packet_labels.append(merged)
            if isinstance(payload.get("candidate_labels"), list):
                candidate_labels.extend([row for row in payload["candidate_labels"] if isinstance(row, dict)])
            if isinstance(payload.get("failure_records"), list):
                failure_records.extend([row for row in payload["failure_records"] if isinstance(row, dict)])
        else:
            rows = [row for row in payload if isinstance(row, dict)]
            if rows and any("category" in row and "severity" in row for row in rows):
                failure_records.extend(rows)
            elif rows and any("candidate_id" in row or "label_type" in row for row in rows):
                candidate_labels.extend(rows)
            elif rows and any("packet_id" in row or "correct_packet" in row for row in rows):
                packet_labels.extend(rows)
    return compile_results, packet_labels, candidate_labels, failure_records


def _mine_domain_alias(candidate_labels: list[dict[str, Any]], *, min_evidence: int) -> list[RuleSuggestion]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_labels:
        label_type = str(row.get("label_type", "")).lower()
        if label_type not in {"entity_alias", "alias", "same_as"}:
            continue
        canonical = str(row.get("canonical_key") or row.get("canonical") or "").strip()
        alias = str(row.get("alias") or row.get("candidate_text") or row.get("value") or "").strip()
        if not canonical or not alias:
            continue
        verdict = _bool_label(
            row.get("approved")
            if "approved" in row
            else row.get("same_entity", row.get("is_correct", row.get("label")))
        )
        if verdict is None:
            continue
        key = (canonical.lower(), alias.lower())
        bucket = buckets.setdefault(
            key,
            {"canonical": canonical, "alias": alias, "pos": 0, "neg": 0, "pos_examples": [], "neg_examples": []},
        )
        example = _row_example(row)
        if verdict:
            bucket["pos"] += 1
            bucket["pos_examples"].append(example)
        else:
            bucket["neg"] += 1
            bucket["neg_examples"].append(example)

    suggestions: list[RuleSuggestion] = []
    for bucket in buckets.values():
        pos = int(bucket["pos"])
        neg = int(bucket["neg"])
        evidence = pos + neg
        if pos < min_evidence:
            continue
        canonical = str(bucket["canonical"])
        alias = str(bucket["alias"])
        suggestions.append(
            RuleSuggestion(
                suggestion_id=stable_id("rule_suggest", "domain_alias", canonical, alias),
                suggestion_type="domain_alias",
                proposed_change={"device_aliases": {canonical: [alias]}},
                evidence_count=evidence,
                positive_examples=sorted(set(bucket["pos_examples"]))[:8],
                negative_examples=sorted(set(bucket["neg_examples"]))[:8],
                confidence=_confidence(pos, neg),
                requires_human_approval=True,
                target_file="app/domain/default_pack.yaml",
            )
        )
    return suggestions


def _mine_parser_header_alias(candidate_labels: list[dict[str, Any]], *, min_evidence: int) -> list[RuleSuggestion]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_labels:
        label_type = str(row.get("label_type", "")).lower()
        if label_type not in {"parser_header_alias", "header_alias"}:
            continue
        target = str(row.get("target") or row.get("field") or "").strip().lower()
        header = str(row.get("header") or row.get("candidate_text") or "").strip()
        if not target or not header:
            continue
        verdict = _bool_label(row.get("approved", row.get("is_correct", row.get("label"))))
        if verdict is None:
            continue
        key = (target, header.lower())
        bucket = buckets.setdefault(
            key,
            {"target": target, "header": header, "pos": 0, "neg": 0, "pos_examples": [], "neg_examples": []},
        )
        example = _row_example(row)
        if verdict:
            bucket["pos"] += 1
            bucket["pos_examples"].append(example)
        else:
            bucket["neg"] += 1
            bucket["neg_examples"].append(example)

    suggestions: list[RuleSuggestion] = []
    for bucket in buckets.values():
        pos = int(bucket["pos"])
        neg = int(bucket["neg"])
        if pos < min_evidence:
            continue
        target = str(bucket["target"])
        header = str(bucket["header"])
        suggestions.append(
            RuleSuggestion(
                suggestion_id=stable_id("rule_suggest", "parser_header_alias", target, header),
                suggestion_type="parser_header_alias",
                proposed_change={"parser_header_aliases": {target: [header]}},
                evidence_count=pos + neg,
                positive_examples=sorted(set(bucket["pos_examples"]))[:8],
                negative_examples=sorted(set(bucket["neg_examples"]))[:8],
                confidence=_confidence(pos, neg),
                requires_human_approval=True,
                target_file="app/domain/default_pack.yaml",
            )
        )
    return suggestions


def _mine_text_pattern_rules(candidate_labels: list[dict[str, Any]], *, min_evidence: int) -> list[RuleSuggestion]:
    pattern_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_labels:
        candidate_type = str(row.get("candidate_type", "")).lower()
        if candidate_type not in {"exclusion", "constraint"}:
            continue
        verdict = _bool_label(row.get("approved", row.get("is_correct", row.get("label"))))
        if verdict is not True:
            continue
        raw_text = str(row.get("raw_text") or row.get("candidate_text") or "").strip()
        if len(raw_text) < 8:
            continue
        normalized = " ".join(raw_text.lower().split())
        normalized = re.sub(r"[^\w\s\-]", "", normalized)
        suggestion_type = "exclusion_pattern" if candidate_type == "exclusion" else "constraint_pattern"
        key = (suggestion_type, normalized)
        bucket = pattern_buckets.setdefault(
            key,
            {"type": suggestion_type, "pattern": normalized, "count": 0, "examples": []},
        )
        bucket["count"] += 1
        bucket["examples"].append(_row_example(row))

    suggestions: list[RuleSuggestion] = []
    for bucket in pattern_buckets.values():
        count = int(bucket["count"])
        if count < min_evidence:
            continue
        suggestion_type = str(bucket["type"])
        pattern = str(bucket["pattern"])
        proposed = (
            {"exclusion_patterns": [pattern]}
            if suggestion_type == "exclusion_pattern"
            else {"constraint_patterns": {"general": [pattern]}}
        )
        suggestions.append(
            RuleSuggestion(
                suggestion_id=stable_id("rule_suggest", suggestion_type, pattern),
                suggestion_type=suggestion_type,  # type: ignore[arg-type]
                proposed_change=proposed,
                evidence_count=count,
                positive_examples=sorted(set(bucket["examples"]))[:8],
                negative_examples=[],
                confidence=1.0,
                requires_human_approval=True,
                target_file="app/domain/default_pack.yaml",
            )
        )
    return suggestions


def _mine_failure_rules(failure_records: list[dict[str, Any]], *, min_evidence: int) -> list[RuleSuggestion]:
    suggestions: list[RuleSuggestion] = []
    false_merge_rows = [
        row
        for row in failure_records
        if str(row.get("category", "")).upper() == "ENTITY_FALSE_MERGE"
    ]
    if len(false_merge_rows) >= min_evidence:
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        examples: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in false_merge_rows:
            message = str(row.get("message", ""))
            quoted = re.findall(r"'([^']+)'", message)
            if len(quoted) >= 2:
                left, right = quoted[0], quoted[1]
            else:
                left, right = "entity_a", "entity_b"
            key = tuple(sorted((left, right)))
            pair_counts[key] += 1
            examples[key].append(_row_example(row))
        for pair, count in pair_counts.items():
            if count < min_evidence:
                continue
            suggestions.append(
                RuleSuggestion(
                    suggestion_id=stable_id("rule_suggest", "entity_normalization_rule", pair[0], pair[1]),
                    suggestion_type="entity_normalization_rule",
                    proposed_change={"do_not_merge": [[pair[0], pair[1]]]},
                    evidence_count=count,
                    positive_examples=sorted(set(examples[pair]))[:8],
                    negative_examples=[],
                    confidence=1.0,
                    requires_human_approval=True,
                    target_file="app/domain/default_pack.yaml",
                )
            )
    return suggestions


def mine_rule_suggestions(
    *,
    compile_results: list[dict[str, Any]] | None = None,
    packet_labels: list[dict[str, Any]] | None = None,
    candidate_labels: list[dict[str, Any]] | None = None,
    failure_records: list[dict[str, Any]] | None = None,
    min_evidence: int = 2,
) -> list[RuleSuggestion]:
    _ = compile_results or []
    packet_labels = packet_labels or []
    candidate_labels = candidate_labels or []
    failure_records = failure_records or []
    suggestions: list[RuleSuggestion] = []
    suggestions.extend(_mine_domain_alias(candidate_labels, min_evidence=min_evidence))
    suggestions.extend(_mine_parser_header_alias(candidate_labels, min_evidence=min_evidence))
    suggestions.extend(_mine_text_pattern_rules(candidate_labels, min_evidence=min_evidence))
    suggestions.extend(_mine_failure_rules(failure_records, min_evidence=min_evidence))

    # Soft signal: repeated incorrect severity labels can suggest default risk tuning.
    severity_no = [
        row
        for row in packet_labels
        if _bool_label(row.get("severity_correct", row.get("correct_severity"))) is False
    ]
    if len(severity_no) >= min_evidence:
        families = sorted(
            {
                str(row.get("family", "unknown"))
                for row in severity_no
                if str(row.get("family", "")).strip()
            }
        )
        suggestions.append(
            RuleSuggestion(
                suggestion_id=stable_id("rule_suggest", "risk_default", *families),
                suggestion_type="risk_default",
                proposed_change={"risk_default_adjustments": {family: "review_thresholds" for family in families}},
                evidence_count=len(severity_no),
                positive_examples=[],
                negative_examples=sorted({_row_example(row) for row in severity_no})[:8],
                confidence=0.5,
                requires_human_approval=True,
                target_file="app/domain/default_pack.yaml",
            )
        )

    suggestions.sort(key=lambda row: (row.suggestion_type, -row.confidence, row.suggestion_id))
    return suggestions
