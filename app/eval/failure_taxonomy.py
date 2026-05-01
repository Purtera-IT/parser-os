from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.core.ids import stable_id
from app.eval.gold import GoldScenario


class FailureCategory(str, Enum):
    SOURCE_REF_MISSING = "SOURCE_REF_MISSING"
    SOURCE_REPLAY_FAILED = "SOURCE_REPLAY_FAILED"
    PARSER_CRASH = "PARSER_CRASH"
    PARSER_MISCLASSIFIED_AUTHORITY = "PARSER_MISCLASSIFIED_AUTHORITY"
    PARSER_MISSED_QUANTITY = "PARSER_MISSED_QUANTITY"
    PARSER_MISSED_EXCLUSION = "PARSER_MISSED_EXCLUSION"
    ENTITY_FALSE_MERGE = "ENTITY_FALSE_MERGE"
    ENTITY_FALSE_SPLIT = "ENTITY_FALSE_SPLIT"
    EDGE_MISSING_CONTRADICTION = "EDGE_MISSING_CONTRADICTION"
    EDGE_FALSE_CONTRADICTION = "EDGE_FALSE_CONTRADICTION"
    PACKET_MISSING_EXPECTED = "PACKET_MISSING_EXPECTED"
    PACKET_FALSE_POSITIVE = "PACKET_FALSE_POSITIVE"
    PACKET_BAD_GOVERNING_ATOM = "PACKET_BAD_GOVERNING_ATOM"
    PACKET_BAD_STATUS = "PACKET_BAD_STATUS"
    PACKET_BAD_SEVERITY = "PACKET_BAD_SEVERITY"
    INVALID_DELETED_TEXT_GOVERNANCE = "INVALID_DELETED_TEXT_GOVERNANCE"
    INVALID_QUOTED_EMAIL_GOVERNANCE = "INVALID_QUOTED_EMAIL_GOVERNANCE"
    NON_DETERMINISTIC_OUTPUT = "NON_DETERMINISTIC_OUTPUT"
    PERF_BUDGET_EXCEEDED = "PERF_BUDGET_EXCEEDED"


class FailureRecord(BaseModel):
    failure_id: str
    category: FailureCategory
    severity: str
    scenario_id: str
    artifact_name: str | None = None
    atom_id: str | None = None
    edge_id: str | None = None
    packet_id: str | None = None
    message: str
    suggested_fix: str
    created_at: str


_SUGGESTED_FIXES: dict[FailureCategory, str] = {
    FailureCategory.SOURCE_REF_MISSING: "Ensure parser emits SourceRef entries for every atom.",
    FailureCategory.SOURCE_REPLAY_FAILED: "Fix locator extraction and replay verification logic for this parser.",
    FailureCategory.PARSER_CRASH: "Add parser guards and exception-safe fallback extraction.",
    FailureCategory.PARSER_MISCLASSIFIED_AUTHORITY: "Improve authority classification rules and parser metadata hints.",
    FailureCategory.PARSER_MISSED_QUANTITY: "Expand numeric extraction patterns and quantity normalization.",
    FailureCategory.PARSER_MISSED_EXCLUSION: "Strengthen exclusion phrase extraction rules for transcripts/emails.",
    FailureCategory.ENTITY_FALSE_MERGE: "Tighten alias merge thresholds for dissimilar entities.",
    FailureCategory.ENTITY_FALSE_SPLIT: "Add canonical alias mappings to reduce duplicate entities.",
    FailureCategory.EDGE_MISSING_CONTRADICTION: "Refine contradiction edge construction rules for quantity mismatches.",
    FailureCategory.EDGE_FALSE_CONTRADICTION: "Add scope-aware guards to prevent spurious contradiction edges.",
    FailureCategory.PACKET_MISSING_EXPECTED: "Adjust parser/packetizer rules so expected packet families are emitted.",
    FailureCategory.PACKET_FALSE_POSITIVE: "Tighten packet creation criteria and de-duplication constraints.",
    FailureCategory.PACKET_BAD_GOVERNING_ATOM: "Refine authority lattice and governing-atom selection logic.",
    FailureCategory.PACKET_BAD_STATUS: "Revisit packet status transitions for contradiction/review conditions.",
    FailureCategory.PACKET_BAD_SEVERITY: "Tune risk scoring and severity thresholds for packet families.",
    FailureCategory.INVALID_DELETED_TEXT_GOVERNANCE: "Block deleted_text atoms from governing decisions.",
    FailureCategory.INVALID_QUOTED_EMAIL_GOVERNANCE: "Prevent quoted_old_email from governing when current customer evidence exists.",
    FailureCategory.NON_DETERMINISTIC_OUTPUT: "Enforce stable ordering and deterministic signatures across compile runs.",
    FailureCategory.PERF_BUDGET_EXCEEDED: "Profile slow stages and optimize parser/replay/packetization hotspots.",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_failure_record(
    *,
    category: FailureCategory,
    severity: str,
    scenario_id: str,
    message: str,
    artifact_name: str | None = None,
    atom_id: str | None = None,
    edge_id: str | None = None,
    packet_id: str | None = None,
    suggested_fix: str | None = None,
) -> FailureRecord:
    failure_id = stable_id(
        "failure",
        category.value,
        scenario_id,
        artifact_name or "",
        atom_id or "",
        edge_id or "",
        packet_id or "",
        message,
    )
    return FailureRecord(
        failure_id=failure_id,
        category=category,
        severity=severity,
        scenario_id=scenario_id,
        artifact_name=artifact_name,
        atom_id=atom_id,
        edge_id=edge_id,
        packet_id=packet_id,
        message=message,
        suggested_fix=suggested_fix or _SUGGESTED_FIXES[category],
        created_at=_now_iso(),
    )


def _extract_id(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1)


def classify_validation_error(message: str, *, scenario_id: str) -> FailureRecord:
    msg = message.strip()
    msg_lower = msg.lower()
    packet_id = _extract_id(r"Packet\s+([a-zA-Z0-9_]+)", msg)
    atom_id = _extract_id(r"Atom\s+([a-zA-Z0-9_]+)", msg)
    edge_id = _extract_id(r"Edge\s+([a-zA-Z0-9_]+)", msg)

    if "no source_refs" in msg_lower:
        category = FailureCategory.SOURCE_REF_MISSING
    elif "failed receipt" in msg_lower:
        category = FailureCategory.SOURCE_REPLAY_FAILED
    elif "deleted_text governing" in msg_lower:
        category = FailureCategory.INVALID_DELETED_TEXT_GOVERNANCE
    elif "quoted_old_email" in msg_lower and "govern" in msg_lower:
        category = FailureCategory.INVALID_QUOTED_EMAIL_GOVERNANCE
    elif "references missing from_atom_id" in msg_lower or "references missing to_atom_id" in msg_lower:
        category = FailureCategory.SOURCE_REF_MISSING
    elif "anchor_signature hash mismatch" in msg_lower:
        category = FailureCategory.NON_DETERMINISTIC_OUTPUT
    else:
        category = FailureCategory.PACKET_BAD_STATUS if packet_id else FailureCategory.PARSER_CRASH

    return make_failure_record(
        category=category,
        severity="high",
        scenario_id=scenario_id,
        message=msg,
        atom_id=atom_id,
        edge_id=edge_id,
        packet_id=packet_id,
    )


def failure_records_from_validation_messages(
    messages: list[str],
    *,
    scenario_id: str,
) -> list[FailureRecord]:
    records: list[FailureRecord] = []
    for message in messages:
        if not str(message).startswith("ERROR:"):
            continue
        records.append(classify_validation_error(message, scenario_id=scenario_id))
    return sorted(records, key=lambda row: (row.category.value, row.failure_id))


def failure_records_from_expected_label_mismatches(
    result: Any,
    gold: GoldScenario,
    *,
    scenario_id: str,
) -> list[FailureRecord]:
    records: list[FailureRecord] = []
    for expected in gold.expected_packets:
        matched = [
            packet
            for packet in result.packets
            if packet.family.value == expected.family and expected.anchor_key_contains in packet.anchor_key
        ]
        if not matched:
            records.append(
                make_failure_record(
                    category=FailureCategory.PACKET_MISSING_EXPECTED,
                    severity="high",
                    scenario_id=scenario_id,
                    message=(
                        f"Expected packet missing for family={expected.family} "
                        f"anchor_contains={expected.anchor_key_contains}"
                    ),
                    packet_id=None,
                )
            )
            continue
        packet = matched[0]
        if expected.expected_status and packet.status.value != expected.expected_status:
            records.append(
                make_failure_record(
                    category=FailureCategory.PACKET_BAD_STATUS,
                    severity="medium",
                    scenario_id=scenario_id,
                    message=(
                        f"Packet {packet.id} expected status {expected.expected_status} "
                        f"but got {packet.status.value}"
                    ),
                    packet_id=packet.id,
                )
            )
    return sorted(records, key=lambda row: (row.category.value, row.failure_id))


def summarize_failure_records(records: list[FailureRecord]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    by_fix: dict[str, int] = {}
    for record in records:
        by_category[record.category.value] = by_category.get(record.category.value, 0) + 1
        by_fix[record.suggested_fix] = by_fix.get(record.suggested_fix, 0) + 1
    return {
        "total_failures": len(records),
        "by_category": [
            {"category": category, "count": count}
            for category, count in sorted(by_category.items(), key=lambda item: (-item[1], item[0]))
        ],
        "by_suggested_fix": [
            {"suggested_fix": fix, "count": count}
            for fix, count in sorted(by_fix.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
