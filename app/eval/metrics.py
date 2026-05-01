from __future__ import annotations

from dataclasses import dataclass

from app.core.schemas import AuthorityClass, PacketStatus, ReviewStatus
from app.eval.gold import GoldScenario


@dataclass
class ScenarioMetricValues:
    packet_family_recall: float
    packet_anchor_recall: float
    governing_accuracy: float
    contradiction_recall: float
    receipt_coverage: float
    verified_receipt_rate: float
    false_active_rate: float
    invalid_governance_count: int
    determinism_pass: bool
    compile_latency_ms: float
    packet_count: int
    atom_count: int
    compile_success: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "packet_family_recall": round(self.packet_family_recall, 4),
            "packet_anchor_recall": round(self.packet_anchor_recall, 4),
            "governing_accuracy": round(self.governing_accuracy, 4),
            "contradiction_recall": round(self.contradiction_recall, 4),
            "receipt_coverage": round(self.receipt_coverage, 4),
            "verified_receipt_rate": round(self.verified_receipt_rate, 4),
            "false_active_rate": round(self.false_active_rate, 4),
            "invalid_governance_count": self.invalid_governance_count,
            "determinism_pass": self.determinism_pass,
            "compile_latency_ms": round(self.compile_latency_ms, 2),
            "packet_count": self.packet_count,
            "atom_count": self.atom_count,
            "compile_success": self.compile_success,
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _packet_match(packet, family: str, anchor_fragment: str) -> bool:
    return packet.family.value == family and anchor_fragment in packet.anchor_key


def _expected_conflict_count(gold: GoldScenario) -> int:
    return sum(1 for entry in gold.expected_packets if "conflict" in entry.family or entry.family == "vendor_mismatch")


def _invalid_governance(result) -> int:
    atom_by_id = {atom.id: atom for atom in result.atoms}
    invalid = 0
    for packet in result.packets:
        governing_atoms = [atom_by_id[aid] for aid in packet.governing_atom_ids if aid in atom_by_id]
        if any(atom.authority_class == AuthorityClass.deleted_text for atom in governing_atoms):
            invalid += 1
        if any(atom.review_status == ReviewStatus.rejected for atom in governing_atoms):
            invalid += 1
        if any(atom.authority_class == AuthorityClass.quoted_old_email for atom in governing_atoms):
            current_customer_exists = any(
                atom.authority_class == AuthorityClass.customer_current_authored
                and packet.anchor_key in atom.entity_keys
                for atom in result.atoms
            )
            if current_customer_exists:
                invalid += 1
    return invalid


def _forbidden_failures(result, gold: GoldScenario) -> list[str]:
    atom_by_id = {atom.id: atom for atom in result.atoms}
    failures: list[str] = []
    for forbidden in gold.forbidden:
        condition = forbidden.condition
        if condition == "deleted_text_governs":
            if any(
                atom_by_id.get(aid) and atom_by_id[aid].authority_class == AuthorityClass.deleted_text
                for packet in result.packets
                for aid in packet.governing_atom_ids
            ):
                failures.append(condition)
        if condition == "quoted_old_email_governs_current_conflict":
            for packet in result.packets:
                governing = [atom_by_id.get(aid) for aid in packet.governing_atom_ids]
                if any(atom and atom.authority_class == AuthorityClass.quoted_old_email for atom in governing):
                    current = any(
                        atom.authority_class == AuthorityClass.customer_current_authored
                        and packet.anchor_key in atom.entity_keys
                        for atom in result.atoms
                    )
                    if current:
                        failures.append(condition)
                        break
    return sorted(set(failures))


def evaluate_scenario_metrics(
    result,
    gold: GoldScenario,
    *,
    compile_latency_ms: float,
    determinism_pass: bool,
    compile_success: bool,
) -> tuple[ScenarioMetricValues, list[str]]:
    atom_by_id = {atom.id: atom for atom in result.atoms}

    family_hits = 0
    anchor_hits = 0
    governing_hits = 0
    contradiction_hits = 0
    for expected in gold.expected_packets:
        family_found = any(packet.family.value == expected.family for packet in result.packets)
        if family_found:
            family_hits += 1
        matched_packets = [packet for packet in result.packets if _packet_match(packet, expected.family, expected.anchor_key_contains)]
        if matched_packets:
            anchor_hits += 1
            if expected.family in {"quantity_conflict", "vendor_mismatch"}:
                contradiction_hits += 1
    for expected in gold.expected_governing:
        matched_packet = next(
            (
                packet
                for packet in result.packets
                if _packet_match(packet, expected.family, expected.anchor_key_contains)
            ),
            None,
        )
        if matched_packet is None:
            continue
        governing_atoms = [atom_by_id.get(aid) for aid in matched_packet.governing_atom_ids]
        if any(atom and atom.authority_class.value == expected.governing_authority for atom in governing_atoms):
            governing_hits += 1

    atoms_with_receipts = [
        atom
        for atom in result.atoms
        if atom.receipts and all(receipt.replay_status in {"verified", "unsupported"} for receipt in atom.receipts)
    ]
    all_receipts = [receipt for atom in result.atoms for receipt in atom.receipts]
    verified_receipts = [receipt for receipt in all_receipts if receipt.replay_status == "verified"]

    active_packets = [packet for packet in result.packets if packet.status == PacketStatus.active]
    bad_active = [
        packet
        for packet in active_packets
        if "contradiction_present" in packet.review_flags or bool(packet.contradicting_atom_ids)
    ]

    values = ScenarioMetricValues(
        packet_family_recall=_safe_ratio(family_hits, len(gold.expected_packets)),
        packet_anchor_recall=_safe_ratio(anchor_hits, len(gold.expected_packets)),
        governing_accuracy=_safe_ratio(governing_hits, len(gold.expected_governing)),
        contradiction_recall=_safe_ratio(contradiction_hits, _expected_conflict_count(gold)),
        receipt_coverage=_safe_ratio(len(atoms_with_receipts), len(result.atoms)),
        verified_receipt_rate=_safe_ratio(len(verified_receipts), len(all_receipts)),
        false_active_rate=_safe_ratio(len(bad_active), len(active_packets)),
        invalid_governance_count=_invalid_governance(result),
        determinism_pass=determinism_pass,
        compile_latency_ms=compile_latency_ms,
        packet_count=len(result.packets),
        atom_count=len(result.atoms),
        compile_success=compile_success,
    )
    failures = _forbidden_failures(result, gold)
    return values, failures
