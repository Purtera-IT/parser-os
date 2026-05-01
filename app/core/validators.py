from __future__ import annotations

import re

from app.core.anchors import make_anchor_signature
from app.core.graph_invariants import check_graph_invariants
from app.eval.failure_taxonomy import FailureRecord, failure_records_from_validation_messages
from app.core.schemas import (
    AuthorityClass,
    CompileResult,
    EvidenceAtom,
    EvidencePacket,
    PacketFamily,
    PacketStatus,
    ReviewStatus,
)


def validate_atom(atom: EvidenceAtom) -> EvidenceAtom:
    if len(atom.source_refs) < 1:
        raise ValueError("Every EvidenceAtom must have at least one SourceRef")
    if not atom.raw_text.strip() or not atom.normalized_text.strip():
        raise ValueError("raw_text and normalized_text cannot be empty")
    if not (0.0 <= atom.confidence <= 1.0):
        raise ValueError("EvidenceAtom confidence must be in range 0..1")
    return atom


def validate_packet(packet: EvidencePacket) -> EvidencePacket:
    requires_governing = packet.status in {PacketStatus.active, PacketStatus.needs_review}
    if requires_governing and len(packet.governing_atom_ids) < 1:
        raise ValueError("Active and needs_review packets require a governing atom")
    return packet


def validate_compile_result(result: CompileResult, source_files_available: bool = False) -> list[str]:
    messages: list[str] = []
    atom_by_id = {atom.id: atom for atom in result.atoms}
    edge_by_id = {edge.id: edge for edge in result.edges}

    if not result.schema_version:
        messages.append("ERROR: CompileResult missing schema_version")
    if not result.compiler_version:
        messages.append("ERROR: CompileResult missing compiler_version")
    if not result.compile_id:
        messages.append("ERROR: CompileResult missing compile_id")
    if result.manifest is None:
        messages.append("ERROR: CompileResult missing manifest")
    else:
        if result.manifest.compile_id != result.compile_id:
            messages.append("ERROR: CompileResult compile_id does not match manifest.compile_id")
        if not result.manifest.input_signature:
            messages.append("ERROR: CompileManifest missing input_signature")
        if not result.manifest.output_signature:
            messages.append("WARNING: CompileManifest missing output_signature")
    if result.candidate_summary is not None:
        summary = result.candidate_summary
        if summary.candidate_count < (summary.accepted_count + summary.rejected_count):
            messages.append("ERROR: candidate_summary counts are inconsistent")
        if summary.needs_review_count > summary.accepted_count:
            messages.append("ERROR: candidate_summary needs_review_count exceeds accepted_count")

    # Errors: atom-level validation
    for atom in result.atoms:
        if not atom.source_refs:
            messages.append(f"ERROR: Atom {atom.id} has no source_refs")
        if not atom.normalized_text.strip():
            messages.append(f"ERROR: Atom {atom.id} has empty normalized_text")
        if source_files_available and atom.source_refs and not atom.receipts:
            messages.append(f"ERROR: Atom {atom.id} has no receipts while source files are available")
        if atom.calibrated_confidence is not None and not (0.0 <= atom.calibrated_confidence <= 1.0):
            messages.append(f"ERROR: Atom {atom.id} calibrated_confidence must be in range 0..1")
        if "calibration_abstain" in atom.review_flags and atom.review_status != ReviewStatus.needs_review:
            messages.append(f"ERROR: Atom {atom.id} calibration_abstain requires needs_review status")
        for receipt in atom.receipts:
            if receipt.replay_status == "failed":
                messages.append(f"ERROR: Atom {atom.id} has failed receipt {receipt.source_ref_id}")
            if receipt.replay_status == "unsupported":
                messages.append(f"WARNING: Atom {atom.id} has unsupported receipt {receipt.source_ref_id}")

    # Errors: packet-level validation + warnings.
    for packet in result.packets:
        certificate = packet.certificate
        if packet.certificate is None:
            messages.append(f"ERROR: Packet {packet.id} is missing certificate")
        if certificate is not None and not certificate.existence_reason.strip():
            messages.append(f"ERROR: Packet {packet.id} certificate existence_reason cannot be empty")

        requires_governing = packet.status in {PacketStatus.active, PacketStatus.needs_review}
        if requires_governing and not packet.governing_atom_ids:
            messages.append(f"ERROR: Packet {packet.id} requires governing_atom_ids for status {packet.status.value}")
        if requires_governing and certificate is not None and not certificate.minimal_sufficient_atom_ids:
            messages.append(
                f"ERROR: Packet {packet.id} requires non-empty certificate minimal_sufficient_atom_ids for status {packet.status.value}"
            )
        if packet.family == PacketFamily.quantity_conflict and not packet.contradicting_atom_ids:
            messages.append(f"ERROR: quantity_conflict packet {packet.id} has no contradicting_atom_ids")

        referenced_ids = (
            list(packet.governing_atom_ids)
            + list(packet.supporting_atom_ids)
            + list(packet.contradicting_atom_ids)
        )
        for atom_id in referenced_ids:
            if atom_id not in atom_by_id:
                messages.append(f"ERROR: Packet {packet.id} references missing atom ID {atom_id}")

        allowed_minimal_ids = set(referenced_ids)
        if certificate is not None:
            for atom_id in certificate.minimal_sufficient_atom_ids:
                if atom_id not in allowed_minimal_ids:
                    messages.append(
                        f"ERROR: Packet {packet.id} certificate minimal atom {atom_id} is not part of packet evidence ids"
                    )
                if atom_id not in atom_by_id:
                    messages.append(f"ERROR: Packet {packet.id} certificate references missing atom ID {atom_id}")

        for atom_id in packet.governing_atom_ids:
            atom = atom_by_id.get(atom_id)
            if atom is None:
                continue
            if atom.authority_class == AuthorityClass.deleted_text:
                messages.append(f"ERROR: Packet {packet.id} has deleted_text governing atom {atom_id}")

        # quoted_old_email governance error when current customer authored atom exists for same anchor.
        has_current_customer_for_anchor = any(
            atom.authority_class == AuthorityClass.customer_current_authored
            and packet.anchor_key in atom.entity_keys
            for atom in result.atoms
        )
        if has_current_customer_for_anchor:
            for atom_id in packet.governing_atom_ids:
                atom = atom_by_id.get(atom_id)
                if atom and atom.authority_class == AuthorityClass.quoted_old_email:
                    messages.append(
                        f"ERROR: Packet {packet.id} governed by quoted_old_email while customer_current_authored exists for anchor {packet.anchor_key}"
                    )

        if packet.status == PacketStatus.needs_review:
            messages.append(f"WARNING: Packet {packet.id} status is needs_review")
        if packet.calibrated_confidence is not None and not (0.0 <= packet.calibrated_confidence <= 1.0):
            messages.append(f"ERROR: Packet {packet.id} calibrated_confidence must be in range 0..1")
        if "calibration_abstain" in packet.review_flags and packet.status != PacketStatus.needs_review:
            messages.append(f"ERROR: Packet {packet.id} calibration_abstain requires needs_review status")
        if packet.contradicting_atom_ids:
            messages.append(f"WARNING: Packet {packet.id} has contradictions")
        if packet.family == PacketFamily.vendor_mismatch:
            messages.append(f"WARNING: Packet {packet.id} vendor_mismatch exists")
        if packet.family in (PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch) and certificate is not None:
            quantities = [
                atom_by_id[aid].value.get("quantity")
                for aid in referenced_ids
                if aid in atom_by_id and atom_by_id[aid].atom_type.value == "quantity"
            ]
            numeric = [q for q in quantities if isinstance(q, (int, float))]
            if len(numeric) >= 2:
                reason_text = (certificate.existence_reason or "") + " " + (certificate.contradiction_summary or "")
                numeric_tokens = re.findall(r"\d+(?:\.\d+)?", reason_text)
                if len(set(numeric_tokens)) < 2:
                    messages.append(
                        f"ERROR: Packet {packet.id} {packet.family.value} certificate must mention both quantity values"
                    )
        if packet.family == PacketFamily.scope_exclusion and certificate is not None:
            if not packet.governing_atom_ids and "vendor_scope_pollution_candidate" in (packet.review_flags or []):
                pass
            else:
                rationale = certificate.governing_rationale.lower()
                if "exclusion" not in rationale and "needs_review" not in rationale:
                    messages.append(
                        f"ERROR: Packet {packet.id} scope_exclusion certificate must explain governing exclusion or needs_review"
                    )

        if packet.anchor_signature is None:
            messages.append(f"ERROR: Packet {packet.id} missing anchor_signature")
        else:
            packet_atoms = [
                atom_by_id[atom_id]
                for atom_id in sorted(
                    set(packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids)
                )
                if atom_id in atom_by_id
            ]
            owner = None
            if packet.family == PacketFamily.action_item:
                action_atom = next((atom for atom in packet_atoms if atom.atom_type.value == "action_item"), None)
                if action_atom is not None:
                    owner = str(action_atom.value.get("owner", "unknown"))
            material_identity = None
            if packet.anchor_signature and packet.anchor_signature.canonical_key.startswith("material:"):
                material_identity = packet.anchor_signature.canonical_key.split(":", 1)[1]
            elif packet.family == PacketFamily.scope_exclusion and packet.anchor_signature:
                ck = packet.anchor_signature.canonical_key
                if "|" in ck:
                    material_identity = ck.split("|", 1)[1]
            elif packet.family == PacketFamily.missing_info and packet.anchor_signature:
                if packet.anchor_signature.canonical_key == "missing_info:raceway_conduit":
                    material_identity = "raceway_conduit"
                elif packet.anchor_signature.canonical_key == "missing_info:requirement:certification":
                    material_identity = "certification"
                elif packet.anchor_signature.canonical_key == "missing_info:access:site_gate":
                    material_identity = "site_access_gate"
            expected_signature = make_anchor_signature(
                packet.family, packet_atoms, owner=owner, material_identity=material_identity
            )
            if packet.anchor_signature.hash != expected_signature.hash:
                messages.append(f"ERROR: Packet {packet.id} anchor_signature hash mismatch")
            if packet.anchor_key != packet.anchor_signature.canonical_key:
                messages.append(f"ERROR: Packet {packet.id} anchor_key does not match anchor_signature canonical_key")

    # Errors: edge references + warnings: low confidence atoms.
    for atom in result.atoms:
        if atom.confidence < 0.75:
            messages.append(f"WARNING: Atom {atom.id} has low confidence {atom.confidence:.2f}")

    for edge in result.edges:
        if edge.from_atom_id not in atom_by_id:
            messages.append(f"ERROR: Edge {edge.id} references missing from_atom_id {edge.from_atom_id}")
        if edge.to_atom_id not in atom_by_id:
            messages.append(f"ERROR: Edge {edge.id} references missing to_atom_id {edge.to_atom_id}")
        if "semantic_candidate_linker" in edge.reason.lower():
            if edge.edge_type.value == "contradicts":
                messages.append(f"ERROR: Semantic linker edge {edge.id} cannot be contradicts")
            if "method=" not in edge.reason.lower():
                messages.append(f"ERROR: Semantic linker edge {edge.id} missing method metadata")

    for packet in result.packets:
        if not packet.related_edge_ids:
            continue
        related_edges = [edge_by_id[edge_id] for edge_id in packet.related_edge_ids if edge_id in edge_by_id]
        if related_edges and all("semantic_candidate_linker" in edge.reason.lower() for edge in related_edges):
            messages.append(
                f"WARNING: Packet {packet.id} uses semantic linker edges as supplementary evidence only"
            )

    messages.extend(check_graph_invariants(result.atoms, result.edges))

    for entity in result.entities:
        if entity.review_status == ReviewStatus.needs_review:
            messages.append(f"WARNING: Entity {entity.id} review_status is needs_review")

    # Deterministic output ordering
    return sorted(set(messages))


def validation_failure_records(result: CompileResult, source_files_available: bool = False) -> list[FailureRecord]:
    messages = validate_compile_result(result, source_files_available=source_files_available)
    return failure_records_from_validation_messages(messages, scenario_id=result.project_id or "validation")
