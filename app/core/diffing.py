from __future__ import annotations

from collections import defaultdict

from app.core.ids import canonical_json_hash
from app.core.invalidation import packet_atom_ids, packet_blast_radius, packet_minimal_ids, should_invalidate_packet
from app.core.schemas import (
    ArtifactFingerprint,
    AtomDiff,
    CompileDiff,
    CompileManifest,
    CompileResult,
    EvidenceAtom,
    EvidencePacket,
    PacketDiff,
    PacketFamily,
)


def _atom_content_hash(atom: EvidenceAtom) -> str:
    payload = {
        "atom_type": atom.atom_type.value,
        "raw_text": atom.raw_text,
        "normalized_text": atom.normalized_text,
        "value": atom.value,
        "entity_keys": sorted(atom.entity_keys),
        "authority_class": atom.authority_class.value,
        "confidence": round(float(atom.confidence), 6),
        "review_status": atom.review_status.value,
        "review_flags": sorted(atom.review_flags),
        "source_refs": [
            {
                "artifact_id": source_ref.artifact_id,
                "artifact_type": source_ref.artifact_type.value,
                "filename": source_ref.filename,
                "locator": source_ref.locator,
                "parser_version": source_ref.parser_version,
            }
            for source_ref in sorted(atom.source_refs, key=lambda ref: ref.id)
        ],
    }
    return canonical_json_hash(payload)


def _packet_content_hash(packet: EvidencePacket) -> str:
    payload = {
        "family": packet.family.value,
        "anchor_key": packet.anchor_key,
        "anchor_signature_hash": packet.anchor_signature.hash if packet.anchor_signature is not None else "",
        "status": packet.status.value,
        "governing_atom_ids": sorted(packet.governing_atom_ids),
        "supporting_atom_ids": sorted(packet.supporting_atom_ids),
        "contradicting_atom_ids": sorted(packet.contradicting_atom_ids),
        "minimal_sufficient_atom_ids": packet_minimal_ids(packet),
        "reason": packet.reason,
    }
    return canonical_json_hash(payload)


def _strong_packet_key(packet: EvidencePacket) -> tuple[str, str]:
    signature_hash = packet.anchor_signature.hash if packet.anchor_signature is not None else packet.anchor_key
    return (packet.family.value, signature_hash)


def _weak_packet_key(packet: EvidencePacket) -> tuple[str, str]:
    canonical_key = packet.anchor_signature.canonical_key if packet.anchor_signature is not None else packet.anchor_key
    return (packet.family.value, canonical_key)


def _build_atom_diffs(before: CompileResult, after: CompileResult) -> tuple[list[AtomDiff], dict[str, str], set[str]]:
    before_atoms = {atom.id: atom for atom in before.atoms}
    after_atoms = {atom.id: atom for atom in after.atoms}
    all_ids = sorted(set(before_atoms).union(after_atoms))

    atom_hash_by_id: dict[str, str] = {}
    unstable_atom_ids: set[str] = set()
    atom_diffs: list[AtomDiff] = []

    for atom_id in all_ids:
        before_atom = before_atoms.get(atom_id)
        after_atom = after_atoms.get(atom_id)

        before_hash = _atom_content_hash(before_atom) if before_atom is not None else None
        after_hash = _atom_content_hash(after_atom) if after_atom is not None else None

        if before_atom is None and after_atom is not None:
            change_type = "added"
            reason = "Atom appears only in the after compile."
        elif before_atom is not None and after_atom is None:
            change_type = "removed"
            reason = "Atom appears only in the before compile."
            unstable_atom_ids.add(atom_id)
        elif before_hash != after_hash:
            change_type = "changed"
            reason = "Stable atom content hash changed."
            unstable_atom_ids.add(atom_id)
        else:
            change_type = "unchanged"
            reason = "Stable atom content hash is unchanged."

        atom_diffs.append(
            AtomDiff(
                atom_id=atom_id,
                change_type=change_type,
                before_hash=before_hash,
                after_hash=after_hash,
                reason=reason,
            )
        )
        if after_hash is not None:
            atom_hash_by_id[atom_id] = after_hash
        elif before_hash is not None:
            atom_hash_by_id[atom_id] = before_hash

    return atom_diffs, atom_hash_by_id, unstable_atom_ids


def diff_compile_results(before: CompileResult, after: CompileResult) -> CompileDiff:
    atom_diffs, _, unstable_atom_ids = _build_atom_diffs(before, after)

    before_by_strong = {_strong_packet_key(packet): packet for packet in before.packets}
    after_by_strong = {_strong_packet_key(packet): packet for packet in after.packets}

    packet_diffs: list[PacketDiff] = []
    invalidated_ids: set[str] = set()
    impacted_consumers: set[str] = set()

    matched_before: set[str] = set()
    matched_after: set[str] = set()

    def append_packet_diff(
        packet_id: str,
        change_type: str,
        before_status: str | None,
        after_status: str | None,
        affected_atom_ids: list[str],
        reason: str,
    ) -> None:
        packet_diffs.append(
            PacketDiff(
                packet_id=packet_id,
                change_type=change_type,  # type: ignore[arg-type]
                before_status=before_status,
                after_status=after_status,
                affected_atom_ids=sorted(set(affected_atom_ids)),
                reason=reason,
            )
        )

    # First pass: exact family + signature-hash matches.
    for key in sorted(set(before_by_strong).union(after_by_strong)):
        before_packet = before_by_strong.get(key)
        after_packet = after_by_strong.get(key)

        if before_packet is None and after_packet is not None:
            append_packet_diff(
                packet_id=after_packet.id,
                change_type="added",
                before_status=None,
                after_status=after_packet.status.value,
                affected_atom_ids=packet_atom_ids(after_packet),
                reason=f"Added packet for family={after_packet.family.value} anchor={after_packet.anchor_key}.",
            )
            continue

        if before_packet is not None and after_packet is None:
            append_packet_diff(
                packet_id=before_packet.id,
                change_type="removed",
                before_status=before_packet.status.value,
                after_status=None,
                affected_atom_ids=packet_atom_ids(before_packet),
                reason=f"Removed/resolved packet for family={before_packet.family.value} anchor={before_packet.anchor_key}.",
            )
            continue

        assert before_packet is not None and after_packet is not None
        matched_before.add(before_packet.id)
        matched_after.add(after_packet.id)

        before_hash = _packet_content_hash(before_packet)
        after_hash = _packet_content_hash(after_packet)
        if before_hash == after_hash:
            append_packet_diff(
                packet_id=after_packet.id,
                change_type="unchanged",
                before_status=before_packet.status.value,
                after_status=after_packet.status.value,
                affected_atom_ids=[],
                reason="Packet content hash unchanged.",
            )
            continue

        if before_packet.family in {PacketFamily.vendor_mismatch, PacketFamily.quantity_conflict}:
            if packet_minimal_ids(before_packet) != packet_minimal_ids(after_packet):
                append_packet_diff(
                    packet_id=before_packet.id,
                    change_type="removed",
                    before_status=before_packet.status.value,
                    after_status=None,
                    affected_atom_ids=packet_atom_ids(before_packet),
                    reason=(
                        f"Removed/resolved packet for family={before_packet.family.value} "
                        "because conflict evidence set changed."
                    ),
                )
                append_packet_diff(
                    packet_id=after_packet.id,
                    change_type="added",
                    before_status=None,
                    after_status=after_packet.status.value,
                    affected_atom_ids=packet_atom_ids(after_packet),
                    reason=(
                        f"Added packet for family={after_packet.family.value} "
                        "because conflict evidence set changed."
                    ),
                )
                impacted_consumers.update(packet_blast_radius(before_packet))
                impacted_consumers.update(packet_blast_radius(after_packet))
                continue

        invalidate, invalidate_reason, affected = should_invalidate_packet(before_packet, after_packet, unstable_atom_ids)
        if invalidate:
            invalidated_ids.add(before_packet.id)
            impacted_consumers.update(packet_blast_radius(before_packet))
            impacted_consumers.update(packet_blast_radius(after_packet))
            append_packet_diff(
                packet_id=before_packet.id,
                change_type="invalidated",
                before_status=before_packet.status.value,
                after_status="invalidated",
                affected_atom_ids=affected,
                reason=f"Invalidated for family={before_packet.family.value}: {invalidate_reason}",
            )
            continue

        append_packet_diff(
            packet_id=after_packet.id,
            change_type="changed",
            before_status=before_packet.status.value,
            after_status=after_packet.status.value,
            affected_atom_ids=sorted(set(packet_atom_ids(before_packet) + packet_atom_ids(after_packet))),
            reason=f"Packet changed for family={after_packet.family.value}, but governing/minimal evidence stayed stable.",
        )
        impacted_consumers.update(packet_blast_radius(before_packet))
        impacted_consumers.update(packet_blast_radius(after_packet))

    # Second pass: weak family + canonical key matching for anchor-signature changes.
    unmatched_before = [packet for packet in before.packets if packet.id not in matched_before]
    unmatched_after = [packet for packet in after.packets if packet.id not in matched_after]
    before_weak = defaultdict(list)
    after_weak = defaultdict(list)
    for packet in unmatched_before:
        before_weak[_weak_packet_key(packet)].append(packet)
    for packet in unmatched_after:
        after_weak[_weak_packet_key(packet)].append(packet)

    for key in sorted(set(before_weak).intersection(after_weak)):
        before_group = sorted(before_weak[key], key=lambda packet: packet.id)
        after_group = sorted(after_weak[key], key=lambda packet: packet.id)
        for idx in range(min(len(before_group), len(after_group))):
            before_packet = before_group[idx]
            after_packet = after_group[idx]
            matched_before.add(before_packet.id)
            matched_after.add(after_packet.id)

            invalidate, invalidate_reason, affected = should_invalidate_packet(before_packet, after_packet, unstable_atom_ids)
            if invalidate:
                invalidated_ids.add(before_packet.id)
                impacted_consumers.update(packet_blast_radius(before_packet))
                impacted_consumers.update(packet_blast_radius(after_packet))
                append_packet_diff(
                    packet_id=before_packet.id,
                    change_type="invalidated",
                    before_status=before_packet.status.value,
                    after_status="invalidated",
                    affected_atom_ids=affected,
                    reason=f"Invalidated for family={before_packet.family.value}: {invalidate_reason}",
                )
            else:
                append_packet_diff(
                    packet_id=after_packet.id,
                    change_type="changed",
                    before_status=before_packet.status.value,
                    after_status=after_packet.status.value,
                    affected_atom_ids=sorted(set(packet_atom_ids(before_packet) + packet_atom_ids(after_packet))),
                    reason=f"Packet changed for family={after_packet.family.value}, anchor signature drifted but core evidence stayed stable.",
                )
                impacted_consumers.update(packet_blast_radius(before_packet))
                impacted_consumers.update(packet_blast_radius(after_packet))

    # Remaining unmatched packets are pure added/removed outcomes.
    for packet in sorted((packet for packet in before.packets if packet.id not in matched_before), key=lambda packet: packet.id):
        append_packet_diff(
            packet_id=packet.id,
            change_type="removed",
            before_status=packet.status.value,
            after_status=None,
            affected_atom_ids=packet_atom_ids(packet),
            reason=f"Removed/resolved packet for family={packet.family.value} anchor={packet.anchor_key}.",
        )
    for packet in sorted((packet for packet in after.packets if packet.id not in matched_after), key=lambda packet: packet.id):
        append_packet_diff(
            packet_id=packet.id,
            change_type="added",
            before_status=None,
            after_status=packet.status.value,
            affected_atom_ids=packet_atom_ids(packet),
            reason=f"Added packet for family={packet.family.value} anchor={packet.anchor_key}.",
        )

    counts = defaultdict(int)
    for packet_diff in packet_diffs:
        counts[packet_diff.change_type] += 1
    blast_radius_summary = {
        "before_packet_count": len(before.packets),
        "after_packet_count": len(after.packets),
        "added_packets": counts["added"],
        "removed_packets": counts["removed"],
        "changed_packets": counts["changed"],
        "invalidated_packets": counts["invalidated"],
        "unchanged_packets": counts["unchanged"],
        "impacted_consumers": sorted(impacted_consumers),
        "invalidated_packet_ids": sorted(invalidated_ids),
    }

    return CompileDiff(
        before_compile_id=before.compile_id,
        after_compile_id=after.compile_id,
        atom_diffs=sorted(atom_diffs, key=lambda diff: (diff.change_type, diff.atom_id)),
        packet_diffs=sorted(packet_diffs, key=lambda diff: (diff.change_type, diff.packet_id)),
        invalidated_packet_ids=sorted(invalidated_ids),
        blast_radius_summary=blast_radius_summary,
    )


def diff_artifact_fingerprints(
    before_manifest: CompileManifest | None,
    after_fingerprints: list[ArtifactFingerprint],
) -> dict[str, list[str]]:
    """Compute artifact-level deltas used by incremental compile cache.

    Two fingerprints are considered reusable when sha256 + parser_version match
    for the same artifact_id.
    """

    before_rows = {row.artifact_id: row for row in (before_manifest.artifact_fingerprints if before_manifest else [])}
    after_rows = {row.artifact_id: row for row in after_fingerprints}

    reused: list[str] = []
    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []

    for artifact_id, after in sorted(after_rows.items(), key=lambda item: item[0]):
        before = before_rows.get(artifact_id)
        if before is None:
            added.append(artifact_id)
            continue
        if before.sha256 == after.sha256 and before.parser_version == after.parser_version:
            reused.append(artifact_id)
        else:
            changed.append(artifact_id)

    for artifact_id in sorted(before_rows.keys()):
        if artifact_id not in after_rows:
            removed.append(artifact_id)

    return {
        "reused": reused,
        "changed": changed,
        "added": added,
        "removed": removed,
    }
