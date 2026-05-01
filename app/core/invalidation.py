from __future__ import annotations

from app.core.schemas import EvidencePacket, PacketFamily


def packet_minimal_ids(packet: EvidencePacket) -> list[str]:
    if packet.certificate is None:
        return []
    return sorted(set(packet.certificate.minimal_sufficient_atom_ids))


def packet_atom_ids(packet: EvidencePacket) -> list[str]:
    return sorted(set(packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids))


def packet_blast_radius(packet: EvidencePacket) -> list[str]:
    if packet.certificate is None:
        return []
    return sorted(set(packet.certificate.blast_radius))


def should_invalidate_packet(
    before_packet: EvidencePacket,
    after_packet: EvidencePacket | None,
    unstable_atom_ids: set[str],
) -> tuple[bool, str, list[str]]:
    before_governing = sorted(set(before_packet.governing_atom_ids))
    before_minimal = packet_minimal_ids(before_packet)
    affected_atoms = sorted(set(before_governing + before_minimal))

    critical_minimal_families = {
        PacketFamily.quantity_conflict,
        PacketFamily.vendor_mismatch,
        PacketFamily.scope_exclusion,
        PacketFamily.customer_override,
        PacketFamily.site_access,
    }

    changed_governing = sorted([aid for aid in before_governing if aid in unstable_atom_ids])
    if changed_governing:
        return (
            True,
            "Governing evidence changed or was removed.",
            sorted(set(changed_governing)),
        )

    changed_minimal = sorted([aid for aid in before_minimal if aid in unstable_atom_ids])
    if changed_minimal and (
        before_packet.family in critical_minimal_families or any(aid in before_governing for aid in changed_minimal)
    ):
        return (
            True,
            "Minimal sufficient evidence changed or was removed.",
            sorted(set(changed_minimal)),
        )

    if after_packet is not None:
        before_anchor_hash = before_packet.anchor_signature.hash if before_packet.anchor_signature is not None else ""
        after_anchor_hash = after_packet.anchor_signature.hash if after_packet.anchor_signature is not None else ""
        if before_anchor_hash != after_anchor_hash:
            return (
                True,
                "Anchor signature changed between compiles.",
                affected_atoms,
            )

        after_minimal = packet_minimal_ids(after_packet)
        minimal_delta = sorted(set(before_minimal).symmetric_difference(set(after_minimal)))
        if minimal_delta and (
            before_packet.family in critical_minimal_families or any(aid in before_governing for aid in minimal_delta)
        ):
            return (
                True,
                "Certificate minimal evidence set changed.",
                minimal_delta,
            )

    return (False, "Critical evidence remains stable.", [])
