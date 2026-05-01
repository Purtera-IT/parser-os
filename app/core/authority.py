from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.core.schemas import AtomType, AuthorityClass, AuthorityScore, EvidenceAtom, PacketFamily, ReviewStatus

AUTHORITY_RANKS: dict[AuthorityClass, int] = {
    AuthorityClass.contractual_scope: 100,
    AuthorityClass.customer_current_authored: 90,
    AuthorityClass.approved_site_roster: 80,
    AuthorityClass.vendor_quote: 65,
    AuthorityClass.meeting_note: 55,
    AuthorityClass.machine_extractor: 40,
    AuthorityClass.quoted_old_email: 10,
    AuthorityClass.deleted_text: 0,
}

SCOPE_ATOM_TYPES = {
    AtomType.scope_item,
    AtomType.exclusion,
    AtomType.customer_instruction,
    AtomType.constraint,
    AtomType.assumption,
}

MEETING_NOTE_NEVER_BEATS = {
    AuthorityClass.contractual_scope,
    AuthorityClass.customer_current_authored,
    AuthorityClass.approved_site_roster,
}


class AuthorityDecision(BaseModel):
    governing_atom_id: str
    losing_atom_id: str
    reason: str
    governing_authority_class: AuthorityClass
    losing_authority_class: AuthorityClass
    governing_score: float | None = None
    losing_score: float | None = None
    governing_dimensions: dict[str, Any] = Field(default_factory=dict)
    losing_dimensions: dict[str, Any] = Field(default_factory=dict)


def authority_rank(authority_class: AuthorityClass) -> int:
    return AUTHORITY_RANKS[authority_class]


def _add_low_confidence_review_flag(atom: EvidenceAtom) -> None:
    if atom.confidence >= 0.75:
        return
    if "low_confidence_needs_review" not in atom.review_flags:
        atom.review_flags.append("low_confidence_needs_review")
    atom.review_status = ReviewStatus.needs_review


def _entity_topic_key(atom: EvidenceAtom) -> str:
    if atom.entity_keys:
        return "|".join(sorted(atom.entity_keys))
    return atom.normalized_text


def _parse_review_boost_penalty(status: ReviewStatus) -> float:
    if status == ReviewStatus.approved:
        return -2.0
    if status == ReviewStatus.needs_review:
        return 4.0
    if status == ReviewStatus.rejected:
        return 1_000.0
    return 0.0


def _parse_locator_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for pattern in (
        None,
        "%a, %d %b %Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            if pattern is None:
                return datetime.fromisoformat(text)
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _atom_timestamp(atom: EvidenceAtom) -> datetime | None:
    for src in atom.source_refs:
        locator = src.locator or {}
        for key in ("timestamp", "sent_at", "date", "updated_at"):
            dt = _parse_locator_timestamp(locator.get(key))
            if dt is not None:
                return dt
    return None


def _recency_score(atom: EvidenceAtom, context_atoms: list[EvidenceAtom]) -> float:
    ts = _atom_timestamp(atom)
    if ts is None:
        return 0.0
    same_class = [candidate for candidate in context_atoms if candidate.authority_class == atom.authority_class]
    if len(same_class) <= 1:
        return 0.0
    stamps = [_atom_timestamp(candidate) for candidate in same_class]
    stamps = [stamp for stamp in stamps if stamp is not None]
    if len(stamps) <= 1:
        return 0.0
    min_ts = min(stamps)
    max_ts = max(stamps)
    if min_ts == max_ts:
        return 0.0
    normalized = (ts - min_ts).total_seconds() / (max_ts - min_ts).total_seconds()
    return round(normalized * 2.0, 4)


def _authorship_score(atom: EvidenceAtom) -> float:
    source = atom.source_refs[0] if atom.source_refs else None
    locator = source.locator if source else {}
    speaker_role = str(locator.get("speaker_role", "")).strip().lower()
    sender = str(locator.get("sender", "")).strip().lower()
    score = 0.0
    if atom.authority_class == AuthorityClass.customer_current_authored:
        score += 2.0
    if speaker_role == "customer" or ("@" in sender and "purtera" not in sender):
        score += 1.0
    if speaker_role == "internal":
        score -= 0.5
    return round(score, 4)


def _artifact_role_score(atom: EvidenceAtom) -> float:
    source = atom.source_refs[0] if atom.source_refs else None
    artifact_type = source.artifact_type.value if source else ""
    score = 0.0
    if atom.authority_class == AuthorityClass.approved_site_roster and atom.atom_type == AtomType.quantity:
        if any(key.startswith("site:") or key.startswith("device:") for key in atom.entity_keys):
            score += 1.5
    if atom.authority_class == AuthorityClass.approved_site_roster and artifact_type in {"xlsx", "csv"}:
        score += 0.5
    if atom.authority_class == AuthorityClass.vendor_quote and artifact_type in {"xlsx", "csv", "txt"}:
        score += 0.5
    return round(score, 4)


def _same_topic_current_customer_exists(atom: EvidenceAtom, context_atoms: list[EvidenceAtom]) -> bool:
    topic = _entity_topic_key(atom)
    return any(
        other.id != atom.id
        and other.authority_class == AuthorityClass.customer_current_authored
        and _entity_topic_key(other) == topic
        for other in context_atoms
    )


def _is_vendor_context(atom: EvidenceAtom) -> bool:
    if any(key.startswith("part:") for key in atom.entity_keys):
        return True
    if atom.atom_type in {AtomType.vendor_line_item, AtomType.quantity}:
        return True
    if isinstance(atom.value, dict):
        if atom.value.get("context") in {"vendor", "vendor_mismatch"}:
            return True
        if any(k in atom.value for k in ("part_number", "unit_price", "lead_time")):
            return True
    return False


def is_scope_impacting_meeting_atom(atom: EvidenceAtom) -> bool:
    return atom.authority_class == AuthorityClass.meeting_note and atom.atom_type in {
        AtomType.scope_item,
        AtomType.exclusion,
        AtomType.customer_instruction,
        AtomType.decision,
        AtomType.meeting_commitment,
        AtomType.quantity,
    }


def score_authority(atom: EvidenceAtom, context_atoms: list[EvidenceAtom] | None = None, context: dict[str, Any] | None = None) -> AuthorityScore:
    context_atoms = context_atoms or [atom]
    context = context or {}
    packet_family = context.get("packet_family")
    if isinstance(packet_family, str):
        packet_family = PacketFamily(packet_family)

    base_rank = authority_rank(atom.authority_class)
    recency_score = _recency_score(atom, context_atoms)
    authorship_score = _authorship_score(atom)
    artifact_role_score = _artifact_role_score(atom)
    review_penalty = _parse_review_boost_penalty(atom.review_status)
    evidence_state_penalty = 0.0
    non_governing = False

    if atom.authority_class == AuthorityClass.deleted_text:
        evidence_state_penalty += 1_000.0
        non_governing = True
    if atom.review_status == ReviewStatus.rejected:
        evidence_state_penalty += 1_000.0
        non_governing = True
    if atom.authority_class == AuthorityClass.quoted_old_email and _same_topic_current_customer_exists(atom, context_atoms):
        evidence_state_penalty += 40.0
    if is_scope_impacting_meeting_atom(atom) and atom.review_status != ReviewStatus.approved:
        evidence_state_penalty += 12.0
    if (
        atom.authority_class == AuthorityClass.vendor_quote
        and atom.atom_type in {AtomType.scope_item, AtomType.exclusion, AtomType.customer_instruction}
        and packet_family in {PacketFamily.scope_inclusion, PacketFamily.scope_exclusion}
    ):
        evidence_state_penalty += 50.0
    if atom.authority_class == AuthorityClass.vendor_quote and packet_family == PacketFamily.vendor_mismatch:
        artifact_role_score += 1.0

    final_score = round(
        base_rank + recency_score + authorship_score + artifact_role_score - evidence_state_penalty - review_penalty,
        4,
    )
    if non_governing:
        final_score = -1_000_000.0

    dimensions = {
        "source_authority": atom.authority_class.value,
        "scope_impacting": is_scope_impacting_meeting_atom(atom),
        "review_status": atom.review_status.value,
        "non_governing": non_governing,
        "topic_key": _entity_topic_key(atom),
        "context_packet_family": packet_family.value if isinstance(packet_family, PacketFamily) else None,
    }
    explanation = (
        f"base={base_rank}, recency={recency_score}, authorship={authorship_score}, "
        f"artifact_role={artifact_role_score}, evidence_penalty={evidence_state_penalty}, "
        f"review_penalty={review_penalty}, final={final_score}"
    )
    return AuthorityScore(
        atom_id=atom.id,
        base_rank=base_rank,
        recency_score=recency_score,
        authorship_score=authorship_score,
        artifact_role_score=artifact_role_score,
        evidence_state_penalty=evidence_state_penalty,
        review_penalty=review_penalty,
        final_score=final_score,
        dimensions=dimensions,
        explanation=explanation,
    )


def is_governing_candidate(atom: EvidenceAtom) -> bool:
    _add_low_confidence_review_flag(atom)

    if atom.authority_class == AuthorityClass.deleted_text:
        return False
    if atom.review_status == ReviewStatus.rejected:
        return False
    if atom.authority_class == AuthorityClass.vendor_quote and atom.atom_type in SCOPE_ATOM_TYPES:
        return False
    if is_scope_impacting_meeting_atom(atom):
        if atom.review_status != ReviewStatus.approved:
            atom.review_status = ReviewStatus.needs_review
        if "verbal_commitment_requires_confirmation" not in atom.review_flags:
            atom.review_flags.append("verbal_commitment_requires_confirmation")
    return True


def compare_atoms(a: EvidenceAtom, b: EvidenceAtom, context: dict[str, Any] | None = None) -> AuthorityDecision:
    scores = {
        a.id: score_authority(a, [a, b], context=context),
        b.id: score_authority(b, [a, b], context=context),
    }
    score_a = scores[a.id]
    score_b = scores[b.id]

    # Rule: customer_current_authored beats quoted_old_email on same topic.
    same_topic = _entity_topic_key(a) == _entity_topic_key(b)
    if same_topic and a.authority_class == AuthorityClass.customer_current_authored and b.authority_class == AuthorityClass.quoted_old_email:
        return AuthorityDecision(
            governing_atom_id=a.id,
            losing_atom_id=b.id,
            reason="current customer authored overrides quoted old email",
            governing_authority_class=a.authority_class,
            losing_authority_class=b.authority_class,
        )
    if same_topic and b.authority_class == AuthorityClass.customer_current_authored and a.authority_class == AuthorityClass.quoted_old_email:
        return AuthorityDecision(
            governing_atom_id=b.id,
            losing_atom_id=a.id,
            reason="current customer authored overrides quoted old email",
            governing_authority_class=b.authority_class,
            losing_authority_class=a.authority_class,
        )

    # Rule: customer exclusion beats approved roster inclusion.
    if (
        same_topic
        and a.atom_type == AtomType.exclusion
        and a.authority_class == AuthorityClass.customer_current_authored
        and b.authority_class == AuthorityClass.approved_site_roster
    ):
        return AuthorityDecision(
            governing_atom_id=a.id,
            losing_atom_id=b.id,
            reason="customer exclusion beats approved site roster inclusion",
            governing_authority_class=a.authority_class,
            losing_authority_class=b.authority_class,
        )
    if (
        same_topic
        and b.atom_type == AtomType.exclusion
        and b.authority_class == AuthorityClass.customer_current_authored
        and a.authority_class == AuthorityClass.approved_site_roster
    ):
        return AuthorityDecision(
            governing_atom_id=b.id,
            losing_atom_id=a.id,
            reason="customer exclusion beats approved site roster inclusion",
            governing_authority_class=b.authority_class,
            losing_authority_class=a.authority_class,
        )

    # Rule: vendor quote does not govern scope changes.
    if a.authority_class == AuthorityClass.vendor_quote and a.atom_type in SCOPE_ATOM_TYPES:
        return AuthorityDecision(
            governing_atom_id=b.id,
            losing_atom_id=a.id,
            reason="vendor quote cannot govern scope changes",
            governing_authority_class=b.authority_class,
            losing_authority_class=a.authority_class,
        )
    if b.authority_class == AuthorityClass.vendor_quote and b.atom_type in SCOPE_ATOM_TYPES:
        return AuthorityDecision(
            governing_atom_id=a.id,
            losing_atom_id=b.id,
            reason="vendor quote cannot govern scope changes",
            governing_authority_class=a.authority_class,
            losing_authority_class=b.authority_class,
        )

    packet_family: PacketFamily | None = None
    if context:
        pf = context.get("packet_family")
        if isinstance(pf, PacketFamily):
            packet_family = pf
        elif isinstance(pf, str):
            try:
                packet_family = PacketFamily(pf)
            except ValueError:
                packet_family = None

    # Rule: approved_site_roster quantity always governs over vendor_quote in quantity_conflict
    # and vendor_mismatch packets (vendor reveals mismatch; roster/addendum governs scoped quantity).
    if (
        packet_family in (PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch)
        and a.atom_type == AtomType.quantity
        and b.atom_type == AtomType.quantity
    ):
        if (
            a.authority_class == AuthorityClass.approved_site_roster
            and b.authority_class == AuthorityClass.vendor_quote
        ):
            return AuthorityDecision(
                governing_atom_id=a.id,
                losing_atom_id=b.id,
                reason="approved_site_roster governs over vendor_quote for scoped quantity vs vendor line items",
                governing_authority_class=a.authority_class,
                losing_authority_class=b.authority_class,
            )
        if (
            b.authority_class == AuthorityClass.approved_site_roster
            and a.authority_class == AuthorityClass.vendor_quote
        ):
            return AuthorityDecision(
                governing_atom_id=b.id,
                losing_atom_id=a.id,
                reason="approved_site_roster governs over vendor_quote for scoped quantity vs vendor line items",
                governing_authority_class=b.authority_class,
                losing_authority_class=a.authority_class,
            )

    # Rule: meeting_note cannot outrank stronger authorities on same topic.
    if same_topic and a.authority_class == AuthorityClass.meeting_note and b.authority_class in MEETING_NOTE_NEVER_BEATS:
        return AuthorityDecision(
            governing_atom_id=b.id,
            losing_atom_id=a.id,
            reason="meeting_note cannot govern over stronger written authority",
            governing_authority_class=b.authority_class,
            losing_authority_class=a.authority_class,
        )
    if same_topic and b.authority_class == AuthorityClass.meeting_note and a.authority_class in MEETING_NOTE_NEVER_BEATS:
        return AuthorityDecision(
            governing_atom_id=a.id,
            losing_atom_id=b.id,
            reason="meeting_note cannot govern over stronger written authority",
            governing_authority_class=a.authority_class,
            losing_authority_class=b.authority_class,
        )
    if (
        same_topic
        and a.authority_class == AuthorityClass.meeting_note
        and b.authority_class == AuthorityClass.vendor_quote
        and (a.atom_type == AtomType.quantity or b.atom_type == AtomType.quantity)
    ):
        return AuthorityDecision(
            governing_atom_id=b.id,
            losing_atom_id=a.id,
            reason="meeting_note quantity cannot govern over vendor quote quantity",
            governing_authority_class=b.authority_class,
            losing_authority_class=a.authority_class,
        )
    if (
        same_topic
        and b.authority_class == AuthorityClass.meeting_note
        and a.authority_class == AuthorityClass.vendor_quote
        and (a.atom_type == AtomType.quantity or b.atom_type == AtomType.quantity)
    ):
        return AuthorityDecision(
            governing_atom_id=a.id,
            losing_atom_id=b.id,
            reason="meeting_note quantity cannot govern over vendor quote quantity",
            governing_authority_class=a.authority_class,
            losing_authority_class=b.authority_class,
        )

    # Lattice score wins.
    if score_a.final_score != score_b.final_score:
        winner, loser = (a, b) if score_a.final_score > score_b.final_score else (b, a)
        winner_score = scores[winner.id]
        loser_score = scores[loser.id]
        return AuthorityDecision(
            governing_atom_id=winner.id,
            losing_atom_id=loser.id,
            reason="authority lattice final_score wins",
            governing_authority_class=winner.authority_class,
            losing_authority_class=loser.authority_class,
            governing_score=winner_score.final_score,
            losing_score=loser_score.final_score,
            governing_dimensions=winner_score.dimensions,
            losing_dimensions=loser_score.dimensions,
        )

    # Tie-breaker: higher confidence wins.
    if a.confidence != b.confidence:
        winner, loser = (a, b) if a.confidence > b.confidence else (b, a)
        winner_score = scores[winner.id]
        loser_score = scores[loser.id]
        return AuthorityDecision(
            governing_atom_id=winner.id,
            losing_atom_id=loser.id,
            reason="tie broken by confidence",
            governing_authority_class=winner.authority_class,
            losing_authority_class=loser.authority_class,
            governing_score=winner_score.final_score,
            losing_score=loser_score.final_score,
            governing_dimensions=winner_score.dimensions,
            losing_dimensions=loser_score.dimensions,
        )

    # Tie-breaker: newer timestamp wins.
    ts_a = _atom_timestamp(a)
    ts_b = _atom_timestamp(b)
    if ts_a and ts_b and ts_a != ts_b:
        winner, loser = (a, b) if ts_a > ts_b else (b, a)
        winner_score = scores[winner.id]
        loser_score = scores[loser.id]
        return AuthorityDecision(
            governing_atom_id=winner.id,
            losing_atom_id=loser.id,
            reason="tie broken by newer source timestamp",
            governing_authority_class=winner.authority_class,
            losing_authority_class=loser.authority_class,
            governing_score=winner_score.final_score,
            losing_score=loser_score.final_score,
            governing_dimensions=winner_score.dimensions,
            losing_dimensions=loser_score.dimensions,
        )

    # Final deterministic fallback by stable id ordering.
    winner, loser = (a, b) if a.id <= b.id else (b, a)
    winner_score = scores[winner.id]
    loser_score = scores[loser.id]
    return AuthorityDecision(
        governing_atom_id=winner.id,
        losing_atom_id=loser.id,
        reason="deterministic fallback by atom id",
        governing_authority_class=winner.authority_class,
        losing_authority_class=loser.authority_class,
        governing_score=winner_score.final_score,
        losing_score=loser_score.final_score,
        governing_dimensions=winner_score.dimensions,
        losing_dimensions=loser_score.dimensions,
    )


def choose_governing_atoms(
    atoms: list[EvidenceAtom],
    context: dict[str, Any] | None = None,
    include_decisions: bool = False,
) -> list[EvidenceAtom] | tuple[list[EvidenceAtom], list[AuthorityDecision]]:
    for atom in atoms:
        _add_low_confidence_review_flag(atom)

    candidate_atoms = [a for a in atoms if is_governing_candidate(a)]
    if not candidate_atoms:
        return []

    has_scope_atom = any(a.atom_type in SCOPE_ATOM_TYPES for a in candidate_atoms if a.authority_class != AuthorityClass.vendor_quote)
    if has_scope_atom:
        candidate_atoms = [
            a
            for a in candidate_atoms
            if not (a.authority_class == AuthorityClass.vendor_quote and _is_vendor_context(a))
        ]

    grouped: dict[str, list[EvidenceAtom]] = {}
    for atom in candidate_atoms:
        key = _entity_topic_key(atom)
        grouped.setdefault(key, []).append(atom)

    governing: list[EvidenceAtom] = []
    decisions: list[AuthorityDecision] = []
    for key in sorted(grouped):
        pool = sorted(grouped[key], key=lambda x: x.id)
        winner = pool[0]
        for challenger in pool[1:]:
            decision = compare_atoms(winner, challenger, context=context)
            decisions.append(decision)
            winner = winner if decision.governing_atom_id == winner.id else challenger
        governing.append(winner)

    governing = sorted(governing, key=lambda a: a.id)
    decisions = sorted(decisions, key=lambda d: (d.governing_atom_id, d.losing_atom_id, d.reason))
    if include_decisions:
        return governing, decisions
    return governing
