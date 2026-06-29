"""Content-aware atom confidence recalibration.

v53: parser-os has been emitting confidence based on PROVENANCE
(which code path created the atom) — hardcoded 0.82 / 0.85 / 0.88
regardless of actual data quality. A bom_line atom with all fields
populated, cross-doc corroborated, from the contractual SOW gets
the same 0.85 as one with empty SKU, single-doc, from a draft note.

This module recalibrates every atom's confidence using SIGNALS that
correlate with actual correctness:

  + 0.15  has stable semantic entity_key (sku, req_id, email, ic_id)
  + 0.00–0.15  value-field completeness ratio (populated/expected)
  + 0.10  cross-doc corroborated (same fact appears in ≥2 artifacts)
  + 0.05–0.20  source authority tier (contractual_final > approved_scope
              > supporting_evidence)
  + 0.10  source_replay receipts verified
  + 0.05  text length ≥ 80 chars (full sentence, not fragment)
  − 0.10  has contradicting edges in the graph
  − 0.10  raw_text < 20 chars (likely fragment)
  − 0.05  no source_refs (orphan atom — shouldn't exist but defensive)

Base score = 0.50 (neutral). Final score clamped to [0.05, 0.99].

This is pure / deterministic / no LLM / no I/O.
Runs as a compiler stage AFTER semantic_dedup and BEFORE
entity_resolution so dedup winners get their refreshed scores.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


# Per-type expected value fields used for completeness scoring.
# When the atom's value dict has these populated, completeness=1.0.
_EXPECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "milestone_phase":              ("phase_id", "name", "start", "end", "owner"),
    "task":                         ("task_id", "name", "owner", "start", "due", "site"),
    "requirement":                  ("req_id", "description", "applies_to", "owner"),
    "bom_line":                     ("item_id", "description", "sku", "qty", "unit_cost"),
    "service_line":                 ("service_id", "description", "unit_price", "qty"),
    "site_allocation":              ("site", "qty", "item_id"),
    "physical_site":                ("id", "site_id", "name", "address"),
    "site_attribute":               ("site", "attribute_kind", "value"),
    "site_access_window":           ("site", "days", "hours", "escort_owner"),
    "site_access_restriction":      ("site", "restriction_kind", "condition"),
    "site_room_mix":                ("site", "room_type", "count"),
    "stakeholder":                  ("name", "title", "role", "email", "org"),
    "signatory":                    ("name", "title", "org", "signatory_type"),
    "approval_authority":           ("approver", "domain", "threshold", "process"),
    "approval_decision":            ("approver", "decision", "scope"),
    "payment_term":                 ("tranche", "percent", "trigger"),
    "commercial_total":             ("category", "amount", "currency"),
    "deal_metadata":                ("field_name", "value"),
    "change_order_rule":            ("trigger_kind", "rate_or_threshold"),
    "lead_time_constraint":         ("sku", "item_id", "lead_time_days"),
    "pricing_assumption":           ("domain", "statement"),
    "electrical_acceptance_test":   ("test", "threshold", "scope"),
    "compliance_classification":    ("classification", "allowed_destinations"),
    "compliance_rule":              ("rule_kind", "condition"),
    "acceptance_criterion":         ("area", "criteria", "threshold"),
    "cutover_step":                 ("step_id", "timing", "owner", "description"),
    "integration_checkpoint":       ("ic_id", "system", "test_description"),
    "blackout_date_range":          ("start", "end", "reason"),
    "deliverable":                  ("name", "due", "owner"),
    "data_flow_step":               ("step_number", "action", "system"),
    "system_mapping":               ("source", "target"),
    "metadata_requirement":         ("system", "key", "expected_value"),
    "mitigation":                   ("risk_id", "mitigation_text", "owner"),
    "dependency":                   ("dependent", "depends_on"),
    "risk":                         ("risk_id", "description", "probability", "impact", "mitigation"),
    "assumption":                   ("assumption", "category"),
}


# Per-type "anchor" fields — populated == has stable semantic key.
_SEMANTIC_KEYS: dict[str, tuple[str, ...]] = {
    "milestone_phase":              ("phase_id", "name"),
    "task":                         ("task_id",),
    "requirement":                  ("req_id",),
    "bom_line":                     ("item_id", "sku"),
    "service_line":                 ("service_id",),
    "site_allocation":              ("site", "item_id"),
    "physical_site":                ("id", "site_id"),
    "stakeholder":                  ("email",),
    "payment_term":                 ("tranche", "percent"),
    "commercial_total":             ("category",),
    "lead_time_constraint":         ("sku", "item_id"),
    "integration_checkpoint":       ("ic_id",),
    "approval_authority":           ("approver", "domain"),
    "approval_decision":            ("approver", "decision"),
    "blackout_date_range":          ("start", "end"),
    "compliance_classification":    ("classification",),
    "risk":                         ("risk_id",),
    "mitigation":                   ("risk_id",),
    "deliverable":                  ("name",),
    "data_flow_step":               ("step_number",),
    "system_mapping":               ("source", "target"),
    "metadata_requirement":         ("system", "key"),
    "cutover_step":                 ("step_id",),
}

# Source authority bonus per tier
_TIER_BONUS: dict[str, float] = {
    "contractual_final":   0.20,
    "approved_scope":      0.10,
    "supporting_evidence": 0.05,
}


def _atype(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    return t.value if hasattr(t, "value") else str(t or "")


def _is_filled(v: Any) -> bool:
    if v is None or v == "":
        return False
    if isinstance(v, (list, tuple, dict, set)) and not v:
        return False
    return True


def _completeness(atom: Any) -> float:
    """Ratio of expected fields populated in atom.value."""
    fields = _EXPECTED_FIELDS.get(_atype(atom), ())
    if not fields:
        return 0.5  # neutral when we don't know what to expect
    val = getattr(atom, "value", None) or {}
    if not isinstance(val, dict):
        return 0.0
    filled = sum(1 for f in fields if _is_filled(val.get(f)))
    return filled / len(fields)


def _has_semantic_key(atom: Any) -> bool:
    keys = _SEMANTIC_KEYS.get(_atype(atom), ())
    if not keys:
        return False
    val = getattr(atom, "value", None) or {}
    if not isinstance(val, dict):
        return False
    return any(_is_filled(val.get(k)) for k in keys)


def _build_cross_doc_index(atoms: list[Any]) -> set[tuple]:
    """Return set of (atype, norm_key) tuples seen in ≥2 distinct artifacts.

    norm_key = first populated semantic-key field's normalized value.
    Used to award the +0.10 cross-doc corroboration bonus.
    """
    import re
    seen: dict[tuple, set[str]] = defaultdict(set)
    for atom in atoms:
        atype = _atype(atom)
        keys = _SEMANTIC_KEYS.get(atype, ())
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        norm_key = ""
        for k in keys:
            v = val.get(k)
            if _is_filled(v):
                norm_key = re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")
                break
        if not norm_key:
            continue
        aid = getattr(atom, "artifact_id", "") or ""
        if aid:
            seen[(atype, norm_key)].add(aid)
    return {key for key, aids in seen.items() if len(aids) >= 2}


def _build_contradiction_set(edges: list[Any]) -> set[str]:
    """Return atom_ids that participate in any contradiction edge."""
    out: set[str] = set()
    for edge in edges or []:
        etype = getattr(edge, "edge_type", None)
        etype_s = etype.value if hasattr(etype, "value") else str(etype or "")
        if etype_s == "contradicts" or "contradict" in etype_s.lower():
            for fld in ("from_atom_id", "to_atom_id"):
                aid = getattr(edge, fld, "") or ""
                if aid:
                    out.add(aid)
    return out


def recalibrate_confidence(
    atoms: list[Any],
    *,
    artifact_authority: dict[str, str] | None = None,
    edges: list[Any] | None = None,
    abstain_threshold: float = 0.70,
) -> int:
    """Update atom.confidence + atom.calibrated_confidence in place.

    Also serializes a deterministic per-atom review gate: an ``auto_accepted``
    atom whose recalibrated confidence is below ``abstain_threshold`` is flipped
    to ``needs_review`` with a ``calibration_abstain`` flag (atomic — validators
    require the flag iff needs_review). This gives PMs a real "check this" signal
    with no ML dependency; the trained calibrator (apply_calibration, later in
    the pipeline) overwrites it with a learned probability when present.

    Args:
        atoms: full atom list
        artifact_authority: artifact_id → tier
            ("contractual_final" | "approved_scope" | "supporting_evidence")
        edges: graph edges (for contradiction detection)

    Returns count of atoms updated.
    """
    if not atoms:
        return 0
    artifact_authority = artifact_authority or {}
    contradictions = _build_contradiction_set(edges or [])
    corroborated = _build_cross_doc_index(atoms)

    import re
    updated = 0
    for atom in atoms:
        atype = _atype(atom)
        score = 0.50  # neutral base

        # +0.15 semantic key anchored
        if _has_semantic_key(atype) if False else _has_semantic_key(atom):
            score += 0.15

        # +0–0.15 completeness
        score += 0.15 * _completeness(atom)

        # +0.10 cross-doc corroboration
        keys = _SEMANTIC_KEYS.get(atype, ())
        val = getattr(atom, "value", None) or {}
        if isinstance(val, dict):
            for k in keys:
                v = val.get(k)
                if _is_filled(v):
                    norm = re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")
                    if (atype, norm) in corroborated:
                        score += 0.10
                        break

        # +0.05–0.20 source authority tier
        aid = getattr(atom, "artifact_id", "") or ""
        tier = artifact_authority.get(aid)
        if tier:
            score += _TIER_BONUS.get(tier, 0.0)

        # +0.10 source_replay verified
        receipts = getattr(atom, "receipts", None) or []
        if receipts:
            try:
                all_verified = all(
                    (getattr(r, "replay_status", None) == "verified") for r in receipts
                )
                if all_verified:
                    score += 0.10
            except Exception:
                pass

        # +0.05 long text vs −0.10 fragment text
        rt = getattr(atom, "raw_text", "") or ""
        if len(rt) >= 80:
            score += 0.05
        elif len(rt) < 20:
            score -= 0.10

        # −0.10 contradicted
        if getattr(atom, "id", "") in contradictions:
            score -= 0.10

        # −0.05 no source refs (defensive)
        if not getattr(atom, "source_refs", None):
            score -= 0.05

        # Clamp
        final = max(0.05, min(0.99, round(score, 3)))
        prev = getattr(atom, "confidence", None)
        try:
            # Always reflect the recalibrated value so the envelope's
            # calibrated_confidence is never null.
            atom.calibrated_confidence = final
            if prev != final:
                atom.confidence = final
                updated += 1
        except Exception:
            pass

        # Deterministic review gate: a low calibrated confidence on an
        # auto_accepted atom becomes a "check this" verdict. Flip status +
        # add the flag atomically (validators require the flag iff needs_review);
        # never downgrade rejected/approved/needs_review.
        if final < abstain_threshold:
            rs = getattr(atom, "review_status", None)
            rs_val = rs.value if hasattr(rs, "value") else rs
            if rs_val == "auto_accepted":
                try:
                    from app.core.schemas import ReviewStatus

                    atom.review_status = ReviewStatus.needs_review
                    flags = getattr(atom, "review_flags", None)
                    if isinstance(flags, list) and "calibration_abstain" not in flags:
                        flags.append("calibration_abstain")
                    updated += 1
                except Exception:
                    pass

    return updated


__all__ = ["recalibrate_confidence"]
