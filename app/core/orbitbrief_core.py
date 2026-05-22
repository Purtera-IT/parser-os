"""OrbitBrief-Core: deterministic PM-ready aggregations over the envelope.

The envelope (orbitbrief.input.v2) carries atoms, edges, packets, and
indexes — everything an LLM needs to reason about the project, but
nothing pre-aggregated for the operator running the show.

This module computes the three deliverables the Purpulse one-pager
attributes to OrbitBrief-Core:

  * ``pm_dashboard``         — what a PM looks at first thing Monday
  * ``sow_readiness_scorecard`` — weighted readiness across dimensions
  * ``srl_missing_checklist`` — required-field gaps before kickoff

These are pure functions over the compile primitives — no I/O, no LLM,
deterministic, and fast. The LLM-synthesis layer consumes them as
authoritative starting points; the PM cockpit renders them directly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EntityRecord,
    EvidenceAtom,
    EvidenceEdge,
    EvidencePacket,
    PacketFamily,
    ReviewStatus,
)


# ─────────────────────────── PM DASHBOARD ───────────────────────────


def build_pm_dashboard(
    *,
    atoms: list[EvidenceAtom],
    packets: list[EvidencePacket],
    edges: list[EvidenceEdge],
    entities: list[EntityRecord] | None = None,
) -> dict[str, Any]:
    """Return the Monday-morning PM view.

    Pre-aggregates the highest-signal facts from atoms/edges/packets so
    a PM (or an LLM rendering for one) sees the actionable surface
    without re-scanning every atom:

      * ``blockers``                — open_question + parse_failure +
                                       contradictions with no resolution
      * ``cross_doc_contradictions``— quantity / scope mismatches
                                       across artifacts, with both sides
      * ``intra_doc_contradictions``— same-artifact self-contradictions
      * ``change_orders``           — customer_instruction atoms with
                                       structured change_delta payloads
      * ``stakeholders_by_role``    — every named stakeholder, role,
                                       email, phone (from c3k-shape rows)
      * ``risks_by_owner``          — risk atoms grouped by owner
      * ``action_items_by_owner``   — action_item atoms grouped by owner
      * ``open_questions``          — open_question atoms with their
                                       artifact + text
      * ``milestones_timeline``     — date+milestone atoms sorted
      * ``sla_summary``             — aggregated structured SLA fields
      * ``exclusions``              — explicit out-of-scope statements
      * ``money_summary``           — money atoms with currency rollup
    """
    entities = entities or []

    blockers: list[dict[str, Any]] = []
    open_qs: list[dict[str, Any]] = []
    cross_doc_contradictions: list[dict[str, Any]] = []
    intra_doc_contradictions: list[dict[str, Any]] = []
    change_orders: list[dict[str, Any]] = []
    stakeholders: list[dict[str, Any]] = []
    risks_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    risks_unowned: list[dict[str, Any]] = []
    action_items_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    action_items_unowned: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    sla_atoms: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    money_atoms: list[dict[str, Any]] = []

    # Two-pass stakeholder build: first sweep collects the RICHEST
    # value per slug (name + role + email + phone), then later atoms
    # referencing the same slug pick up the merged metadata. Without
    # this, a stakeholder whose first occurrence is a "risk owner"
    # cell (with no email/phone) renders as half-empty even though a
    # later contact row carries the full record.
    stakeholder_record: dict[str, dict[str, str]] = {}
    for atom in atoms:
        value = atom.value if isinstance(atom.value, dict) else {}
        for k in atom.entity_keys or []:
            if not k.startswith("stakeholder:"):
                continue
            slug = k[len("stakeholder:"):]
            current = stakeholder_record.setdefault(slug, {"slug": slug})
            if isinstance(value, dict):
                for field in ("name", "role", "email", "phone"):
                    v = value.get(field)
                    if isinstance(v, str) and v and not current.get(field):
                        current[field] = v
                # Risk-row sourced stakeholders carry ``owner`` but no
                # ``name``. Promote owner -> name so the dashboard has
                # a human-readable label even when the only mention
                # was in a risk-register Owner cell.
                if not current.get("name"):
                    owner_v = value.get("owner")
                    if isinstance(owner_v, str) and owner_v:
                        current["name"] = owner_v
    seen_stakeholder_slugs: set[str] = set()

    for atom in atoms:
        atom_type = atom.atom_type.value if hasattr(atom.atom_type, "value") else str(atom.atom_type)
        value = atom.value if isinstance(atom.value, dict) else {}
        text = atom.raw_text or ""

        # Open questions — the canonical "PM needs to chase someone" signal.
        if atom_type == "open_question":
            open_qs.append({
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "text": text[:200],
                "review_status": _review_status_str(atom.review_status),
            })
            blockers.append({
                "kind": "open_question",
                "atom_id": atom.id,
                "summary": text[:200],
            })

        # Change orders — customer_instruction with a structured delta.
        if atom_type == "customer_instruction":
            change_delta = value.get("change_delta") if isinstance(value, dict) else None
            entry = {
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "text": text[:240],
            }
            if isinstance(change_delta, dict):
                entry["change_delta"] = change_delta
            change_orders.append(entry)

        # Stakeholders — emit once per slug, pulling the richest
        # metadata collected in the two-pass sweep above.
        for k in atom.entity_keys or []:
            if k.startswith("stakeholder:"):
                slug = k[len("stakeholder:"):]
                if slug in seen_stakeholder_slugs:
                    continue
                seen_stakeholder_slugs.add(slug)
                stakeholders.append(dict(stakeholder_record.get(slug, {"slug": slug})))

        # Risk atoms — group by owner (stakeholder slug or raw owner string).
        if atom_type == "risk":
            risk_entry = {
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "risk_id": value.get("risk_id") if isinstance(value, dict) else None,
                "summary": value.get("risk_summary") if isinstance(value, dict) else text[:200],
                "severity": value.get("severity") if isinstance(value, dict) else None,
                "mitigation": value.get("mitigation") if isinstance(value, dict) else None,
            }
            owner = (value.get("owner") if isinstance(value, dict) else None) or ""
            owner_slug = _owner_slug_from_atom(atom, owner)
            if owner_slug:
                risk_entry["owner"] = owner
                risk_entry["owner_slug"] = owner_slug
                risks_by_owner[owner_slug].append(risk_entry)
            else:
                risks_unowned.append(risk_entry)

        # Action items — group by owner.
        if atom_type == "action_item":
            ai_entry = {
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "text": text[:240],
            }
            owner_slug = _owner_slug_from_atom(atom, "")
            if owner_slug:
                ai_entry["owner_slug"] = owner_slug
                action_items_by_owner[owner_slug].append(ai_entry)
            else:
                action_items_unowned.append(ai_entry)

        # Milestones — dates with milestone-context.
        for k in atom.entity_keys or []:
            if k.startswith("milestone:"):
                iso = k[len("milestone:"):]
                milestones.append({
                    "atom_id": atom.id,
                    "iso": iso,
                    "text": text[:160],
                })

        # SLAs — constraint atoms with structured sla payload.
        if atom_type == "constraint" and isinstance(value, dict) and isinstance(value.get("sla"), dict):
            sla_atoms.append({
                "atom_id": atom.id,
                "sla": value["sla"],
                "text": text[:200],
            })

        # Exclusions — explicit out-of-scope.
        if atom_type == "exclusion":
            exclusions.append({
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "text": text[:240],
            })

        # Money — currency rollup.
        for k in atom.entity_keys or []:
            if k.startswith("money:"):
                amount_raw = k[len("money:"):]
                try:
                    amount = float(amount_raw)
                except ValueError:
                    continue
                money_atoms.append({
                    "atom_id": atom.id,
                    "amount": amount,
                    "text": text[:160],
                })
                break

    # Cross-doc contradictions from edges. Dedupe (a vs b) and (b vs
    # a) into a single symmetric record so the dashboard doesn't
    # double-count the same disagreement.
    seen_contradictions: set[tuple[str, frozenset[str]]] = set()
    import re as _re
    for edge in edges:
        meta = edge.metadata or {}
        fam = meta.get("edge_family") if isinstance(meta, dict) else None
        if fam not in ("device_quantity_cross_doc", "part_number_quantity_conflict"):
            continue
        reason = edge.reason or ""
        # Symmetric key: (device_key, frozenset of qty values).
        device_match = _re.search(r"device:[a-z_]+", reason)
        nums = tuple(sorted({int(m) for m in _re.findall(r"\b(\d{1,5})\b", reason) if int(m) >= 2}))
        if device_match and nums:
            dedupe_key = (device_match.group(0), frozenset(nums))
            if dedupe_key in seen_contradictions:
                continue
            seen_contradictions.add(dedupe_key)
        entry = {
            "from_atom_id": edge.from_atom_id,
            "to_atom_id": edge.to_atom_id,
            "reason": reason,
        }
        if fam == "part_number_quantity_conflict":
            entry["kind"] = "part_number"
            cross_doc_contradictions.append(entry)
        elif "intra-doc" in reason.lower():
            intra_doc_contradictions.append(entry)
        else:
            cross_doc_contradictions.append(entry)

    # Add each cross-doc contradiction to the blockers stream.
    for c in cross_doc_contradictions:
        blockers.append({
            "kind": "cross_doc_contradiction",
            "summary": c.get("reason", "")[:240],
            "from_atom_id": c.get("from_atom_id"),
            "to_atom_id": c.get("to_atom_id"),
        })

    # Milestones sorted by ISO date.
    milestones.sort(key=lambda m: (m.get("iso") or ""))

    # Currency rollup over money atoms.
    money_total = sum(m["amount"] for m in money_atoms) if money_atoms else 0.0

    return {
        "blockers": blockers,
        "cross_doc_contradictions": cross_doc_contradictions,
        "intra_doc_contradictions": intra_doc_contradictions,
        "change_orders": change_orders,
        "stakeholders": stakeholders,
        "risks_by_owner": {k: v for k, v in risks_by_owner.items()},
        "risks_unowned": risks_unowned,
        "action_items_by_owner": {k: v for k, v in action_items_by_owner.items()},
        "action_items_unowned": action_items_unowned,
        "open_questions": open_qs,
        "milestones_timeline": milestones,
        "sla_summary": sla_atoms,
        "exclusions": exclusions,
        "money_summary": {
            "total": money_total,
            "atoms": money_atoms,
        },
    }


# ─────────────────────── SOW READINESS SCORECARD ───────────────────────


_READINESS_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("scope_clarity", "Scope is unambiguous (no unresolved cross-doc contradictions)"),
    ("stakeholder_coverage", "Key stakeholders named with role and contact"),
    ("schedule_definition", "Project milestones and key dates are present"),
    ("risk_coverage", "Risks identified with owners and mitigations"),
    ("pricing_clarity", "Money / commercial figures captured"),
    ("acceptance_defined", "Acceptance criteria or completion gates stated"),
    ("site_definition", "Sites / locations enumerated"),
    ("constraints_captured", "Site access, escort, SLA constraints noted"),
)


def build_sow_readiness_scorecard(
    *,
    atoms: list[EvidenceAtom],
    packets: list[EvidencePacket],
    edges: list[EvidenceEdge],
    entities: list[EntityRecord] | None = None,
) -> dict[str, Any]:
    """Return a weighted SOW-readiness scorecard.

    Each dimension is computed deterministically from the atom/edge/
    entity stream; the overall ``readiness_score`` is the simple mean
    of dimension scores. The accompanying ``signals`` list records
    which evidence drove each score so a PM can audit the call.
    """
    entities = entities or []
    by_atom_type: Counter[str] = Counter(_atom_type_str(a) for a in atoms)
    site_keys = {k for a in atoms for k in (a.entity_keys or []) if k.startswith("site:")}
    stakeholder_keys = {
        k for a in atoms for k in (a.entity_keys or []) if k.startswith("stakeholder:")
    }
    date_keys = {k for a in atoms for k in (a.entity_keys or []) if k.startswith("date:") or k.startswith("milestone:")}
    money_keys = {k for a in atoms for k in (a.entity_keys or []) if k.startswith("money:")}

    cross_doc_conflicts = sum(
        1 for e in edges
        if isinstance(e.metadata, dict)
        and e.metadata.get("edge_family") == "device_quantity_cross_doc"
    )

    risks = by_atom_type.get("risk", 0)
    risks_with_owner = sum(
        1 for a in atoms
        if _atom_type_str(a) == "risk"
        and isinstance(a.value, dict)
        and a.value.get("owner")
    )

    constraints = by_atom_type.get("constraint", 0)
    acceptance_signals = sum(
        1 for a in atoms
        if "acceptance" in (a.raw_text or "").lower()
        or "sign-off" in (a.raw_text or "").lower()
        or "signoff" in (a.raw_text or "").lower()
        or _atom_type_str(a) == "decision"
    )

    # Per-dimension scoring (0.0 - 1.0). Each dimension is bounded
    # individually and the overall score is the unweighted mean.
    dimensions: dict[str, dict[str, Any]] = {}

    # 1. scope_clarity — penalize cross-doc contradictions.
    scope_total = by_atom_type.get("scope_item", 0) + by_atom_type.get("quantity", 0)
    scope_penalty = min(1.0, cross_doc_conflicts / max(1.0, scope_total) * 4.0)
    dimensions["scope_clarity"] = {
        "score": round(max(0.0, 1.0 - scope_penalty), 3),
        "signals": {
            "scope_atom_count": scope_total,
            "cross_doc_quantity_conflicts": cross_doc_conflicts,
        },
    }

    # 2. stakeholder_coverage — number of named stakeholders.
    dimensions["stakeholder_coverage"] = {
        "score": round(min(1.0, len(stakeholder_keys) / 4.0), 3),
        "signals": {
            "stakeholder_count": len(stakeholder_keys),
            "stakeholder_slugs": sorted(stakeholder_keys)[:10],
        },
    }

    # 3. schedule_definition — milestone atoms present.
    milestone_atoms = sum(
        1 for a in atoms
        for k in (a.entity_keys or [])
        if k.startswith("milestone:")
    )
    dimensions["schedule_definition"] = {
        "score": round(min(1.0, len(date_keys) / 3.0), 3),
        "signals": {
            "date_count": len(date_keys),
            "milestone_atom_count": milestone_atoms,
        },
    }

    # 4. risk_coverage — risks with owners.
    if risks == 0:
        risk_score = 0.0
    else:
        coverage = risks_with_owner / risks if risks else 0.0
        risk_score = min(1.0, (risks / 3.0) * 0.5 + coverage * 0.5)
    dimensions["risk_coverage"] = {
        "score": round(risk_score, 3),
        "signals": {
            "risk_count": risks,
            "risks_with_owner": risks_with_owner,
        },
    }

    # 5. pricing_clarity — money atoms captured.
    dimensions["pricing_clarity"] = {
        "score": round(min(1.0, len(money_keys) / 3.0), 3),
        "signals": {
            "money_atom_count": len(money_keys),
        },
    }

    # 6. acceptance_defined — acceptance / sign-off / decision atoms.
    dimensions["acceptance_defined"] = {
        "score": round(min(1.0, acceptance_signals / 3.0), 3),
        "signals": {
            "acceptance_signal_count": acceptance_signals,
        },
    }

    # 7. site_definition — sites enumerated.
    dimensions["site_definition"] = {
        "score": round(min(1.0, len(site_keys) / 2.0), 3),
        "signals": {
            "site_count": len(site_keys),
            "site_slugs": sorted(site_keys)[:10],
        },
    }

    # 8. constraints_captured — site access / SLA / escort.
    dimensions["constraints_captured"] = {
        "score": round(min(1.0, constraints / 3.0), 3),
        "signals": {
            "constraint_atom_count": constraints,
        },
    }

    score_values = [d["score"] for d in dimensions.values()]
    overall = sum(score_values) / max(1, len(score_values))

    # Grade banding.
    if overall >= 0.85:
        grade = "ready_to_sow"
    elif overall >= 0.65:
        grade = "almost_ready"
    elif overall >= 0.40:
        grade = "needs_work"
    else:
        grade = "discovery_only"

    return {
        "readiness_score": round(overall, 3),
        "grade": grade,
        "dimensions": dimensions,
        "description_by_dimension": dict(_READINESS_DIMENSIONS),
    }


# ──────────────────── SRL MISSING-FIELDS CHECKLIST ────────────────────


# Lightweight subset of the 707-field SOW Requirements Library. These
# are the must-have fields for a managed-services engagement to leave
# discovery. The full SRL plugs into this same predicate framework.
_SRL_REQUIRED_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "field_id": "project_sponsor",
        "label": "Project Sponsor (named, with role)",
        "kind": "stakeholder",
        "predicate": "any_stakeholder",
    },
    {
        "field_id": "technical_lead",
        "label": "Technical / Engineering Lead",
        "kind": "stakeholder",
        "predicate": "stakeholder_role_match",
        "role_terms": ("technical lead", "tech lead", "engineer", "architect"),
    },
    {
        "field_id": "site_count",
        "label": "Sites / locations enumerated",
        "kind": "site",
        "predicate": "any_site",
    },
    {
        "field_id": "device_inventory",
        "label": "Device inventory captured",
        "kind": "device",
        "predicate": "any_device",
    },
    {
        "field_id": "kickoff_date",
        "label": "Kickoff / start date",
        "kind": "milestone",
        "predicate": "any_milestone",
    },
    {
        "field_id": "cutover_date",
        "label": "Cutover / go-live date",
        "kind": "milestone",
        "predicate": "milestone_text_match",
        "text_terms": ("cutover", "go-live", "go live", "rollout"),
    },
    {
        "field_id": "acceptance_criteria",
        "label": "Acceptance criteria stated",
        "kind": "acceptance",
        "predicate": "text_match",
        "text_terms": ("acceptance", "sign-off", "signoff", "completion criteria"),
    },
    {
        "field_id": "site_access_terms",
        "label": "Site access / escort / badge terms",
        "kind": "constraint",
        "predicate": "text_match",
        "text_terms": ("escort", "badge", "after-hours", "after hours", "background check"),
    },
    {
        "field_id": "sla_targets",
        "label": "SLA targets stated (uptime / response / resolution)",
        "kind": "sla",
        "predicate": "sla_present",
    },
    {
        "field_id": "pricing_baseline",
        "label": "Commercial pricing baseline captured",
        "kind": "money",
        "predicate": "any_money",
    },
    {
        "field_id": "risk_register",
        "label": "Risk register present with owners",
        "kind": "risk",
        "predicate": "risks_with_owners",
    },
    {
        "field_id": "escalation_path",
        "label": "Escalation path / on-call contact",
        "kind": "constraint",
        "predicate": "text_match",
        "text_terms": ("escalation", "on-call", "l1", "l2", "l3", "tier 1", "tier 2"),
    },
)


def build_srl_missing_checklist(
    *,
    atoms: list[EvidenceAtom],
    documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return required-field gaps for SOW readiness.

    Each field is evaluated via a deterministic predicate against the
    atom stream. Present fields are noted with the evidence that
    satisfied them; missing fields surface in ``missing`` with the
    label a PM should chase down.
    """
    documents = documents or []

    # Pre-compute facts once.
    stakeholders = [
        a for a in atoms
        if any(k.startswith("stakeholder:") for k in (a.entity_keys or []))
    ]
    sites = sorted({
        k for a in atoms for k in (a.entity_keys or []) if k.startswith("site:")
    })
    devices = sorted({
        k for a in atoms for k in (a.entity_keys or []) if k.startswith("device:")
    })
    milestones = [
        a for a in atoms
        if any(k.startswith("milestone:") for k in (a.entity_keys or []))
    ]
    money_atoms = [
        a for a in atoms
        if any(k.startswith("money:") for k in (a.entity_keys or []))
    ]
    risks = [a for a in atoms if _atom_type_str(a) == "risk"]
    risks_with_owner = [
        a for a in risks
        if isinstance(a.value, dict) and a.value.get("owner")
    ]
    sla_atoms = [
        a for a in atoms
        if _atom_type_str(a) == "constraint"
        and isinstance(a.value, dict)
        and isinstance(a.value.get("sla"), dict)
    ]
    all_text_lower = " ".join((a.raw_text or "").lower() for a in atoms)

    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for field in _SRL_REQUIRED_FIELDS:
        predicate = field["predicate"]
        satisfied = False
        evidence: dict[str, Any] = {}

        if predicate == "any_stakeholder":
            satisfied = len(stakeholders) > 0
            evidence = {"stakeholder_atom_count": len(stakeholders)}
        elif predicate == "stakeholder_role_match":
            terms = field.get("role_terms", ())
            hits = [
                a for a in stakeholders
                if isinstance(a.value, dict)
                and any(t in (a.value.get("role") or "").lower() for t in terms)
            ]
            satisfied = len(hits) > 0
            evidence = {"matching_role_count": len(hits)}
        elif predicate == "any_site":
            satisfied = len(sites) > 0
            evidence = {"site_count": len(sites)}
        elif predicate == "any_device":
            satisfied = len(devices) > 0
            evidence = {"device_count": len(devices)}
        elif predicate == "any_milestone":
            satisfied = len(milestones) > 0
            evidence = {"milestone_count": len(milestones)}
        elif predicate == "milestone_text_match":
            terms = field.get("text_terms", ())
            hits = [
                a for a in milestones
                if any(t in (a.raw_text or "").lower() for t in terms)
            ]
            satisfied = len(hits) > 0
            evidence = {"matching_milestone_count": len(hits)}
        elif predicate == "text_match":
            terms = field.get("text_terms", ())
            satisfied = any(t in all_text_lower for t in terms)
            evidence = {"matched_terms": [t for t in terms if t in all_text_lower]}
        elif predicate == "sla_present":
            satisfied = len(sla_atoms) > 0
            evidence = {"sla_atom_count": len(sla_atoms)}
        elif predicate == "any_money":
            satisfied = len(money_atoms) > 0
            evidence = {"money_atom_count": len(money_atoms)}
        elif predicate == "risks_with_owners":
            satisfied = len(risks_with_owner) >= 1
            evidence = {"risk_count": len(risks), "risks_with_owner": len(risks_with_owner)}

        entry = {
            "field_id": field["field_id"],
            "label": field["label"],
            "kind": field["kind"],
            "evidence": evidence,
        }
        if satisfied:
            present.append(entry)
        else:
            missing.append(entry)

    coverage = len(present) / max(1, len(present) + len(missing))
    return {
        "checklist_version": "srl_lite_v1",
        "field_count": len(_SRL_REQUIRED_FIELDS),
        "present_count": len(present),
        "missing_count": len(missing),
        "coverage": round(coverage, 3),
        "present": present,
        "missing": missing,
    }


# ─────────────────────── helpers ───────────────────────


def _atom_type_str(atom: EvidenceAtom) -> str:
    t = atom.atom_type
    return t.value if hasattr(t, "value") else str(t)


def _review_status_str(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _owner_slug_from_atom(atom: EvidenceAtom, owner_raw: str) -> str | None:
    """Find a stakeholder slug for the atom's owner.

    Prefers an explicit ``stakeholder:*`` key on the atom, falls back
    to slugifying the raw owner string. Returns None when no owner
    signal is present at all (so the dashboard separates owned vs
    unowned items).
    """
    for k in atom.entity_keys or []:
        if k.startswith("stakeholder:"):
            return k[len("stakeholder:"):]
    if owner_raw and isinstance(owner_raw, str):
        import re as _re
        slug = _re.sub(r"[^a-z0-9]+", "_", owner_raw.lower()).strip("_")
        if slug and slug not in {"owner", "tbd", "unknown"}:
            return slug
    return None


__all__ = [
    "build_pm_dashboard",
    "build_sow_readiness_scorecard",
    "build_srl_missing_checklist",
]
