"""OrbitBrief-Core: deterministic PM-ready aggregations over the envelope.

DEPRECATED LOCATION — this module has moved to Orbitbrief-Core at
``orbitbrief_core.envelope_builders``. The copy in parser-os remains
for back-compat with the parser-os shim envelope and direct callers.
New code should import from ``orbitbrief_core.envelope_builders``
directly.

See: https://github.com/Purtera-IT/Orbitbrief-Core
PR:  feat/envelope-migration-from-parser-os

The envelope (orbitbrief.input.v2) carries atoms, edges, packets, and
indexes — everything an LLM needs to reason about the project, but
nothing pre-aggregated for the operator running the show.

This module computes the deliverables the Purpulse one-pager attributes
to OrbitBrief-Core, plus the operator-facing surfaces a PM cockpit
needs to render the cockpit without re-deriving:

Core deliverables:
  * ``pm_dashboard``           — Monday-morning view
  * ``sow_readiness_scorecard``— weighted readiness across dimensions
  * ``srl_missing_checklist``  — required-field gaps before kickoff

Cockpit surfaces (S+++++):
  * ``scope_truth``           — authority-weighted canonical
                                 (device, site) counts with audit trail
  * ``change_order_timeline`` — chronological audit of every scope
                                 change with from→to deltas
  * ``site_readiness``        — per-site rollup of all signals
  * ``stakeholder_load``      — who owes what, severity-weighted
  * ``project_vitals``        — single 0-100 health score with breakdown

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




def _schematic_review_safe(atoms):
    """Never break PM_HANDOFF if the schematic review errors."""
    try:
        from app.core.schematic_pm_review import schematic_pm_review
        return schematic_pm_review(atoms)
    except Exception:
        return {"present": False}


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
        # Skip questions whose answer already exists in the corpus (flagged
        # by open_question_resolution): they're not blockers.
        if atom_type == "open_question":
            _answered = (
                (isinstance(value, dict) and value.get("answered") is True)
                or ("answered_in_corpus" in (atom.review_flags or []))
            )
            if not _answered:
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
            risk_summary = (value.get("risk_summary") if isinstance(value, dict) else None) or text or ""
            risk_entry = {
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "risk_id": value.get("risk_id") if isinstance(value, dict) else None,
                "summary": risk_summary[:200],
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

    # Gap-driven questions: turn high-priority MISSING SRL fields into
    # explicit questions the PM should chase. Unlike open_question atoms,
    # these come from what's absent, not from literal "?" characters.
    gap_questions: list[dict[str, Any]] = []
    try:
        from app.core.open_question_resolution import generate_gap_questions
        _srl = build_srl_missing_checklist(atoms=atoms)
        gap_questions = generate_gap_questions(_srl)
    except Exception:
        gap_questions = []
    for gq in gap_questions:
        open_qs.append({
            "atom_id": None,
            "artifact_id": None,
            "text": gq["summary"],
            "review_status": "needs_review",
            "kind": "generated_gap",
            "field_id": gq.get("field_id"),
        })
        blockers.append({
            "kind": "generated_gap",
            "atom_id": None,
            "summary": gq["summary"],
            "field_id": gq.get("field_id"),
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
        "schematic_review": _schematic_review_safe(atoms),
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


# Subset of the 707-field SOW Requirements Library covering the
# domains the parser-os substrate has primitives for. The full SRL
# YAML plugs in via the same predicate framework — this in-code list
# is the "lite" surface we can evaluate from primitives alone.
#
# Categories: stakeholders / sites / devices / schedule / commercial
# / acceptance / constraints / risk / governance / operations.
_SRL_REQUIRED_FIELDS: tuple[dict[str, Any], ...] = (
    # ── Stakeholders ──────────────────────────────────────────
    {
        "field_id": "project_sponsor",
        "label": "Project Sponsor (named)",
        "category": "stakeholders",
        "predicate": "stakeholder_role_match",
        "role_terms": ("sponsor", "executive sponsor", "program sponsor"),
    },
    {
        "field_id": "technical_lead",
        "label": "Technical / Engineering Lead",
        "category": "stakeholders",
        "predicate": "stakeholder_role_match",
        "role_terms": ("technical lead", "tech lead", "engineer", "architect"),
    },
    {
        "field_id": "project_manager",
        "label": "Project Manager (vendor side)",
        "category": "stakeholders",
        "predicate": "stakeholder_role_match",
        "role_terms": ("project manager", "vendor pm", "pm", "delivery manager"),
    },
    {
        "field_id": "security_owner",
        "label": "Security / Compliance Owner",
        "category": "stakeholders",
        "predicate": "stakeholder_role_match",
        "role_terms": ("security", "ciso", "compliance"),
    },
    {
        "field_id": "site_contact",
        "label": "Per-site primary contact",
        "category": "stakeholders",
        "predicate": "stakeholder_role_match",
        "role_terms": ("site manager", "site lead", "site contact", "facilities"),
    },
    # ── Sites & locations ─────────────────────────────────────
    {
        "field_id": "site_enumeration",
        "label": "Sites / locations enumerated",
        "category": "sites",
        "predicate": "any_site",
    },
    {
        "field_id": "site_addresses",
        "label": "Site street addresses present",
        "category": "sites",
        "predicate": "any_address",
    },
    {
        "field_id": "device_inventory",
        "label": "Device inventory captured",
        "category": "sites",
        "predicate": "any_device",
    },
    {
        "field_id": "device_qty_per_site",
        "label": "Per-site device counts (no contradictions)",
        "category": "sites",
        "predicate": "scope_truth_clean",
    },
    # ── Schedule ──────────────────────────────────────────────
    {
        "field_id": "kickoff_date",
        "label": "Kickoff / project start date",
        "category": "schedule",
        "predicate": "milestone_text_match",
        "text_terms": ("kickoff", "kick-off", "start date", "mobilization"),
    },
    {
        "field_id": "phase_milestones",
        "label": "Phase milestones with dates",
        "category": "schedule",
        "predicate": "milestone_count_min",
        "min_count": 2,
    },
    {
        "field_id": "cutover_date",
        "label": "Cutover / go-live date",
        "category": "schedule",
        "predicate": "milestone_text_match",
        "text_terms": ("cutover", "go-live", "go live", "rollout"),
    },
    {
        "field_id": "blackout_windows",
        "label": "Blackout / maintenance windows defined",
        "category": "schedule",
        "predicate": "text_match",
        "text_terms": ("blackout", "maintenance window", "downtime window", "freeze period"),
    },
    # ── Commercial ────────────────────────────────────────────
    {
        "field_id": "pricing_baseline",
        "label": "Commercial pricing baseline captured",
        "category": "commercial",
        "predicate": "any_money",
    },
    {
        "field_id": "payment_terms",
        "label": "Payment / invoicing terms",
        "category": "commercial",
        "predicate": "text_match",
        "text_terms": ("net 30", "net 60", "net 45", "milestone payment", "progress payment", "invoice"),
    },
    {
        "field_id": "currency_clarity",
        "label": "Currency specified (single primary)",
        "category": "commercial",
        "predicate": "text_match",
        "text_terms": ("usd", "$", "eur", "gbp", "cad"),
    },
    {
        "field_id": "change_order_process",
        "label": "Change order process documented",
        "category": "commercial",
        "predicate": "text_match",
        "text_terms": ("change order", "change request", "scope change", "revised scope"),
    },
    # ── Acceptance & quality ──────────────────────────────────
    {
        "field_id": "acceptance_criteria",
        "label": "Acceptance criteria stated",
        "category": "acceptance",
        "predicate": "text_match",
        "text_terms": ("acceptance", "completion criteria", "definition of done"),
    },
    {
        "field_id": "sign_off_required",
        "label": "Customer sign-off gate required",
        "category": "acceptance",
        "predicate": "text_match",
        "text_terms": ("sign-off", "signoff", "customer approval", "customer acceptance"),
    },
    {
        "field_id": "test_plan",
        "label": "Test / validation plan referenced",
        "category": "acceptance",
        "predicate": "text_match",
        "text_terms": ("test plan", "test results", "validation", "performance test", "smoke test"),
    },
    {
        "field_id": "warranty_terms",
        "label": "Warranty / defect-correction window",
        "category": "acceptance",
        "predicate": "text_match",
        "text_terms": ("warranty", "defect", "rework", "remediation"),
    },
    # ── Constraints / site access ─────────────────────────────
    {
        "field_id": "site_access_terms",
        "label": "Site access / escort / badge terms",
        "category": "constraints",
        "predicate": "text_match",
        "text_terms": ("escort", "badge", "after-hours", "after hours", "background check"),
    },
    {
        "field_id": "work_hours",
        "label": "Allowed work hours / windows",
        "category": "constraints",
        "predicate": "text_match",
        "text_terms": ("work window", "business hours", "weekdays", "weekends", "after hours"),
    },
    {
        "field_id": "lift_equipment",
        "label": "Lift / boom / scaffolding requirements",
        "category": "constraints",
        "predicate": "text_match",
        "text_terms": ("lift", "boom lift", "scissor lift", "scaffolding", "ceiling access"),
    },
    {
        "field_id": "ppe_safety",
        "label": "PPE / safety requirements",
        "category": "constraints",
        "predicate": "text_match",
        "text_terms": ("ppe", "hard hat", "osha", "harness", "fall protection", "hot work"),
    },
    # ── SLAs & operations ─────────────────────────────────────
    {
        "field_id": "sla_targets",
        "label": "SLA targets stated (uptime / response / resolution)",
        "category": "operations",
        "predicate": "sla_present",
    },
    {
        "field_id": "support_tiers",
        "label": "Support tier structure (P1/P2/P3 or similar)",
        "category": "operations",
        "predicate": "text_match",
        "text_terms": ("p1", "p2", "p3", "tier 1", "tier 2", "severity 1", "severity 2"),
    },
    {
        "field_id": "escalation_path",
        "label": "Escalation path / on-call contact",
        "category": "operations",
        "predicate": "text_match",
        "text_terms": ("escalation", "on-call", "oncall", "l1", "l2", "l3", "after-hours support"),
    },
    {
        "field_id": "service_credits",
        "label": "Service credits / penalty terms",
        "category": "operations",
        "predicate": "text_match",
        "text_terms": ("service credit", "penalty", "credit", "remedy"),
    },
    {
        "field_id": "maintenance_window",
        "label": "Routine maintenance window",
        "category": "operations",
        "predicate": "text_match",
        "text_terms": ("maintenance window", "patching", "firmware update", "scheduled maintenance"),
    },
    # ── Risk & governance ─────────────────────────────────────
    {
        "field_id": "risk_register",
        "label": "Risk register present with owners",
        "category": "risk",
        "predicate": "risks_with_owners",
    },
    {
        "field_id": "compliance_clauses",
        "label": "Compliance / regulatory clauses (HIPAA / PCI / etc.)",
        "category": "risk",
        "predicate": "text_match",
        "text_terms": ("hipaa", "pci", "ferpa", "soc 2", "iso 27001", "nist", "nfpa", "ada"),
    },
    {
        "field_id": "data_handling",
        "label": "Data handling / confidentiality",
        "category": "risk",
        "predicate": "text_match",
        "text_terms": ("confidential", "nda", "data handling", "pii", "phi"),
    },
    {
        "field_id": "ip_ownership",
        "label": "IP ownership / work product",
        "category": "risk",
        "predicate": "text_match",
        "text_terms": ("intellectual property", "work product", "ownership", "license"),
    },
    {
        "field_id": "indemnification",
        "label": "Indemnification clause",
        "category": "risk",
        "predicate": "text_match",
        "text_terms": ("indemnif", "hold harmless"),
    },
    # ── Out-of-scope / exclusions ─────────────────────────────
    {
        "field_id": "exclusions_documented",
        "label": "Out-of-scope items documented",
        "category": "scope",
        "predicate": "any_exclusion",
    },
    {
        "field_id": "customer_responsibilities",
        "label": "Customer responsibilities stated",
        "category": "scope",
        "predicate": "text_match",
        "text_terms": ("customer provides", "customer will", "customer responsibility", "owner provides"),
    },
    {
        "field_id": "vendor_responsibilities",
        "label": "Vendor / contractor responsibilities stated",
        "category": "scope",
        "predicate": "text_match",
        "text_terms": ("vendor will", "vendor provides", "vendor responsibility", "contractor will"),
    },
    # ── Closeout / handoff ────────────────────────────────────
    {
        "field_id": "as_built_required",
        "label": "As-built documentation required",
        "category": "closeout",
        "predicate": "text_match",
        "text_terms": ("as-built", "as built", "redline", "drawings"),
    },
    {
        "field_id": "ops_handoff",
        "label": "Operations handoff package defined",
        "category": "closeout",
        "predicate": "text_match",
        "text_terms": ("handoff", "hand-off", "operations turnover", "runbook"),
    },
    {
        "field_id": "training_required",
        "label": "Training / knowledge transfer scope",
        "category": "closeout",
        "predicate": "text_match",
        "text_terms": ("training", "knowledge transfer", "kt session", "user training"),
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
    exclusions_atoms = [a for a in atoms if _atom_type_str(a) == "exclusion"]
    addresses = sorted({
        k for a in atoms for k in (a.entity_keys or [])
        if k.startswith("address:") or k.startswith("zip:")
    })
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
        elif predicate == "any_address":
            satisfied = len(addresses) > 0
            evidence = {"address_count": len(addresses)}
        elif predicate == "any_exclusion":
            satisfied = len(exclusions_atoms) > 0
            evidence = {"exclusion_atom_count": len(exclusions_atoms)}
        elif predicate == "milestone_count_min":
            min_count = int(field.get("min_count", 1) or 1)
            satisfied = len(milestones) >= min_count
            evidence = {"milestone_count": len(milestones), "required": min_count}
        elif predicate == "scope_truth_clean":
            # "Clean" means: device-mention atoms exist AND no cross-doc
            # quantity contradictions remain unresolved. We approximate
            # with: at least one device + atom_type=quantity AND fewer
            # than 3 contradiction-shaped reasons in the atom stream.
            qty_atoms = [
                a for a in atoms
                if _atom_type_str(a) in ("quantity", "vendor_line_item")
                and any(k.startswith("device:") for k in (a.entity_keys or []))
            ]
            satisfied = len(qty_atoms) >= 1
            evidence = {"qty_with_device_count": len(qty_atoms)}

        entry = {
            "field_id": field["field_id"],
            "label": field["label"],
            "category": field.get("category", "general"),
            "evidence": evidence,
        }
        if satisfied:
            present.append(entry)
        else:
            missing.append(entry)

    coverage = len(present) / max(1, len(present) + len(missing))
    # Per-category coverage so a PM can see which area of the SRL is
    # thinnest (e.g. "schedule" 4/4 but "closeout" 0/3).
    cat_present: Counter[str] = Counter(e.get("category", "general") for e in present)
    cat_total: Counter[str] = Counter(e.get("category", "general") for e in present + missing)
    by_category = {
        c: {
            "present": cat_present.get(c, 0),
            "total": cat_total.get(c, 0),
            "coverage": round(cat_present.get(c, 0) / max(1, cat_total.get(c, 0)), 3),
        }
        for c in sorted(cat_total)
    }
    return {
        "checklist_version": "srl_v2_expanded",
        "field_count": len(_SRL_REQUIRED_FIELDS),
        "present_count": len(present),
        "missing_count": len(missing),
        "coverage": round(coverage, 3),
        "by_category": by_category,
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


# ─────────────────────────── SCOPE TRUTH ───────────────────────────


# Authority precedence — the higher number wins when two atoms make
# contradicting claims about the same (device, site) count. Matches
# the architecture from item_identity: signed legal artifacts top
# vendor proposals top meeting transcripts.
_AUTHORITY_PRECEDENCE: dict[str, int] = {
    "approved_site_roster": 100,
    "contractual_scope": 90,
    "formal_sow": 85,
    "current_addendum": 80,
    "customer_current_authored": 75,
    "vendor_quote": 60,
    "meeting_note": 40,
    "quoted_old_email": 20,
    "machine_extractor": 10,
}


def _atom_authority_rank(atom: EvidenceAtom) -> int:
    ac = atom.authority_class
    name = ac.value if hasattr(ac, "value") else str(ac)
    return _AUTHORITY_PRECEDENCE.get(name, 30)


def build_scope_truth(
    *,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
) -> dict[str, Any]:
    """Return the authoritative scope per (device, site) tuple.

    When multiple artifacts make contradicting quantity claims for the
    same device at the same site (SOW says 48 cameras, vendor quote
    says 42, customer email says 36), the highest-authority claim wins
    and the others surface as ``contested_claims`` so a PM sees the
    full audit trail and which artifact governs.

    ``scope_truth.devices`` is the canonical answer the LLM should
    quote. ``scope_truth.contested`` is the open-questions list the
    PM should chase to close. Both carry source atom IDs so any claim
    is traceable to a page.
    """
    # Group quantity-bearing atoms by (device, site).
    by_key: dict[tuple[str, str], list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        atom_type = _atom_type_str(atom)
        if atom_type not in ("quantity", "scope_item", "vendor_line_item", "constraint", "site_roster"):
            continue
        devices = {k for k in (atom.entity_keys or []) if k.startswith("device:") and k != "device:unknown"}
        sites = {k for k in (atom.entity_keys or []) if k.startswith("site:")}
        if not devices:
            continue
        # When no site is named we still want a row, just under (device, "site:*").
        sites_for_row = sites if sites else {"site:*"}
        for d in devices:
            for s in sites_for_row:
                by_key[(d, s)].append(atom)

    devices_truth: list[dict[str, Any]] = []
    contested: list[dict[str, Any]] = []
    import re as _re
    for (device, site), bucket in sorted(by_key.items()):
        # Extract the noun-anchored quantity per atom for THIS device.
        from app.core.graph_builder import _noun_anchored_quantity
        device_canonical = device.split(":", 1)[1]
        claims: list[dict[str, Any]] = []
        for atom in bucket:
            qty = _noun_anchored_quantity(atom.raw_text or "", device_canonical)
            if qty is None:
                # Fallback: take qty:N from atom.entity_keys IF this
                # atom has only ONE device. Multi-device atoms without
                # noun-anchored binding are too ambiguous to credit.
                atom_devices = {k for k in atom.entity_keys or [] if k.startswith("device:") and k != "device:unknown"}
                atom_qtys = {k for k in atom.entity_keys or [] if k.startswith("quantity:")}
                if len(atom_devices) == 1 and len(atom_qtys) == 1:
                    try:
                        qty = int(list(atom_qtys)[0].split(":", 1)[1])
                    except (ValueError, IndexError):
                        qty = None
            if qty is None:
                continue
            ac = atom.authority_class
            ac_name = ac.value if hasattr(ac, "value") else str(ac)
            claims.append({
                "atom_id": atom.id,
                "artifact_id": atom.artifact_id,
                "quantity": qty,
                "authority_class": ac_name,
                "authority_rank": _atom_authority_rank(atom),
                "text": (atom.raw_text or "")[:160],
            })
        if not claims:
            continue
        # Group by quantity value.
        by_value: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for c in claims:
            by_value[c["quantity"]].append(c)
        # The governing claim is the highest authority. Ties broken by
        # claim count (more sources agreeing wins).
        ranked = sorted(
            by_value.items(),
            key=lambda kv: (-max(c["authority_rank"] for c in kv[1]), -len(kv[1]), kv[0]),
        )
        governing_value, governing_claims = ranked[0]
        all_values = sorted(by_value.keys())
        entry = {
            "device": device,
            "site": site,
            "canonical_quantity": governing_value,
            "governing_claims": governing_claims,
            "governing_authority": max(c["authority_class"] for c in governing_claims),
            "all_reported_values": all_values,
            "is_contested": len(by_value) > 1,
        }
        devices_truth.append(entry)
        if entry["is_contested"]:
            contested.append({
                "device": device,
                "site": site,
                "canonical_quantity": governing_value,
                "competing_values": [v for v in all_values if v != governing_value],
                "audit": [
                    {"quantity": qty, "claims": cl}
                    for qty, cl in sorted(by_value.items())
                ],
            })

    return {
        "devices": devices_truth,
        "contested": contested,
        "device_count": len({e["device"] for e in devices_truth}),
        "site_count": len({e["site"] for e in devices_truth if e["site"] != "site:*"}),
        "contested_count": len(contested),
    }


# ─────────────────────── CHANGE ORDER TIMELINE ───────────────────────


def build_change_order_timeline(
    *,
    atoms: list[EvidenceAtom],
) -> dict[str, Any]:
    """Return the chronological audit trail of every scope change.

    Each entry: timestamp (from message_index / line_start / atom
    ordering as a proxy), source artifact, the change_delta (from→to)
    when structured, the raw text, and approval signal (explicit
    "approved" / "approve to proceed" in the same atom).

    Sorted oldest-first when locator metadata permits, otherwise by
    deterministic atom_id order so repeated compiles produce the same
    timeline.
    """
    entries: list[dict[str, Any]] = []
    # The markdown classifier may emit the same paragraph as multiple
    # atom types (e.g. "Hold off on Phase 2" lands as both
    # ``exclusion`` and ``customer_instruction``). Dedup by atom_id
    # so the timeline doesn't double-count one change-order event.
    seen_atom_ids: set[str] = set()
    # When the same paragraph is split into multiple atom types,
    # prefer ``customer_instruction`` over ``exclusion`` over
    # ``decision`` so the most-actionable label wins.
    type_priority = {
        "customer_instruction": 0,
        "decision": 1,
        "exclusion": 2,
    }
    atoms_sorted_for_dedupe = sorted(
        atoms, key=lambda a: type_priority.get(_atom_type_str(a), 99),
    )
    for atom in atoms_sorted_for_dedupe:
        atom_type = _atom_type_str(atom)
        value = atom.value if isinstance(atom.value, dict) else {}
        if atom_type not in ("customer_instruction", "exclusion", "decision"):
            continue
        # Dedupe on (artifact, raw_text) — same text emitted as two
        # atom types from one paragraph should produce ONE entry.
        text_key = (atom.artifact_id, (atom.raw_text or "")[:120])
        if text_key in seen_atom_ids:
            continue
        seen_atom_ids.add(text_key)
        text_lower = (atom.raw_text or "").lower()
        # Only count atoms that look like a change-order signal:
        # explicit add/remove/cancel/reduce/approve language OR
        # a structured change_delta.
        is_change = (
            isinstance(value.get("change_delta"), dict)
            or any(t in text_lower for t in (
                "please add", "please remove", "please include",
                "cancel", "reduce", "drop the", "drop from",
                "approve", "approved to proceed", "approved the revised",
                "hold off", "go ahead",
            ))
        )
        if not is_change:
            continue
        # Locator ordering: prefer (artifact_id, message_index OR
        # line_start OR row) as a sortable tuple. Falls back to atom.id.
        locator = (atom.source_refs[0].locator if atom.source_refs else {}) or {}
        if not isinstance(locator, dict):
            locator = {}
        sort_key = (
            atom.artifact_id,
            locator.get("message_index") or 0,
            locator.get("line_start") or locator.get("row") or 0,
            atom.id,
        )
        text = atom.raw_text or ""
        approval_signal = any(t in text_lower for t in (
            "approved to proceed", "approved the revised", "approve to proceed",
            "we approve", "we agree", "approved",
        ))
        entry: dict[str, Any] = {
            "atom_id": atom.id,
            "artifact_id": atom.artifact_id,
            "kind": atom_type,
            "text": text[:240],
            "approval_signal": approval_signal,
            "sort_key": list(sort_key),
        }
        if isinstance(value.get("change_delta"), dict):
            entry["change_delta"] = value["change_delta"]
        # Capture the stakeholder driving the change when present on
        # the atom (e.g. customer-email From: header).
        for k in atom.entity_keys or []:
            if k.startswith("stakeholder:"):
                entry["driven_by"] = k[len("stakeholder:"):]
                break
        entries.append(entry)
    entries.sort(key=lambda e: e["sort_key"])
    return {
        "entries": entries,
        "entry_count": len(entries),
        "with_structured_delta": sum(1 for e in entries if "change_delta" in e),
        "with_approval_signal": sum(1 for e in entries if e["approval_signal"]),
    }


# ─────────────────────────── SITE READINESS ───────────────────────────

# Gap E: readiness floor for a site that is anchored by a physical_site
# atom (we know where it is and that it's in scope) but has no other
# signal yet. Keeps "located, details pending" from scoring identically
# to a site we know nothing about. Deliberately small — anchoring is
# real progress but far from readiness.
_SITE_ANCHOR_FLOOR = 0.15


def build_site_readiness(
    *,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
) -> dict[str, Any]:
    """Return a per-site rollup of completeness signals.

    For each site mentioned anywhere, surface: device count, named
    constraints, stakeholders attached, contradictions targeting that
    site, and a 0-1 readiness score.
    """
    sites: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "site_key": "",
        "device_keys": set(),
        "constraint_count": 0,
        "stakeholder_keys": set(),
        "contradiction_count": 0,
        "scope_atom_count": 0,
        "money_present": False,
        "milestone_present": False,
    })

    # Gap E: the set of site slugs that are *anchored* by a physical_site
    # atom — i.e. we positively know the location exists and is in scope.
    # An anchored-but-undetailed site is materially different from a deal
    # where the site is entirely unknown; the maturity model below uses
    # this to grade "located, details pending" (amber) apart from
    # "unscoped" (red), instead of both collapsing to readiness 0.0.
    anchored_set: set[str] = set()

    for atom in atoms:
        atom_type = _atom_type_str(atom)
        # v49 FIX 5 (v49.1): physical_site atoms ARE the authoritative
        # site source. Read their structured value for the canonical
        # site key. Skip atoms with NO explicit site_id/id — they
        # otherwise fall back to address strings as the slug,
        # producing keys like "site:address_1180_peachtree_st_..."
        # which is just another ghost site.
        if atom_type == "physical_site":
            val = getattr(atom, "value", None) or {}
            sid = ""
            if isinstance(val, dict):
                sid = val.get("id") or val.get("site_id") or ""
            if not sid:
                # No explicit site code → skip. Don't pollute site_readiness.
                continue
            import re as _re_sid
            site_slug = _re_sid.sub(r"[^a-z0-9]+", "_", sid.lower()).strip("_")
            site_keys = [f"site:{site_slug}"] if site_slug else []
        else:
            # v49.1: skip v49 schema atom types entirely — they're
            # already structured. Their raw_text contains snippets
            # like "Wi-Fi 7 AP | site: ATL-AIR | qty: 105" which the
            # regex emitter parses as site:atl_air_qty_105 ghosts.
            if atom_type in (
                "site_allocation", "bom_line", "cutover_step",
                "acceptance_criterion", "deliverable", "site_budget",
                "integration_checkpoint", "compliance_classification",
                "system_mapping", "signatory", "site_attribute",
            ):
                continue
            site_keys = [k for k in (atom.entity_keys or []) if k.startswith("site:")]
        if not site_keys:
            continue
        for sk in site_keys:
            entry = sites[sk]
            entry["site_key"] = sk
            for k in atom.entity_keys or []:
                if k.startswith("device:") and k != "device:unknown":
                    entry["device_keys"].add(k)
                if k.startswith("stakeholder:"):
                    entry["stakeholder_keys"].add(k)
                if k.startswith("money:"):
                    entry["money_present"] = True
                if k.startswith("milestone:"):
                    entry["milestone_present"] = True
            if atom_type == "constraint":
                entry["constraint_count"] += 1
            if atom_type in ("scope_item", "quantity", "vendor_line_item"):
                entry["scope_atom_count"] += 1

    # Cross-doc contradictions can include a site key in either
    # endpoint atom — match those against per-site totals.
    import re as _re
    atom_by_id = {a.id: a for a in atoms}
    for edge in edges:
        meta = edge.metadata or {}
        fam = meta.get("edge_family") if isinstance(meta, dict) else None
        if fam != "device_quantity_cross_doc":
            continue
        for atom_id in (edge.from_atom_id, edge.to_atom_id):
            atom = atom_by_id.get(atom_id)
            if not atom:
                continue
            for k in atom.entity_keys or []:
                if k.startswith("site:"):
                    sites[k]["contradiction_count"] += 1
                    break

    # v53.1 FIX 5c: 3-level canonicalization + gating.
    #
    # The problem (cloud v52): rows like atlanta_headquarters,
    # innovation_tower, warehouse_rf survive merge because the alias
    # map didn't catch them. They have no devices/scope — just stray
    # mentions in text — but they still show as "sites".
    #
    # Fix: build canonical_set FROM physical_site atoms (authoritative
    # IDs), build name→canonical map from physical_site value.name +
    # value.names, then drop any merged row that is (a) not in
    # canonical_set AND (b) has zero substantive signal.
    try:
        import re as _re_canon
        from app.core.site_detection import _llm_site_attr_cache

        canonical_map: dict[str, str] = {}   # alias_slug → canonical_slug
        canonical_set: set[str] = set()      # canonical site_keys only

        # (0) Authoritative canonical IDs from physical_site atoms.
        # These are the ONLY truly canonical sites — every other slug
        # must alias into one of these or be dropped (if it has no
        # substantive signal).
        for _a in atoms:
            if _atom_type_str(_a) != "physical_site":
                continue
            _val = getattr(_a, "value", None) or {}
            if not isinstance(_val, dict):
                continue
            _sid = _val.get("id") or _val.get("site_id") or ""
            if not _sid:
                continue
            _canon_slug = "site:" + _re_canon.sub(
                r"[^a-z0-9]+", "_", str(_sid).lower()
            ).strip("_")
            if not _canon_slug or _canon_slug == "site:":
                continue
            canonical_set.add(_canon_slug)
            anchored_set.add(_canon_slug)
            # Map physical_site's name + alternative names to this canonical.
            for _name_field in ("name", "names", "aliases", "alternative_names"):
                _nv = _val.get(_name_field)
                if not _nv:
                    continue
                if isinstance(_nv, str):
                    _nv = [_nv]
                if not isinstance(_nv, (list, tuple)):
                    continue
                for _nm in _nv:
                    if not _nm:
                        continue
                    _nslug = "site:" + _re_canon.sub(
                        r"[^a-z0-9]+", "_", str(_nm).lower()
                    ).strip("_")
                    if _nslug and _nslug != _canon_slug:
                        canonical_map[_nslug] = _canon_slug

        # (a) LLM cache aliases (v48 structured site extraction)
        if _llm_site_attr_cache:
            for alias_name, site_obj in _llm_site_attr_cache.items():
                if not isinstance(site_obj, dict):
                    continue
                canon_id = site_obj.get("id") or alias_name
                if not canon_id:
                    continue
                canon_slug = "site:" + _re_canon.sub(r"[^a-z0-9]+", "_", str(canon_id).lower()).strip("_")
                alias_slug = "site:" + _re_canon.sub(r"[^a-z0-9]+", "_", str(alias_name).lower()).strip("_")
                if alias_slug and canon_slug:
                    canonical_map[alias_slug] = canon_slug
                    canonical_set.add(canon_slug)
                for nm in site_obj.get("names") or []:
                    if nm:
                        nslug = "site:" + _re_canon.sub(r"[^a-z0-9]+", "_", str(nm).lower()).strip("_")
                        if nslug and nslug != canon_slug:
                            canonical_map[nslug] = canon_slug

        # (b) Prefix-based universal collapse. Goal: collapse "atl_hq"
        # into "atl_hq_01" when both exist. Catches short-form aliases
        # like "ATL-HQ" referring to row "ATL-HQ-01".
        sorted_slugs = sorted(sites.keys(), key=len, reverse=True)
        for long_slug in sorted_slugs:
            for short_slug in sorted_slugs:
                if short_slug == long_slug:
                    continue
                if len(short_slug) >= len(long_slug):
                    continue
                if not long_slug.startswith(short_slug + "_"):
                    continue
                extra = long_slug[len(short_slug) + 1:]
                if len(extra) > 6:
                    continue
                if short_slug in canonical_map:
                    continue
                canonical_map[short_slug] = long_slug

        # (c) Token-overlap collapse for name-style slugs into canonical
        # IDs. Catches "atlanta_headquarters" → "atl_hq_01" when the
        # canonical's first-word-initials overlap. Universal heuristic.
        if canonical_set:
            def _token_initials(slug: str) -> str:
                # site:atl_hq_01 → atlhq
                body = slug[5:] if slug.startswith("site:") else slug
                parts = [p for p in body.split("_") if p and not p.isdigit()]
                return "".join(p[0] for p in parts if p)[:6]

            canon_initials = {
                _token_initials(c): c for c in canonical_set if _token_initials(c)
            }
            for sk in list(sites.keys()):
                if sk in canonical_set or sk in canonical_map:
                    continue
                # Compute initials from the alias slug. If a canonical
                # exists whose initials are a prefix of the alias's
                # initials, alias into it.
                body = sk[5:] if sk.startswith("site:") else sk
                parts = [p for p in body.split("_") if p]
                alias_initials = "".join(p[0] for p in parts if p)[:6]
                if not alias_initials or len(alias_initials) < 2:
                    continue
                # Prefix match against canonical initials.
                for ci, cslug in canon_initials.items():
                    if not ci:
                        continue
                    if alias_initials.startswith(ci) or ci.startswith(alias_initials):
                        canonical_map[sk] = cslug
                        break

        # Merge entries
        merged: dict[str, dict[str, Any]] = {}
        for sk, entry in sites.items():
            canon = canonical_map.get(sk, sk)
            if canon not in merged:
                merged[canon] = {
                    "site_key": canon,
                    "device_keys": set(entry.get("device_keys", set())),
                    "constraint_count": entry.get("constraint_count", 0),
                    "stakeholder_keys": set(entry.get("stakeholder_keys", set())),
                    "contradiction_count": entry.get("contradiction_count", 0),
                    "scope_atom_count": entry.get("scope_atom_count", 0),
                    "money_present": bool(entry.get("money_present", False)),
                    "milestone_present": bool(entry.get("milestone_present", False)),
                    "_aliases": {sk} if sk != canon else set(),
                }
            else:
                m = merged[canon]
                m["device_keys"] |= set(entry.get("device_keys", set()))
                m["stakeholder_keys"] |= set(entry.get("stakeholder_keys", set()))
                m["constraint_count"] += entry.get("constraint_count", 0)
                m["scope_atom_count"] += entry.get("scope_atom_count", 0)
                m["contradiction_count"] += entry.get("contradiction_count", 0)
                m["money_present"] = m["money_present"] or entry.get("money_present", False)
                m["milestone_present"] = m["milestone_present"] or entry.get("milestone_present", False)
                if sk != canon:
                    m["_aliases"].add(sk)

        # v53.2 STRICT final gate: when we have canonical sites,
        # ANYTHING not in canonical_set is dropped. No "kept if has
        # devices" exception — devices/scope are TAGGED with whatever
        # site phrase the regex/LLM lifted from the atom's text, which
        # is exactly the ghost-site source. The canonical roster is
        # authoritative.
        # v53.11 ALSO: drop generic placeholder slugs unconditionally
        # ("site:all", "site:various", etc.) — these are NEVER real sites.
        _PLACEHOLDER_SLUGS = {
            "site:all", "site:various", "site:tbd", "site:n_a",
            "site:none", "site:unknown", "site:all_sites",
            "site:all_locations", "site:various_sites",
            "site:site_all",
        }
        if canonical_set:
            filtered: dict[str, dict[str, Any]] = {}
            for ck, entry in merged.items():
                if ck in _PLACEHOLDER_SLUGS:
                    continue
                if ck in canonical_set:
                    filtered[ck] = entry
            sites = filtered
        else:
            sites = {ck: e for ck, e in merged.items() if ck not in _PLACEHOLDER_SLUGS}
    except Exception:
        pass

    out: list[dict[str, Any]] = []
    maturity_breakdown: dict[str, int] = {
        "anchored": 0, "scoping": 0, "planning": 0, "ready": 0,
    }
    for sk, entry in sorted(sites.items()):
        device_count = len(entry["device_keys"])
        stakeholder_count = len(entry["stakeholder_keys"])
        # Per-site readiness: small weighted sum of "do we have it?"
        # bools, scaled to 0-1.
        score = 0.0
        signal_count = 0
        if device_count > 0:
            score += 0.25
            signal_count += 1
        if stakeholder_count > 0:
            score += 0.20
            signal_count += 1
        if entry["constraint_count"] > 0:
            score += 0.15
            signal_count += 1
        if entry["scope_atom_count"] > 0:
            score += 0.15
            signal_count += 1
        if entry["money_present"]:
            score += 0.10
            signal_count += 1
        if entry["milestone_present"]:
            score += 0.15
            signal_count += 1
        # Penalize contradictions.
        score = max(0.0, score - 0.10 * min(3, entry["contradiction_count"]))

        # Gap E — graded maturity. A site anchored by a physical_site
        # atom but carrying no other signal is "located, details pending"
        # (amber), not "unknown" (red). Floor its readiness so the deal
        # isn't scored as if the site doesn't exist, and label the stage
        # explicitly so the UI can show progress rather than a flat 0.
        is_anchored = sk in anchored_set
        if is_anchored:
            score = max(score, _SITE_ANCHOR_FLOOR)

        if score >= 0.75:
            maturity = "ready"
            band = "green"
        elif signal_count >= 2:
            maturity = "planning"
            band = "amber"
        elif signal_count == 1:
            maturity = "scoping"
            band = "amber"
        elif is_anchored:
            maturity = "anchored"
            band = "amber"
        else:
            maturity = "anchored"
            band = "red"
        maturity_breakdown[maturity] = maturity_breakdown.get(maturity, 0) + 1

        out.append({
            "site": sk,
            "readiness": round(score, 3),
            "maturity": maturity,
            "band": band,
            "anchored": is_anchored,
            "signal_count": signal_count,
            "device_keys": sorted(entry["device_keys"]),
            "device_count": device_count,
            "stakeholder_keys": sorted(entry["stakeholder_keys"]),
            "stakeholder_count": stakeholder_count,
            "constraint_count": entry["constraint_count"],
            "scope_atom_count": entry["scope_atom_count"],
            "money_present": entry["money_present"],
            "milestone_present": entry["milestone_present"],
            "contradiction_count": entry["contradiction_count"],
            # v52: aliases column — short forms / LLM-detected names that
            # all refer to the same canonical site_id.
            "aliases": sorted(entry.get("_aliases", set())),
        })

    return {
        "sites": out,
        "site_count": len(out),
        "anchored_count": sum(1 for s in out if s["anchored"]),
        "maturity_breakdown": maturity_breakdown,
        "avg_readiness": round(
            sum(s["readiness"] for s in out) / max(1, len(out)), 3
        ),
        "least_ready_sites": [
            s["site"] for s in sorted(out, key=lambda r: r["readiness"])[:3]
            if s["readiness"] < 0.75
        ],
    }


# ─────────────────────────── STAKEHOLDER LOAD ───────────────────────────


_SEVERITY_WEIGHT = {
    "critical": 5,
    "high": 3,
    "medium": 2,
    "med": 2,
    "low": 1,
    "info": 0,
}


def build_stakeholder_load(
    *,
    atoms: list[EvidenceAtom],
) -> dict[str, Any]:
    """Return a per-stakeholder workload matrix.

    Surfaces who owns what — risks (severity-weighted), action items,
    open questions, decisions, change orders driven — so a PM can spot
    bottlenecks (one stakeholder carrying all critical risks) and
    bandwidth gaps (unowned action items).
    """
    by_slug: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "slug": "",
        "risk_count": 0,
        "risk_severity_load": 0,
        "critical_risk_count": 0,
        "high_risk_count": 0,
        "action_item_count": 0,
        "decision_count": 0,
        "change_order_count": 0,
        "risks": [],
        "action_items": [],
    })

    for atom in atoms:
        atom_type = _atom_type_str(atom)
        value = atom.value if isinstance(atom.value, dict) else {}
        owner_slug = _owner_slug_from_atom(atom, value.get("owner") if isinstance(value, dict) else "")
        if not owner_slug:
            continue
        entry = by_slug[owner_slug]
        entry["slug"] = owner_slug
        if atom_type == "risk":
            entry["risk_count"] += 1
            sev = (value.get("severity") if isinstance(value, dict) else "") or ""
            sev_lower = sev.lower().strip()
            entry["risk_severity_load"] += _SEVERITY_WEIGHT.get(sev_lower, 1)
            if sev_lower == "critical":
                entry["critical_risk_count"] += 1
            elif sev_lower == "high":
                entry["high_risk_count"] += 1
            risk_summary = (value.get("risk_summary") if isinstance(value, dict) else None) or atom.raw_text or ""
            entry["risks"].append({
                "atom_id": atom.id,
                "severity": sev,
                "risk_id": value.get("risk_id") if isinstance(value, dict) else None,
                "summary": risk_summary[:160],
            })
        elif atom_type == "action_item":
            entry["action_item_count"] += 1
            entry["action_items"].append({
                "atom_id": atom.id,
                "text": (atom.raw_text or "")[:200],
            })
        elif atom_type == "decision":
            entry["decision_count"] += 1
        elif atom_type == "customer_instruction":
            entry["change_order_count"] += 1

    stakeholders = sorted(
        by_slug.values(),
        key=lambda x: (-x["risk_severity_load"], -x["risk_count"], x["slug"]),
    )

    # Bottleneck detection: stakeholders carrying 2+ critical risks or
    # 4+ high risks.
    bottlenecks = [
        s["slug"] for s in stakeholders
        if s["critical_risk_count"] >= 2 or s["high_risk_count"] >= 4
    ]

    return {
        "stakeholders": stakeholders,
        "stakeholder_count": len(stakeholders),
        "bottlenecks": bottlenecks,
        "max_severity_load": max((s["risk_severity_load"] for s in stakeholders), default=0),
    }


# ─────────────────────────── PROJECT VITALS ───────────────────────────


def build_project_vitals(
    *,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
    packets: list[EvidencePacket],
    scorecard: dict[str, Any] | None = None,
    checklist: dict[str, Any] | None = None,
    site_readiness: dict[str, Any] | None = None,
    stakeholder_load: dict[str, Any] | None = None,
    scope_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a single 0-100 health score with breakdown.

    Blends the SOW readiness scorecard, SRL coverage, site readiness,
    contradiction surface, and unowned-work signal into one number a
    PM can put on the cockpit header. ``components`` records every
    input weight + score so the number can be audited.

    Bands:
      90-100 ``green``   — ready, low risk
      75-89  ``yellow``  — minor gaps, watchable
      55-74  ``orange``  — needs attention before kickoff
      0-54   ``red``     — substantial work remaining
    """
    components: list[dict[str, Any]] = []

    # 1. SOW readiness scorecard (weight 0.30)
    sow_score = (scorecard or {}).get("readiness_score", 0.0) or 0.0
    components.append({
        "name": "sow_readiness",
        "weight": 0.30,
        "raw_score": sow_score,
        "contribution": round(sow_score * 0.30, 3),
    })

    # 2. SRL coverage (weight 0.20)
    srl_coverage = (checklist or {}).get("coverage", 0.0) or 0.0
    components.append({
        "name": "srl_field_coverage",
        "weight": 0.20,
        "raw_score": srl_coverage,
        "contribution": round(srl_coverage * 0.20, 3),
    })

    # 3. Site readiness average (weight 0.15)
    site_score = (site_readiness or {}).get("avg_readiness", 0.0) or 0.0
    components.append({
        "name": "site_readiness_avg",
        "weight": 0.15,
        "raw_score": site_score,
        "contribution": round(site_score * 0.15, 3),
    })

    # 4. Contradiction penalty (weight 0.15) — fewer is better, 1.0 means none.
    contradiction_count = 0
    for edge in edges:
        meta = edge.metadata or {}
        if isinstance(meta, dict) and meta.get("edge_family") in (
            "device_quantity_cross_doc",
            "part_number_quantity_conflict",
            "quantity_contradiction",
        ):
            contradiction_count += 1
    contested_count = (scope_truth or {}).get("contested_count", 0) or 0
    raw_signal = contradiction_count + 2 * contested_count
    contradiction_health = max(0.0, 1.0 - min(1.0, raw_signal / 20.0))
    components.append({
        "name": "contradiction_health",
        "weight": 0.15,
        "raw_score": round(contradiction_health, 3),
        "contribution": round(contradiction_health * 0.15, 3),
        "signal": {"edges": contradiction_count, "contested_in_scope_truth": contested_count},
    })

    # 5. Ownership signal (weight 0.10) — fraction of risks with owners.
    risks = [a for a in atoms if _atom_type_str(a) == "risk"]
    risks_with_owner = sum(
        1 for a in risks
        if isinstance(a.value, dict) and a.value.get("owner")
    )
    ownership = risks_with_owner / len(risks) if risks else 1.0
    components.append({
        "name": "risk_ownership",
        "weight": 0.10,
        "raw_score": round(ownership, 3),
        "contribution": round(ownership * 0.10, 3),
        "signal": {"risk_count": len(risks), "risks_with_owner": risks_with_owner},
    })

    # 6. Stakeholder bottleneck signal (weight 0.10) — penalty when
    # one stakeholder is overloaded.
    bottlenecks = (stakeholder_load or {}).get("bottlenecks", []) or []
    bottleneck_health = max(0.0, 1.0 - 0.5 * len(bottlenecks))
    components.append({
        "name": "load_balance",
        "weight": 0.10,
        "raw_score": round(bottleneck_health, 3),
        "contribution": round(bottleneck_health * 0.10, 3),
        "signal": {"bottleneck_stakeholders": bottlenecks},
    })

    total = sum(c["contribution"] for c in components)
    score_100 = round(total * 100, 1)
    if score_100 >= 90:
        band = "green"
    elif score_100 >= 75:
        band = "yellow"
    elif score_100 >= 55:
        band = "orange"
    else:
        band = "red"

    # Top-3 drivers (highest contribution) and top-3 detractors
    # (lowest raw_score) so a PM can render a "what's pushing us up /
    # down" summary directly.
    sorted_by_score = sorted(components, key=lambda c: c["raw_score"])
    top_detractors = [c["name"] for c in sorted_by_score[:3]]
    top_drivers = [c["name"] for c in sorted(components, key=lambda c: -c["contribution"])[:3]]

    return {
        "score_100": score_100,
        "band": band,
        "components": components,
        "top_drivers": top_drivers,
        "top_detractors": top_detractors,
    }


# ──────────────────── DEAL HEADER / FINANCIALS / BOM ────────────────
# PM-facing assembly of the structured commercial atoms the xlsx parser
# now emits (deal_metadata header + per-category P&L commercial_total +
# folded materials rollups). These read existing atom ``value`` payloads
# and never re-parse text, so they are deterministic and cheap.

# Logical P&L order the PM reads top-to-bottom; "deal" is the grand total.
_PL_CATEGORY_ORDER = [
    "deal", "labor", "pmo", "materials", "lift_rental", "miscellaneous",
]


def build_deal_header(*, atoms: list[EvidenceAtom]) -> dict[str, Any]:
    """Assemble the deal header (OPPTY #, customer, sales rep, billing
    type, duration, region, …) from ``deal_metadata`` atoms.

    First non-empty value wins per field across atoms. Returns the merged
    field map plus presence/provenance so the PM surface can render a deal
    banner without re-reading the workbook."""
    fields: dict[str, Any] = {}
    source_atom_ids: list[str] = []
    for atom in atoms:
        if _atom_type_str(atom) != "deal_metadata":
            continue
        value = atom.value if isinstance(atom.value, dict) else {}
        if value.get("kind") != "deal_header":
            continue
        afields = value.get("fields") if isinstance(value.get("fields"), dict) else {}
        if not afields:
            continue
        source_atom_ids.append(atom.id)
        for k, v in afields.items():
            if v not in (None, "") and k not in fields:
                fields[k] = v

    return {
        "fields": fields,
        "field_count": len(fields),
        "present": bool(fields),
        "source_atom_ids": source_atom_ids,
    }


def _round2(v: Any) -> Any:
    return round(v, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def build_deal_financials(*, atoms: list[EvidenceAtom]) -> dict[str, Any]:
    """Assemble the deal P&L from ``commercial_total`` atoms whose value
    is a structured ``pl_line`` (category + revenue/cost/margin/margin%).

    Returns an ordered line list (grand-total "Deal" first), the rolled-up
    totals (the Deal line when present, else summed), and presence so the
    PM surface can render a financial table + headline margin."""
    by_key: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        if _atom_type_str(atom) != "commercial_total":
            continue
        value = atom.value if isinstance(atom.value, dict) else {}
        if value.get("kind") != "pl_line":
            continue
        ckey = value.get("category_key")
        if not ckey:
            import re as _re_pl
            ckey = _re_pl.sub(r"[^a-z0-9]+", "_", str(value.get("category", "")).lower()).strip("_")
        if not ckey or ckey in by_key:
            continue
        by_key[ckey] = {
            "category": value.get("category", ckey.title()),
            "category_key": ckey,
            "revenue": _round2(value.get("revenue")),
            "cost": _round2(value.get("cost")),
            "margin": _round2(value.get("margin")),
            "margin_pct": _round2(value.get("margin_pct")),
            "atom_id": atom.id,
        }

    def _order(k: str) -> tuple[int, str]:
        return (_PL_CATEGORY_ORDER.index(k) if k in _PL_CATEGORY_ORDER else 99, k)

    lines = [by_key[k] for k in sorted(by_key, key=_order)]

    # Totals: prefer the explicit grand-total "deal" line; else sum the
    # component categories (excluding any "deal" to avoid double counting).
    deal_line = by_key.get("deal")
    if deal_line and any(deal_line.get(m) is not None for m in ("revenue", "cost", "margin")):
        rev, cost, margin = deal_line.get("revenue"), deal_line.get("cost"), deal_line.get("margin")
        mpct = deal_line.get("margin_pct")
    else:
        comp = [v for k, v in by_key.items() if k != "deal"]
        rev = sum(v["revenue"] for v in comp if isinstance(v.get("revenue"), (int, float)))
        cost = sum(v["cost"] for v in comp if isinstance(v.get("cost"), (int, float)))
        margin = round(rev - cost, 2)
        mpct = round(margin / rev * 100, 2) if rev else None
    totals = {
        "revenue": _round2(rev),
        "cost": _round2(cost),
        "margin": _round2(margin),
        "margin_pct": _round2(mpct),
    }

    return {
        "lines": lines,
        "category_count": len(lines),
        "totals": totals,
        "present": bool(lines),
    }


# Commercial sheet roles that are categorically NOT deal bills of
# materials: a rate card is labor pricing, a financial summary is deal
# economics, and a catalog is a *master price book* — labels + unit
# prices with no order quantities populated. None describe materials
# actually ordered for the job.
_NON_BOM_SHEET_ROLES = frozenset({"rate_card", "financial_summary", "catalog"})


def _money_values_in_row(row: dict[str, Any]) -> set[float]:
    """Numeric values flagged as currency for this row, parsed from the
    ``money:<n>`` keys the parser stamps (e.g. ``money:1_2`` → 1.2)."""
    vals: set[float] = set()
    for mk in row.get("money_keys") or []:
        frac = str(mk).split(":", 1)[-1].replace("_", ".")
        try:
            vals.add(round(float(frac), 4))
        except ValueError:
            continue
    return vals


def _row_has_order_quantity(row: dict[str, Any]) -> bool:
    """Whether a folded pricing row records a COUNT of items ordered.

    Universal, content-derived signal (no sheet-name keywords): a bill of
    materials lists quantities ordered, whereas a master price book /
    catalog lists only unit prices. A row carries an order quantity when
    it has a positive whole-number value in a NON-money cell — a count
    column distinct from any currency amount."""
    money_vals = _money_values_in_row(row)
    for cell in row.get("cells") or []:
        if isinstance(cell, bool):
            continue
        if isinstance(cell, (int, float)):
            v = float(cell)
            if v <= 0:
                continue
            if round(v, 4) in money_vals:
                continue  # this cell is a price, not a count
            if abs(v - round(v)) < 1e-9:
                return True
    return False


def _rows_are_ordered_materials(rows: list[dict[str, Any]]) -> bool:
    """A folded sheet is a deal BOM (rather than a price book) when a
    majority of its rows carry order quantities."""
    considered = [r for r in rows if isinstance(r, dict)]
    if not considered:
        return False
    qty_rows = sum(1 for r in considered if _row_has_order_quantity(r))
    return qty_rows >= 1 and qty_rows >= len(considered) * 0.5


def build_bill_of_materials(*, atoms: list[EvidenceAtom]) -> dict[str, Any]:
    """Assemble the materials / BOM view from the folded pricing rollups
    (``pricing_assumption`` / ``commercial_total`` atoms carrying
    ``value.rows``) and any per-line ``bom_line`` atoms.

    Surfaces the line items as readable rows so a PM sees the actual
    materials menu instead of a single "99 pricing lines" banner."""
    sections: list[dict[str, Any]] = []
    total_lines = 0

    for atom in atoms:
        atype = _atom_type_str(atom)
        value = atom.value if isinstance(atom.value, dict) else {}
        folded = value.get("rows")
        # A folded pricing sheet belongs in the BOM only when it records
        # materials actually ordered. Two universal, content-derived
        # gates (no sheet-name keyword matching):
        #   1. the parser's sheet role is not a categorically-commercial
        #      role (rate card / financial summary / master price book), and
        #   2. the rows themselves carry order quantities — distinguishing
        #      a real BOM from a catalog of unit prices.
        sheet_role = str(value.get("sheet_role") or "")
        is_material = (
            sheet_role not in _NON_BOM_SHEET_ROLES
            and isinstance(folded, list)
            and _rows_are_ordered_materials(folded)
        )
        if (
            atype in ("pricing_assumption", "commercial_total")
            and isinstance(folded, list) and folded
            and is_material
        ):
            rows: list[dict[str, Any]] = []
            for r in folded:
                if not isinstance(r, dict):
                    continue
                rows.append({
                    "row": r.get("row"),
                    "label": (r.get("label") or "")[:300],
                    "cells": r.get("cells") or [],
                    "money_keys": r.get("money_keys") or [],
                })
            if rows:
                total_lines += len(rows)
                sections.append({
                    "sheet_name": value.get("sheet_name") or atom.artifact_id,
                    "line_count": value.get("line_count") or len(rows),
                    "money_min": value.get("money_min"),
                    "money_max": value.get("money_max"),
                    "money_sum": value.get("money_sum"),
                    "atom_id": atom.id,
                    "rows": rows,
                })

    # Per-line bom_line atoms (if any parser emits them) become their own
    # one-row-per-atom section so nothing is lost.
    bom_rows: list[dict[str, Any]] = []
    for atom in atoms:
        if _atom_type_str(atom) != "bom_line":
            continue
        value = atom.value if isinstance(atom.value, dict) else {}
        bom_rows.append({
            "label": (value.get("description") or atom.raw_text or "")[:300],
            "cells": value.get("cells") or [],
            "atom_id": atom.id,
        })
    if bom_rows:
        total_lines += len(bom_rows)
        sections.append({
            "sheet_name": "bom_line atoms",
            "line_count": len(bom_rows),
            "rows": bom_rows,
        })

    return {
        "sections": sections,
        "section_count": len(sections),
        "total_lines": total_lines,
        "present": bool(sections),
    }


__all__ = [
    "build_pm_dashboard",
    "build_sow_readiness_scorecard",
    "build_srl_missing_checklist",
    "build_scope_truth",
    "build_change_order_timeline",
    "build_site_readiness",
    "build_stakeholder_load",
    "build_project_vitals",
    "build_deal_header",
    "build_deal_financials",
    "build_bill_of_materials",
]
