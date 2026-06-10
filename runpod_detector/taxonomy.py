"""Canonical atom-type taxonomy — ONE source of truth for facets + micro-types.

The boss audit found 3 conflicting collapse maps (label_stats vs sample_rows vs
RUBRIC.md/trainer), plus teacher-emitted types none of them knew (addendum_qa,
mitigation, metadata_requirement, signatory). Result: the facet of an atom
depended on which file computed it, so imbalance/coverage/per-facet metrics were
not comparable. This module is the single import for the prod prompt, the stats
tooling, the relabeler, the gold-eval builder, and every trainer.

Design (per the architecture review):
  Level 0  GATE   : _keep vs typed
  Level 1  FACET  : the 7 dashboard sections (the TRAINED target — learnable,
                    ~0.85 two-model reproducible)
  Level 2  MICRO  : the ~40 micro-types (PRESERVED in data/gold for future fine
                    heads where a consumer needs the split; NOT a trained target
                    at current data volume)

No silent drops: a micro-type not in MICRO_TO_FACET maps to REVIEW (surfaced,
never dropped) so new teacher vocabulary is caught instead of vanishing.
"""
from __future__ import annotations

TAXONOMY_VERSION = "v1-2026-06-10"

# The 7 facets = the PM dashboard sections. KEEP is the gate's negative class.
FACETS = ("SITE", "COMMERCIAL", "WORK", "COMPLIANCE", "PARTY", "TIMING", "META")
KEEP = "_keep"
REVIEW = "REVIEW"  # unmapped micro-type -> surfaced for taxonomy owner, never dropped

# micro-type -> facet. Sources tagged: [R]=RUBRIC.md collapse map, [F]=fuzzy
# ruling in RUBRIC.md, [D]=seen in training data, [N]=new/straggler teacher type.
MICRO_TO_FACET: dict[str, str] = {
    # --- SITE ---
    "physical_site": "SITE",            # [R][D]
    "site_attribute": "SITE",           # [R][D]
    "site_access_restriction": "SITE",  # [R]
    "site_room_mix": "SITE",            # [R]
    "site_infrastructure": "SITE",      # [R]
    # --- COMMERCIAL ---
    "service_line": "COMMERCIAL",       # [R][D]
    "bom_line": "COMMERCIAL",           # [R][D]
    "payment_term": "COMMERCIAL",       # [R][D]
    "commercial_total": "COMMERCIAL",   # [R][D]
    "pricing_assumption": "COMMERCIAL", # [R][D]
    "site_budget": "COMMERCIAL",        # [R]
    # --- WORK ---
    "requirement": "WORK",              # [R][D]
    "task": "WORK",                     # [R][D]
    "deliverable": "WORK",              # [R][D]
    "acceptance_criterion": "WORK",     # [R][D]
    "milestone_phase": "WORK",          # [R][D]
    "cutover_step": "WORK",             # [R]
    "electrical_acceptance_test": "WORK",  # [R]
    "site_implementation_note": "WORK", # [R][D]
    "site_access_window": "WORK",       # [R] (rubric routes the *work* here; timing in value)
    "exclusion": "WORK",                # [F] negative scope still defines scope
    "integration_checkpoint": "WORK",   # [F] technical validation step
    "data_flow_step": "WORK",           # [D] technical step
    "system_mapping": "WORK",           # [D] technical mapping
    "mitigation": "WORK",               # [N] a mitigation is a remediating action
    "constraint": "WORK",               # [N] scope constraint
    # --- COMPLIANCE ---
    "compliance_rule": "COMPLIANCE",        # [R][D]
    "compliance_classification": "COMPLIANCE",  # [R][D]
    "approval_authority": "COMPLIANCE",     # [R]
    "submission_req": "COMPLIANCE",         # [R]
    "change_order_rule": "COMPLIANCE",      # [F][D] governs the contract process
    "bonding_insurance": "COMPLIANCE",      # [F][D] legal/regulatory obligation
    "contract_term": "COMPLIANCE",          # [D] contractual obligation
    # --- PARTY ---
    "stakeholder": "PARTY",             # [F][D] acts-in-deal; bare contact -> META (relabel rules)
    # --- TIMING ---
    "blackout_date_range": "TIMING",    # [R]
    "lead_time_constraint": "TIMING",   # [R][D]
    "deadline": "TIMING",               # [F][D] the facet is about *when*
    "dependency": "TIMING",             # [F] sequencing/precondition gating *when*
    # --- META ---
    "deal_metadata": "META",            # [R][D]
    "eval_criterion": "META",           # [R]
    "approval_decision": "META",        # [R]
    "metadata_requirement": "META",     # [N] schema/field metadata
    "signatory": "META",                # [N] sales signatory (not an actor) -> meta
    "addendum_qa": "META",              # [N] deal Q&A addendum metadata
    # --- KEEP passes through unchanged ---
    KEEP: KEEP,
}

# Micro-types whose label IS the deliverable (no rich value payload) — eligible
# for direct head assignment that skips the LLM. Kept here so the runtime and the
# trainer agree on one list (it was hardcoded in typed_atom_classifier before).
VALUE_LIGHT = frozenset({
    "requirement", "exclusion", "contract_term", "deal_metadata",
    "acceptance_criterion", "task", "change_order_rule", "constraint",
    "dependency", "mitigation", "compliance_rule", "submission_req", "addendum_qa",
})


def to_facet(micro: str | None) -> str:
    """Map a micro-type to its facet. KEEP passes through; unknown -> REVIEW
    (surfaced, never silently dropped)."""
    if not micro:
        return REVIEW
    return MICRO_TO_FACET.get(micro.strip(), REVIEW)


def all_micro_types() -> tuple[str, ...]:
    return tuple(k for k in MICRO_TO_FACET if k != KEEP)


def unknown_types(seen: list[str]) -> list[str]:
    """Return micro-types present in `seen` that the map doesn't cover (i.e. would
    route to REVIEW) — for catching new teacher vocabulary."""
    return sorted({s for s in seen if s and s.strip() not in MICRO_TO_FACET})


if __name__ == "__main__":
    print(f"taxonomy {TAXONOMY_VERSION}: {len(FACETS)} facets, "
          f"{len(all_micro_types())} micro-types")
    # audit against the local DBs: any teacher type we don't map?
    import sqlite3
    import sys
    for db in sys.argv[1:] or ["_training_coarse.db", "_training_deepseek.db", "_training_cloud.db"]:
        try:
            con = sqlite3.connect(db)
        except Exception:  # noqa: BLE001
            continue
        seen = [r[0] for r in con.execute(
            "SELECT DISTINCT label FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL")]
        unk = unknown_types(seen)
        print(f"  {db}: {len(seen)} distinct labels, UNMAPPED={unk or 'none'}")
