"""Semantic key-based atom deduplication.

v52: Single biggest overcount source — three independent paths each emit
the same fact:
  1. Table schema registry (from xlsx/docx tables)
  2. Prose-list splitter (from multi-fact paragraphs)
  3. multi_entity_llm → entity bridge (from LLM extraction)

When all three fire on doc 01's milestone paragraph, OPTBOT's Phase 0
becomes THREE atoms: schema-row, prose-split, LLM-bridge. Same for
requirements, BOM rows, stakeholders, etc.

The v48 collapse_duplicate_atoms() catches IDENTICAL or 92%-similar
raw_text but the three paths produce DIFFERENT text shapes:
  schema: "Phase | Name | Start | End"
  prose:  "Phase 0 Discovery and intake | 2026-05-20 to 2026-05-29 | ..."
  bridge: "Discovery and intake"

Text-similarity dedup misses these. This module dedupes by the
SEMANTIC KEY each atom carries in its ``value`` dict:

  Type                       Key fields tried (first non-empty wins)
  -------------------------- ----------------------------------------
  milestone_phase            phase_id, name, start
  task                       task_id, name
  requirement                req_id, description
  bom_line                   item_id, sku, description
  service_line               service_id, description
  site_allocation            (bom_item, site)
  signatory                  name, role
  stakeholder                email, name
  payment_term               tranche, percent + trigger
  electrical_acceptance_test test, threshold
  cutover_step               step_id, description
  integration_checkpoint     ic_id, system, test
  compliance_classification  classification
  approval_authority         (approver, domain)
  approval_decision          (approver, decision)
  deal_metadata              field_name
  commercial_total           category
  lead_time_constraint       sku, item_id
  blackout_date_range        (start, end)
  dependency                 (dependent, depends_on)
  mitigation                 risk_id, mitigation_text
  physical_site              id, site_id

When the key matches, atoms collapse into the highest-confidence one.
The dropped atoms' fields are merged in (union of provenance, longest
non-empty values per field). No data is lost — duplicates are collapsed
into the richest representation.

Pure function, no LLM, no I/O.
"""
from __future__ import annotations

import re
from typing import Any


def _norm_key(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _value_key(atom: Any) -> tuple | None:
    """Return a hashable key describing the atom's identity.

    The key includes atom_type so the same fact_id under two types
    doesn't collide (a task_id and a req_id can both be "001").
    Returns None when no key field is populated — those atoms aren't
    eligible for semantic dedup (they go through the v48 text-based
    pass only).
    """
    atom_type = getattr(atom, "atom_type", None)
    atype = atom_type.value if hasattr(atom_type, "value") else str(atom_type or "")
    val = getattr(atom, "value", None) or {}
    if not isinstance(val, dict):
        return None

    def _first(*fields: str) -> str:
        for f in fields:
            v = val.get(f)
            if v and isinstance(v, (str, int, float)):
                k = _norm_key(v)
                if k and len(k) >= 2:
                    return k
        return ""

    def _first_trunc(*fields: str, n: int = 40) -> str:
        """Like _first, but truncate string values to N chars BEFORE
        normalizing. Used for description-style fallback keys so two
        LLM-paraphrased descriptions of the same fact collapse instead
        of each surviving as a distinct key.

        Example:
          "Network outage during business hours could ..."   first 40 → "network_outage_during_business_hours_coul"
          "Network outage during business hours might ..."   first 40 → "network_outage_during_business_hours_migh"
        Still collapse-friendly at 32-char normalized prefix.
        """
        for f in fields:
            v = val.get(f)
            if v and isinstance(v, (str, int, float)):
                s = str(v)[:n]
                k = _norm_key(s)
                # Use a shorter prefix of the normalized key for fuzzier
                # collapse (LLM paraphrases of same fact diverge after
                # ~30 chars of normalized content).
                if k and len(k) >= 8:
                    return k[:32]
        return ""

    if atype == "milestone_phase":
        key = _first("phase_id", "name", "start")
        return (atype, key) if key else None
    if atype == "task":
        key = _first("task_id", "name")
        return (atype, key) if key else None
    if atype == "requirement":
        # ID first (stable), fall back to TRUNCATED description so LLM
        # paraphrases of the same requirement collapse.
        key = _first("req_id")
        if not key:
            key = _first_trunc("description", "text", "requirement", "criterion")
        return (atype, key) if key else None
    if atype == "bom_line":
        key = _first("item_id", "sku")
        if not key:
            key = _first_trunc("description")
        return (atype, key) if key else None
    if atype == "service_line":
        key = _first("service_id")
        if not key:
            key = _first_trunc("description")
        return (atype, key) if key else None
    if atype == "site_allocation":
        item = _first("bom_item", "item_id", "description", "sku")
        site = _first("site")
        if item and site:
            return (atype, item, site)
        return None
    if atype == "signatory":
        key = _first("name", "role")
        return (atype, key) if key else None
    if atype == "stakeholder":
        # Email is the natural unique key for a person; fall back to name.
        email = _first("email")
        if email:
            return (atype, email)
        key = _first("name")
        return (atype, key) if key else None
    if atype == "payment_term":
        tranche = _first("tranche", "trigger")
        pct = val.get("percent")
        if tranche and pct is not None:
            return (atype, tranche, str(pct))
        if tranche:
            return (atype, tranche)
        return None
    if atype == "electrical_acceptance_test":
        key = _first("test")
        return (atype, key) if key else None
    if atype == "cutover_step":
        key = _first("step_id")
        if not key:
            key = _first_trunc("description", "action", "step")
        return (atype, key) if key else None
    if atype == "integration_checkpoint":
        ic_id = _first("ic_id")
        if ic_id:
            return (atype, ic_id)
        sys_test = (_first("system"), _first("test_description", "test"))
        if any(sys_test):
            return (atype, sys_test)
        return None
    if atype == "compliance_classification":
        key = _first("classification")
        return (atype, key) if key else None
    if atype == "compliance_rule":
        key = _first("rule_kind")
        if not key:
            key = _first_trunc("condition", "rule", "description", "statement")
        return (atype, key) if key else None
    if atype == "approval_authority":
        appr = _first("approver")
        dom = _first("domain", "scope")
        if appr and dom:
            return (atype, appr, dom)
        if dom:
            return (atype, dom)
        return None
    if atype == "approval_decision":
        appr = _first("approver")
        dec = _first("decision")
        if appr:
            return (atype, appr, dec)
        return None
    if atype == "deal_metadata":
        key = _first("field_name", "value")
        return (atype, key) if key else None
    if atype == "commercial_total":
        cat = _first("category")
        amt = val.get("amount")
        if cat:
            return (atype, cat)
        return None
    if atype == "lead_time_constraint":
        key = _first("sku", "item_id", "description")
        return (atype, key) if key else None
    if atype == "blackout_date_range":
        start = _first("start")
        end = _first("end")
        if start or end:
            return (atype, start, end)
        return None
    if atype == "dependency":
        dep = _first("dependent")
        on = _first("depends_on")
        if dep and on:
            return (atype, dep, on)
        return None
    if atype == "mitigation":
        rid = _first("risk_id")
        if rid:
            return (atype, rid)
        return None
    if atype == "physical_site":
        key = _first("id", "site_id")
        return (atype, key) if key else None
    if atype == "deliverable":
        key = _first("name")
        return (atype, key) if key else None
    if atype == "site_room_mix":
        site = _first("site")
        rt = _first("room_type")
        if site and rt:
            return (atype, site, rt)
        return None
    if atype == "site_attribute":
        site = _first("site")
        kind = _first("attribute_kind", "kind")
        if site and kind:
            return (atype, site, kind)
        return None
    if atype == "site_access_window":
        site = _first("site")
        if site:
            return (atype, site)
        return None
    if atype == "site_access_restriction":
        site = _first("site")
        rk = _first("restriction_kind")
        if site and rk:
            return (atype, site, rk)
        if site:
            return (atype, site)
        return None
    if atype == "site_infrastructure":
        site = _first("site")
        kind = _first("kind")
        if site and kind:
            return (atype, site, kind)
        return None
    if atype == "site_implementation_note":
        site = _first("site")
        kind = _first("kind")
        detail_short = _first("detail")[:30]
        if site and (kind or detail_short):
            return (atype, site, kind or detail_short)
        return None
    if atype == "assumption":
        key = _first("assumption")[:60]
        return (atype, key) if key else None
    if atype == "data_flow_step":
        step = _first("step_number", "sequence")
        src = _first("source_system")
        if step:
            return (atype, step)
        if src:
            return (atype, src)
        return None
    if atype == "system_mapping":
        src = _first("source")
        tgt = _first("target")
        if src or tgt:
            return (atype, src, tgt)
        return None
    if atype == "metadata_requirement":
        sys = _first("system")
        key = _first("key", "metadata_key")
        if sys and key:
            return (atype, sys, key)
        return None
    if atype == "risk":
        # Stable risk_id first. When LLM didn't emit one, collapse on
        # truncated description so paraphrased duplicates (60+ from
        # phased LLM extraction) fold to ~5-10 unique risks.
        key = _first("risk_id", "id")
        if not key:
            key = _first_trunc(
                "description", "risk", "risk_summary",
                "text", "summary", "title",
            )
        return (atype, key) if key else None
    if atype == "acceptance_criterion":
        # No natural ID — collapse on truncated criterion text.
        key = _first("criterion_id", "ac_id")
        if not key:
            key = _first_trunc(
                "criterion", "test", "acceptance", "criteria",
                "check", "item", "description", "statement",
            )
        return (atype, key) if key else None
    if atype == "change_order_rule":
        key = _first("trigger_kind", "rule_id")
        if not key:
            key = _first_trunc("rate_or_threshold", "condition", "description")
        return (atype, key) if key else None
    if atype == "pricing_assumption":
        key = _first("domain")
        if not key:
            key = _first_trunc("statement", "assumption", "description")
        return (atype, key) if key else None

    return None


def _confidence(atom: Any) -> float:
    try:
        return float(getattr(atom, "confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def _merge_values(winner: Any, loser: Any) -> None:
    """Best-effort merge: take longest non-empty value per field from
    loser into winner. Doesn't override populated winner fields.
    """
    wv = getattr(winner, "value", None)
    lv = getattr(loser, "value", None)
    if not isinstance(wv, dict) or not isinstance(lv, dict):
        return
    for k, lval in lv.items():
        if not lval:
            continue
        wval = wv.get(k)
        if wval is None or wval == "":
            wv[k] = lval
        elif isinstance(wval, str) and isinstance(lval, str) and len(lval) > len(wval):
            wv[k] = lval
        elif isinstance(wval, (list, tuple)) and isinstance(lval, (list, tuple)):
            # union lists
            merged = list(wval)
            for x in lval:
                if x not in merged:
                    merged.append(x)
            wv[k] = merged


def semantic_dedup_atoms(atoms: list[Any]) -> list[Any]:
    """Collapse atoms that share a semantic key into one (highest-
    confidence wins; loser values merged into winner).

    Returns a new list. Atoms without a semantic key (no value, or
    no recognized atom_type) pass through unchanged.
    """
    if not atoms:
        return atoms

    # Group by key; track the winner per key
    by_key: dict[tuple, Any] = {}
    unkeyed: list[Any] = []

    # Sort by confidence desc so first-encountered per key is the winner.
    sorted_atoms = sorted(atoms, key=_confidence, reverse=True)

    for atom in sorted_atoms:
        key = _value_key(atom)
        if key is None:
            unkeyed.append(atom)
            continue
        if key not in by_key:
            by_key[key] = atom
        else:
            _merge_values(by_key[key], atom)

    return list(by_key.values()) + unkeyed


__all__ = ["semantic_dedup_atoms"]
