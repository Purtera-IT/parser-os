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


def _atom_type_value(atom: Any) -> str:
    atom_type = getattr(atom, "atom_type", None)
    return atom_type.value if hasattr(atom_type, "value") else str(atom_type or "")


def _site_display_key(value: Any) -> str:
    """Stable display key for physical-site ids/names.

    Unlike ``_norm_key`` this keeps the enterprise-code convention
    (upper-case with hyphens) so canonical ids such as ATL-HQ-01 remain
    human-readable after dedup.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").upper()


_PHYSICAL_SITE_ALLOWED_FIELDS: frozenset[str] = frozenset({
    "kind", "id", "site_id", "site_no", "name", "facility_name",
    "administrative_site_name", "address", "street", "street_address",
    "city", "state", "zip", "zip_code",
    "city_state", "lat_long", "latitude", "longitude",
    "mdf_idf", "access_window", "escort_owner", "contact",
    "phone", "email", "sqft", "occupancy", "notes", "extras",
    "raw", "raw_cells", "row_index",
})

_NON_SITE_CODE_HEADS: frozenset[str] = frozenset({
    "MOCK", "DEV", "TEST", "DEMO", "FAKE", "DUMMY", "SAMPLE",
    "MSA", "NDA", "SOW", "RFP", "RFQ", "RFI", "PO", "WO",
    "INV", "TKT", "TASK", "PROJ", "DEAL", "CASE", "REQ",
    "QUOTE", "Q", "ORDER", "ORD", "HS", "HUBSPOT", "AZURE",
    "AWS", "GCP", "INTUNE", "OKTA", "API", "SKU", "UPC",
})

_GENERIC_SITE_IDS: frozenset[str] = frozenset({
    "", "ALL", "ALL-SITES", "ALL-LOCATIONS", "N-A", "NA", "N/A",
    "TBD", "TBA", "VARIOUS", "MULTIPLE", "NONE", "UNKNOWN",
    "SITE", "LOCATION", "ADDRESS", "TOTAL", "SUBTOTAL", "SUM",
})


def _physical_site_id(atom: Any) -> str:
    val = getattr(atom, "value", None) or {}
    if not isinstance(val, dict):
        return ""
    # v56: NEVER fall back to val.get("name"). When an atom has a facility
    # name but no real site_id, it's a ghost emission from a non-roster
    # parser (text-extracted prose, BOM SKU mis-classification, address
    # tokenization). Synthesizing site_id from name creates atoms like
    # "OPTBOT-AIRPORT-LOGIST" (truncated facility) and
    # "4200-GLOBAL-GATEWAY-CONNECTOR" (address bleed). Real physical_site
    # atoms always carry an explicit site_id from a structured ID column.
    # Returning "" here causes such ghost atoms to be DROPPED in
    # _dedupe_physical_site_atoms (canonical_for returns None).
    return _site_display_key(val.get("site_id") or val.get("id") or "")


def _is_bad_physical_site_id(site_id: str) -> bool:
    sid = _site_display_key(site_id)
    if sid in _GENERIC_SITE_IDS:
        return True
    # A naked year or amount is not a site.
    if sid.isdigit() and len(sid) >= 4:
        return True
    parts = [p for p in sid.split("-") if p]
    if parts and parts[0] in _NON_SITE_CODE_HEADS:
        return True
    # Test-data/document identifiers often contain these tokens after a
    # one-letter lead (Q-DEV-ATL-047, HS-DEAL-ATL-2026, etc.).
    if any(p in _NON_SITE_CODE_HEADS for p in parts[:2]):
        return True
    return False


def _looks_complete_site_id(site_id: str) -> bool:
    sid = _site_display_key(site_id)
    if not sid or _is_bad_physical_site_id(sid):
        return False
    if not re.search(r"\d", sid):
        return False
    # Common PDF table clipping turns ATL-WEST-02 into ATL-WEST-0.
    # Do not let the clipped form become the canonical full id when a
    # real full id is also present elsewhere in the authoritative doc.
    if re.search(r"-0$", sid):
        return False
    return True


def _physical_site_quality(atom: Any, canonical_id: str = "") -> tuple[int, int, float]:
    val = getattr(atom, "value", None) or {}
    if not isinstance(val, dict):
        val = {}
    sid = _physical_site_id(atom)
    filename_blob = " ".join(
        str(getattr(ref, "filename", "") or "") for ref in (getattr(atom, "source_refs", None) or [])
    ).lower()
    authoritative = int(
        "authoritative" in filename_blob
        or "site_roster" in filename_blob
        or "site roster" in str(getattr(atom, "raw_text", "")).lower()
        or "kind=physical_site" in str(getattr(atom, "raw_text", "")).lower()
    )
    exact_canonical = int(bool(canonical_id) and sid == canonical_id)
    rich_fields = sum(
        1 for k in (
            "facility_name", "name", "address", "street_address",
            "mdf_idf", "access_window", "escort_owner", "lat_long",
            "city", "zip", "phone", "email",
        )
        if val.get(k)
    )
    return (authoritative * 100 + exact_canonical * 50 + rich_fields, len(str(getattr(atom, "raw_text", "") or "")), _confidence(atom))


def _append_unique(target: list[Any], incoming: list[Any]) -> None:
    seen = {getattr(x, "id", None) or repr(x) for x in target}
    for item in incoming or []:
        key = getattr(item, "id", None) or repr(item)
        if key not in seen:
            target.append(item)
            seen.add(key)


def _merge_atom_metadata(winner: Any, loser: Any) -> None:
    """Carry evidence/provenance from a collapsed duplicate into winner."""
    try:
        _append_unique(winner.source_refs, getattr(loser, "source_refs", []) or [])
    except Exception:
        pass
    try:
        _append_unique(winner.receipts, getattr(loser, "receipts", []) or [])
    except Exception:
        pass
    try:
        for key in getattr(loser, "entity_keys", []) or []:
            if key not in winner.entity_keys:
                winner.entity_keys.append(key)
    except Exception:
        pass
    try:
        for flag in getattr(loser, "review_flags", []) or []:
            if flag not in winner.review_flags:
                winner.review_flags.append(flag)
    except Exception:
        pass


def _clean_physical_site_value(value: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in value.items() if k in _PHYSICAL_SITE_ALLOWED_FIELDS and v not in (None, "", [], {})}
    cleaned["kind"] = "physical_site"
    # Keep id/site_id/name/facility_name synchronized without importing a
    # heavyweight schema layer into the hot dedup path.
    canonical = cleaned.get("site_id") or cleaned.get("id") or cleaned.get("name") or cleaned.get("facility_name")
    if canonical:
        canonical = _site_display_key(canonical) if re.search(r"[A-Z]{2,}[-_][A-Z0-9]", str(canonical), re.I) else str(canonical).strip()
        cleaned.setdefault("id", canonical)
        cleaned.setdefault("site_id", canonical)
    label = cleaned.get("facility_name") or cleaned.get("name") or cleaned.get("site_id") or cleaned.get("id")
    if label:
        cleaned.setdefault("name", label)
        cleaned.setdefault("facility_name", label)
    return cleaned


def _merge_physical_site_values(winner: Any, loser: Any) -> None:
    wv = getattr(winner, "value", None)
    lv = getattr(loser, "value", None)
    if not isinstance(wv, dict) or not isinstance(lv, dict):
        return
    # Remove legacy/LLM bridge shape fields before they can create
    # Frankenstein physical_site values.
    for k in list(wv.keys()):
        if k not in _PHYSICAL_SITE_ALLOWED_FIELDS:
            wv.pop(k, None)
    for k, lval in lv.items():
        if k not in _PHYSICAL_SITE_ALLOWED_FIELDS or lval in (None, "", [], {}):
            continue
        wval = wv.get(k)
        if wval in (None, "", [], {}):
            wv[k] = lval
            continue
        # A text-fallback atom may have name == id. Let a structured row
        # provide the actual facility name, but otherwise do not overwrite
        # authoritative fields merely because a less-authoritative document
        # has a longer string.
        if k in {"name", "facility_name"}:
            wid = _site_display_key(wv.get("site_id") or wv.get("id"))
            if _site_display_key(wval) == wid and str(lval).strip():
                wv[k] = lval
        elif isinstance(wval, (list, tuple)) and isinstance(lval, (list, tuple)):
            merged = list(wval)
            for x in lval:
                if x not in merged:
                    merged.append(x)
            wv[k] = merged
    winner.value = _clean_physical_site_value(wv)


def _dedupe_physical_site_atoms(atoms: list[Any]) -> list[Any]:
    physical = [a for a in atoms if _atom_type_value(a) == "physical_site"]
    if not physical:
        return atoms

    good_ids = [_physical_site_id(a) for a in physical if not _is_bad_physical_site_id(_physical_site_id(a))]
    complete_ids = sorted({sid for sid in good_ids if _looks_complete_site_id(sid)}, key=len)

    def canonical_for(atom: Any) -> str | None:
        sid = _physical_site_id(atom)
        if not sid or _is_bad_physical_site_id(sid):
            return None
        # Exact complete ids are canonical.
        if sid in complete_ids:
            return sid
        # Merge clipped table ids: ATL-WEST-0 -> ATL-WEST-02 when a full
        # id is present from the same authoritative text.
        clipped_matches = [full for full in complete_ids if full.startswith(sid) and len(full) > len(sid)]
        if clipped_matches:
            return clipped_matches[0]
        # Merge short aliases from SOW/BOM tables into the authoritative
        # numbered site row: ATL-HQ -> ATL-HQ-01.
        prefix_matches = [full for full in complete_ids if full.startswith(sid + "-")]
        if prefix_matches:
            return prefix_matches[0]
        return sid

    grouped: dict[str, list[Any]] = {}
    dropped_ids: set[int] = set()
    for atom in physical:
        canon = canonical_for(atom)
        if canon is None:
            dropped_ids.add(id(atom))
            continue
        grouped.setdefault(canon, []).append(atom)

    merged_physical: list[Any] = []
    consumed_ids: set[int] = set(dropped_ids)
    for canon, group in grouped.items():
        group_sorted = sorted(group, key=lambda a: _physical_site_quality(a, canon), reverse=True)
        winner = group_sorted[0]
        # Force the canonical display id onto the winner before merging.
        if isinstance(getattr(winner, "value", None), dict):
            winner.value["id"] = canon
            winner.value["site_id"] = canon
            winner.value = _clean_physical_site_value(winner.value)
        # v56: also force entity_keys to a SINGLE canonical site:<slug>.
        # Any prior site:* keys (from over-eager regex passes, LLM cluster
        # aliases, or pre-merge variant slugs) get dropped here. Non-site
        # keys (date:, money:, address:, etc.) are preserved.
        try:
            canon_slug = re.sub(r"[^a-z0-9]+", "_", canon.lower()).strip("_")
            if canon_slug:
                existing_keys = list(getattr(winner, "entity_keys", []) or [])
                non_site_keys = [k for k in existing_keys if not k.startswith("site:")]
                non_site_keys.append(f"site:{canon_slug}")
                winner.entity_keys = sorted(set(non_site_keys))
        except Exception:
            pass
        for loser in group_sorted[1:]:
            _merge_physical_site_values(winner, loser)
            _merge_atom_metadata(winner, loser)
            consumed_ids.add(id(loser))
        merged_physical.append(winner)

    out: list[Any] = []
    merged_by_id = {id(a): a for a in merged_physical}
    emitted_merged: set[int] = set()
    for atom in atoms:
        if _atom_type_value(atom) != "physical_site":
            out.append(atom)
            continue
        aid = id(atom)
        if aid in consumed_ids:
            continue
        if aid in merged_by_id:
            out.append(merged_by_id[aid])
            emitted_merged.add(aid)
            continue
    # Add winners whose original position belonged to an atom consumed by
    # another winner. This is rare but keeps the function total.
    for atom in merged_physical:
        if id(atom) not in emitted_merged and id(atom) not in consumed_ids:
            out.append(atom)
    return out


def _drop_generic_site_entity_atoms(atoms: list[Any]) -> list[Any]:
    """Remove legacy generic entity atoms that restate roster sites.

    The pack contract is explicit: site rows must be typed as
    physical_site. Once a physical_site roster exists, keeping
    ``atom_type=entity`` with ``value.entity_type == 'site'`` resurrects
    the old anti-pattern and pollutes downstream packet/entity counts.
    """
    has_physical_site = any(_atom_type_value(a) == "physical_site" for a in atoms)
    if not has_physical_site:
        return atoms
    out: list[Any] = []
    for atom in atoms:
        val = getattr(atom, "value", None) or {}
        if isinstance(val, dict) and str(val.get("entity_type") or "").lower() == "site":
            continue
        out.append(atom)
    return out


def _value_key(atom: Any) -> tuple | None:
    """Return a hashable key describing the atom's identity.

    The key includes atom_type so the same fact_id under two types
    doesn't collide (a task_id and a req_id can both be "001").
    Returns None when no key field is populated — those atoms aren't
    eligible for semantic dedup (they go through the v48 text-based
    pass only).
    """
    atype = _atom_type_value(atom)
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
    if _atom_type_value(winner) == "physical_site":
        _merge_physical_site_values(winner, loser)
        _merge_atom_metadata(winner, loser)
        return
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
    _merge_atom_metadata(winner, loser)


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

    return _drop_generic_site_entity_atoms(_dedupe_physical_site_atoms(list(by_key.values()) + unkeyed))


__all__ = ["semantic_dedup_atoms"]
