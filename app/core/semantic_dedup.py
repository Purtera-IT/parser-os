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


# Bookkeeping / provenance keys that never distinguish one fact from another.
# Excluded from the structural signature so they don't add noise (or, worse,
# split a genuine duplicate because one copy carries a provenance stamp).
_SIGNATURE_SKIP_FIELDS: frozenset[str] = frozenset({
    "kind", "type", "atom_type", "raw", "raw_cells", "row_index",
    "source", "provenance", "receipts", "confidence", "names", "aliases",
    "notes", "note", "extras",
})
# A scalar string longer than this is description-like prose, not an identity
# token — it is already represented by the truncated description key, so we
# don't fold it into the signature.
_SIGNATURE_MAX_STR = 48
_SIGNATURE_MAX_FIELDS = 8


def _scalar_signature(val: dict[str, Any], exclude: frozenset[str] | set[str]) -> str:
    """Compact, deterministic fingerprint of an atom value's *distinguishing*
    scalar fields (numbers and short identity strings), excluding the
    description-style fields the caller already keyed on.

    Two table rows that share a boilerplate description ("Video bar,
    scheduling panel, …") but carry different structured fields (room name,
    quantity, site) produce different signatures, so they are NOT collapsed.
    Two LLM paraphrases of the *same* fact carry the same structured fields,
    so their signature matches and they still collapse. Returns "" when the
    atom has no distinguishing scalar field — pure-prose atoms (assumption,
    plain risk text, …) behave exactly as before.
    """
    parts: list[str] = []
    for k in sorted(val):
        if len(parts) >= _SIGNATURE_MAX_FIELDS:
            break
        if k in exclude or k in _SIGNATURE_SKIP_FIELDS or k.startswith("_"):
            continue
        v = val.get(k)
        if isinstance(v, bool) or v in (None, "", [], {}):
            continue
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif isinstance(v, str):
            s = v.strip()
            if s and len(s) <= _SIGNATURE_MAX_STR:
                nk = _norm_key(s)
                if nk:
                    parts.append(f"{k}={nk}")
    return "|".join(parts)


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
    # v56: aliases / surface forms collected by the v55 LLM-cluster
    # merge step in entity_extraction._entities_to_atoms. Without
    # this, _clean_physical_site_value silently drops them.
    "names", "aliases",
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
    # v56f: DETERMINISTIC aliases — only the SAME-ROW identity signals
    # (site_id, facility_name, street_address). NEVER mdf_idf,
    # access_window, escort_owner — those are SEPARATE FIELDS for a
    # reason. NEVER carry over aliases from prior LLM merges since
    # those mixed multiple-row data ("OPTBOT Facil" from row 1 ending
    # up as an alias of ATL-WEST-02). One row = one site = three
    # identity strings tops. The atom already has every column in its
    # proper field; aliases exist only for cross-doc text matching.
    names_out: list[str] = []
    for f in ("site_id", "id", "facility_name", "name", "street_address", "address"):
        v = cleaned.get(f)
        if isinstance(v, str) and v.strip() and v not in names_out:
            names_out.append(v.strip())
    if names_out:
        cleaned["names"] = names_out
    else:
        cleaned.pop("names", None)
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


def _is_hallucinated_physical_site_value(value: Any) -> bool:
    """v57.2 — independent invariant guard. True if the atom value smells
    like an LLM hallucination of a physical_site row.

    Real roster atoms come from a structured row with three DISTINCT
    identity strings: site_id (``ATL-HQ-01``), facility_name
    (``OPTBOT Atlanta HQ``), address (``1200 Peachtree St NE...``).
    When the LLM fails to separate cell-bleed text from a paragraph
    block, it fills ``name`` / ``address`` / ``facility_name`` with the
    SAME synthesized string (and often appends ``v5`` / ``2026``).
    This invariant catches both shapes regardless of source path —
    typed_atom_classifier, LLM bridge, vision extractor, etc.
    """
    if not isinstance(value, dict):
        return False
    name = (value.get("name") or "").strip()
    address = (value.get("address") or value.get("street_address") or "").strip()
    facility = (value.get("facility_name") or "").strip()
    # All three identity fields identical — only fire when ALL THREE are
    # populated (mirrors typed_atom_classifier guard). A clean text-roster
    # atom legitimately has name == facility with address empty; that's
    # not a ghost.
    if name and address and facility and name == address == facility:
        return True
    # Also catch the ``v\d+`` / 4-digit-year suffix on site_id directly.
    sid = (value.get("site_id") or value.get("id") or "").strip()
    if sid and (re.search(r"-V\d+$", sid, re.IGNORECASE) or re.search(r"\s\d{4}$", sid)):
        return True
    return False


def _dedupe_physical_site_atoms(atoms: list[Any]) -> list[Any]:
    physical = [a for a in atoms if _atom_type_value(a) == "physical_site"]
    if not physical:
        return atoms

    # v57.2: kill the LLM hallucination shape unconditionally before
    # canonical_for resolves anything. Independent of the address/facility
    # lookup below — even when those indexes are empty (no complete
    # structural atoms), this catches OPTBOT-WEST-CAMPUS-V5-style ghosts.
    # Atoms that don't smell hallucinated continue through the existing
    # canonical_for resolution + winner-merge logic unchanged.
    before_hallucination = len(physical)
    physical = [a for a in physical if not _is_hallucinated_physical_site_value(getattr(a, "value", None))]
    if before_hallucination != len(physical):
        # Update the atoms list too so the new list reflects the drop.
        dropped_ids = {id(a) for a in atoms if _atom_type_value(a) == "physical_site"} - {id(a) for a in physical}
        atoms = [a for a in atoms if id(a) not in dropped_ids]
        if not physical:
            return atoms

    good_ids = [_physical_site_id(a) for a in physical if not _is_bad_physical_site_id(_physical_site_id(a))]
    complete_ids = sorted({sid for sid in good_ids if _looks_complete_site_id(sid)}, key=len)

    # v57.1: address/facility lookup against complete atoms. typed_atom_
    # classification and other downstream promoters create physical_site
    # atoms with synthetic site_ids derived from facility names (e.g.
    # "OPTBOT-ATLANTA-HQ" from "OPTBOT Atlanta HQ"). These have the SAME
    # street_address as the canonical roster atom ("ATL-HQ-01") but a
    # non-canonical site_id, so the old by-site-id grouping kept them
    # as separate sites. We now build a normalized address+facility
    # index over the complete atoms and use it as a fallback when the
    # site_id alone doesn't resolve. Without this, the OPTBOT cockpit
    # shows 10 sites (5 clean + 5 ghosts) instead of 5.
    def _nf(s: Any) -> str:
        if not isinstance(s, str):
            return ""
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    addr_to_canonical: dict[str, str] = {}
    facility_to_canonical: dict[str, str] = {}
    for a in physical:
        sid_a = _physical_site_id(a)
        if not _looks_complete_site_id(sid_a):
            continue
        v = getattr(a, "value", None) or {}
        if not isinstance(v, dict):
            continue
        for field in ("street_address", "address"):
            key = _nf(v.get(field))
            if key and key not in addr_to_canonical:
                addr_to_canonical[key] = sid_a
        for field in ("facility_name", "name"):
            key = _nf(v.get(field))
            if key and key not in facility_to_canonical:
                facility_to_canonical[key] = sid_a

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
        # v57.1: address/facility lookup. Catches typed_atom_classifier
        # promotions where site_id was derived from facility_name (e.g.
        # "OPTBOT-ATLANTA-HQ") but address/facility match a complete
        # atom ("ATL-HQ-01"). Without this, the dedup keeps both as
        # separate sites — 10 atoms displayed instead of 5.
        v = getattr(atom, "value", None) or {}
        if isinstance(v, dict) and complete_ids:
            for field in ("street_address", "address"):
                ak = _nf(v.get(field))
                if ak and ak in addr_to_canonical:
                    return addr_to_canonical[ak]
            for field in ("facility_name", "name"):
                fk = _nf(v.get(field))
                if fk and fk in facility_to_canonical:
                    return facility_to_canonical[fk]
            # No match against any complete atom — ghost emission with a
            # synthetic site_id and no canonical address/facility to
            # merge into. Drop it. Safe: a genuine new site would have
            # either a canonical-shape site_id (caught above) or be in a
            # document with NO structural roster (complete_ids empty,
            # this branch skipped).
            return None
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
        for loser in group_sorted[1:]:
            _merge_physical_site_values(winner, loser)
            _merge_atom_metadata(winner, loser)
            consumed_ids.add(id(loser))
        # v56: AFTER the merge loop, force entity_keys to a SINGLE
        # canonical site:<slug>. Doing this before the merge would have
        # been undone by _merge_atom_metadata (line 205) which copies
        # loser's entity_keys onto winner. Order MUST be: merge then
        # dedup-keys. Address/quantity/etc keys carried by losers are
        # preserved (only site:* gets collapsed). Other physical_site
        # atoms that mention this site's address in their raw_text may
        # still pick up a site:* variant via cross-doc joins later, but
        # at this point the canonical roster atoms are clean.
        try:
            canon_slug = re.sub(r"[^a-z0-9]+", "_", canon.lower()).strip("_")
            if canon_slug:
                existing_keys = list(getattr(winner, "entity_keys", []) or [])
                non_site_keys = [k for k in existing_keys if not k.startswith("site:")]
                non_site_keys.append(f"site:{canon_slug}")
                winner.entity_keys = sorted(set(non_site_keys))
        except Exception:
            pass
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

        Structure-awareness: a truncated description is a WEAK key. Table
        rows often share a boilerplate description ("Video bar, scheduling
        panel, …") while differing only in structured fields (room name,
        quantity, site). Keying on description alone collapses those
        distinct rows into one — silent data loss. So when we fall back to
        a description key we append a signature of the atom's OTHER scalar
        fields; distinct rows then get distinct keys and survive. This only
        ever *splits* fuzzy groups (keeps more atoms), never merges more —
        the guess-free / keep-don't-drop direction. The stable-ID path
        (``_first``) is unaffected: a real req_id/sku still collapses
        regardless of field jitter.
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
                    base = k[:32]
                    sig = _scalar_signature(val, frozenset(fields))
                    return f"{base}#{sig}" if sig else base
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


# ── cross-type dedup ────────────────────────────────────────────────
#
# The same source sentence often reaches the atom list under several
# types: a table row gets emitted as a ``raw_table_row`` AND typed into a
# ``service_line`` AND swept into ``scope_item`` AND tagged a ``task``.
# semantic_dedup keys *with* atom_type (by design — a task_id and a
# req_id of "001" must not collide), so those never collapse. This pass
# folds same-text-different-type duplicates into the single most-specific
# type. A fact that is verbatim-identical across types is one fact; we
# keep the richest representation and merge the rest's provenance in.

# Higher rank wins. raw_table_row is the raw extraction and always loses
# to anything typed; scope_item is the generic catch-all; structured
# commercial/service types are the most specific.
_CROSS_TYPE_PRIORITY: dict[str, int] = {
    "raw_table_row": 0,
    "scope_item": 2,
    "site_attribute": 5,
    "deal_metadata": 6,
    "deliverable": 4,
    "task": 4,
    "open_question": 4,
    "requirement": 5,
    "exclusion": 5,
    "constraint": 6,
    "service_line": 7,
    "bom_line": 7,
    "pricing_assumption": 7,
    "site_budget": 7,
    "payment_term": 8,
    "commercial_total": 8,
}
_CROSS_TYPE_DEFAULT_PRIORITY = 3

# Money/quantity tokens are stripped before keying so "…| $5,390.00" and
# "…| 5390" collapse, and so a typed atom that dropped the trailing total
# still matches the raw row.
_CROSS_TYPE_STRIP_RE = re.compile(r"[$£€]|\b\d[\d,.]*\b|[^a-z0-9\s]")


def _cross_type_text_key(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    norm = _CROSS_TYPE_STRIP_RE.sub(" ", str(raw).lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    # Structure-aware scoping: a table cell is identified by (artifact, table,
    # row). When the atom comes from a table, prefix the key with that cell so
    # the same-cell twins (raw_table_row + scope_item + typed schema atom)
    # collapse, but identical-looking rows from DIFFERENT cells/sites never do.
    # This matters because the key strips quantities — without the cell prefix,
    # "Large conference | 4" (HQ) and "| 2" (WEST) and "| 2" (AIR) would share
    # one key and three sites' room counts would collapse to one. The cell
    # prefix keeps each site's per-row payload distinct.
    cell = _atom_cell_locator(atom)
    if cell:
        return f"{cell}|{norm[:80]}"
    if len(norm) < 8:
        return ""
    # Cap so trailing paraphrase divergence doesn't split a shared fact.
    return norm[:80]


def _atom_cell_locator(atom: Any) -> str:
    """Return 'artifact:table:row' for a table/sheet-cell atom, else '' (prose).

    Covers both shapes: docx tables (``table_index``) and xlsx sheets
    (``sheet`` + ``row``/``row_index``), so identical-looking rows from
    different cells/sites/sheets never collapse into one another.
    """
    for ref in (getattr(atom, "source_refs", None) or []):
        loc = getattr(ref, "locator", None) or {}
        if not isinstance(loc, dict):
            continue
        row = loc.get("row")
        if row is None:
            row = loc.get("row_index")
        if row is None:
            continue
        table = loc.get("sheet")
        if table is None and loc.get("table_index") is not None:
            table = f"t{loc['table_index']}"
        if table is None:
            continue
        art = getattr(atom, "artifact_id", "") or ""
        return f"{art}:{table}:r{row}"
    return ""


def _cross_type_priority(atom: Any) -> int:
    return _CROSS_TYPE_PRIORITY.get(_atom_type_value(atom), _CROSS_TYPE_DEFAULT_PRIORITY)


def cross_type_dedup_atoms(atoms: list[Any]) -> list[Any]:
    """Collapse the *same sentence* emitted under multiple atom types.

    Groups atoms by a money/quantity-stripped text key. Within any group
    that spans more than one atom_type, the highest-priority type wins
    (ties broken by confidence); the losers' provenance is merged in and
    they are dropped. Groups that are all one type are left untouched
    (intra-type dedup is semantic_dedup's job). Pure function, no I/O.
    """
    if not atoms:
        return atoms

    groups: dict[str, list[Any]] = {}
    order: list[str] = []
    passthrough: list[Any] = []
    for atom in atoms:
        # An open_question is a distinct speech act, not a lossy duplicate of a
        # declarative fact. The text key strips punctuation — including the
        # trailing "?" that *makes* it a question — so "MDF badge access?"
        # (open_question) and "MDF badge access" (constraint/scope) would
        # otherwise collapse, dropping the only atom type that drives the
        # missing_info packet. Never collapse questions across types; let them
        # survive on their own axis (intra-type dups are semantic_dedup's job).
        if _atom_type_value(atom) == "open_question":
            passthrough.append(atom)
            continue
        key = _cross_type_text_key(atom)
        if not key:
            passthrough.append(atom)
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(atom)

    # Decide survivors per group (and merge losers' provenance into the winner)
    # WITHOUT building a new order: a dedup pass must not shuffle the atoms it
    # keeps. Clustering members at the first-occurrence position previously
    # reordered identical-text prose (e.g. the same site note under 3 sites all
    # jumped adjacent), so the stream stopped reading in document order even
    # though nothing was dropped.
    survivors: set[int] = set()
    for key in order:
        members = groups[key]
        if len(members) == 1 or len({_atom_type_value(a) for a in members}) == 1:
            # Single atom, or all one type — not a cross-type duplicate; keep all.
            survivors.update(id(m) for m in members)
            continue
        winner = max(members, key=lambda a: (_cross_type_priority(a), _confidence(a)))
        for loser in members:
            if loser is winner:
                continue
            _merge_atom_metadata(winner, loser)
        survivors.add(id(winner))

    # Emit in ORIGINAL input order: each atom survives if it's a passthrough
    # (open_question / unkeyed) or the kept member of its group.
    passthrough_ids = {id(a) for a in passthrough}
    result = [a for a in atoms if id(a) in passthrough_ids or id(a) in survivors]
    return _suppress_line_item_doubles(result)


def _suppress_line_item_doubles(atoms: list[Any]) -> list[Any]:
    """Drop the bom_line/service_line DOUBLE of a quote row that also produced a
    vendor_line_item.

    A quote sheet routes through BOTH the quote parser (vendor_line_item — feeds
    the commercial summary + the vendor-mismatch contradiction graph) and the
    schema registry (bom_line/service_line — feeds the BOM section). They are
    two rich, full-row copies of the same line at the same (sheet,row) cell.
    Keep vendor_line_item (the BOM renderer reads it too now) and drop the
    registry double, merging its provenance in so nothing is lost. docx BOM
    tables have no vendor_line_item, so their bom_line is untouched.
    """
    vli_by_cell: dict[str, Any] = {}
    for a in atoms:
        if _atom_type_value(a) == "vendor_line_item":
            cell = _atom_cell_locator(a)
            if cell:
                vli_by_cell.setdefault(cell, a)
    if not vli_by_cell:
        return atoms
    out: list[Any] = []
    for a in atoms:
        if _atom_type_value(a) in ("bom_line", "service_line"):
            cell = _atom_cell_locator(a)
            winner = vli_by_cell.get(cell) if cell else None
            if winner is not None:
                _merge_atom_metadata(winner, a)
                continue
        out.append(a)
    return out


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


__all__ = ["semantic_dedup_atoms", "cross_type_dedup_atoms"]
