"""Universal typed-atom classifier — v47 taxonomy.

After ``enrich_atoms`` populates entity_keys on every atom, this
classifier promotes the catch-all ``scope_item`` (and other under-
specified types) into the rich v47 taxonomy:

  milestone_phase, task, deliverable, cutover_step,
  stakeholder, approval_authority, approval_decision,
  bom_line, service_line, site_allocation, site_budget,
  commercial_total, payment_term, change_order_rule,
  physical_site, site_attribute, site_access_window,
  site_room_mix, site_infrastructure,
  requirement, acceptance_criterion, electrical_acceptance_test,
  compliance_classification, compliance_rule,
  mitigation, dependency, blackout_date_range,
  data_flow_step, system_mapping, metadata_requirement,
  lead_time_constraint, integration_checkpoint,
  ...

The classifier is INTENTIONALLY universal — no regex over column
headers, no hardcoded customer-name lists. A small Ollama model
(qwen2.5:3b by default) reads the atom text + section_path and
returns the type plus a structured ``value`` payload. Because the
model has the v47 taxonomy as its instruction set, it generalises
across customer terminology variations ("Phase|Sprint|Wave|Stage|
Milestone" all map to milestone_phase; "Owner|Lead|Responsible|
Accountable|Approver" all collapse into the same role field).

Behaviour:
  * runs AFTER enrich_atoms, so entity_keys are populated
  * BATCHED — 25 atoms per call, parallel via ThreadPoolExecutor
  * STABLE — model output is keyed by atom_id; on parse failure the
    atom keeps its original type
  * RESPECTFUL of existing types — already-typed atoms (risk, exclusion,
    decision, assumption, schematic_*, etc.) are NOT reclassified unless
    the model is highly confident the existing type is wrong
  * NO-OP when LLM unreachable — atoms keep their original types

Configuration:
  SOWSMITH_TYPED_CLASSIFIER_DISABLE=1   skip entirely
  SOWSMITH_TYPED_CLASSIFIER_MODEL       default qwen2.5:3b
  SOWSMITH_TYPED_CLASSIFIER_BATCH       default 25
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import urllib.request
from typing import Any

DEFAULT_HOST = "http://100.114.102.122:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT = 180
DEFAULT_BATCH_SIZE = 12  # dev proxy drops responses >~3KB; 12 atoms stays well under
DEFAULT_PARALLEL = 4

# Types the classifier WILL promote scope_item / entity into.
# Other existing types (risk, exclusion, decision, assumption,
# schematic_*, port_vlan_assignment, etc.) are preserved.
_PROMOTABLE_FROM = frozenset({
    "scope_item",
    "entity",
    "customer_instruction",
})

# Universal v47 taxonomy taught to the model. Each entry: name +
# one-line description + (optional) example fields the model should
# emit in ``value`` for that type. The model can also emit
# ``"_keep"`` to signal the atom should retain its current type.
_TAXONOMY: dict[str, dict[str, Any]] = {
    # Tier 1 — deal context
    "deal_metadata": {
        "desc": "Header-level fact about the deal itself: customer name, deal ID, quote/PO/MSA references, contract type, currency, target close date, deal stage.",
        "fields": ["field_name", "value"],
    },
    "commercial_total": {
        "desc": "An aggregate price line: hardware subtotal, services subtotal, freight/logistics, contingency, taxes, grand total. Each is one atom.",
        "fields": ["category", "amount", "currency"],
    },
    "payment_term": {
        "desc": "One row of a billing/payment schedule, e.g. '30% at order acceptance'. Each tier is its own atom.",
        "fields": ["tier", "percent", "trigger"],
    },
    "change_order_rule": {
        "desc": "A rule that governs change orders — when one is required, T&M rate caps, materials markup, after-hours rate.",
        "fields": ["trigger_kind", "rate_or_threshold"],
    },

    # Tier 2 — sites
    "physical_site": {
        "desc": "Authoritative site declaration: site_id + facility name + street address. One per real building / location.",
        "fields": ["site_id", "name", "address"],
    },
    "site_attribute": {
        "desc": "A scalar attribute of a site: user count, room count, square feet, priority, floors.",
        "fields": ["site", "attribute_kind", "value"],
    },
    "site_access_window": {
        "desc": "When a site is accessible — days + hours + escort owner.",
        "fields": ["site", "days", "hours", "escort_owner"],
    },
    "site_access_restriction": {
        "desc": "A constraint on access: blackout window, restricted hours, badged escort required, weekend-only cutover, etc.",
        "fields": ["site", "restriction_kind", "window", "condition"],
    },
    "site_infrastructure": {
        "desc": "Site-bound infrastructure facts: MDF/IDF id, bandwidth/circuit, cable plant condition.",
        "fields": ["site", "kind", "value"],
    },
    "site_room_mix": {
        "desc": "A row from a per-site room-mix table: room type + count + standard build + validation test.",
        "fields": ["site", "room_type", "count", "build_spec", "validation"],
    },
    "site_implementation_note": {
        "desc": "Site-specific implementation instruction (asset-tag prefix, receiving rule, install rule).",
        "fields": ["site", "kind", "detail"],
    },

    # Tier 3 — schedule
    "milestone_phase": {
        "desc": "A row from a project phase/milestone table — phase number + name + start date + end date + owner + exit criteria. Synonyms accepted: phase, sprint, wave, milestone, stage.",
        "fields": ["phase_id", "name", "start", "end", "owner", "exit_criteria"],
    },
    "task": {
        "desc": "A row from a detailed-task table: task_id + site + phase + description + owner + dates + dependency + status.",
        "fields": ["task_id", "site", "phase", "name", "owner", "start", "due", "dependency", "status"],
    },
    "deliverable": {
        "desc": "A named deliverable that a phase produces (e.g. 'Network closet readiness checklist').",
        "fields": ["phase", "name", "owner"],
    },
    "cutover_step": {
        "desc": "A row from a cutover/runbook checklist — step number + timing relative to cutover + owner + action + evidence required.",
        "fields": ["step_num", "timing", "owner", "action", "evidence"],
    },
    "integration_checkpoint": {
        "desc": "A row from a system-to-system integration checkpoint table: checkpoint id + system + expected input/output + owner + due.",
        "fields": ["checkpoint_id", "system", "input", "output", "owner", "due"],
    },
    "blackout_date_range": {
        "desc": "A no-work date window (holiday freeze, exec blackout, peak travel exclusion).",
        "fields": ["start", "end", "reason", "applies_to"],
    },

    # Tier 4 — stakeholders / authority
    "stakeholder": {
        "desc": "A named person on the deal with their role and email. Synonyms accepted: owner, lead, sponsor, accountable, contact.",
        "fields": ["name", "role", "email", "authority_class"],
    },
    "approval_authority": {
        "desc": "A rule that some role/threshold requires a specific approver (e.g. 'CFO signoff required over $1.5M').",
        "fields": ["approver", "scope", "threshold", "condition"],
    },
    "approval_decision": {
        "desc": "A recorded approval/conditional approval/rejection statement by a named stakeholder.",
        "fields": ["approver", "decision", "condition"],
    },
    "signatory": {
        "desc": "A signature-block entry — role + named person who signs.",
        "fields": ["role", "name"],
    },

    # Tier 5 — BOM / pricing
    "bom_line": {
        "desc": "A hardware-BOM table row: item id + description + SKU + qty + unit price + extended cost + lead time + serial-capture flag.",
        "fields": ["item_id", "description", "sku", "qty", "unit_price", "extended_cost", "lead_time_days"],
    },
    "site_allocation": {
        "desc": "Per-site qty breakdown of a single BOM line (e.g. 'HW-001: ATL-HQ:52; ATL-WEST:27; ATL-AIR:15').",
        "fields": ["bom_item", "site", "qty"],
    },
    "service_line": {
        "desc": "A services-row line item: service id + description + unit + qty + unit price + extended cost.",
        "fields": ["service_id", "description", "unit", "qty", "unit_price", "extended_cost"],
    },
    "site_budget": {
        "desc": "Per-site budget category amount (hardware budget / services budget / logistics estimate per site).",
        "fields": ["site", "category", "amount"],
    },
    "lead_time_constraint": {
        "desc": "Lead time + expediting policy for an item class (core switches, APs, AV kits, fiber panels, custom millwork).",
        "fields": ["item_class", "lead_days", "expedite_terms", "stock_location"],
    },
    "pricing_assumption": {
        "desc": "An explicit pricing assumption (taxes excluded, hardware substitutions require approval, etc.).",
        "fields": ["domain", "statement"],
    },

    # Tier 6 — requirements / acceptance / compliance
    "requirement": {
        "desc": "A formally numbered requirement (REQ-001 style) with applies-to + owner + verification method.",
        "fields": ["req_id", "description", "applies_to", "owner", "verification"],
    },
    "acceptance_criterion": {
        "desc": "A row from an acceptance-criteria table — area + criteria + pass threshold.",
        "fields": ["area", "criteria", "threshold"],
    },
    "electrical_acceptance_test": {
        "desc": "A measurable electrical/network acceptance test (Megger, ground resistance, load burn-in, witness/ATP).",
        "fields": ["test_kind", "measurable_threshold", "applies_to"],
    },
    "compliance_classification": {
        "desc": "A document/data classification + its allowed and blocked destinations (Confidential, Internal, Mock, Production).",
        "fields": ["classification", "allowed_destinations", "blocked_destinations"],
    },
    "compliance_rule": {
        "desc": "A rule about access, data handling, retention, credential use, or auditing.",
        "fields": ["rule_kind", "condition"],
    },

    # Tier 7 — risks / dependencies
    "mitigation": {
        "desc": "A mitigation paired with a specific risk (separate from the risk atom itself).",
        "fields": ["risk_id", "mitigation_text", "owner"],
    },
    "dependency": {
        "desc": "A dependency on a customer-provided input, third party, or infrastructure readiness.",
        "fields": ["kind", "description", "needed_by"],
    },

    # Tier 8 — integration / system
    "data_flow_step": {
        "desc": "One step of a data-flow pipeline (source system → sink system, what payload).",
        "fields": ["sequence", "source_system", "sink_system", "payload"],
    },
    "system_mapping": {
        "desc": "A field mapping between two systems (e.g. HubSpot dealname → 'OPTBOT Atlanta Office Refresh').",
        "fields": ["system", "field_name", "canonical_value"],
    },
    "metadata_requirement": {
        "desc": "A required metadata key/value to set on an object (blob metadata, CRM custom field).",
        "fields": ["system", "key", "expected_value"],
    },
}


def classify_atoms(atoms: list[Any]) -> int:
    """Promote atoms from the v47 taxonomy where confident.

    Mutates ``atom.atom_type`` and ``atom.value`` in place. Returns the
    number of atoms reclassified.

    No-op on LLM unreachable / malformed response — atoms keep their
    original types. NEVER demotes a domain-specific type (risk,
    exclusion, schematic_*, port_vlan_assignment, etc.).
    """
    if not atoms:
        return 0
    if os.environ.get("SOWSMITH_TYPED_CLASSIFIER_DISABLE"):
        return 0

    promotable = [a for a in atoms if _atom_type_str(a) in _PROMOTABLE_FROM]
    if not promotable:
        return 0

    if not _ollama_reachable():
        return 0

    batch_size = int(os.environ.get("SOWSMITH_TYPED_CLASSIFIER_BATCH", str(DEFAULT_BATCH_SIZE)))
    parallel = int(os.environ.get("SOWSMITH_LLM_PARALLEL", str(DEFAULT_PARALLEL)))
    batches = [promotable[i:i + batch_size] for i in range(0, len(promotable), batch_size)]

    results_by_atom_id: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        future_to_batch = {pool.submit(_classify_batch, b): b for b in batches}
        for fut in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = fut.result()
            except Exception:
                continue
            for atom_id, payload in batch_results.items():
                results_by_atom_id[atom_id] = payload

    promoted = 0
    for atom in promotable:
        atom_id = _atom_id(atom)
        if not atom_id or atom_id not in results_by_atom_id:
            continue
        payload = results_by_atom_id[atom_id]
        new_type = payload.get("atom_type")
        if not new_type or new_type == "_keep":
            continue
        if new_type not in _TAXONOMY:
            continue
        # v57.2: reject promotion to physical_site when the source text or
        # the LLM-emitted value smells like a hallucination. The OPTBOT
        # site-roster PDF has cell-bleed paragraph blocks like
        # "Mon-Fri 07:00-18: OPTBOT Facil ATL-WEST-0 OPTBOT West Campus
        # 3100 Interstate N Pkwy, Atla" — six columns from one row mashed
        # into one string. qwen3:14b sees the chaos and synthesizes
        # site_id="OPTBOT-WEST-CAMPUS-V5" with name == address ==
        # facility_name == "OPTBOT West Campus v5" (all three identical,
        # mathematically impossible for a real roster row). These ghosts
        # survive every other downstream guard because their hallucinated
        # address (literally the facility name) doesn't match any
        # canonical street address in the dedup index. Kill at the source.
        new_value = payload.get("value")
        if new_type == "physical_site":
            if _is_hallucinated_physical_site(atom, new_value):
                continue
        try:
            from app.core.schemas import AtomType
            atom.atom_type = AtomType(new_type)
        except (ImportError, ValueError):
            continue
        if isinstance(new_value, dict) and new_value:
            existing_value = getattr(atom, "value", None)
            if isinstance(existing_value, dict):
                merged = {**existing_value, **new_value}
                atom.value = merged
            else:
                atom.value = new_value
        promoted += 1

    return promoted


# v57.2 — hallucination invariants for typed_atom_classifier physical_site
# promotions. ALL checks operate on the VALUE shape, never on raw_text,
# because the structured table parser legitimately synthesizes raw_text
# strings like "Mon-Fri 07:00-18 | OPTBOT Facil" for cross-doc text
# matching — those are clean atoms despite the time-window content.
_GHOST_NAME_PATTERNS = (
    re.compile(r"\sv\d+$", re.IGNORECASE),    # "OPTBOT West Campus v5"
    re.compile(r"\s\d{4}$"),                  # "Atl Hq 2026"  (v53.7 pattern)
    re.compile(r"-v\d+$", re.IGNORECASE),     # "OPTBOT-WEST-CAMPUS-V5"
)
# v57.2 — site_id shapes that betray the LLM mistakenly classifying a
# street address as the ID column. Real site IDs are short codes
# (ATL-HQ-01, SITE-042); addresses-as-id contain ZIP codes or building
# words. The actual address goes in value.address — never in value.site_id.
_ADDRESS_AS_ID_PATTERNS = (
    re.compile(r"\d{5}(?:-\d{4})?\b"),                          # ZIP code in id
    re.compile(r"\bBUILDING\b", re.IGNORECASE),                 # "BUILDING-C"
    re.compile(r"\b(STREET|AVENUE|BOULEVARD|PARKWAY|DRIVE)\b", re.IGNORECASE),
)


def _is_hallucinated_physical_site(atom: Any, new_value: Any) -> bool:
    """True if the LLM-promoted physical_site atom shows hallucination tells.

    Three value-only invariants — never inspects ``raw_text`` because the
    structured table parser legitimately synthesizes row-summary strings
    that contain time windows and column labels.

    1. **All identity fields identical.** Real roster rows have distinct
       ``site_id``, ``facility_name``, and ``address``. When the LLM
       emits the SAME string for ``name``, ``address``, and
       ``facility_name``, it forfeited parsing — that string is just
       whatever it could grab. Catches every ``OPTBOT-XXX-V5`` ghost.

    2. **Ghost suffix on site_id / name / facility.** Trailing ``-V5`` /
       ``v\\d+`` / 4-digit year is a schema-version or row-number
       hallucination. Real site IDs come from the structured ID column.

    3. **Address-shape site_id.** When ``value.site_id`` contains a ZIP
       code or a street-type word, the LLM swapped the ID and address
       columns. Catches ``4200-GLOBAL-GATEWAY-...-COLLEGE-PARK-GA-30337``
       and similar.
    """
    if not isinstance(new_value, dict):
        return False

    name = (new_value.get("name") or "").strip()
    address = (new_value.get("address") or new_value.get("street_address") or "").strip()
    facility = (new_value.get("facility_name") or "").strip()
    site_id = (new_value.get("site_id") or new_value.get("id") or "").strip()

    # 1. Identical identity fields — mathematically impossible for a real
    # row. Requires ALL THREE of name/address/facility to be non-empty
    # AND all equal. A clean text-fallback atom with name == facility
    # but address empty is NOT a ghost (the v53.8 text-roster extractor
    # legitimately emits that shape: it only extracts site_id + facility
    # name from a section header, never an address). Only fire when the
    # LLM filled all three slots with the same string — that's the
    # forfeit shape (couldn't separate columns, copied one value into
    # every field).
    if name and address and facility and name == address == facility:
        return True

    # 2. Ghost suffix on site_id, name, or facility_name.
    for s in (site_id, name, facility):
        if not s:
            continue
        for pat in _GHOST_NAME_PATTERNS:
            if pat.search(s):
                return True

    # 3. Address-shape site_id — model swapped ID and address columns.
    if site_id:
        for pat in _ADDRESS_AS_ID_PATTERNS:
            if pat.search(site_id):
                return True

    return False


# ────────────────────────── internals ──────────────────────────


def _atom_id(atom: Any) -> str | None:
    aid = getattr(atom, "id", None)
    return str(aid) if aid else None


def _atom_type_str(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    if hasattr(t, "value"):
        return str(t.value)
    return str(t) if t else ""


def _atom_text(atom: Any) -> str:
    text = getattr(atom, "raw_text", None) or ""
    if not text:
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            text = val.get("text") or val.get("content") or ""
    return str(text or "")[:600]


def _atom_section_path(atom: Any) -> str:
    try:
        refs = getattr(atom, "source_refs", None) or []
        if refs:
            loc = getattr(refs[0], "locator", None) or {}
            if isinstance(loc, dict):
                sp = loc.get("section_path")
                if isinstance(sp, list) and sp:
                    return " > ".join(str(x) for x in sp if x)[:200]
    except Exception:
        pass
    return ""


def _build_prompt(batch: list[Any]) -> str:
    taxonomy_lines = []
    for name, meta in _TAXONOMY.items():
        flds = ",".join(meta.get("fields", []))
        taxonomy_lines.append(f"  {name}: {meta['desc']} | value fields: {flds}")
    taxonomy_block = "\n".join(taxonomy_lines)

    atom_lines = []
    for atom in batch:
        atom_id = _atom_id(atom) or "?"
        text = _atom_text(atom).replace("\n", " ")
        section = _atom_section_path(atom)
        suffix = f" [section: {section}]" if section else ""
        atom_lines.append(f"  {atom_id}: {text}{suffix}")
    atom_block = "\n".join(atom_lines)

    return f"""You are classifying parsed deal-document atoms into a typed taxonomy. Each atom is a fact extracted from a deal packet (SOW, BOM, site roster, schedule, contracting, etc.).

For EACH atom below, return EITHER:
  - the most specific matching ``atom_type`` from the taxonomy AND a structured ``value`` payload extracting the named fields from the atom text, OR
  - ``"_keep"`` if no taxonomy entry fits (the atom will keep its current type).

Be precise — a phase row is milestone_phase, NOT a generic deal_metadata. A "30% at order acceptance" line is payment_term, not commercial_total (that's for aggregate subtotals). A row of an REQ-001 table is requirement. A named person's signature block entry is signatory.

TAXONOMY:
{taxonomy_block}

ATOMS TO CLASSIFY:
{atom_block}

OUTPUT — strict JSON, one line per atom, in the form:
{{"results": [{{"atom_id": "...", "atom_type": "<taxonomy_name_or__keep>", "value": {{...extracted fields...}}}}, ...]}}

If a field is not present in the atom text, omit it from value. Do not invent values. Do not echo example fields.

/no_think"""


def _classify_batch(batch: list[Any]) -> dict[str, dict[str, Any]]:
    if not batch:
        return {}
    prompt = _build_prompt(batch)
    response_text = _call_ollama(prompt)
    if not response_text:
        return {}
    return _parse_response(response_text)


def _parse_response(response_text: str) -> dict[str, dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\"results\"[\s\S]*\}", response_text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        # Try to recover by trimming trailing text after the last ``}``
        cleaned = match.group(0).rsplit("}", 1)[0] + "}"
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
    results = parsed.get("results")
    if not isinstance(results, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        aid = item.get("atom_id")
        atype = item.get("atom_type")
        if not aid or not atype:
            continue
        out[str(aid)] = {
            "atom_type": str(atype),
            "value": item.get("value") if isinstance(item.get("value"), dict) else {},
        }
    return out


# ──────────────────────── HTTP transport ────────────────────────


def _ollama_reachable() -> bool:
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _call_ollama(prompt: str, *, max_tokens: int = 4096) -> str:
    """POST to /api/generate, robust against proxy mid-stream drops.

    The dev Ollama HTTPS proxy occasionally truncates non-streaming
    responses (urllib raises IncompleteRead). We capture the partial
    body via IncompleteRead.partial, fall through to the parser, and
    let the partial-body recovery in _parse_response do its job.
    Streaming mode is no better — the proxy emits one NDJSON chunk
    then drops the connection.
    """
    import http.client
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = os.environ.get("SOWSMITH_TYPED_CLASSIFIER_MODEL") or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    timeout = int(os.environ.get("SOWSMITH_LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    body = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except http.client.IncompleteRead as exc:
        # Proxy dropped mid-stream — keep what we got.
        try:
            body = (exc.partial or b"").decode("utf-8", errors="ignore")
        except Exception:
            body = ""
    except Exception:
        return ""
    if not body:
        return ""
    # body is the FULL Ollama envelope: {"model":..., "response":"...", "done":..., ...}
    # When truncated, the outer JSON itself may be broken — fall back to
    # extracting the "response" substring directly.
    try:
        result = json.loads(body)
        return str(result.get("response") or "")
    except json.JSONDecodeError:
        # Try to pull "response":"<...>" via raw scan even from broken JSON.
        m = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
        if m:
            # JSON-unescape the captured group.
            try:
                return json.loads('"' + m.group(1) + '"')
            except json.JSONDecodeError:
                return m.group(1)
        return ""


__all__ = ["classify_atoms"]
