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
import sys
import time
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

    # Tier 9 — procurement / solicitation (RFP/RFQ/ITB)
    "deadline": {
        "desc": "A hard date/time the bidder must meet: proposal/bid due date, questions-due date, pre-bid meeting, site visit. NOT a project phase end (that is milestone_phase).",
        "fields": ["kind", "date", "time", "location"],
    },
    "submission_req": {
        "desc": "A required element of the proposal submission: number of copies, file format, required form (SF330, W-9), page limit, sealed-bid labeling, delivery method.",
        "fields": ["requirement", "value", "mandatory"],
    },
    "eval_criterion": {
        "desc": "A scored evaluation/award criterion with its weight: 'Technical approach 40%', 'Price 30%', 'Past performance 20%'. One per criterion.",
        "fields": ["criterion", "weight", "max_points", "basis"],
    },
    "bonding_insurance": {
        "desc": "A bonding or insurance requirement: bid bond, performance bond, payment bond, general liability limit, workers comp, professional liability.",
        "fields": ["kind", "amount_or_percent", "condition"],
    },
    "contract_term": {
        "desc": "A contractual term governing the resulting agreement: contract length, renewal/option years, payment terms (net 30), warranty period, retainage, liquidated-damages clause reference.",
        "fields": ["term_kind", "value", "detail"],
    },
    "addendum_qa": {
        "desc": "An addendum, amendment, or question-and-answer item issued during the solicitation that modifies or clarifies the original documents.",
        "fields": ["reference", "issued_date", "summary"],
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


def _atom_type_deflect_enabled() -> bool:
    return os.environ.get(
        "SOWSMITH_ATOM_TYPE_DEFLECT", ""
    ).strip().lower() not in ("", "0", "false", "no", "off")


def _typed_student_enabled() -> bool:
    """Grounded-Extractor #70: let the trained student front the LLM call.

    OFF by default. Flip it ON (in deploy/env) only for relations the shadow
    harness (:mod:`app.core.shadow_eval`) has certified ``ready`` — that is the
    cutover gate. With it off, or with an empty training log, the student
    abstains everywhere and this stage is byte-identical to the LLM-only path.
    """
    return os.environ.get(
        "SOWSMITH_TYPED_STUDENT", ""
    ).strip().lower() not in ("", "0", "false", "no", "off")


# Process-wide student over the warm-base training log. Lazily built; reused so
# the (cached) embeddings of the log's masked rows aren't recomputed per compile.
_TYPED_STUDENT = None
_TYPED_STUDENT_TRIED = False


def _get_typed_student():
    """The atom-type student, or None when no training log is configured."""
    global _TYPED_STUDENT, _TYPED_STUDENT_TRIED
    if _TYPED_STUDENT_TRIED:
        return _TYPED_STUDENT
    _TYPED_STUDENT_TRIED = True
    try:
        from app.core.extractor_student import ExtractionStudent
        from app.core.training_log import get_training_log
        log = get_training_log()
        if log is None:
            return None
        _TYPED_STUDENT = ExtractionStudent(log)  # production: all rows in memory
    except Exception:
        _TYPED_STUDENT = None
    return _TYPED_STUDENT


_ATOM_TYPE_RELATION = "atom_type"
_ATOM_TYPE_INSTRUCTION = (
    "Classify this parsed deal-document atom into the typed taxonomy, or _keep "
    "if no taxonomy entry fits and it should retain its current type."
)


def _atom_type_candidates() -> list[str]:
    return list(_TAXONOMY) + ["_keep"]


def _atom_row_view(atom: Any) -> tuple[list[str], list[Any]] | None:
    """(headers, values) for a per-row table atom, handling both emitted shapes
    (``value._columns``/``value._row`` and ``value.cells``); else ``None``."""
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        return None
    cells = val.get("cells")
    if isinstance(cells, dict) and cells:
        return [str(k) for k in cells], [cells[k] for k in cells]
    cols = val.get("_columns") or val.get("columns")
    row = val.get("_row")
    if cols and row is not None:
        return [str(c) for c in cols], list(row)
    return None


def _atom_bound_text(atom: Any) -> str | None:
    """Render a table row as ``Header: value | Header: value`` so every cell carries
    its column meaning to the classifier. A bare ``Focus/phone room | 6 | ...`` row
    becomes ``Room Type: Focus/phone room | Count: 6 | ...`` — the values stop being
    a meaningless pipe-string. Returns ``None`` when the atom is not a recoverable
    table row (prose/bullet atoms fall through to raw text)."""
    rv = _atom_row_view(atom)
    if not rv:
        return None
    headers, values = rv
    pairs: list[str] = []
    for idx, (k, v) in enumerate(zip(headers, values)):
        k = str(k).strip()
        v = str(v).replace("\n", " ").strip()
        if not v or k.lower().startswith("col_") or k.startswith("_"):
            continue
        # Summary / total rows ("Subtotal", "Recommended fixed fee hours",
        # "Safer bid hours", "Grand Total") are NOT a value of the first column's
        # header — render the label BARE so it reads "Subtotal | Labor Hours: 458.5"
        # instead of "Task Category: Subtotal …".
        if idx == 0 and re.match(
            r"^(sub-?\s*totals?|totals?|grand total|safer bid\b|.*\bfixed fee\b)",
            v, re.I,
        ):
            pairs.append(v)
            continue
        if not k:
            continue
        pairs.append(f"{k}: {v}")
    return " | ".join(pairs)[:600] if pairs else None


def _atom_table_ref(atom: Any) -> str:
    """A stable table-group ref (+ row) for a table-row atom, read from whatever
    the parser already stamped in the locator: pdf -> table ``block_id``,
    xlsx -> ``sheet``, docx/schema -> ``table_index``. Rows of the SAME table
    share this ref, so the head knows the row is structured tabular data (one of
    a group) and downstream can re-assemble the table by grouping on it. Returns
    '' for non-table atoms (prose/bullets)."""
    for r in (getattr(atom, "source_refs", None) or []):
        loc = getattr(r, "locator", None)
        if not isinstance(loc, dict):
            continue
        tid = loc.get("block_id") or loc.get("sheet")
        if tid is None and loc.get("table_index") is not None:
            tid = f"t{loc.get('table_index')}"
        if tid is None:
            continue
        row = loc.get("row")
        if row is None:
            row = loc.get("row_index")
        return f"{str(tid)[:24]} r{row}" if row is not None else str(tid)[:24]
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict) and val.get("_table_idx") is not None:
        return f"t{val.get('_table_idx')} r{val.get('_row_idx')}"
    return ""


def _atom_decide_text(atom: Any) -> str:
    bound = None
    if os.environ.get("SOWSMITH_ATOM_BIND_HEADERS", "1") != "0":
        bound = _atom_bound_text(atom)
    text = bound or _atom_text(atom).replace("\n", " ").strip()
    table_ref = _atom_table_ref(atom)
    section = _atom_section_path(atom)
    if table_ref:
        text = f"{text} [table: {table_ref}]"
    return f"{text} [section: {section}]" if section else text


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
    # Honour the global LLM kill-switch: this stage drives promotion via an
    # /api/generate call, so SOWSMITH_DISABLE_LLM must short-circuit it (it
    # previously ignored the flag and spent ~46s/compile hitting a reachable
    # but slow remote model even in "no-LLM" runs).
    if os.environ.get("SOWSMITH_DISABLE_LLM"):
        return 0

    promotable = [a for a in atoms if _atom_type_str(a) in _PROMOTABLE_FROM]
    if not promotable:
        return 0

    # --- #70 deflect-layer instrumentation ---------------------------------
    # Each cascade layer below removes confidently-_keep atoms from the LLM
    # batch, but historically NONE logged how many — so we could not tell which
    # layer (if any) was firing, nor how the typing wall-time split between
    # deflection and the LLM. This emits one structured event per call so a
    # compile shows, per layer: deflected counts, the residual LLM batch size,
    # promoted count, and total vs LLM-only milliseconds. Pure observability.
    _dfl = {"store": 0, "student": 0, "type_head": 0,
            "contrastive": 0, "rubric_gate": 0}
    _dfl_ms = {"store": 0.0, "student": 0.0, "type_head": 0.0,
               "contrastive": 0.0, "rubric_gate": 0.0, "post": 0.0}
    _dfl_input = len(promotable)
    _t_start = time.perf_counter()
    _t_llm = 0.0

    def _lap():
        return time.perf_counter()

    def _emit_deflect(*, llm_batch: int, promoted: int, reached_llm: bool) -> None:
        try:
            print(json.dumps({
                "event": "typed_atom_deflect",
                "stage": "typed_atom_classification",
                "input": _dfl_input,
                "deflected": dict(_dfl),
                "deflected_total": sum(_dfl.values()),
                "llm_batch": llm_batch,
                "promoted": promoted,
                "reached_llm": reached_llm,
                "total_ms": round((time.perf_counter() - _t_start) * 1000, 1),
                "llm_ms": round(_t_llm * 1000, 1),
                "layer_ms": {k: round(v * 1000, 1) for k, v in _dfl_ms.items()},
            }, ensure_ascii=True), file=sys.stderr)
        except Exception:
            pass

    # upgrade #3: warm-store deflection on the atom-TYPE decision. STORE-ONLY
    # (llm=False) pre-filter: an atom the store CONFIDENTLY classifies ``_keep``
    # (no taxonomy entry fits → keep current type) is dropped from the LLM batch
    # entirely — no round-trip, no value extraction needed. One-sided: the store
    # can only ever REMOVE a promotion candidate (a no-op keep), never fabricate
    # a promotion (those still go to the LLM, which alone synthesizes the value
    # payload). Worst case is a missed promotion (recall, not correctness), and
    # it's PM-correctable. OFF by default — production is byte-identical.
    deflect = _atom_type_deflect_enabled()
    if deflect:
        _t = _lap()
        try:
            from app.core.decide import decide
            kept_by_store = 0
            survivors: list[Any] = []
            cands = _atom_type_candidates()
            for a in promotable:
                d = decide(
                    _ATOM_TYPE_RELATION,
                    _atom_decide_text(a),
                    cands,
                    instruction=_ATOM_TYPE_INSTRUCTION,
                    llm=False,
                )
                if d.source == "store" and d.verdict == "_keep":
                    kept_by_store += 1
                    _dfl["store"] += 1
                    continue
                survivors.append(a)
            promotable = survivors
        except Exception:
            pass
        _dfl_ms["store"] += _lap() - _t
        if not promotable:
            _emit_deflect(llm_batch=0, promoted=0, reached_llm=False)
            return 0

    # Grounded-Extractor #70: STUDENT deflection — the trained head fronts the
    # LLM. Same one-sided contract as the store deflect above: the student may
    # only DROP a candidate it confidently classifies ``_keep`` (no taxonomy
    # entry fits → no value payload needed), never fabricate a promotion (those
    # require the LLM's value synthesis and stay in the batch). Worst case is a
    # missed promotion (recall, PM-correctable), never a wrong write. The
    # student abstains when unsure / log empty / embedder down, so OFF or
    # cold → byte-identical to the LLM-only path. This is the call we are
    # replacing; every confident keep here is one fewer atom in the 98s stage.
    student_deflected = 0
    if _typed_student_enabled():
        _t = _lap()
        student = _get_typed_student()
        if student is not None:
            try:
                cands = _atom_type_candidates()
                survivors = []
                for a in promotable:
                    pred = student.classify(
                        _atom_decide_text(a), _ATOM_TYPE_RELATION, candidates=cands,
                    )
                    if not pred.abstained and pred.label == "_keep":
                        student_deflected += 1
                        _dfl["student"] += 1
                        continue
                    survivors.append(a)
                promotable = survivors
            except Exception:
                pass
        _dfl_ms["student"] += _lap() - _t
        if not promotable:
            _emit_deflect(llm_batch=0, promoted=0, reached_llm=False)
            return 0

    # Grounded-Extractor #70 (partial cutover): the TRAINED, learnable type head
    # ASSIGNS a confident specific type and skips the LLM for that atom — but
    # only for VALUE-LIGHT types (the label is the deliverable; no rich value
    # payload to synthesize). Value-heavy types (commercial_total, milestone,
    # bom_line, quantity, payment_term, stakeholder, site_*) still go to the LLM.
    # Eval-gated + learnable (app.core.type_head): precision-first (~0.92 @
    # conf>=0.85), abstains when unsure, retrains as the log grows. OFF by
    # default; cold/abstain -> byte-identical to the LLM-only path.
    head_deflected = 0
    if os.environ.get("SOWSMITH_TYPE_HEAD_DEFLECT", "").strip().lower() in ("1", "true", "yes", "on"):
        _t = _lap()
        try:
            from app.core.schemas import AtomType as _AT
            from app.core.type_head import load_promoted_head

            _VALUE_LIGHT = {
                "requirement", "exclusion", "contract_term", "deal_metadata",
                "acceptance_criterion", "task", "change_order_rule", "constraint",
                "dependency", "mitigation", "compliance_rule", "submission_req",
                "addendum_qa",
            }
            head = load_promoted_head()
            if head is not None:
                survivors = []
                for a in promotable:
                    res = head.classify(_atom_decide_text(a))
                    if res is not None and res[0] in _VALUE_LIGHT:
                        try:
                            a.atom_type = _AT(res[0])
                            head_deflected += 1
                            _dfl["type_head"] += 1
                            continue
                        except (ValueError, ImportError):
                            pass
                    survivors.append(a)
                promotable = survivors
        except Exception:
            pass
        _dfl_ms["type_head"] += _lap() - _t
        if not promotable:
            _emit_deflect(llm_batch=0, promoted=head_deflected, reached_llm=False)
            return head_deflected

    # Contrastive kNN keep-gate (Layer 2 of the cascade) — confidently-_keep atoms
    # skip the LLM typing call and remain _keep. Guess-free + safe by direction: we
    # only act on a confident _keep verdict (never emit a positive type here), so a
    # wrong abstain just falls through to the LLM as before. Instant-learning store;
    # OFF by default; cold/abstain -> byte-identical to the LLM-only path.
    if os.environ.get("SOWSMITH_CONTRASTIVE_TYPE", "").strip().lower() in ("1", "true", "yes", "on"):
        _t = _lap()
        try:
            from app.core.contrastive_type_knn import load_promoted as _load_cknn

            ck = _load_cknn()
            if ck is not None and ck.mode in ("unified", "gate"):
                verdicts = ck.classify_batch([_atom_decide_text(a) for a in promotable])
                survivors = []
                for a, res in zip(promotable, verdicts):
                    if res is not None and res[0] == "_keep":
                        head_deflected += 1
                        _dfl["contrastive"] += 1
                        continue
                    survivors.append(a)
                promotable = survivors
        except Exception:
            pass
        _dfl_ms["contrastive"] += _lap() - _t
        if not promotable:
            _emit_deflect(llm_batch=0, promoted=head_deflected, reached_llm=False)
            return head_deflected

    # Rubric GATE (the 0.864 bge-base keep-vs-typed classifier) — deflects
    # confidently-_keep atoms off the LLM typing stage; they stay _keep. Guess-free
    # + safe by direction (only acts on a confident _keep verdict). OFF by default;
    # abstains (no-op) if torch/transformers or the model are absent.
    if os.environ.get("SOWSMITH_RUBRIC_GATE", "").strip().lower() in ("1", "true", "yes", "on"):
        _t = _lap()
        try:
            from app.core.rubric_gate import keep_deflect_flags

            flags = keep_deflect_flags([_atom_decide_text(a) for a in promotable])
            if any(flags):
                survivors = []
                for a, deflect in zip(promotable, flags):
                    if deflect:
                        head_deflected += 1
                        _dfl["rubric_gate"] += 1
                        continue
                    survivors.append(a)
                promotable = survivors
        except Exception:
            pass
        _dfl_ms["rubric_gate"] += _lap() - _t
        if not promotable:
            _emit_deflect(llm_batch=0, promoted=head_deflected, reached_llm=False)
            return head_deflected

    if not _ollama_reachable():
        _emit_deflect(llm_batch=0, promoted=0, reached_llm=False)
        return head_deflected

    batch_size = int(os.environ.get("SOWSMITH_TYPED_CLASSIFIER_BATCH", str(DEFAULT_BATCH_SIZE)))
    parallel = int(os.environ.get("SOWSMITH_LLM_PARALLEL", str(DEFAULT_PARALLEL)))
    batches = [promotable[i:i + batch_size] for i in range(0, len(promotable), batch_size)]

    _t_llm0 = time.perf_counter()
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
    _t_llm = time.perf_counter() - _t_llm0
    _llm_batch_size = len(promotable)
    _t_post = _lap()

    promoted = 0
    # upgrade #3: APPLIED outcome per atom (the verdict actually enacted, after
    # hallucination guards) — fed back to the store so the SAME shape deflects
    # next run. We teach the enacted decision, never the raw LLM label, so a
    # rejected ghost teaches "_keep" (what we kept), not the bad promotion.
    applied_verdict: dict[str, str] = {}
    for atom in promotable:
        atom_id = _atom_id(atom)
        if not atom_id or atom_id not in results_by_atom_id:
            continue
        payload = results_by_atom_id[atom_id]
        new_type = payload.get("atom_type")
        if not new_type or new_type == "_keep":
            applied_verdict[atom_id] = "_keep"
            continue
        if new_type not in _TAXONOMY:
            applied_verdict[atom_id] = "_keep"
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
                applied_verdict[atom_id] = "_keep"  # we kept the type
                continue
        try:
            from app.core.schemas import AtomType
            atom.atom_type = AtomType(new_type)
        except (ImportError, ValueError):
            applied_verdict[atom_id] = "_keep"
            continue
        if isinstance(new_value, dict) and new_value:
            existing_value = getattr(atom, "value", None)
            if isinstance(existing_value, dict):
                merged = {**existing_value, **new_value}
                atom.value = merged
            else:
                atom.value = new_value
        applied_verdict[atom_id] = new_type
        promoted += 1

    # Grounded Extractor (#68): log every enacted atom-type verdict as an LLM
    # (silver) training row so the Type head (#70) can one day serve this stage
    # instead of qwen3:14b. Pure logging — no behavior change, no-op unless
    # SOWSMITH_TRAINING_LOG_DB is set. We log the ENACTED verdict (post
    # hallucination-guard), never the raw LLM label, mirroring the self-teach
    # contract above. raw_text is delexicalized into the feature on write.
    try:
        from app.core.training_log import TEACHER_LLM, TrainingRow, log_rows
        _by_id = {_atom_id(a): a for a in promotable}
        _rows = []
        for _aid, _verdict in applied_verdict.items():
            _a = _by_id.get(_aid)
            if _a is None:
                continue
            _rows.append(
                TrainingRow(
                    relation=_ATOM_TYPE_RELATION,
                    label=_verdict,
                    raw_text=_atom_decide_text(_a),
                    label_kind="type",
                    teacher=TEACHER_LLM,
                    confidence=0.9,
                    deal_id=str(getattr(_a, "project_id", "") or ""),
                    project_id=str(getattr(_a, "project_id", "") or ""),
                    provenance={"stage": "typed_atom_classification", "source": "llm_batch"},
                )
            )
        log_rows(_rows)
    except Exception:
        pass

    # upgrade #3: self-teach the enacted type decisions so the store warms and
    # deflects matching shapes next run (gated by the same teacher-cache flag
    # decide() uses for its own LLM tier). Both classes accumulate so the
    # relation head can calibrate; only a confident learned "_keep" ever
    # deflects a candidate out of the LLM batch.
    if deflect and applied_verdict:
        try:
            from app.core.decide import (
                DecisionScope,
                _teacher_cache_enabled,
                get_store,
            )
            store = get_store()
            if (
                store is not None
                and _teacher_cache_enabled()
                and hasattr(store, "learn_from_teacher")
            ):
                by_id = {_atom_id(a): a for a in promotable}
                for atom_id, verdict in applied_verdict.items():
                    a = by_id.get(atom_id)
                    if a is None:
                        continue
                    store.learn_from_teacher(
                        relation=_ATOM_TYPE_RELATION,
                        text=_atom_decide_text(a),
                        verdict=verdict,
                        confidence=0.9,
                        scope=DecisionScope(),
                        instruction=_ATOM_TYPE_INSTRUCTION,
                    )
        except Exception:
            pass

    _dfl_ms["post"] += _lap() - _t_post
    _emit_deflect(llm_batch=_llm_batch_size, promoted=promoted, reached_llm=True)
    return promoted + head_deflected


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
                # xlsx rows carry no heading chain, but the SHEET is their
                # section: it's the structural context a spreadsheet groups by
                # (docx headings ≈ xlsx sheets). Surface it so a BOM / deal-
                # summary row tells the head which sheet it came from.
                sheet = loc.get("sheet")
                if sheet:
                    return str(sheet)[:200]
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
    # Hosted-teacher route (default-off): if TEACHER_API_BASE is set, classify
    # via the OpenAI-compatible client; otherwise use the local Ollama below.
    from app.core import llm_client
    if llm_client.teacher_api_enabled():
        return llm_client.complete(prompt, max_tokens=max_tokens)
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
