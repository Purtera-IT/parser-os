"""Bounded v2 auxiliary contexts. Forecast execution is deliberately unsupported.

Provenance is a reviewed declaration, not a semantic certificate inferred by
this module. A v2 artifact describes exactly ONE physical source context.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

VERSION = 2
LITERAL_HEADS = frozenset({
    "col:label_type", "col:about", "col:supplier", "col:admission", "read:doc_kind",
    "read:hours_stated", "read:labor_hours", "read:crew_size", "read:equipment_qty",
    "read:display_size", "read:run_length", "read:work_window", "read:start_date",
    "read:site", "read:address_level", "read:price_basis", "read:rate", "read:rate_for",
    "read:billing_type",
    "read:stage_std", "read:deal_stage",
})
NOISE_HEADS = frozenset({"col:label_type", "col:admission", "read:noise_class"})
PROFILES = {"source_local_literal": LITERAL_HEADS, "source_local_noise": NOISE_HEADS}
MAX_CONTEXT_ATOMS = 2048
DIRECT_LOSSES = frozenset({"heads", "numbers", "literals"})
LOSS_TERMS = frozenset("""absence adversary card_alone card_contrast changes citations claims
clause_use co_why_alone co_why_contrast coherence conduct deal_numbers derived_cards
described_alone entities entity_swap explanation_only governs heads hindsight judgment_rationale
judgment_why_alone judgments keyword_dropout layout_lock layout_why link_why_alone link_why_contrast
literals meaning_flip near_misses negatives node_pointers number_why_alone numbers pointers policy_l1
relations residual rule_links table_kind variance why_align why_alone why_contrast why_echo
why_sufficiency why_swap""".split())
LOSS_CONFIG_FIELDS = frozenset("""card_cap company_r_grad described_alone_p free_h_out head
keyword_dropout_broad keyword_dropout_cap keyword_dropout_p layout_why_cap layout_why_p link_why_hard_negative near_miss_margin space
why_hard_negative why_swap_cap""".split())


def validate_loss_census(weights_type):
    if set(weights_type.__dataclass_fields__) != LOSS_TERMS | LOSS_CONFIG_FIELDS:
        raise ValueError("unclassified loss/config field: auxiliary disposition must be reviewed")


def digest(value):
    def encode(v):
        if hasattr(v, "tolist"):
            return v.tolist()
        raise TypeError(type(v).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=encode).encode()).hexdigest()


def source_fingerprint(atom):
    """Fingerprint the exact source occurrence before projection, not just its text."""
    return digest(atom)


def validate_profile(profile):
    if (not isinstance(profile, dict) or not isinstance(profile.get("task_id"), str)
            or profile["task_id"] not in PROFILES):
        raise ValueError("unsupported/uncertified task profile; causal forecast execution is not supported")
    if (set(profile) != {"task_id", "execute", "as_of", "context", "allow_heads", "target_provenance"}
            or profile["execute"] is not True or profile["as_of"] is not None):
        raise ValueError("auxiliary task must execute with no temporal/as_of claim and exact v2 fields")
    context = profile["context"]
    if (not isinstance(context, dict)
            or not {"document_id", "source_keys", "source_fingerprints", "evidence_ref"} <= set(context)
            or set(context) - {"document_id", "source_keys", "source_fingerprints", "evidence_ref", "observed_stage_evidence"}
            or not isinstance(context["document_id"], str) or not context["document_id"]
            or not isinstance(context["evidence_ref"], str) or not context["evidence_ref"]):
        raise ValueError("task needs an explicit source-local context and evidence reference")
    keys = context["source_keys"]
    fingerprints = context["source_fingerprints"]
    if (not isinstance(keys, list) or not keys or len(keys) > MAX_CONTEXT_ATOMS
            or not all(isinstance(k, str) and k for k in keys) or len(set(keys)) != len(keys)
            or not isinstance(fingerprints, dict) or set(fingerprints) != set(keys)
            or not all(isinstance(h, str) and len(h) == 64
                       and all(c in "0123456789abcdef" for c in h) for h in fingerprints.values())):
        raise ValueError("task needs exact bounded source keys and fingerprints")
    stages = context.get("observed_stage_evidence", {})
    if (not isinstance(stages, dict) or not set(stages) <= set(keys)
            or not all(isinstance(v, str) and v for v in stages.values())):
        raise ValueError("observed source stage needs per-occurrence evidence references")
    from .schema import LINE, RELATION, load_schema

    by = load_schema().by_key()
    heads = profile["allow_heads"]
    if (not isinstance(heads, list) or not heads
            or not all(isinstance(k, str) for k in heads) or len(set(heads)) != len(heads)
            or not all(k in PROFILES[profile["task_id"]]
                       and k in by and by[k].universal and by[k].size == LINE
                       and by[k].kind != RELATION for k in heads)):
        raise ValueError("task head allowlist is unsupported; no forecast/deal/group heads are eligible")
    targets = profile["target_provenance"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("task needs explicit per-target source provenance")
    seen = set()
    for t in targets:
        if (not isinstance(t, dict)
                or set(t) != {"source_key", "head", "evidence_keys", "evidence_ref"}
                or not isinstance(t["source_key"], str) or not isinstance(t["head"], str)
                or t["source_key"] not in keys or t["head"] not in heads
                or not isinstance(t["evidence_keys"], list)
                or not all(isinstance(k, str) for k in t["evidence_keys"])
                or t["source_key"] not in t["evidence_keys"]
                or not set(t["evidence_keys"]) <= set(keys)
                or not isinstance(t["evidence_ref"], str) or not t["evidence_ref"]
                or (t["source_key"], t["head"]) in seen):
            raise ValueError("invalid or cross-context target provenance")
        seen.add((t["source_key"], t["head"]))
    return seen


def project_blob(blob, atoms):
    """Physically select context before labels, derivation, features or windows."""
    contract = blob.get("supervision_holds") or {}
    if (set(contract) != {"version", "groups", "task_profile"}
            or type(contract["version"]) is not int or contract["version"] != VERSION):
        raise ValueError("unsupported v2 task contract")
    p = copy.deepcopy(contract["task_profile"])
    targets = validate_profile(p)
    context = p["context"]
    by_key = {a.get("label_key"): a for a in atoms}
    if len(by_key) != len(atoms):
        raise ValueError("duplicate task source occurrences")
    selected = []
    for key in context["source_keys"]:
        a = by_key.get(key)
        if (a is None or a.get("doc_id") != context["document_id"]
                or source_fingerprint(a) != context["source_fingerprints"][key]):
            raise ValueError("task source missing, cross-artifact or fingerprint changed")
        # Positive input allowlist: no embedded labels, promoted/global readings,
        # rewrites or extra source-supervision payload can reach the Atom.
        a = {k: copy.deepcopy(v) for k, v in a.items() if k in {
            "label_key", "atom_id", "text", "entered_at", "doc_id", "doc_kind", "section", "order",
            "speaker_role", "speaker_side", "speaker_org", "doc_direction", "image_pixels",
            "bbox", "table_row", "table_col", "row_span", "col_span", "loss_weight"}}
        a["stage_at"] = (by_key[key].get("stage_at") or "") if key in context.get("observed_stage_evidence", {}) else ""
        if key in context.get("observed_stage_evidence", {}) and not a["stage_at"]:
            raise ValueError("observed stage admission names a missing source stage")
        a["view"] = ""
        selected.append(a)
    keys = set(context["source_keys"])
    result = copy.deepcopy(blob)
    from .schema import load_schema
    from .data import excluded, select_labels
    by = load_schema().by_key()
    labels = []
    # Resolve conflicts/person precedence and all existing source exclusions
    # before removing notes/metadata; provenance never overrides an exclusion.
    chosen = select_labels([r for r in result.get("labels") or [] if r.get("label_key") in keys])
    for row in chosen.values():
        key = row.get("label_key")
        if key not in keys:
            continue
        if row.get("promoted_from"):
            raise ValueError("promoted target is not certified source-local supervision")
        clean = {k: row[k] for k in ("id", "label_key", "labeler", "purpose", "supersedes", "weight_tier") if k in row}
        if excluded(row):
            clean["weight_tier"] = "exclude"
        clean["reads_set"] = {}
        for source, head in targets:
            if source != key:
                continue
            opp = by[head]
            if opp.source == "read":
                if opp.field in (row.get("reads_set") or {}):
                    clean["reads_set"][opp.field] = row["reads_set"][opp.field]
            elif opp.field in row:
                clean[opp.field] = row[opp.field]
        labels.append(clean)
    result.update(atoms=selected, labels=labels, links=[], judgments=[], deal_answers=[],
                  missed_history=[], outcome={}, company_policy="")
    result["supervision_holds"] = {"version": 1, "groups": copy.deepcopy(contract["groups"])}
    return result, selected, p


def atom_input_digest(atom):
    from dataclasses import asdict
    # copy_of (C3_COPY_OF) joins the digest only when a row names one, so a
    # profile materialized before it keeps its digest.
    return digest({k: v for k, v in asdict(atom).items()
                   if k not in {"label", "row_weight", "loss_weight"}
                   and not (k == "copy_of" and not v)})


def supervision_digest(batch):
    from dataclasses import asdict
    return digest({"keys": batch.keys, "targets": batch.targets,
                   "numbers": batch.numbers_target, "sets": batch.multi_targets,
                   "literals": [asdict(t) for t in batch.literals], "head_scale": batch.head_scale})


def validate_deal(deal):
    """Windowing may subset inputs or strip labels, never add/rewrite context."""
    validate_profile(deal.task_profile)
    if deal.task_profile_digest != digest(deal.task_profile) or not deal.task_atom_digests:
        raise ValueError("task must originate in the v2 materializing loader")
    for atom in deal.atoms:
        if atom_input_digest(atom) != deal.task_atom_digests.get(atom.key):
            raise ValueError("task source context changed before featurization")
        # Windows may remove a label to make an overlap context-only, but
        # cannot rewrite or restore supervision from a different source row.
        if atom.label is not None and digest(atom.label) != deal.task_label_digests.get(atom.key):
            raise ValueError("task source supervision changed before featurization")


def constrain_batch(batch, profile, schema):
    """Explicit closure: only reviewed per-line direct targets/literals remain."""
    eligible = validate_profile(profile)
    by = schema.by_key()
    batch.task_profile = copy.deepcopy(profile)
    permitted = {(batch.keys.index(k), h) for k, h in eligible if k in batch.keys}
    for key, col in batch.targets.items():
        for i in range(len(col)):
            opp = by[key]
            if (i, key) not in permitted or (col[i] >= 0 and opp.answers[col[i]].value == "_absent"):
                col[i] = -100
    for key, col in batch.numbers_target.items():
        for i in range(len(col)):
            if (i, key) not in permitted:
                col[i] = None
    for key, col in batch.multi_targets.items():
        for i in range(len(col)):
            if ((i, key) not in permitted or (col[i] is not None and any(
                    by[key].answers[v].value == "_absent" for v in col[i]))):
                col[i] = None
    batch.head_scale = {o.key: (batch.head_scale.get(o.key, 1.) if o.key in profile["allow_heads"] else 0.)
                        for o in schema.opportunities}
    batch.literals = [t for t in batch.literals if (t.line, t.key) in permitted
                      and batch.head_scale.get(t.key, 0.)]
    batch.why = [None] * len(batch)
    batch.policy_note = [None] * len(batch)
    batch.why_weights = [0.] * len(batch)
    batch.field_notes = [{} for _ in batch.keys]
    for name in ("edges", "link_elsewhere", "link_ignored", "negatives", "parser_targets", "deal_numbers", "outcome"):
        setattr(batch, name, {})
    for name in ("judged", "reason_notes", "unplaced_notes", "edge_notes", "rule_links", "flips",
                 "near_misses", "twins", "layout_rewrites"):
        setattr(batch, name, [])
    batch.hint_lines = [[] for _ in batch.keys]
    batch.entities = [[] for _ in batch.keys]
    batch.changes = [None] * len(batch)
    batch.company_policy = ""
    batch.context = {k: [None] * len(batch) for k in batch.context}
    batch.task_input_digest = digest({"profile": batch.task_profile, "inputs": batch.inputs()})
    batch.task_supervision_digest = supervision_digest(batch)
    validate_batch(batch)
    return batch


def validate_batch(batch):
    """Refuse mutable inputs/targets that bypass the materialized task contract."""
    p = batch.task_profile
    eligible = validate_profile(p)
    if not set(batch.keys) <= set(p["context"]["source_keys"]):
        raise ValueError("task contains context outside its admitted source keys")
    if batch.task_input_digest != digest({"profile": p, "inputs": batch.inputs()}):
        raise ValueError("task input context changed after physical materialization")
    if batch.task_supervision_digest != supervision_digest(batch):
        raise ValueError("task supervision/holds changed after materialization")
    for key, col in batch.targets.items():
        if any(v != -100 and (batch.keys[i], key) not in eligible for i, v in enumerate(col)):
            raise ValueError("task contains unapproved head supervision")
    for key, col in batch.numbers_target.items():
        if any(v is not None and (batch.keys[i], key) not in eligible for i, v in enumerate(col)):
            raise ValueError("task contains unapproved numeric supervision")
    for key, col in batch.multi_targets.items():
        if any(v is not None and (batch.keys[i], key) not in eligible for i, v in enumerate(col)):
            raise ValueError("task contains unapproved set supervision")
    from .schema import load_schema
    by = load_schema().by_key()
    for key, col in batch.multi_targets.items():
        if any(v is not None and any(by[key].answers[j].value == "_absent" for j in v) for v in col):
            raise ValueError("task contains absence set supervision")
    for key, col in batch.targets.items():
        if any(v >= 0 and by[key].answers[v].value == "_absent" for v in col):
            raise ValueError("task contains absence supervision")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1
           or (key not in p["allow_heads"] and v != 0) for key, v in batch.head_scale.items()) \
            or any(batch.head_scale.get(key, 1.) != 0 for key in by if key not in p["allow_heads"]):
        raise ValueError("task head scales escaped allowlist")
    if (batch.judged or batch.deal_numbers or batch.edges or batch.edge_notes or batch.reason_notes
            or batch.unplaced_notes or any(batch.field_notes) or any(batch.why) or any(batch.policy_note)
            or batch.rule_links or batch.flips or batch.negatives or any(batch.entities) or any(batch.hint_lines)
            or batch.outcome or batch.parser_targets or batch.near_misses or batch.twins
            or batch.layout_rewrites or any(batch.changes) or batch.company_policy
            or any(batch.why_weights) or batch.__dict__.get("_table_facts")):
        raise ValueError("task contains dependent auxiliary/forecast supervision")
    if any((batch.keys[t.line], t.key) not in eligible for t in batch.literals):
        raise ValueError("task contains unapproved literal supervision")


def reject_teacher(batch):
    if getattr(batch, "task_profile", None) is not None or getattr(batch, "task_input_digest", ""):
        raise ValueError("teacher/lesson execution is unsupported for isolated auxiliary tasks")
