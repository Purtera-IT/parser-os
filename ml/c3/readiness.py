"""Read-only, fail-closed supervision preflight; no model or training imports.

An export is not a training consumer. This report distinguishes usable
targets, privileged rationale and deliberately deferred tab fields. Reports
contain counts and codes, never customer text or the raw supervision packet.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .data import (DealExample, _Lines, _derived, _is_a_person, _judged_lines,
                   excluded, field_note_targets, is_deal_kit, select_labels, judgment_query, judgment_subject, question_field_targets)
from .schema import CAUSAL_RELATIONS, LINE, NUMBER, PAIR, RELATION, Schema, load_schema


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def validate_split(manifest: dict[str, Any]) -> list[str]:
    """Frozen deal and source-group separation, with untouched final holdout."""
    errors = []
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for split in ("train", "validation", "holdout"):
        rows = manifest.get(split)
        if not isinstance(rows, list) or not rows:
            errors.append(f"missing_{split}")
            continue
        for row in rows:
            if not isinstance(row, dict):
                errors.append("invalid_manifest_row")
                continue
            deal = row.get("deal_id")
            if not deal or deal in seen:
                errors.append("duplicate_or_missing_deal")
            seen.add(deal)
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("snapshot_sha256") or "")):
                errors.append("missing_snapshot_hash")
            if row.get("audit_status") != "verified":
                errors.append("unverified_deal")
            if not isinstance(row.get("source_groups"), list) or not row["source_groups"]:
                errors.append("missing_source_groups")
            # Related versions / duplicate source documents travel together.
            for group in row.get("source_groups", []):
                if group in groups and groups[group] != split:
                    errors.append("source_group_leakage")
                groups[group] = split
    if manifest.get("holdout_used_for_tuning") is not False:
        errors.append("holdout_not_certified_untouched")
    return sorted(set(errors))


def audit_contract(blob: dict[str, Any], atoms: list[dict[str, Any]], schema: Schema, *,
                   deal_id: str, max_atoms: int = 4096) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    classes: dict[str, Counter[str]] = {}
    fields: dict[str, Counter[str]] = {}
    by_key = schema.by_key()
    universal = frozenset(o.field for o in schema.opportunities if o.source == "read" and o.universal)
    note_fields = field_note_targets(schema)
    auxiliary = {"why_author", "changes", "entity_keys", "follows_rule", "exception_to"}
    label_metadata = {"id", "deal_id", "label_key", "labeler", "atom_id", "compile_id",
                      "created_at", "updated_at", "saved_at", "labeled_at", "labelled_at",
                      "text", "origin", "doc_id", "doc_kind", "artifact_id", "section", "order",
                      "entered_at", "source_date", "consumer", "correction_id"}
    label_auxiliary = {"note", "weight_tier", "hint_refs", "entity_keys", "follows_rules",
                       "exception_to", "reads_shown", "rejected_reads", "rejected", "parser_type"}

    def account(key: str, status: str):
        fields.setdefault(key, Counter())[status] += 1

    def target(key: str, value: Any):
        targets[key] += 1
        classes.setdefault(key, Counter())[str(value)] += 1

    try:
        selected = select_labels(blob.get("labels", []))
        deal = DealExample.from_training_blob(blob, atoms, deal_id=deal_id)
    except (ValueError, KeyError, TypeError):
        return {"deal_id": deal_id, "ready": False, "blockers": {"invalid_or_conflicting_labels": 1},
                "snapshot_sha256": digest({"blob": blob, "atoms": atoms})}
    lines = _Lines(deal.atoms)
    if len(deal.atoms) > max_atoms:
        blockers["dense_attention_budget_exceeded"] += 1
    used: dict[tuple[str, tuple[int, ...]], int] = {}
    for label in selected.values():
        if excluded(label):
            counts["excluded_labels"] += 1
            continue
        counts["selected_labels"] += 1
        if str(label["label_key"]) not in lines.by_key:
            blockers["label_without_source_occurrence"] += 1
            continue
        doc_kind = deal.atoms[lines.by_key[str(label["label_key"])]].doc_kind
        derived = _derived(label, universal, is_deal_kit(doc_kind))
        for name, value in (derived.get("reads_set") or {}).items():
            if value in (None, "", []):
                continue
            opp = schema.of_field("read", name)
            if name in note_fields:
                account(f"read:{name}", "rationale")
            elif name in auxiliary:
                account(f"read:{name}", "auxiliary")
            elif opp is None:
                account(f"read:{name}", "deferred")
                blockers[f"unconsumed_read:{name}"] += 1
            else:
                values = value if isinstance(value, list) else [value]
                if len(values) > 1 and not opp.multiple:
                    blockers[f"unexpected_multi_value:{opp.key}"] += 1
                for item in values:
                    if opp.index(item) is None:
                        blockers[f"unknown_answer:{opp.key}"] += 1
                    elif opp.kind == NUMBER:
                        try:
                            n = float(str(item).replace(",", "").split()[0])
                            import math
                            if not math.isfinite(n) or n < 0:
                                raise ValueError()
                        except (ValueError, TypeError, IndexError):
                            blockers[f"invalid_number:{opp.key}"] += 1
                            continue
                    ix = opp.index(item)
                    if ix is not None:
                        target(opp.key, opp.answers[ix].value)
                account(opp.key, "direct")
        for opp in schema.select():
            if opp.source != "column":
                continue
            value = derived.get(opp.field)
            if value in (None, "", []):
                continue
            values = value if isinstance(value, list) else [value]
            if len(values) > 1 and not opp.multiple:
                blockers[f"unexpected_multi_value:{opp.key}"] += 1
            for item in values:
                ix = opp.index(item)
                if ix is None:
                    blockers[f"unknown_answer:{opp.key}"] += 1
                else:
                    target(opp.key, opp.answers[ix].value)
                    if not opp.multiple:
                        used[(opp.key, (lines.by_key[str(label["label_key"])],))] = ix
        # Account for top-level supervision too, rather than accepting a rich
        # source row while silently dropping its non-registry fields.
        columns = {o.field for o in schema.opportunities if o.source == "column"}
        for name, value in label.items():
            if value in (None, "", [], {}) or name in columns or name == "reads_set":
                continue
            if name in label_metadata:
                account(f"label:{name}", "metadata")
            elif name in label_auxiliary:
                account(f"label:{name}", "rationale" if name == "note" else "auxiliary")
            else:
                account(f"label:{name}", "deferred")
                blockers[f"unconsumed_label_field:{name}"] += 1

    for j in deal.judgments:
        if not _is_a_person(j.get("labeler")) or excluded(j):
            counts["excluded_or_proposal_cards"] += 1
            continue
        counts["saved_cards"] += 1
        head, verdict = str(j.get("head") or ""), str(j.get("verdict") or "")
        key = f"jdg:{head}"
        if head in schema.judgment_aliases:
            key, mapping = schema.judgment_aliases[head]
            verdict = mapping.get(verdict, "")
        opp = by_key.get(key)
        if opp is None:
            if head == "rule":
                blockers["separate_rule_export_required"] += 1
                account(key, "external_pending")
            else:
                blockers[f"unconsumed_tab:{head}"] += 1
                account(key, "deferred")
            continue
        ix, idx = opp.index(verdict), judgment_subject(j, opp.size, lines)
        if ix is None:
            blockers[f"unknown_verdict:{head}"] += 1
            continue
        if idx is None:
            blockers[f"unresolved_occurrence:{head}"] += 1
            continue
        if any(excluded(deal.atoms[i].label) for i in idx) and opp.size in (LINE, PAIR):
            counts["cards_on_excluded_lines"] += 1
            continue
        identity = (key, idx, judgment_query(j), str(j.get("target_key") or "")) if judgment_query(j) else (key, idx)
        if identity in used:
            # Equal answers still lose card-specific rationale in the current
            # line-target consumer. Flag rather than silently count it twice.
            code = "target_conflict" if used[identity] != ix else "colliding_card_rationale"
            blockers[f"{code}:{head}"] += 1
        used[identity] = ix
        target(key, verdict)
        account(key, "direct")
        consumed = set()
        if judgment_query(j):
            try:
                for field_key, indexes in question_field_targets(j, schema):
                    consumed.add(by_key[field_key].field)
                    target(field_key, "set" if by_key[field_key].multiple else by_key[field_key].answers[indexes[0]].value)
                    account(f"card:{by_key[field_key].field}", "direct")
            except ValueError:
                blockers[f"unknown_card_field_value:{head}"] += 1
        if any(v not in (None, "", []) and k not in consumed for k, v in (j.get("fields") or {}).items()):
            blockers[f"unconsumed_card_fields:{head}"] += 1
    for name in ("deal_answers", "missed_history"):
        if blob.get(name):
            blockers[f"unconsumed_{name}"] += 1
    relations = {o.field for o in schema.opportunities if o.kind == RELATION}
    for src, dst, relation in deal.edges:
        if relation not in relations:
            blockers[f"unconsumed_relation:{relation}"] += 1
        i, j = lines.by_key.get(src), lines.by_key.get(dst)
        if i is not None and j is not None and relation in CAUSAL_RELATIONS and i <= j:
            blockers[f"masked_causal_relation:{relation}"] += 1
    for code in ("unresolved", "card_no_line", "self_links"):
        if deal.link_stats.get(code):
            blockers[f"evidence_{code}"] += deal.link_stats[code]
    if targets.get("jdg:image"):
        blockers["vision_input_not_implemented"] += 1
    return {"deal_id": deal_id, "ready": not blockers, "counts": dict(counts),
            "blockers": dict(sorted(blockers.items())), "targets": dict(sorted(targets.items())),
            "classes": {k: dict(v) for k, v in sorted(classes.items())},
            "field_accounting": {k: dict(v) for k, v in sorted(fields.items())},
            "snapshot_sha256": digest({"blob": blob, "atoms": atoms}),
            "schema_sha256": digest([asdict(o) for o in schema.opportunities])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blob", type=Path, required=True)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--deal-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path)
    args = p.parse_args()
    report = audit_contract(json.loads(args.blob.read_text()), json.loads(args.atoms.read_text()),
                            load_schema(), deal_id=args.deal_id)
    manifest = json.loads(args.split_manifest.read_text()) if args.split_manifest else {}
    errors = validate_split(manifest) if manifest else ["missing_split_manifest"]
    rows = [r for split in ("train", "validation", "holdout") for r in manifest.get(split, [])
            if isinstance(r, dict) and r.get("deal_id") == args.deal_id]
    if len(rows) != 1 or rows[0].get("snapshot_sha256") != report["snapshot_sha256"]:
        errors.append("snapshot_not_pinned_in_manifest")
    report["split_blockers"] = errors
    report["ready"] = report["ready"] and not errors
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
