"""The super-heads registry: every head, which layer it is in, what it replaces
in the parser and OrbitBrief, whether it is ready, its fallback, and what
labeling it needs. Re-run after every retrain or labeling round:

    python -m ml.c3.heads_registry --holdout-log runC.holdout.log \\
        --validation runC.pt.validation.json --labels 'train/*.json' 010043.json \\
        --out c3_record/heads.json

Layers: ``base`` heads say what a line IS and are the same for any company;
``purtera`` heads say what PURTERA does with it, and another company refits
only that layer. Status is computed, never hand-set:

* ``live``          wired into a parser stage and switched on;
* ``ready``         beats always-guessing-the-common-answer by 10+ points on the
                    unseen holdout (n >= 50), or no baseline but >= 0.85 there
                    (not for links: their exact score is mostly "no link");
* ``one_sided``     its labels are (nearly) all one answer, so it learned "always
                    that": label the other answer too;
* ``thin``          fewer than 100 labels or 5 deals;
* ``no_labels``     never labeled;
* ``weak``          scored on the holdout but not clearly better than guessing;
* ``unproven``      labeled, but not scored on the holdout (too few unseen lines).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

from .schema import load_schema

#: What each head decides in place of a parser stage or an OrbitBrief step,
#: and the fast fallback used until it is ready. Keys are head keys.
STAGES: dict[str, dict] = {
    "col:label_type": {"replaces": ["parser: typed_atom_classification (LLM)", "orbit: brain section/type"],
                       "fallback": "the typing LLM for unsure lines", "live": True},
    "jdg:document_job": {"replaces": ["parser: document_job_scope (LLM judge)"],
                         "fallback": "the judge LLM for conversations the base does not keep", "live": True},
    "read:co_action": {"replaces": ["parser: noise_suppression / admission (lessons)", "orbit: brain keep/reject"],
                       "fallback": "the parser's own type filter (small_talk, chatter)"},
    "read:co_reason": {"replaces": ["orbit: why a line was left out", "deal kit: teach reason chip"],
                       "fallback": "no reason shown"},
    "read:sow_section": {"replaces": ["orbit: brain section placement", "sow: section of each fact"],
                         "fallback": "section by line type"},
    "read:sow_fill": {"replaces": ["orbit: planner SOW template vs filled"], "fallback": "treat as filled"},
    "read:sow_coverage": {"replaces": ["orbit: planner SOW coverage flags"], "fallback": "no coverage flag"},
    "read:noise_class": {"replaces": ["parser: noise_suppression", "orbit: brain filter"],
                         "fallback": "deal_chatter rules"},
    "read:scope_facet": {"replaces": ["orbit: work steps vs schedule vs materials grouping"],
                         "fallback": "group by line type"},
    "deal:primary_service": {"replaces": ["orbit: pack_prior LLM", "orbit: scope_router LLM", "orbit: planner pack status"],
                             "fallback": "parser pack routing (domain pack)"},
    "rel:answers": {"replaces": ["parser: open_question_resolution", "orbit: question queue (answered?)"],
                    "fallback": "key-overlap closing"},
    "rel:contradicts": {"replaces": ["parser: cross_document_conflicts", "orbit: planner review flags"],
                        "fallback": "rule-based conflicts"},
    "jdg:conflict": {"replaces": ["parser: supply/cross-document conflicts"], "fallback": "rule-based conflicts"},
    "rel:same_as": {"replaces": ["parser: semantic_dedup / stakeholder_dedup"], "fallback": "embedding dedup"},
    "read:intake_gap": {"replaces": ["orbit: question queue (which questions)"],
                        "fallback": "parser open_question lines + gap judgments"},
    "col:wants": {"replaces": ["orbit: question queue (chase / confirm / decide)"],
                  "fallback": "parser open_question lines"},
    "read:needed_by": {"replaces": ["orbit: question owner"], "fallback": "project_manager"},
    "jdg:gap": {"replaces": ["orbit: gap neural head (LLM teacher+judge)", "parser: gap_valid lessons"],
                "fallback": "lessons"},
    "jdg:question_answered": {"replaces": ["orbit: question answered?"], "fallback": "key-overlap closing"},
    "read:blocked_on": {"replaces": ["orbit: blocked flags", "parser: deal_signals regex"], "fallback": "deal_signals regex"},
    "read:lead_time": {"replaces": ["orbit: lead-time flags"], "fallback": "regex"},
    "read:superseded": {"replaces": ["orbit: superseded facts", "parser: reconcile"], "fallback": "newest document wins"},
    "jdg:site": {"replaces": ["parser: entity_resolution site fusion", "orbit: site_reality name"], "fallback": "lessons + LLM"},
    "jdg:site_role": {"replaces": ["parser: site_geo_fallback vendor address (LLM)"], "fallback": "lessons + LLM"},
    "jdg:facility": {"replaces": ["parser: site_facility_head"], "fallback": "lessons"},
    "col:supplier": {"replaces": ["parser: bom_owner (lessons)"], "fallback": "lessons"},
    "jdg:task_tier": {"replaces": ["parser: task_tier_classification"], "fallback": "rules"},
    "read:task_owner_role": {"replaces": ["parser: work_order owner", "runbook: owner"], "fallback": "LLM"},
    "jdg:billing_type": {"replaces": ["parser: commercial_terms", "orbit: commercial head (LLM)"], "fallback": "rules"},
    "jdg:commercial": {"replaces": ["orbit: commercial head (LLM)", "deal kit: billing model"], "fallback": "rules"},
    "jdg:tier": {"replaces": ["deal kit: tier"], "fallback": "kit default"},
    "read:co_deal_kit_route": {"replaces": ["deal kit: which parser a line trains"], "fallback": "none"},
    "read:co_hours_estimate": {"replaces": ["deal kit: hours"], "fallback": "kit rules"},
    "read:co_crew_rule": {"replaces": ["deal kit: crew"], "fallback": "kit rules"},
    "read:co_tech_base_miles": {"replaces": ["deal kit: travel"], "fallback": "kit rules"},
    "read:scope_category": {"replaces": ["deal kit: scope grouping"], "fallback": "by type"},
    "read:delivery_field": {"replaces": ["runbook: field"], "fallback": "LLM"},
    "col:admission": {"replaces": ["parser: span_admission (lessons)"], "fallback": "lessons"},
}

#: What to label, by status.
TODO = {
    "one_sided": "label the OTHER answer too (it learned 'always {top}')",
    "thin": "label it on more deals (aim: 10+ deals, 300+ lines)",
    "no_labels": "start labeling it",
    "unproven": "label it on the next unseen deal so it can be scored",
    "weak": "label disagreements: where it is wrong, correct it (it is near guess level)",
    "ready": "keep labeling as you go",
    "live": "keep labeling as you go",
}


def _facts(holdout_log: str, validation: str, label_files: list[str]):
    log = Path(holdout_log).read_text(encoding="utf-8")
    hold = log[log.rfind("\nholdout\n"):]
    val = json.loads(Path(validation).read_text(encoding="utf-8"))
    vl = val.get("open_links") or val.get("lines") or {}
    cnt, deals, mix = collections.Counter(), collections.defaultdict(set), collections.defaultdict(collections.Counter)
    for f in label_files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        dn = Path(f).stem[:6]
        for lab in d.get("labels") or []:
            for k, v in (lab.get("reads_set") or {}).items():
                if v not in (None, "", []):
                    cnt["read:" + k] += 1; deals["read:" + k].add(dn); mix["read:" + k][str(v)[:30]] += 1
            for k, col in (("label_type", "col:label_type"), ("about", "col:about"), ("wants", "col:wants"),
                           ("supplier", "col:supplier")):
                if lab.get(k):
                    cnt[col] += 1; deals[col].add(dn); mix[col][str(lab[k])[:30]] += 1
        for k in d.get("links") or []:
            h = "rel:" + str(k.get("relation")); cnt[h] += 1; deals[h].add(dn)
        for j in d.get("judgments") or []:
            h = "jdg:" + str(j.get("head")); cnt[h] += 1; deals[h].add(dn)
        for a in d.get("deal_answers") or []:
            for k in ("primary_service", "declared_site_count"):
                if a.get(k) not in (None, ""):
                    cnt["deal:" + k] += 1; deals["deal:" + k].add(dn)
    return hold, vl, cnt, deals, mix


def build(holdout_log: str, validation: str, label_files: list[str]) -> list[dict]:
    schema = load_schema()
    hold, vl, cnt, deals, mix = _facts(holdout_log, validation, label_files)
    out = []
    for o in schema.opportunities:
        if o.key.endswith(":reason") and o.key != "read:co_reason":
            continue
        m = re.search(rf"\n\s+{re.escape(o.key)} exact=([\d.]+) n=(\d+)(?: majority=([\d.]+))?", hold)
        h = {"exact": float(m.group(1)), "n": int(m.group(2)),
             "guess": float(m.group(3)) if m and m.group(3) else None} if m else None
        n, nd, top = cnt.get(o.key, 0), len(deals.get(o.key, ())), mix.get(o.key)
        share = (top.most_common(1)[0][1] / max(1, sum(top.values()))) if top else 0.0
        stage = STAGES.get(o.key, {})
        if stage.get("live"):
            status = "live"
        elif n == 0:
            status = "no_labels"
        elif top and share >= 0.97 and o.kind != "number":
            status = "one_sided"
        elif n < 100 or nd < 5:
            status = "thin"
        elif h and h["n"] >= 50 and ((h["guess"] is not None and h["exact"] >= h["guess"] + 0.10)
                                     or (h["guess"] is None and h["exact"] >= 0.85 and o.kind != "relation")):
            status = "ready"
        elif h and h["n"] >= 50 and not (o.kind == "relation" and h["guess"] is None):
            status = "weak"
        else:
            status = "unproven"
        todo = TODO[status].format(top=top.most_common(1)[0][0] if top else "")
        out.append({
            "key": o.key, "layer": "base" if o.universal else "purtera",
            "size": o.size, "kind": o.kind, "answers": [a.value for a in o.answers][:30],
            "status": status, "replaces": stage.get("replaces", []), "fallback": stage.get("fallback", ""),
            "holdout": h, "validation": ({"exact": round(vl[o.key]["exact"], 3), "n": int(vl[o.key]["n"])}
                                         if o.key in vl else None),
            "labels": n, "label_deals": nd, "label_mix": dict(top.most_common(4)) if top else {},
            "labeling_task": todo,
        })
    order = {"live": 0, "ready": 1, "weak": 2, "unproven": 3, "one_sided": 4, "thin": 5, "no_labels": 6}
    out.sort(key=lambda x: (x["layer"] != "base", order[x["status"]], -len(x["replaces"]), x["key"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--holdout-log", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="training-format deal files (globs ok)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    files = sorted({f for g in a.labels for f in glob.glob(g)} | {f for f in a.labels if Path(f).is_file()})
    reg = build(a.holdout_log, a.validation, files)
    Path(a.out).write_text(json.dumps({"version": 1, "heads": reg}, indent=1), encoding="utf-8")
    c = collections.Counter((h["layer"], h["status"]) for h in reg)
    print(json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(c.items())}, indent=1))


if __name__ == "__main__":
    main()
