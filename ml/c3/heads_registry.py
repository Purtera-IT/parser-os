"""The super-heads registry: every head, which layer it is in, what it replaces
in the parser and OrbitBrief, whether it is ready, its fallback, and what
labeling it needs. Re-run after every retrain or labeling round:

    python -m ml.c3.heads_registry --holdout-log runC.holdout.log \\
        --validation runC.pt.validation.json --labels 'train/*.json' 010043.json \\
        --out c3_record/heads.json

Test A onward the holdout evidence is the graded checkpoint (``--graded``), not a training log:

    python -m ml.c3.heads_registry \
        --graded tables_<NAME>/per_deal_tables.json --gate gate_score.json \
        --items golden/<NAME>/items.json heldout/<NAME>/items.json \
        --labels '<train dirs>/*.json' --out c3_record/heads.json

The scored set (``--graded-set``, default heldout7) gives each head's accuracy (``exact``) and live's
on the same items (``live``); the majority answer of the same gold is ``guess``; the swap gate's pass is
``swap`` (recorded, and named in the labeling task; it does not change the status). A head the grade
measures worse than live is never ``ready`` (it is ``weak``): new evidence can never hand a decision
to a head that lost to the one in service.
A head the grade does not score keeps the training log's evidence when one is given (``evidence``
says which).

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


def _facts(holdout_log: str | None, validation: str | None, label_files: list[str]):
    log = Path(holdout_log).read_text(encoding="utf-8") if holdout_log else ""
    hold = log[log.rfind("\nholdout\n"):] if log else ""
    val = json.loads(Path(validation).read_text(encoding="utf-8")) if validation else {}
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


def graded(tables: str, gate: str | None = None, items: list[str] = (), *, set_name: str = "heldout7",
           pair: str = "X:live", source: str = "") -> dict[str, dict]:
    """Per head, the graded checkpoint's held-out evidence: ``exact`` (its accuracy), ``live`` (the
    service's on the same items), ``n``, ``verdict`` (paired bootstrap), ``guess`` (majority share of the
    same gold answers in ``items``) and ``swap`` (the swap gate's pass, None when not gated)."""
    t = json.loads(Path(tables).read_text(encoding="utf-8"))["pairs"][pair]
    heads = t["sets"][set_name]["heads"]
    deals = set(t["sets"][set_name].get("deals") or [])
    gold: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for f in items:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for dn, deal in (d.get("deals") or {}).items():
            if deals and dn not in deals:
                continue
            for item, row in ((deal.get("kinds") or {}).get("clean") or {}).items():
                g = row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else None
                if g is None or item.startswith(("num|", "glink|")):
                    continue
                gold[item.split("|")[1]][str(g)] += 1
    swap: dict[str, bool] = {}
    if gate:
        for r in (json.loads(Path(gate).read_text(encoding="utf-8")).get("swap") or {}).get("rows") or []:
            if r.get("gated"):
                swap[r["head"]] = bool(r.get("pass"))
    out = {}
    for k, r in heads.items():
        g = gold.get(k)
        out[k] = {"exact": round(float(r["cand"]), 3), "n": int(r["n"]),
                  "guess": round(g.most_common(1)[0][1] / sum(g.values()), 3) if g else None,
                  "live": round(float(r["live"]), 3), "verdict": r.get("verdict"), "swap": swap.get(k),
                  "evidence": source or f"graded {set_name}"}
    return out


def _worse_than_live(h: dict | None) -> bool:
    """The grade measured this head worse than the service's on the same held-out items (paired
    verdict "worse", or lower accuracy at all)."""
    if not h or h.get("live") is None:
        return False
    return h.get("verdict") == "worse" or h["exact"] < h["live"]


def build(holdout_log: str | None, validation: str | None, label_files: list[str],
          grade: dict[str, dict] | None = None) -> list[dict]:
    schema = load_schema()
    hold, vl, cnt, deals, mix = _facts(holdout_log, validation, label_files)
    grade = grade or {}
    out = []
    for o in schema.opportunities:
        if o.key.endswith(":reason") and o.key != "read:co_reason":
            continue
        m = re.search(rf"\n\s+{re.escape(o.key)} exact=([\d.]+) n=(\d+)(?: majority=([\d.]+))?", hold)
        h = {"exact": float(m.group(1)), "n": int(m.group(2)),
             "guess": float(m.group(3)) if m and m.group(3) else None, "evidence": "holdout log"} if m else None
        if o.key in grade:
            h = dict(grade[o.key])
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
        elif h and h["n"] >= 50 and not _worse_than_live(h) and (
                (h["guess"] is not None and h["exact"] >= h["guess"] + 0.10)
                or (h["guess"] is None and h["exact"] >= 0.85 and o.kind != "relation")):
            status = "ready"
        elif h and h["n"] >= 50 and not (o.kind == "relation" and h["guess"] is None):
            status = "weak"
        else:
            status = "unproven"
        todo = TODO[status].format(top=top.most_common(1)[0][0] if top else "")
        if h and _worse_than_live(h):
            todo = f"worse than live on the held-out grade ({h['exact']} vs {h['live']}): " + todo
        if h and h.get("swap") is False:
            todo += "; fails the swap gate (answers move under name / keyword / layout swaps)"
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
    p.add_argument("--holdout-log", help="a training holdout log (Run C and before)")
    p.add_argument("--validation")
    p.add_argument("--graded", help="per_deal_tables.json of a graded checkpoint (test A onward)")
    p.add_argument("--graded-set", default="heldout7")
    p.add_argument("--gate", help="gate_score.json (its swap rows)")
    p.add_argument("--items", nargs="*", default=[], help="the graded items.json files (gold, for the guess)")
    p.add_argument("--source", default="", help="what the graded evidence is, e.g. 'test A N2pick_ablA heldout7'")
    p.add_argument("--labels", nargs="+", required=True, help="training-format deal files (globs ok)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not (a.holdout_log or a.graded):
        p.error("give --holdout-log or --graded")
    files = sorted({f for g in a.labels for f in glob.glob(g)} | {f for f in a.labels if Path(f).is_file()})
    grade = graded(a.graded, a.gate, a.items, set_name=a.graded_set, source=a.source) if a.graded else None
    reg = build(a.holdout_log, a.validation, files, grade)
    Path(a.out).write_text(json.dumps({"version": 2 if grade else 1, "evidence": a.source or a.holdout_log,
                                       "heads": reg}, indent=1), encoding="utf-8")
    c = collections.Counter((h["layer"], h["status"]) for h in reg)
    print(json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(c.items())}, indent=1))


if __name__ == "__main__":
    main()
