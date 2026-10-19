"""Can a SOW writer rule be retired onto the rubric judge? (rule_retire.py, for jdg:sow_section)

    python -m ml.c3.rule_retire_judge --envelopes '/outside/final_*_env.json' \
        --cache /outside/judge/rule_cache.jsonl --report /outside/rule_retire_judge.json \
        [--skip-deals 000002 000123 ...] [--workers 2]

rule_retire.py scores a refit checkpoint on held-out deals. The judge never
trains, so every deal outside its prompt-tuning set (``--skip-deals``) is held
out for it. For each rule decision about a line's section, party or supplier
(``c3_sow.rule_decisions``), the judge reads the decision's first cited atom
(its text, heading and the deal's job family) and the report says how often
its answer is the rule's (``value``) and how often it is what the rule moved
it away from (``was``). READY = >=95% agreement over >=30 decisions, the same
bar as rule_retire. Keep/drop decisions are not the judge's question and are
left to rule_retire. Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .labels_from_sow import SUPPLIER_OF_OWNER
from .rule_retire import READY_AGREE, READY_MIN

HEADS = ("sow_section", "sow_party", "sow_supplier")


def judged_answer(head: str, verdict: dict) -> str | None:
    """The judge's answer to a rule decision's question."""
    if head == "sow_section":
        return verdict.get("section")
    if head == "sow_party":
        return verdict.get("owner")
    if head == "sow_supplier":
        return SUPPLIER_OF_OWNER.get(str(verdict.get("owner")))
    return None


def score(items: list[tuple[str, dict, dict]]) -> list[dict]:
    """items: (deal, decision, judge verdict). One row per (rule, head)."""
    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for _, d, v in items:
        groups[(str(d.get("rule")), str(d.get("head")))].append((d, v))
    out = []
    for (rule, head), rows in sorted(groups.items()):
        n = len(rows)
        agree = sum(judged_answer(head, v) == d.get("value") for d, v in rows)
        was = sum(judged_answer(head, v) == d.get("was") for d, v in rows if d.get("was"))
        rate = agree / n if n else 0.0
        out.append({"rule": rule, "head": head, "decisions": n, "judge_agrees": round(rate, 3),
                    "judge_says_was": was,
                    "status": "ready" if n >= READY_MIN and rate >= READY_AGREE else "learning",
                    "blocker": "" if n >= READY_MIN and rate >= READY_AGREE else
                    (f"n={n} of {READY_MIN} needed" if n < READY_MIN else f"agreement {rate:.0%} < {READY_AGREE:.0%}")})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--envelopes", nargs="+", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--skip-deals", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    from app.core import sow_rubric_judge as J

    cache: dict[str, dict] = {}
    if os.path.exists(a.cache):
        for line in open(a.cache):
            r = json.loads(line)
            if r.get("section"):
                cache[f"{r['deal']}|{r['id']}"] = r
    lock = threading.Lock()
    asks, items = [], []
    for path in sorted(x for pat in a.envelopes for x in glob.glob(pat)):
        env = json.loads(Path(path).read_text())
        deal = str((env.get("crm") or {}).get("deal_name") or "")[:6] or Path(path).stem
        if deal in a.skip_deals:
            continue
        job = str((env.get("c3_sow") or {}).get("job_family") or "")
        atoms = {str(x.get("id")): x for x in env.get("atoms") or []}
        for d in (env.get("c3_sow") or {}).get("rule_decisions") or []:
            if d.get("head") not in HEADS:
                continue
            aid = str((d.get("atom_ids") or [""])[0])
            atom = atoms.get(aid)
            if not atom:
                continue
            path_ = [str(x) for x in atom.get("section_path") or [] if str(x).strip()]
            line = {"deal": deal, "id": aid, "job": job, "under": path_[-1][:120] if path_ else "",
                    "text": " ".join(str(atom.get("text") or "").split())[:300]}
            items.append((deal, d, line))
            if f"{deal}|{aid}" not in cache:
                asks.append(line)

    def one(line):
        k = f"{line['deal']}|{line['id']}"
        with lock:
            if k in cache:
                return
        v = J.judge(line["text"], job=line["job"], under=line["under"], deal_id=line["deal"])
        if v.section is None:
            return
        rec = {"deal": line["deal"], "id": line["id"], "section": v.section, "owner": v.owner}
        with lock:
            cache[k] = rec
            with open(a.cache, "a") as f:
                f.write(json.dumps(rec) + "\n")

    uniq = list({f"{x['deal']}|{x['id']}": x for x in asks}.values())
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, uniq))
    scored = [(deal, d, cache[f"{deal}|{line['id']}"]) for deal, d, line in items
              if f"{deal}|{line['id']}" in cache]
    report = score(scored)
    Path(a.report).write_text(json.dumps({"head": "jdg:sow_section", "model": J.judge_model(),
                                          "unjudged": len(items) - len(scored), "rules": report}, indent=1))
    for r in report:
        print(f"{r['rule']:>22} {r['head']:<13} n={r['decisions']:<4} agree={r['judge_agrees']:.3f} "
              f"was={r['judge_says_was']:<3} {r['status']} {r['blocker']}")


if __name__ == "__main__":
    main()
