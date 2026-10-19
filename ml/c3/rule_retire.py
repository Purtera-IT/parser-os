"""Can a SOW writer rule be retired? Score a checkpoint's heads against each
rule's decisions on deals the checkpoint never trained on.

    python -m ml.c3.rule_retire --checkpoint /path/refit.pt \
        --envelopes /outside/final_*_env.json [--reviews /outside/sow_reviews.jsonl] \
        --report /outside/rule_retire.json

For every rule in ``c3_sow.rule_decisions`` (grouped by rule name and head):
how many decisions, how many a PM reviewed, and how often the head gives the
same answer. A rule is READY to retire when the head agrees on >= 95% of at
least 30 decisions and on every decision a PM accepted, and disagrees with
every decision a PM rejected; otherwise it is LEARNING. SowSmith's Heads panel
reads the report. Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from .head_fit import _cache
from .labels_from_sow import _read, _rows
from .loop import live_student, load_split
from .readiness import digest
from .schema import load_schema

READY_AGREE = 0.95
READY_MIN = 30


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--envelopes", nargs="+", required=True)
    p.add_argument("--reviews", default=None)
    p.add_argument("--report", required=True)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--max-atoms", type=int, default=8192)
    a = p.parse_args()
    from .gates import load_checkpoint

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    schema = load_schema()
    model = live_student(schema, a.encoder).to(dev)
    load_checkpoint(a.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    model.eval()
    reviews = [json.loads(x) for x in open(a.reviews)] if a.reviews and os.path.exists(a.reviews) else []
    verdict = {(str(r.get("deal")), str(r.get("atom_id")), str(r.get("head"))): str(r.get("verdict")) for r in reviews}

    # One record per (rule, head), labels = the rule's own calls, so the
    # head's answer on each labeled line can be compared to the rule's.
    groups: dict[tuple[str, str], list[tuple[str, dict, list[dict]]]] = defaultdict(list)
    for path in [x for pat in a.envelopes for x in glob.glob(pat)]:
        env = json.loads(Path(path).read_text())
        deal = str((env.get("crm") or {}).get("deal_name") or "")[:6] or Path(path).stem
        rows = None
        for d in (env.get("c3_sow") or {}).get("rule_decisions") or []:
            read = _read(str(d.get("head")), str(d.get("value")), str(d.get("rule")))
            if not read:
                continue
            rows = rows if rows is not None else _rows(env)
            groups[(str(d.get("rule")), read[0])].append((deal, d, rows))
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        for (rule, key), items in sorted(groups.items()):
            files, rv = [], {"accepted": [0, 0], "rejected": [0, 0]}
            by_deal: dict[str, list] = defaultdict(list)
            for deal, d, rows in items:
                by_deal[deal].append((d, rows))
            for deal, ds in by_deal.items():
                rows = ds[0][1]
                have = {r["key"]: r["text"] for r in rows}
                labels = {}
                for d, _ in ds:
                    aid = (d.get("atom_ids") or [None])[0]
                    read = _read(str(d.get("head")), str(d.get("value")), str(d.get("rule")))
                    if aid in have and read:
                        labels[aid] = {"label_key": aid, "atom_id": aid, "labeler": "rule:" + rule, "origin": "labeler",
                                       "text": have[aid], "reads_set": {read[0]: read[1]}}
                if labels:
                    f = Path(tmp, f"{rule}-{key}-{deal}.json")
                    f.write_text(json.dumps({"deal_id": deal, "company": "purtera",
                                             "atoms": [dict(r, label_key=r["key"]) for r in rows],
                                             "labels": list(labels.values()), "judgments": [], "links": [],
                                             "deal_answers": []}))
                    files.append(str(f))
            if not files:
                continue
            parts, _ = load_split(files, schema, a.max_atoms)
            c = _cache(model, parts, [f"read:{key}"], dev).get(f"read:{key}")
            if not c or not len(c["y"]):
                continue
            agree = (c["base"].argmax(-1) == c["y"]).float()
            n, k = len(c["y"]), float(agree.sum())
            for deal, d, _ in items:
                v = verdict.get((deal, str((d.get("atom_ids") or [""])[0]), str(d.get("head"))))
                if v in rv:
                    rv[v][1] += 1
            rate = k / n
            ready = n >= READY_MIN and rate >= READY_AGREE
            report.append({"rule": rule, "head": f"read:{key}", "decisions": n, "head_agrees": round(rate, 3),
                           "reviewed_accepted": rv["accepted"][1], "reviewed_rejected": rv["rejected"][1],
                           "status": "ready" if ready else "learning"})
    Path(a.report).write_text(json.dumps({"checkpoint": os.path.basename(a.checkpoint), "rules": report}, indent=1))
    for r in report:
        print(f"{r['rule']:>14} {r['head']:<18} n={r['decisions']:<4} agree={r['head_agrees']:.3f} {r['status']}")


if __name__ == "__main__":
    main()
