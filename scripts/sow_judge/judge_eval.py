"""Measure jdg:sow_section (app/core/sow_rubric_judge.py) against the rubric labels.

    python3.11 scripts/sow_judge/judge_eval.py --rubric '/outside/rubric/*.A.jsonl' --out /outside/judge/v1 \
        [--split test] [--workers 2] [--limit N] [--variant v1]

Held out BY DEAL: deals hashing into the ``dev`` split (about a quarter) are the
only ones looked at while the prompt is iterated; the gate (>=93% section,
>=95% owner) is read on ``test``. Every answer is cached in
``<out>/pred.jsonl`` so a run resumes. The report gives agreement, the
always-majority baseline, per-section precision/recall and the confusion
matrix, and seconds per line. Line text is never printed.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))  # the repo root: app/ and ml/ beside each other

import argparse
import glob
import hashlib
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SECTION_GATE = 0.93
OWNER_GATE = 0.95


def split_of(deal: str) -> str:
    """dev (prompt tuning, ~1/4 of deals) | testA | testB: two held-out halves, so a
    gate missed on testA can be retried once on deals nobody has looked at."""
    h = int(hashlib.sha1(f"judge:{deal}".encode()).hexdigest(), 16)
    if h % 4 == 0:
        return "dev"
    # Salt 131 halves the held-out lines evenly (2,872 / 2,896), chosen on counts only.
    return "testA" if int(hashlib.sha1(f"half131:{deal}".encode()).hexdigest(), 16) % 2 == 0 else "testB"


def load(patterns: list[str]) -> list[dict]:
    """Rubric lines with their labels; a line two labelers split on is left out."""
    rows = []
    for path in sorted(x for p in patterns for x in glob.glob(p)):
        if not path.endswith(".A.jsonl"):
            continue
        batch = path[: -len(".A.jsonl")] + ".jsonl"
        other = path[: -len(".A.jsonl")] + ".B.jsonl"
        src = {json.loads(l)["id"]: json.loads(l) for l in open(batch)}
        second = {json.loads(l)["id"]: json.loads(l) for l in open(other)} if os.path.exists(other) else {}
        for l in open(path):
            r = json.loads(l)
            s = src.get(r.get("id"))
            if not s:
                continue
            b = second.get(r["id"])
            if b and (b.get("section") != r.get("section") or b.get("owner") != r.get("owner")):
                continue
            rows.append(dict(s, section=r["section"], owner=r["owner"]))
    return rows


def _in(split: str, deal: str) -> bool:
    s = split_of(deal)
    return split == "all" or s == split or (split == "test" and s.startswith("test"))


def report(rows: list[dict], pred: dict[str, dict]) -> dict:
    out = {}
    for split in ("dev", "testA", "testB", "test", "all"):
        rs = [r for r in rows if _in(split, r["deal"]) and r["id"] in pred]
        if not rs:
            continue
        n = len(rs)
        sec_ok = sum(pred[r["id"]].get("section") == r["section"] for r in rs)
        own_ok = sum(pred[r["id"]].get("owner") == r["owner"] for r in rs)
        undecided = sum(pred[r["id"]].get("section") is None for r in rs)
        maj_s = Counter(r["section"] for r in rs).most_common(1)[0]
        maj_o = Counter(r["owner"] for r in rs).most_common(1)[0]
        conf = defaultdict(Counter)
        for r in rs:
            conf[r["section"]][pred[r["id"]].get("section") or "UNDECIDED"] += 1
        per = {}
        for s in sorted(set(r["section"] for r in rs) | {p.get("section") for p in pred.values() if p.get("section")}):
            tp = conf[s][s]
            gold = sum(conf[s].values())
            said = sum(conf[g][s] for g in conf)
            per[s] = {"gold": gold, "said": said, "recall": round(tp / gold, 3) if gold else None,
                      "precision": round(tp / said, 3) if said else None}
        secs = [pred[r["id"]].get("secs", 0.0) for r in rs]
        out[split] = {
            "n": n, "deals": len({r["deal"] for r in rs}),
            "section_agree": round(sec_ok / n, 4), "section_majority": round(maj_s[1] / n, 4),
            "owner_agree": round(own_ok / n, 4), "owner_majority": round(maj_o[1] / n, 4),
            "undecided": undecided,
            "section_gate": "pass" if sec_ok / n >= SECTION_GATE else "fail",
            "owner_gate": "pass" if own_ok / n >= OWNER_GATE else "fail",
            "secs_per_line": round(sum(secs) / n, 2),
            "per_section": per,
            "confusion": {g: dict(c) for g, c in conf.items()},
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rubric", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["dev", "testA", "testB", "test", "all"], default="dev")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default="")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    from app.core import sow_rubric_judge as J

    rows = load(a.rubric)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "pred.jsonl"
    pred = {}
    if cache.exists():
        for l in open(cache):
            r = json.loads(l)
            if r.get("section") is not None:  # an undecided answer is retried
                pred[r["id"]] = r
    todo = [r for r in rows if _in(a.split, r["deal"]) and r["id"] not in pred]
    if a.limit:
        todo = todo[: a.limit]
    lock = threading.Lock()

    def one(r):
        t = time.time()
        v = J.judge(r["text"], job=r.get("job") or "", under=r.get("under") or "", deal_id=r["deal"],
                    model=a.model or "")
        rec = {"id": r["id"], "deal": r["deal"], "section": v.section, "owner": v.owner,
               "section_conf": v.section_conf, "source": v.source, "secs": round(time.time() - t, 2)}
        with lock:
            with open(cache, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["section"] is not None:
                pred[r["id"]] = rec
        return rec

    if not a.report_only and todo:
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(one, todo), 1):
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)} {(time.time() - t0) / i:.2f}s/line", flush=True)
    rep = report(rows, pred)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    for split, r in rep.items():
        print(f"{split:>4} n={r['n']} deals={r['deals']} section {r['section_agree']:.3f} (majority "
              f"{r['section_majority']:.3f}) {r['section_gate']} | owner {r['owner_agree']:.3f} (majority "
              f"{r['owner_majority']:.3f}) {r['owner_gate']} | undecided {r['undecided']} | {r['secs_per_line']}s/line")


if __name__ == "__main__":
    main()
