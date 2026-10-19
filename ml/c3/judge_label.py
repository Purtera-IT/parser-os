"""Label deal lines with the rubric judge (jdg:sow_section), for the fast heads to learn from.

    python -m ml.c3.judge_label --envelopes '/outside/envs/*.json' --out /outside/judge_labels \
        [--per-deal 150] [--skip-deals 010043 ...] [--workers 2] [--max-lines N]

Per deal: the envelope's lines (atoms), deduplicated by text, a seeded sample
of ``--per-deal``. Each sampled line is written to ``<out>/<deal>.jsonl`` in
the rubric batch shape (deal, id, job, under, text) and judged; answers go to
``<out>/<deal>.J.jsonl`` ({"id", "deal", "section", "owner", "model",
"section_conf"}). These are MODEL DRAFTS: ``labels_from_sow --judge`` reads
them as their own source (created_by=judge, slight tier) and any human
answer on the same line wins. Nothing here writes to the feedback store.
Resumes: a judged line is never asked again.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MIN_CHARS = 12
MAX_CHARS = 300  # the rubric labelers saw the first 300 characters


def deal_of(env: dict, path: str) -> str:
    return str((env.get("crm") or {}).get("deal_name") or "")[:6] or Path(path).stem


def sample_lines(env: dict, deal: str, per_deal: int, seed: int = 0) -> list[dict]:
    job = str((env.get("c3_sow") or {}).get("job_family") or "")
    seen, lines = set(), []
    for a in env.get("atoms") or []:
        text = " ".join(str(a.get("text") or "").split())
        aid = a.get("id")
        if not aid or len(text) < MIN_CHARS:
            continue
        k = text.lower()
        if k in seen:
            continue
        seen.add(k)
        path = [str(x) for x in a.get("section_path") or [] if str(x).strip()]
        lines.append({"deal": deal, "id": aid, "job": job, "under": path[-1][:120] if path else "",
                      "text": text[:MAX_CHARS]})
    rnd = random.Random(f"{seed}:{deal}")
    rnd.shuffle(lines)
    return lines[:per_deal]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--envelopes", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-deal", type=int, default=150)
    p.add_argument("--skip-deals", nargs="*", default=[])
    p.add_argument("--skip-file", default="", help="a file of deal numbers to skip (the rubric-labeled deals)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-lines", type=int, default=0, help="stop after judging this many lines (0 = all)")
    p.add_argument("--prepare-only", action="store_true", help="write the batch files, judge nothing")
    p.add_argument("--model", default="")
    a = p.parse_args()
    from app.core import sow_rubric_judge as J

    model = a.model or J.judge_model() or "default"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = set(a.skip_deals)
    if a.skip_file:
        skip |= {x.strip() for x in open(a.skip_file) if x.strip()}
    todo, queued = [], set()
    for path in sorted(x for pat in a.envelopes for x in glob.glob(pat)):
        try:
            env = json.loads(Path(path).read_text())
        except Exception:
            continue
        deal = deal_of(env, path)
        if not deal or deal in skip or not (deal[:1].isdigit()):
            continue
        batch = out / f"{deal}.jsonl"
        if batch.exists():
            lines = [json.loads(l) for l in open(batch)]
        else:
            lines = sample_lines(env, deal, a.per_deal)
            batch.write_text("".join(json.dumps(r) + "\n" for r in lines))
        done = set()
        jf = out / f"{deal}.J.jsonl"
        if jf.exists():
            done = {json.loads(l)["id"] for l in open(jf)}
        for r in lines:
            if r["id"] not in done and (deal, r["id"]) not in queued:
                queued.add((deal, r["id"]))
                todo.append(r)
    print(f"{len(todo)} lines to judge", flush=True)
    if a.prepare_only:
        return
    if a.max_lines:
        todo = todo[: a.max_lines]
    lock = threading.Lock()
    t0 = time.time()

    def one(r):
        v = J.judge(r["text"], job=r.get("job") or "", under=r.get("under") or "", deal_id=r["deal"],
                    model=a.model or "")
        if v.section is None:
            return None  # undecided: asked again next run
        rec = {"id": r["id"], "deal": r["deal"], "section": v.section, "owner": v.owner, "model": model,
               "section_conf": round(v.section_conf, 3), "source": v.source}
        with lock:
            with open(out / f"{r['deal']}.J.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} {(time.time() - t0) / i:.2f}s/line", flush=True)
    print("JUDGE_LABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
