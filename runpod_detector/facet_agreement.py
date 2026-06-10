"""Gate A — cross-FAMILY facet-rubric agreement (boss audit #11). Runs the SAME
facet rubric through two independent model families on a sample of train atoms and
reports agreement. Purpose: don't spend the $20 full relabel if the rubric only
gets ~0.82 cross-model — fix the rubric first.

  DeepSeek (API)  vs  qwen2.5vl:32b (Ollama proxy)
  >= ~0.85 on non-abstained -> rubric is relabel-worthy
  <  ~0.85                  -> fix the rubric, don't relabel yet

Env: TEACHER_API_KEY/BASE/MODEL (deepseek), OLLAMA_PROXY, QWEN_MODEL, N, PAR.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import os
import sqlite3
import urllib.request

try:
    from _split_util import load_split_map, split_of
    from rubric_relabel_facets import FEWSHOT_BLOCK, RUBRIC
    from taxonomy import FACETS
except ImportError:
    from runpod_detector._split_util import load_split_map, split_of
    from runpod_detector.rubric_relabel_facets import FEWSHOT_BLOCK, RUBRIC
    from runpod_detector.taxonomy import FACETS

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
DS_BASE = os.environ.get("TEACHER_API_BASE", "https://api.deepseek.com").rstrip("/")
DS_KEY = os.environ.get("TEACHER_API_KEY", "")
DS_MODEL = os.environ.get("TEACHER_API_MODEL", "deepseek-chat")
PROXY = os.environ.get("OLLAMA_PROXY",
    "https://ollama-mac-proxy-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io").rstrip("/")
QWEN = os.environ.get("QWEN_MODEL", "qwen2.5vl:32b")
N = int(os.environ.get("N", "300"))
PAR = int(os.environ.get("PAR", "12"))
ORDER = [*FACETS, "_keep", "ambiguous"]


def _parse(c: str):
    try:
        c = json.loads(c[c.index("{"):c.rindex("}") + 1]).get("facet", c)
    except Exception:
        pass
    c = str(c).strip().lower()
    for cand in ORDER:
        if c == cand.lower():
            return cand
    hits = [cand for cand in ORDER if cand.lower() in c]
    return hits[0] if len(hits) == 1 else None


def ask(base, key, model, text, prev, nxt):
    ctx = f"PREV: {(prev or '(none)')[:200]}\nNEXT: {(nxt or '(none)')[:200]}\n"
    content = f"{RUBRIC}\n\n{FEWSHOT_BLOCK}\n\n{ctx}CLAUSE:\n{text[:600]}\n\nJSON:"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({"model": model, "temperature": 0.0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, headers=headers)
    for _ in range(2):
        try:
            o = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return _parse(o["choices"][0]["message"]["content"])
        except Exception:
            continue
    return None


def main():
    if not DS_KEY:
        print("set TEACHER_API_KEY first"); return
    con = sqlite3.connect(DB)
    smap = load_split_map(con)
    rows = con.execute(
        "SELECT deal_id, COALESCE(raw_text,''), COALESCE(label,'') FROM training_rows "
        "WHERE relation='atom_type' AND COALESCE(raw_text,'')!='' ORDER BY deal_id, id").fetchall()
    con.close()
    by_deal = collections.defaultdict(list)
    for deal, text, label in rows:
        by_deal[deal].append((text, label))
    sample = []
    for deal, seq in by_deal.items():
        if split_of(deal, smap) != "train":
            continue
        for i, (text, label) in enumerate(seq):
            sample.append((text, seq[i-1][0] if i > 0 else "", seq[i+1][0] if i < len(seq)-1 else ""))
    # deterministic stride sample of N
    step = max(1, len(sample) // N)
    sample = sample[::step][:N]
    print(f"Gate A: {len(sample)} train atoms | DeepSeek({DS_MODEL}) vs qwen({QWEN})")

    def both(item):
        t, p, n = item
        return ask(DS_BASE, DS_KEY, DS_MODEL, t, p, n), ask(PROXY, "", QWEN, t, p, n)

    pairs = []
    with cf.ThreadPoolExecutor(max_workers=PAR) as pool:
        for i, r in enumerate(pool.map(both, sample), 1):
            pairs.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(sample)}", flush=True)

    ds_ok = sum(1 for a, b in pairs if a)
    qw_ok = sum(1 for a, b in pairs if b)
    both_ans = [(a, b) for a, b in pairs if a and b and a != "ambiguous" and b != "ambiguous"]
    agree = sum(1 for a, b in both_ans if a == b)
    amb = sum(1 for a, b in pairs if a == "ambiguous" or b == "ambiguous")
    print(f"\nDeepSeek answered {ds_ok}/{len(pairs)} | qwen answered {qw_ok}/{len(pairs)}")
    print(f"either-ambiguous: {amb} ({amb/len(pairs):.1%})")
    print(f"AGREEMENT on non-ambiguous (n={len(both_ans)}): {agree/max(len(both_ans),1):.3f}")
    # confusions
    conf = collections.Counter((a, b) for a, b in both_ans if a != b)
    print("top disagreements (deepseek -> qwen):")
    for (a, b), c in conf.most_common(8):
        print(f"   {a:>11} vs {b:<11} x{c}")
    # dump disagreement EXAMPLES so rubric rulings are grounded in real atoms
    dump = []
    for (t, p, n), (a, b) in zip(sample, pairs):
        if a and b and a != b and "ambiguous" not in (a, b):
            dump.append({"pair": f"{a}|{b}", "text": t[:200], "prev": p[:80], "next": n[:80]})
    json.dump(dump, open("_gateA_disagreements.json", "w"), indent=1)
    print(f"wrote _gateA_disagreements.json ({len(dump)} examples)")
    print("\nVERDICT:", "RELABEL-WORTHY (>=0.85)" if agree/max(len(both_ans),1) >= 0.85
          else "FIX RUBRIC FIRST (<0.85) — do NOT spend the $20 yet")


if __name__ == "__main__":
    main()
