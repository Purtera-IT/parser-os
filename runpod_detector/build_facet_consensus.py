"""Cross-FAMILY consensus relabel (boss #11 — kill single-model systematic bias).

DeepSeek already wrote facet_clean. This re-judges each unique train text with qwen
(different family) using the SAME v2 rubric, then writes facet_consensus:
  - DeepSeek facet == qwen facet      -> that facet (high-confidence, two families agree)
  - DeepSeek was AMBIGUOUS, or models DISAGREE -> AMBIGUOUS (abstain; don't train a guess)
The strongest label set obtainable without humans. Holdout untouched; micro preserved.

  python runpod_detector/build_facet_consensus.py            # DB=_training_facet.db
Env: QWEN_MODEL, OLLAMA_PROXY, PAR, LIMIT (0=all; >0 = timing/subset test).
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import os
import sqlite3
import time

try:
    from facet_agreement import PROXY, QWEN, ask
except ImportError:
    from runpod_detector.facet_agreement import PROXY, QWEN, ask

DB = os.environ.get("FACET_DB", "_training_facet.db")
PAR = int(os.environ.get("PAR", "10"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def main():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, deal_id, COALESCE(raw_text,''), COALESCE(facet_clean,'') "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(facet_clean,'')!='' "
        "ORDER BY deal_id, id").fetchall()
    # prev/next per deal for context
    by_deal = collections.defaultdict(list)
    allrows = con.execute(
        "SELECT id, deal_id, COALESCE(raw_text,'') FROM training_rows "
        "WHERE relation='atom_type' AND COALESCE(raw_text,'')!='' ORDER BY deal_id, id").fetchall()
    pos = {}
    for did, deal, text in allrows:
        pos.setdefault(deal, []).append((did, text))
    ctx = {}
    for deal, seq in pos.items():
        for i, (did, text) in enumerate(seq):
            ctx[did] = (seq[i-1][1] if i > 0 else "", seq[i+1][1] if i < len(seq)-1 else "")

    # dedup by text: one qwen call per unique text, carry deepseek facet
    uniq = {}  # text -> (prev, next, deepseek_facet)
    text_rows = collections.defaultdict(list)
    for rid, deal, text, ds in rows:
        text_rows[text].append(rid)
        if text not in uniq:
            p, n = ctx.get(rid, ("", ""))
            uniq[text] = (p, n, ds)
    texts = list(uniq)
    if LIMIT:
        texts = texts[:LIMIT]
    print(f"consensus: {len(texts)} unique texts vs qwen({QWEN}) | PAR={PAR}")

    t0 = time.time()
    qwen_facet = {}

    def q(t):
        p, n, _ = uniq[t]
        r = ask(PROXY, "", QWEN, t, p, n)
        return t, (r if r and r != "ambiguous" else "AMBIGUOUS")

    with cf.ThreadPoolExecutor(max_workers=PAR) as pool:
        done = 0
        for t, qf in pool.map(q, texts):
            qwen_facet[t] = qf; done += 1
            if done % 200 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(texts)} | {rate:.1f}/s | eta {int((len(texts)-done)/max(rate,1e-9))}s", flush=True)

    if LIMIT:  # timing test only — don't write
        rate = len(texts) / max(time.time() - t0, 1e-9)
        full = con.execute("SELECT COUNT(DISTINCT raw_text) FROM training_rows "
                           "WHERE relation='atom_type' AND COALESCE(facet_clean,'')!=''").fetchone()[0]
        print(f"\nTIMING: {rate:.2f} texts/s -> full {full} texts ~ {int(full/max(rate,1e-9)/60)} min")
        con.close(); return

    cols = [r[1] for r in con.execute("PRAGMA table_info(training_rows)")]
    if "facet_consensus" not in cols:
        con.execute("ALTER TABLE training_rows ADD COLUMN facet_consensus TEXT")
    agree = amb = 0
    for text, (p, n, ds) in uniq.items():
        qf = qwen_facet.get(text)
        if ds == "AMBIGUOUS" or qf is None or qf != ds:
            cons = "AMBIGUOUS"; amb += 1
        else:
            cons = ds; agree += 1
        for rid in text_rows[text]:
            con.execute("UPDATE training_rows SET facet_consensus=? WHERE id=?", (cons, rid))
    con.commit(); con.close()
    tot = agree + amb
    print(f"\nfacet_consensus written. consensus(both agree)={agree} ({agree/tot:.1%}) "
          f"AMBIGUOUS={amb} ({amb/tot:.1%})")
    print("train on facet_consensus for the strongest labels; AMBIGUOUS -> abstain.")


if __name__ == "__main__":
    main()
