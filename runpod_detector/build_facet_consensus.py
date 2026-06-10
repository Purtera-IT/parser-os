"""Cross-FAMILY consensus relabel (boss #11) — HARDENED / resumable.

DeepSeek wrote facet_clean. This re-judges each unique train text with qwen
(different family, same v2 rubric) and writes facet_consensus:
  - DeepSeek facet == qwen facet      -> that facet (two families agree; strongest)
  - DeepSeek was AMBIGUOUS, or disagree -> AMBIGUOUS (abstain; never train a guess)
  - qwen call FAILS (proxy blip)      -> leave NULL, retried on the next run

Robustness (why the first version wedged and this won't):
  * writes to the DB AS EACH text completes -> resumable; a kill/blip loses nothing.
  * as_completed (not map) -> one hung request can't block the others.
  * short per-call timeout (QTIMEOUT, default 30s, 1 retry) -> a blip costs seconds.
  * on resume, skips texts that already have facet_consensus.
Holdout untouched; micro preserved.

  python runpod_detector/build_facet_consensus.py   # resumes; run again after any blip
Env: QWEN_MODEL, OLLAMA_PROXY, PAR, QTIMEOUT, COMMIT_EVERY.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import os
import sqlite3
import urllib.request

try:
    from facet_agreement import PROXY, QWEN, _parse
except ImportError:
    from runpod_detector.facet_agreement import PROXY, QWEN, _parse
try:
    from rubric_relabel_facets import FEWSHOT_BLOCK, RUBRIC
except ImportError:
    from runpod_detector.rubric_relabel_facets import FEWSHOT_BLOCK, RUBRIC

DB = os.environ.get("FACET_DB", "_training_facet.db")
PAR = int(os.environ.get("PAR", "10"))
QTIMEOUT = int(os.environ.get("QTIMEOUT", "30"))
COMMIT_EVERY = int(os.environ.get("COMMIT_EVERY", "40"))


def qwen_ask(text, prev, nxt):
    ctx = f"PREV: {(prev or '(none)')[:200]}\nNEXT: {(nxt or '(none)')[:200]}\n"
    content = f"{RUBRIC}\n\n{FEWSHOT_BLOCK}\n\n{ctx}CLAUSE:\n{text[:600]}\n\nJSON:"
    body = json.dumps({"model": QWEN, "temperature": 0.0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"{PROXY}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for _ in range(2):  # 1 retry, short timeout -> a blip fails fast (text retried next run)
        try:
            o = json.loads(urllib.request.urlopen(req, timeout=QTIMEOUT).read())
            return _parse(o["choices"][0]["message"]["content"])
        except Exception:
            continue
    return None  # failure -> leave NULL, resume retries it


def main():
    con = sqlite3.connect(DB)
    cols = [r[1] for r in con.execute("PRAGMA table_info(training_rows)")]
    if "facet_consensus" not in cols:
        con.execute("ALTER TABLE training_rows ADD COLUMN facet_consensus TEXT")
        con.commit()

    # context by row id
    allrows = con.execute(
        "SELECT id, deal_id, COALESCE(raw_text,'') FROM training_rows "
        "WHERE relation='atom_type' AND COALESCE(raw_text,'')!='' ORDER BY deal_id, id").fetchall()
    pos = collections.defaultdict(list)
    for rid, deal, text in allrows:
        pos[deal].append((rid, text))
    ctx = {}
    for deal, seq in pos.items():
        for i, (rid, text) in enumerate(seq):
            ctx[rid] = (seq[i-1][1] if i > 0 else "", seq[i+1][1] if i < len(seq)-1 else "")

    # unique train texts that still need consensus (resume-aware)
    rows = con.execute(
        "SELECT id, COALESCE(raw_text,''), COALESCE(facet_clean,''), COALESCE(facet_consensus,'') "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(facet_clean,'')!=''").fetchall()
    text_rows, uniq = collections.defaultdict(list), {}
    done_text = set()
    for rid, text, ds, cons in rows:
        text_rows[text].append(rid)
        if cons:
            done_text.add(text)
        if text not in uniq:
            uniq[text] = (ctx.get(rid, ("", ""))[0], ctx.get(rid, ("", ""))[1], ds)
    todo = [t for t in uniq if t not in done_text]
    print(f"consensus(hardened): {len(uniq)} unique texts, {len(done_text)} already done, "
          f"{len(todo)} to do | qwen={QWEN} PAR={PAR} timeout={QTIMEOUT}s", flush=True)

    def work(t):
        p, n, ds = uniq[t]
        if ds == "AMBIGUOUS":
            return t, "AMBIGUOUS"            # already ambiguous; no qwen needed
        qf = qwen_ask(t, p, n)
        if qf is None:
            return t, None                   # blip -> leave NULL, retry next run
        qf = "AMBIGUOUS" if qf == "ambiguous" else qf
        return t, (ds if qf == ds else "AMBIGUOUS")

    written = agree = amb = fail = 0
    with cf.ThreadPoolExecutor(max_workers=PAR) as pool:
        futs = [pool.submit(work, t) for t in todo]
        for fut in cf.as_completed(futs):
            t, cons = fut.result()
            if cons is None:
                fail += 1; continue
            for rid in text_rows[t]:
                con.execute("UPDATE training_rows SET facet_consensus=? WHERE id=?", (cons, rid))
            written += 1
            agree += int(cons != "AMBIGUOUS"); amb += int(cons == "AMBIGUOUS")
            if written % COMMIT_EVERY == 0:
                con.commit()
                print(f"  written {written}/{len(todo)} | agree {agree} amb {amb} fail {fail}", flush=True)
    con.commit()

    tot = con.execute("SELECT COUNT(*) FROM training_rows WHERE COALESCE(facet_consensus,'')!=''").fetchone()[0]
    ca = con.execute("SELECT COUNT(*) FROM training_rows WHERE facet_consensus='AMBIGUOUS'").fetchone()[0]
    con.close()
    print(f"\nDONE this run: written={written} fail(retry next run)={fail}")
    print(f"facet_consensus rows total: {tot} | AMBIGUOUS {ca} ({100*ca/max(tot,1):.1f}%)")
    if fail:
        print(f"{fail} texts failed (proxy blips) — just re-run to finish them (resumes).")


if __name__ == "__main__":
    main()
