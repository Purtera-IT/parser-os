"""Relabel keep-vs-typed with the RUBRIC via the DeepSeek API (no Ollama needed).

WHY: the contrastive gate caps ~0.66 because the stored teacher labels draw the
keep/typed boundary inconsistently (59% reproducible). The rubric makes the
boundary ~95% reproducible. This re-judges atoms with the rubric so we can test
whether CLEAN labels break the ceiling — the make-or-break experiment.

Runs on the pod with only internet + your DeepSeek key (no GPU, no Ollama):
  export TEACHER_API_BASE=https://api.deepseek.com
  export TEACHER_API_KEY=sk-...            # YOU set this; never commit it
  export TEACHER_API_MODEL=deepseek-chat
  python runpod_detector/rubric_relabel_deepseek.py

Output: _rubric_gate_data.json  (train/test rubric labels, held-out by deal) ->
feed to runpod_detector/train_gate_rubric.py. Also prints how far the rubric
disagrees with the OLD labels = how noisy the originals were.

Env: MAX_TRAIN=3000 MAX_TEST=900 (0 = all), PAR=16, HOLDOUT=0.25
"""
import os, sqlite3, hashlib, json, urllib.request, concurrent.futures as cf

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
BASE = os.environ.get("TEACHER_API_BASE", "https://api.deepseek.com").rstrip("/")
KEY = os.environ.get("TEACHER_API_KEY", "")
MODEL = os.environ.get("TEACHER_API_MODEL", "deepseek-chat")
HOLDOUT = float(os.environ.get("HOLDOUT", "0.25"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "3000"))
MAX_TEST = int(os.environ.get("MAX_TEST", "900"))
PAR = int(os.environ.get("PAR", "16"))

RUBRIC = """You label clauses from B2B managed-services deal documents (SOWs/quotes) across ANY
trade (AV, cabling, electrical, fire-alarm, managed-IT). Decide ONE thing by ROLE, not topic:
is this a SUBSTANTIVE deal item (TYPED) or scaffolding/boilerplate (KEEP)?
1. Document navigation (index, table of contents, section/sheet/page refs) -> KEEP.
2. Legend / notation key / definition table -> KEEP. Exception: a stated equipment QUANTITY -> TYPED.
3. Schema/field metadata ("ServiceNow Fields:", "Available Fields:", "Index: N | <field>") -> KEEP.
4. Form label/scaffold ("Project Name:", "Provider Name:", bare "Key |") -> KEEP the label; TYPED only if real fact value.
5. Contact/signature block (Name|Title|Email) -> TYPED only if the person ACTS in this deal; sales rep/signatory -> KEEP.
6. Names an action/service/requirement/site/price/milestone/condition -> TYPED, even if terse.
Reply with ONLY {"decision":"typed"} or {"decision":"keep"}."""


def split(deal_id):
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "test" if (h % 100) / 100.0 < HOLDOUT else "train"


def ask(text):
    body = json.dumps({
        "model": MODEL, "temperature": 0,
        "messages": [{"role": "user", "content": f"{RUBRIC}\n\nCLAUSE:\n{text[:600]}\n\nJSON:"}],
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for _ in range(3):
        try:
            o = json.loads(urllib.request.urlopen(req, timeout=120).read())
            c = o["choices"][0]["message"]["content"].lower()
            return 1 if ('"typed"' in c or c.rfind("typed") > c.rfind("keep")) else 0
        except Exception:
            continue
    return None


def main():
    if not KEY:
        print("ERROR: set TEACHER_API_KEY (and TEACHER_API_BASE/MODEL). I won't store your key."); return
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT COALESCE(NULLIF(masked_text,''),raw_text) AS t, label, deal_id "
        "FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL "
        "AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()

    buckets = {"train": {}, "test": {}}   # text -> (old_typed, deal)
    for t, label, deal in rows:
        t = (t or "").strip()
        if not t:
            continue
        buckets[split(deal)].setdefault(t, (0 if label == "_keep" else 1, deal or ""))
    for s, cap in (("train", MAX_TRAIN), ("test", MAX_TEST)):
        items = list(buckets[s].items())
        if cap and len(items) > cap:
            items = items[:cap]
        buckets[s] = dict(items)
    print(f"relabeling: train={len(buckets['train'])} test={len(buckets['test'])} via {MODEL} (rubric)")

    out = {"train": [], "test": []}
    for s in ("train", "test"):
        texts = list(buckets[s].keys())
        results = {}
        with cf.ThreadPoolExecutor(max_workers=PAR) as pool:
            futs = {pool.submit(ask, t): t for t in texts}
            done = 0
            for fut in cf.as_completed(futs):
                t = futs[fut]; results[t] = fut.result()
                done += 1
                if done % 200 == 0:
                    print(f"  {s}: {done}/{len(texts)}", flush=True)
        agree = tot = 0
        for t in texts:
            r = results.get(t)
            if r is None:
                continue
            old = buckets[s][t][0]
            tot += 1; agree += int(r == old)
            out[s].append({"text": t, "label": r})   # 1=typed, 0=keep (rubric)
        print(f"  {s}: rubric vs OLD label agreement = {agree/max(tot,1):.3f} "
              f"({tot-agree}/{tot} rows the old teacher had inconsistent)")

    json.dump(out, open("_rubric_gate_data.json", "w"))
    print(f"wrote _rubric_gate_data.json -> now run: python runpod_detector/train_gate_rubric.py")


if __name__ == "__main__":
    main()
