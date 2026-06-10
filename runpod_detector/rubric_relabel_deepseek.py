"""Relabel keep-vs-typed with the RUBRIC via the DeepSeek API (no Ollama needed) —
the highest-quality clean-label pass we can build. This is the make-or-break fix:
the gate caps ~0.66 because the stored labels draw the keep/typed boundary
59%-consistently; the rubric makes it ~95%-consistent. We re-judge with every
quality lever stacked:

  1. FEW-SHOT rubric — worked edge cases (legend-with-quantity, signatory-who-acts,
     form-label-with-a-real-value) on top of the rules, where models flip most.
  2. ABSTAIN — the model may answer "ambiguous"; those rows are DROPPED, not forced.
     The genuinely 50/50 clauses are the irreducible boundary — labeling them would
     re-inject the noise we're removing. Guess-free by construction.
  3. SELF-CONSISTENCY — judge each clause VOTES times; keep ONLY rows where the
     votes agree at AGREE_FRAC (default unanimous). Disagreement = ambiguous = drop.
  4. CONTEXT — each clause is judged WITH its previous/next clause (role depends on
     surroundings: "Project Name:" is keep in a form, typed in a scope sentence).

Pod-friendly: internet only, no GPU/Ollama. Output _rubric_gate_data.json (schema
matches train_gate_rubric.py: rows of {"text","y"}). Prints how far the rubric
disagrees with the OLD labels (= the noise we were training on) and how many rows
were dropped as ambiguous.

  export TEACHER_API_BASE=https://api.deepseek.com
  export TEACHER_API_KEY=sk-...            # YOU set this; never committed
  export TEACHER_API_MODEL=deepseek-chat
  python runpod_detector/rubric_relabel_deepseek.py

Env: MAX_TRAIN=3000 MAX_TEST=900 (0=all) PAR=16 VOTES=3 AGREE_FRAC=1.0
     CONTEXT=1 HOLDOUT=0.25
"""
import os, sqlite3, hashlib, json, collections, urllib.request, concurrent.futures as cf

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
BASE = os.environ.get("TEACHER_API_BASE", "https://api.deepseek.com").rstrip("/")
KEY = os.environ.get("TEACHER_API_KEY", "")
MODEL = os.environ.get("TEACHER_API_MODEL", "deepseek-chat")
HOLDOUT = float(os.environ.get("HOLDOUT", "0.25"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "3000"))
MAX_TEST = int(os.environ.get("MAX_TEST", "900"))
PAR = int(os.environ.get("PAR", "16"))
VOTES = int(os.environ.get("VOTES", "3"))
AGREE_FRAC = float(os.environ.get("AGREE_FRAC", "1.0"))   # 1.0 = unanimous; 0.67 = 2/3
CONTEXT = os.environ.get("CONTEXT", "1") not in ("0", "false", "no")

RUBRIC = """You label clauses from B2B managed-services deal documents (SOWs/quotes) across ANY
trade (AV, cabling, electrical, fire-alarm, managed-IT). Decide ONE thing by ROLE, not topic:
is this a SUBSTANTIVE deal item (TYPED) or scaffolding/boilerplate (KEEP)?

Apply in order:
1. Document navigation (index, table of contents, "Section 3.2", sheet/page refs) -> KEEP.
2. Legend / notation key / definition table -> KEEP. Exception: a stated equipment QUANTITY
   ("48-PORT PATCH PANEL x 12") -> TYPED (it's a BOM line).
3. Schema/field metadata ("ServiceNow Fields:", "Available Fields:", "Index: N | <field>") -> KEEP.
4. Form label/scaffold ("Project Name:", "Provider Name:", bare "Key |") -> KEEP the label;
   TYPED only if it carries a real fact value ("Project Name: Marriott Downtown").
5. Contact/signature block (Name|Title|Email) -> KEEP if it's a sales rep/signatory; TYPED only
   if the person ACTS in this deal (site contact who escorts installers, approver, etc.).
6. Names an action/service/requirement/site/price/milestone/condition -> TYPED, even if terse.

Use the surrounding context (PREV/NEXT clauses) to judge ROLE. If after applying the rules it
is genuinely 50/50, answer "ambiguous" rather than guessing.
Reply with ONLY {"decision":"typed"} or {"decision":"keep"} or {"decision":"ambiguous"}."""

FEWSHOT = [
    ("Table of Contents .................. 3", "keep"),
    ("See Section 3.2 for cabling specifications", "keep"),
    ("Legend: WAP = Wireless Access Point", "keep"),
    ("48-PORT PATCH PANEL x 12", "typed"),
    ("Project Name: ____________", "keep"),
    ("Project Name: Marriott Downtown Tower", "typed"),
    ("John Smith, Account Executive, john@vendor.com", "keep"),
    ("Site contact: Maria Lopez, Facilities Mgr — escorts installers on site", "typed"),
    ("Provider shall mount cameras at each stairwell landing", "typed"),
    ("Index: 4 | service_line", "keep"),
]
FEWSHOT_BLOCK = "\n".join(f'EXAMPLE: {t}\n{{"decision":"{d}"}}' for t, d in FEWSHOT)


try:
    from _split_util import load_split_map, split_of
except ImportError:
    from runpod_detector._split_util import load_split_map, split_of

_SPLIT_MAP = None


def split(deal_id):
    """Canonical split (recorded column, holdout-wins, hash fallback)."""
    global _SPLIT_MAP
    if _SPLIT_MAP is None:
        _c = sqlite3.connect(DB)
        try:
            _SPLIT_MAP = load_split_map(_c)
        finally:
            _c.close()
    return "test" if split_of(deal_id, _SPLIT_MAP, HOLDOUT) == "holdout" else "train"


def ask(text, prev, nxt, temp):
    ctx = ""
    if CONTEXT:
        ctx = f"PREV: {(prev or '(none)')[:200]}\nNEXT: {(nxt or '(none)')[:200]}\n"
    content = f"{RUBRIC}\n\n{FEWSHOT_BLOCK}\n\n{ctx}CLAUSE:\n{text[:600]}\n\nJSON:"
    body = json.dumps({"model": MODEL, "temperature": temp,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for _ in range(3):
        try:
            o = json.loads(urllib.request.urlopen(req, timeout=120).read())
            c = o["choices"][0]["message"]["content"].lower()
            if "ambiguous" in c:
                return "ambiguous"
            return "typed" if ('"typed"' in c or c.rfind("typed") > c.rfind("keep")) else "keep"
        except Exception:
            continue
    return None


def judge(text, prev, nxt):
    """VOTES-way self-consistency. Returns 'typed'/'keep' if votes agree at
    AGREE_FRAC and none ambiguous; else None (drop)."""
    votes = []
    for v in range(VOTES):
        temp = 0.0 if VOTES == 1 else (0.0 if v == 0 else 0.5)
        r = ask(text, prev, nxt, temp)
        if r in (None, "ambiguous"):
            return None          # any abstain/failure -> drop (guess-free)
        votes.append(r)
    win, n = collections.Counter(votes).most_common(1)[0]
    return win if (n / VOTES) >= AGREE_FRAC else None


def main():
    if not KEY:
        print("ERROR: set TEACHER_API_KEY (+ TEACHER_API_BASE/MODEL). I won't store your key."); return

    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT rowid, COALESCE(NULLIF(masked_text,''),raw_text) AS t, label, deal_id "
        "FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL "
        "AND COALESCE(masked_text,raw_text,'')!='' ORDER BY deal_id, rowid").fetchall()
    con.close()

    # per-deal ordered lists -> prev/next context; first occurrence of each text wins
    by_deal = collections.defaultdict(list)
    for rid, t, label, deal in rows:
        by_deal[deal or ""].append(((t or "").strip(), 0 if label == "_keep" else 1))
    ctx_of, info = {}, {}
    for deal, seq in by_deal.items():
        for i, (t, old) in enumerate(seq):
            if not t or t in info:
                continue
            prev = seq[i - 1][0] if i > 0 else ""
            nxt = seq[i + 1][0] if i + 1 < len(seq) else ""
            ctx_of[t] = (prev, nxt)
            info[t] = (old, split(deal))

    pools = {"train": [], "test": []}
    for t, (old, s) in info.items():
        pools[s].append(t)
    for s, cap in (("train", MAX_TRAIN), ("test", MAX_TEST)):
        if cap and len(pools[s]) > cap:
            pools[s] = pools[s][:cap]
    print(f"relabeling: train={len(pools['train'])} test={len(pools['test'])} | model={MODEL} "
          f"| VOTES={VOTES} agree>={AGREE_FRAC} context={CONTEXT}")

    out = {"train": [], "test": []}
    for s in ("train", "test"):
        texts = pools[s]
        results = {}
        with cf.ThreadPoolExecutor(max_workers=PAR) as pool:
            futs = {pool.submit(judge, t, *ctx_of.get(t, ("", ""))): t for t in texts}
            done = 0
            for fut in cf.as_completed(futs):
                results[futs[fut]] = fut.result(); done += 1
                if done % 200 == 0:
                    print(f"  {s}: {done}/{len(texts)}", flush=True)
        kept = dropped = agree = 0
        for t in texts:
            r = results.get(t)
            if r is None:
                dropped += 1; continue
            y = 1 if r == "typed" else 0
            agree += int(y == info[t][0])
            out[s].append({"text": t, "y": y})
            kept += 1
        tf = (sum(x["y"] for x in out[s]) / kept) if kept else 0.0
        print(f"  {s}: kept={kept} dropped_ambiguous={dropped} "
              f"| rubric-vs-OLD agreement={agree/max(kept,1):.3f} (lower = noisier originals) "
              f"| typed-frac={tf:.2f}")

    json.dump(out, open("_rubric_gate_data.json", "w"))
    print("wrote _rubric_gate_data.json -> now run: python runpod_detector/train_gate_rubric.py")


if __name__ == "__main__":
    main()
