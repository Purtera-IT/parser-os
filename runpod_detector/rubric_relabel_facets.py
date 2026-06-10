"""Facet relabel via the rubric (DeepSeek API) — the label-ceiling fix for TYPING.

Same proven machinery as rubric_relabel_deepseek.py (few-shot + abstain +
self-consistency vote + prev/next context) but routes each TYPED atom to one of
the 7 facets (or _keep, or AMBIGUOUS). Writes a `facet_clean` column on a COPY
of the training DB:
  - TRAIN-split rows only (canonical split) get relabeled; HOLDOUT is never
    touched (its truth is the human gold set, not the LLM).
  - micro `label` is PRESERVED — facet is the trained target, micro stays for
    future fine heads (the "perfect base", both levels).
  - AMBIGUOUS is written, NOT dropped (it trains abstention; boss item 6).

  export TEACHER_API_BASE=https://api.deepseek.com
  export TEACHER_API_KEY=sk-...          # you set this; never committed/stored
  export TEACHER_API_MODEL=deepseek-chat
  python runpod_detector/rubric_relabel_facets.py            # -> _training_facet.db
Env: MAX_TRAIN=0(all) PAR=16 VOTES=3 AGREE_FRAC=1.0 CONTEXT=1
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import os
import shutil
import sqlite3
import urllib.request

try:
    from _split_util import load_split_map, split_of
    from taxonomy import FACETS, TAXONOMY_VERSION, to_facet
except ImportError:
    from runpod_detector._split_util import load_split_map, split_of
    from runpod_detector.taxonomy import FACETS, TAXONOMY_VERSION, to_facet

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
OUT_DB = os.environ.get("OUT_DB", "_training_facet.db")
BASE = os.environ.get("TEACHER_API_BASE", "https://api.deepseek.com").rstrip("/")
KEY = os.environ.get("TEACHER_API_KEY", "")
MODEL = os.environ.get("TEACHER_API_MODEL", "deepseek-chat")
PAR = int(os.environ.get("PAR", "16"))
VOTES = int(os.environ.get("VOTES", "3"))
AGREE_FRAC = float(os.environ.get("AGREE_FRAC", "1.0"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "0"))  # 0 = all
CONTEXT = os.environ.get("CONTEXT", "1") not in ("0", "false", "no")

VALID = set(FACETS) | {"_keep", "ambiguous"}

RUBRIC = """You label clauses from B2B managed-services deal documents (SOWs/quotes), ANY trade
(AV, cabling, electrical, fire-alarm, managed-IT). Route each clause to ONE bucket by its ROLE
in the deal, not its topic:

_keep  : scaffolding/boilerplate — document navigation, legends, schema/field metadata, empty
         form labels, a sales rep / quote signatory. Carries no deal commitment.
SITE       : a place, site access, or physical-site attribute.
COMMERCIAL : a price, rate, quantity/BOM line, payment term, or total.
WORK       : an action/service/task/requirement/deliverable/milestone/acceptance — incl. a
             negative scope statement (exclusion) and technical validation steps.
COMPLIANCE : a rule, certification, approval authority, insurance/bonding, change-order process,
             or regulatory/contractual obligation.
PARTY      : a person/org that ACTS in this deal's execution (PM, site contact, approver).
             A bare name/contact on a header with no action -> _keep.
TIMING     : a deadline, blackout window, or lead-time/sequencing/dependency constraint.
META       : deal-level metadata (project name, provider name, document field definitions).

Use the PREV/NEXT context to judge ROLE. If after the rules it is genuinely multi-bucket or
50/50, answer "ambiguous" rather than guessing.
Reply with ONLY JSON: {"facet":"SITE|COMMERCIAL|WORK|COMPLIANCE|PARTY|TIMING|META|_keep|ambiguous"}."""

FEWSHOT = [
    ("Table of Contents .................. 3", "_keep"),
    ("Legend: WAP = Wireless Access Point", "_keep"),
    ("Provider Name: ____________", "_keep"),
    ("John Smith, Account Executive, john@vendor.com", "_keep"),
    ("48 PORT CATEGORY 6 PATCH PANEL x 12", "COMMERCIAL"),
    ("Net 30 from invoice date", "COMMERCIAL"),
    ("Provider shall mount cameras at each stairwell landing", "WORK"),
    ("Exclusions: conduit and core drilling are by others", "WORK"),
    ("Provide access to all 23 dwellings and installation locations", "SITE"),
    ("Site contact: Maria Lopez, Facilities Mgr — escorts installers", "PARTY"),
    ("All work must be UL-listed and meet NEC 2020", "COMPLIANCE"),
    ("Cutover blackout: no changes Dec 20 - Jan 2", "TIMING"),
    ("Project Name: Marriott Downtown Tower", "META"),
]
FEWSHOT_BLOCK = "\n".join(f'EXAMPLE: {t}\n{{"facet":"{d}"}}' for t, d in FEWSHOT)


def ask(text, prev, nxt, temp):
    ctx = f"PREV: {(prev or '(none)')[:200]}\nNEXT: {(nxt or '(none)')[:200]}\n" if CONTEXT else ""
    content = f"{RUBRIC}\n\n{FEWSHOT_BLOCK}\n\n{ctx}CLAUSE:\n{text[:600]}\n\nJSON:"
    body = json.dumps({"model": MODEL, "temperature": temp,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for _ in range(3):
        try:
            o = json.loads(urllib.request.urlopen(req, timeout=120).read())
            c = o["choices"][0]["message"]["content"]
            try:
                f = json.loads(c[c.index("{"):c.rindex("}") + 1]).get("facet", "")
            except Exception:
                f = c
            f = f.strip()
            for cand in VALID:  # tolerant match
                if cand.lower() in f.lower():
                    return "ambiguous" if cand == "ambiguous" else cand
            return None
        except Exception:
            continue
    return None


def judge(text, prev, nxt):
    """VOTES-way self-consistency. Returns a facet if votes agree at AGREE_FRAC,
    'AMBIGUOUS' if any vote abstains or votes split, None on API failure."""
    votes = []
    for v in range(VOTES):
        temp = 0.0 if VOTES == 1 else (0.0 if v == 0 else 0.5)
        r = ask(text, prev, nxt, temp)
        if r is None:
            return None              # API failure -> leave unlabeled (falls back to micro map)
        if r == "ambiguous":
            return "AMBIGUOUS"       # first-class: trains abstention, not dropped
        votes.append(r)
    win, cnt = collections.Counter(votes).most_common(1)[0]
    return win if (cnt / VOTES) >= AGREE_FRAC else "AMBIGUOUS"


def main():
    if not KEY:
        print("ERROR: set TEACHER_API_KEY (+ TEACHER_API_BASE/MODEL). I won't store your key.")
        return
    con = sqlite3.connect(DB)
    smap = load_split_map(con)
    rows = con.execute(
        "SELECT id, deal_id, COALESCE(raw_text,''), COALESCE(label,'') "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(raw_text,'')!='' "
        "ORDER BY deal_id, id").fetchall()
    con.close()

    by_deal = collections.defaultdict(list)
    for rid, deal, text, label in rows:
        by_deal[deal].append((rid, text.strip(), label))
    train = []  # (id, text, prev, next, teacher_micro)
    for deal, seq in by_deal.items():
        if split_of(deal, smap) != "train":
            continue  # HOLDOUT never touched
        for i, (rid, text, label) in enumerate(seq):
            prev = seq[i - 1][1] if i > 0 else ""
            nxt = seq[i + 1][1] if i < len(seq) - 1 else ""
            train.append((rid, text, prev, nxt, label))
    if MAX_TRAIN:
        train = train[:MAX_TRAIN]
    print(f"facet-relabel: {len(train)} TRAIN atoms (holdout untouched) | model={MODEL} "
          f"| VOTES={VOTES} agree>={AGREE_FRAC} | taxonomy {TAXONOMY_VERSION}")

    results = {}
    with cf.ThreadPoolExecutor(max_workers=PAR) as pool:
        futs = {pool.submit(judge, t, p, n): rid for rid, t, p, n, _ in train}
        done = 0
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result(); done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(train)}", flush=True)

    shutil.copy2(DB, OUT_DB)
    out = sqlite3.connect(OUT_DB)
    cols = [r[1] for r in out.execute("PRAGMA table_info(training_rows)")]
    if "facet_clean" not in cols:
        out.execute("ALTER TABLE training_rows ADD COLUMN facet_clean TEXT")
    agree = ambiguous = labeled = failed = 0
    teacher = {rid: micro for rid, _, _, _, micro in train}
    for rid, fac in results.items():
        if fac is None:
            failed += 1; continue
        out.execute("UPDATE training_rows SET facet_clean=? WHERE id=?", (fac, rid))
        labeled += 1
        if fac == "AMBIGUOUS":
            ambiguous += 1
        elif fac == to_facet(teacher.get(rid)):
            agree += 1
    out.commit(); out.close()

    nonamb = labeled - ambiguous
    print(f"\nwrote {OUT_DB} (facet_clean on train rows; micro preserved; holdout untouched)")
    print(f"  labeled={labeled} api_failed={failed}")
    print(f"  AMBIGUOUS={ambiguous} ({ambiguous/max(labeled,1):.1%})  "
          f"<- ~10-18% expected; >>that = rubric gaps, <<that = rubric guessing")
    print(f"  rubric-vs-teacher facet agreement (non-ambiguous) = {agree/max(nonamb,1):.3f}")
    print("  next: train the contrastive/LR head on facet_clean; score on the gold set.")


if __name__ == "__main__":
    main()
