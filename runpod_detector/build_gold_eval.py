"""Gold eval — the keystone. Build a FROZEN, human-adjudicated truth set so every
model is measured against ground truth, not the noisy teacher. Two modes:

  build : sample atoms from the CANONICAL holdout deals -> a PM-friendly CSV to
          adjudicate + a frozen manifest (reproducible/auditable).
  score : given the adjudicated CSV + a model's predictions, report per-facet
          precision/coverage on gold (the only honest number).

Sampling (per the architecture review):
  - ONLY canonical-holdout deals (split_util, holdout-wins) so training can never
    leak into the bar.
  - FREQUENCY-WEIGHTED (not text-deduped): recurring boilerplate is real volume.
  - STRATIFIED: guarantee >= MIN_PER_FACET for every facet incl. starved ones,
    oversample a NOVEL-text slice + (if a CL flag file is given) flagged atoms.
  - CONTEXT: prev/next clause by (deal_id, rowid) — the rubric judges in context.

Adjudicator fills `gold_facet` (one of the 7, or AMBIGUOUS) and optionally
`gold_micro`. AMBIGUOUS is first-class: it trains abstention, it is not a failure.

  python runpod_detector/build_gold_eval.py build   --db _training_deepseek.db --n 1800
  # PM fills gold_facet in gold_eval_v1_TOADJUDICATE.csv ...
  python runpod_detector/build_gold_eval.py score   --gold gold_eval_v1_DONE.csv --pred preds.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict

try:
    from _split_util import load_split_map, split_of
    from taxonomy import FACETS, KEEP, TAXONOMY_VERSION, to_facet
except ImportError:
    from runpod_detector._split_util import load_split_map, split_of
    from runpod_detector.taxonomy import FACETS, KEEP, TAXONOMY_VERSION, to_facet

ADJ_FACETS = list(FACETS) + [KEEP, "AMBIGUOUS"]
SEED = 1310  # fixed so the sample is reproducible/auditable (no Date/random drift)


def _rows_with_context(db: str):
    """Holdout rows with prev/next text reconstructed by (deal_id, rowid)."""
    con = sqlite3.connect(db)
    smap = load_split_map(con)
    rows = con.execute(
        "SELECT id, deal_id, COALESCE(raw_text,''), COALESCE(label,'') "
        "FROM training_rows WHERE relation='atom_type' "
        "AND COALESCE(raw_text,'')!='' ORDER BY deal_id, id"
    ).fetchall()
    con.close()
    by_deal = defaultdict(list)
    for rid, deal, text, label in rows:
        by_deal[deal].append((rid, text, label))
    out = []
    for deal, seq in by_deal.items():
        if split_of(deal, smap) != "holdout":
            continue
        for i, (rid, text, label) in enumerate(seq):
            prev = seq[i - 1][1] if i > 0 else ""
            nxt = seq[i + 1][1] if i < len(seq) - 1 else ""
            out.append({
                "id": rid, "deal_id": deal, "prev": prev, "text": text,
                "next": nxt, "teacher_micro": label, "teacher_facet": to_facet(label),
            })
    return out


def build(db: str, n: int, min_per_facet: int, novel_frac: float, flagged_ids: set[int] | None):
    rng = random.Random(SEED)
    pool = _rows_with_context(db)
    if not pool:
        print("no holdout rows found — check the split column"); return
    # text-frequency across the whole corpus (to mark recurring boilerplate + a novel slice)
    freq = Counter(r["text"].strip() for r in pool)
    for r in pool:
        r["text_freq"] = freq[r["text"].strip()]
        r["is_novel"] = freq[r["text"].strip()] == 1

    picked: dict[int, dict] = {}

    def take(rows, k):
        rng.shuffle(rows)
        added = 0
        for r in rows:
            if len(picked) >= n or added >= k:
                break
            if r["id"] not in picked:
                picked[r["id"]] = r
                added += 1

    # 1) guarantee minority-facet coverage
    by_facet = defaultdict(list)
    for r in pool:
        by_facet[r["teacher_facet"]].append(r)
    for f in ADJ_FACETS:
        take(list(by_facet.get(f, [])), min_per_facet)
    # 2) flagged (CL/disagreement) atoms if provided
    if flagged_ids:
        take([r for r in pool if r["id"] in flagged_ids], int(n * 0.25))
    # 3) a novel-text slice (generalization probe)
    take([r for r in pool if r["is_novel"]], int(n * novel_frac))
    # 4) fill the rest frequency-weighted (sample WITH text repetition = real volume)
    take(list(pool), n)

    sample = list(picked.values())
    rng.shuffle(sample)

    csv_path = "gold_eval_v1_TOADJUDICATE.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "deal_id", "prev", "text", "next",
                    "teacher_facet", "teacher_micro",
                    "gold_facet (FILL: %s)" % "|".join(ADJ_FACETS),
                    "gold_micro (optional)", "notes"])
        for r in sample:
            w.writerow([r["id"], r["deal_id"], r["prev"][:200], r["text"][:400],
                        r["next"][:200], r["teacher_facet"], r["teacher_micro"], "", "", ""])

    manifest = {
        "taxonomy_version": TAXONOMY_VERSION, "db": db, "seed": SEED,
        "n_requested": n, "n_sampled": len(sample),
        "facet_distribution": dict(Counter(r["teacher_facet"] for r in sample)),
        "novel_count": sum(1 for r in sample if r["is_novel"]),
        "ids": sorted(r["id"] for r in sample),
    }
    with open("gold_eval_v1_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"wrote {csv_path}: {len(sample)} atoms (holdout-only, frozen seed={SEED})")
    print(f"  facet mix: {manifest['facet_distribution']}")
    print(f"  novel-text atoms: {manifest['novel_count']}")
    print("  PM fills the gold_facet column (7 facets / _keep / AMBIGUOUS), then run `score`.")


def score(gold_csv: str, pred_csv: str):
    """gold_csv: adjudicated (id, gold_facet). pred_csv: model (id, pred_facet, confidence)."""
    gold = {}
    with open(gold_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gf = (row.get("gold_facet") or row.get("gold_facet (FILL: %s)" % "|".join(ADJ_FACETS)) or "").strip().upper()
            if gf:
                gold[str(row["id"]).strip()] = gf
    preds = {}
    with open(pred_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[str(row["id"]).strip()] = (
                (row.get("pred_facet") or "").strip().upper(),
                float(row.get("confidence") or 0.0))
    # per-facet precision/coverage among DEFLECTED (non-abstain) preds on gold
    per = defaultdict(lambda: {"deflected": 0, "correct": 0, "gold_total": 0})
    n_gold = 0
    for rid, gf in gold.items():
        if gf in ("AMBIGUOUS",):
            continue
        n_gold += 1
        per[gf]["gold_total"] += 1
        pf, conf = preds.get(rid, ("", 0.0))
        if pf and pf != "ABSTAIN":
            per[pf]["deflected"] += 1
            if pf == gf:
                per[pf]["correct"] += 1
    print(f"GOLD score (taxonomy {TAXONOMY_VERSION}) — {n_gold} non-ambiguous gold atoms")
    print(f"{'facet':12} {'deflected':>9} {'precision':>9} {'coverage':>9}")
    tot_def = tot_cor = tot_gold = 0
    for f in FACETS:
        d = per[f]
        prec = d["correct"] / d["deflected"] if d["deflected"] else 0.0
        cov = d["deflected"] / d["gold_total"] if d["gold_total"] else 0.0
        tot_def += d["deflected"]; tot_cor += d["correct"]; tot_gold += d["gold_total"]
        print(f"{f:12} {d['deflected']:9d} {prec:9.3f} {cov:9.3f}")
    print(f"{'TOTAL':12} {tot_def:9d} {tot_cor/tot_def if tot_def else 0:9.3f} "
          f"{tot_def/tot_gold if tot_gold else 0:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--db", default=os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db"))
    b.add_argument("--n", type=int, default=1800)
    b.add_argument("--min-per-facet", type=int, default=120)
    b.add_argument("--novel-frac", type=float, default=0.20)
    b.add_argument("--flagged", default="", help="optional JSON list of CL-flagged row ids")
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        flagged = set(json.load(open(a.flagged))) if a.flagged and os.path.exists(a.flagged) else None
        build(a.db, a.n, a.min_per_facet, a.novel_frac, flagged)
    else:
        score(a.gold, a.pred)


if __name__ == "__main__":
    main()
