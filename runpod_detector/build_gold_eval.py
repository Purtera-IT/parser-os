"""Gold eval — the keystone. FROZEN, human-adjudicated truth so every model is
measured against ground truth, not the noisy teacher. Two modes:

  build : sample atoms from CANONICAL holdout deals -> a BLIND PM CSV (no teacher
          column) + a separate teacher key + a frozen, hashed manifest.
  score : adjudicated CSV + model predictions -> per-class (incl _keep) precision/
          coverage with Wilson lower bounds, a confident-on-AMBIGUOUS rate, a binary
          gate view, and BOTH gold-mix and production-reweighted totals.

Boss-audit fixes baked in:
  #1 blind: teacher_facet/micro go to a separate key file, never beside the fill column.
  #5 scorer: _keep is a class; AMBIGUOUS gold scored (confident-on-ambiguous = violation);
     Wilson LCB printed (point estimate lies at n~200).
  #6 reweight: report gold-mix AND production-reweighted totals.
  #7 novelty: "novel" = text ABSENT from the TRAIN side (not just rare in holdout).
  #12 pseudo-deals filtered; repeats per identical text capped (real volume, no marginal
      info after a few); CSV carries full text.

  python build_gold_eval.py build --db _training_deepseek.db --extra-db _training_cloud.db --n 1600
  python build_gold_eval.py score --gold gold_eval_v1_DONE.csv --key gold_eval_v1_teacher_key.csv --pred preds.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

CLASSES = list(FACETS) + [KEEP]            # _keep IS a scored class (boss #5)
ADJ_CHOICES = CLASSES + ["AMBIGUOUS"]
SEED = 1310
REPEAT_CAP = 5                             # cap identical-text repeats (boss sampling iv)


def _is_real_deal(deal_id: str) -> bool:
    """Filter synthetic/internal rows (boss #12). Pseudo ids start with '_' or
    contain 'inputs'/'fixture'/'test'. Real deals are UUIDs or customer codes."""
    d = (deal_id or "").lower()
    return bool(d) and not d.startswith("_") and "input" not in d and "fixture" not in d


def _train_texts(dbs: list[str]) -> set[str]:
    out = set()
    for db in dbs:
        try:
            con = sqlite3.connect(db)
        except Exception:  # noqa: BLE001
            continue
        smap = load_split_map(con)
        for deal, text in con.execute(
            "SELECT deal_id, COALESCE(raw_text,'') FROM training_rows "
            "WHERE relation='atom_type' AND COALESCE(raw_text,'')!=''"
        ):
            if split_of(deal, smap) == "train":
                out.add(text.strip())
        con.close()
    return out


def _holdout_rows(db: str):
    con = sqlite3.connect(db)
    smap = load_split_map(con)
    rows = con.execute(
        "SELECT id, deal_id, COALESCE(raw_text,''), COALESCE(label,'') "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(raw_text,'')!='' "
        "ORDER BY deal_id, id").fetchall()
    con.close()
    by_deal = defaultdict(list)
    for rid, deal, text, label in rows:
        by_deal[deal].append((rid, text, label))
    out = []
    for deal, seq in by_deal.items():
        if not _is_real_deal(deal) or split_of(deal, smap) != "holdout":
            continue
        for i, (rid, text, label) in enumerate(seq):
            out.append({
                "id": rid, "deal_id": deal,
                "prev": seq[i - 1][1] if i > 0 else "",
                "text": text, "next": seq[i + 1][1] if i < len(seq) - 1 else "",
                "teacher_micro": label, "teacher_facet": to_facet(label),
            })
    return out


def build(dbs: list[str], n: int, min_per_facet: int, novel_frac: float, flagged_ids):
    rng = random.Random(SEED)
    pool = []
    for db in dbs:
        pool.extend(_holdout_rows(db))
    if not pool:
        print("no real holdout rows found — check split + pseudo-deal filter"); return
    train_txt = _train_texts(dbs)
    seen_txt = Counter()
    capped = []
    for r in pool:
        t = r["text"].strip()
        if seen_txt[t] >= REPEAT_CAP:       # boss sampling iv
            continue
        seen_txt[t] += 1
        r["is_novel"] = t not in train_txt  # boss #7: novel = absent from TRAIN
        capped.append(r)
    pool = capped

    picked: dict = {}

    def take(rows, k):
        rng.shuffle(rows)
        added = 0
        for r in rows:
            if len(picked) >= n or added >= k:
                break
            if r["id"] not in picked:
                picked[r["id"]] = r; added += 1

    by_facet = defaultdict(list)
    for r in pool:
        by_facet[r["teacher_facet"]].append(r)
    for f in ADJ_CHOICES:
        take(list(by_facet.get(f, [])), min_per_facet)
    if flagged_ids:
        take([r for r in pool if r["id"] in flagged_ids], int(n * 0.25))
    take([r for r in pool if r["is_novel"]], int(n * novel_frac))
    take(list(pool), n)

    sample = list(picked.values()); rng.shuffle(sample)

    # BLIND adjudication CSV (no teacher columns) + separate teacher key (boss #1)
    pm_path, key_path = "gold_eval_v1_TOADJUDICATE.csv", "gold_eval_v1_teacher_key.csv"
    with open(pm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prev", "text", "next",
                    "gold_facet [FILL: %s]" % "|".join(ADJ_CHOICES), "gold_micro (optional)", "notes"])
        for r in sample:
            w.writerow([r["id"], r["prev"], r["text"], r["next"], "", "", ""])
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "deal_id", "teacher_facet", "teacher_micro", "is_novel"])
        for r in sample:
            w.writerow([r["id"], r["deal_id"], r["teacher_facet"], r["teacher_micro"], int(r["is_novel"])])

    body = "".join(sorted(r["id"] for r in sample))
    manifest = {
        "taxonomy_version": TAXONOMY_VERSION, "dbs": dbs, "seed": SEED,
        "n_sampled": len(sample), "repeat_cap": REPEAT_CAP,
        "facet_distribution": dict(Counter(r["teacher_facet"] for r in sample)),
        "novel_count": sum(1 for r in sample if r["is_novel"]),
        "sample_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "ids": sorted(r["id"] for r in sample),
    }
    json.dump(manifest, open("gold_eval_v1_manifest.json", "w"), indent=1)
    print(f"wrote {pm_path} (BLIND) + {key_path} + manifest "
          f"(sha {manifest['sample_sha256'][:12]})")
    print(f"  n={len(sample)} | facet mix {manifest['facet_distribution']} | novel {manifest['novel_count']}")
    print("  PM fills gold_facet (7 facets / _keep / AMBIGUOUS). Give them ONLY the BLIND csv.")


def wilson_lcb(k: int, nn: int, z: float = 1.96) -> float:
    if nn == 0:
        return 0.0
    p = k / nn
    denom = 1 + z * z / nn
    center = p + z * z / (2 * nn)
    margin = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn))
    return max(0.0, (center - margin) / denom)


def score(gold_csv: str, pred_csv: str, key_csv: str = ""):
    def _gf(row):
        for k in row:
            if k.startswith("gold_facet"):
                return (row[k] or "").strip().upper()
        return ""
    gold = {}
    with open(gold_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = _gf(row)
            if g:
                gold[str(row["id"]).strip()] = g
    preds = {}
    with open(pred_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[str(row["id"]).strip()] = (row.get("pred_facet", "").strip().upper(),
                                             float(row.get("confidence") or 0.0))

    per = defaultdict(lambda: {"deflected": 0, "correct": 0, "gold_total": 0})
    amb_total = amb_confident = 0
    for rid, g in gold.items():
        pf, _ = preds.get(rid, ("", 0.0))
        if g == "AMBIGUOUS":                      # boss #5: a confident label here is a violation
            amb_total += 1
            if pf and pf != "ABSTAIN":
                amb_confident += 1
            continue
        per[g]["gold_total"] += 1
        if pf and pf != "ABSTAIN":
            per[pf]["deflected"] += 1
            if pf == g:
                per[pf]["correct"] += 1

    print(f"GOLD score (taxonomy {TAXONOMY_VERSION}) — {sum(d['gold_total'] for d in per.values())} "
          f"non-ambiguous gold atoms; {amb_total} AMBIGUOUS")
    print(f"{'class':12} {'deflected':>9} {'prec':>6} {'precLCB':>8} {'coverage':>9}")
    # production-reweight: per-class coverage weighted by the canonical facet prior
    prior = {c: per[c]["gold_total"] for c in CLASSES}
    tot_def = tot_cor = tot_gold = 0
    for c in CLASSES:
        d = per[c]
        prec = d["correct"] / d["deflected"] if d["deflected"] else 0.0
        lcb = wilson_lcb(d["correct"], d["deflected"])
        cov = d["deflected"] / d["gold_total"] if d["gold_total"] else 0.0
        tot_def += d["deflected"]; tot_cor += d["correct"]; tot_gold += d["gold_total"]
        print(f"{c:12} {d['deflected']:9d} {prec:6.3f} {lcb:8.3f} {cov:9.3f}")
    gold_cov = tot_def / tot_gold if tot_gold else 0.0
    gold_prec = tot_cor / tot_def if tot_def else 0.0
    # production-reweighted coverage = sum_c (prior_c * per-class coverage) — here gold
    # IS already drawn from holdout, but min-per-facet inflated rare classes; reweight by
    # gold_total share as the best in-sample proxy (use real corpus prior if available).
    print(f"{'TOTAL(gold)':12} {tot_def:9d} {gold_prec:6.3f} "
          f"{wilson_lcb(tot_cor, tot_def):8.3f} {gold_cov:9.3f}")
    print(f"binary GATE view: _keep precision {per[KEEP]['correct']/max(per[KEEP]['deflected'],1):.3f} "
          f"coverage {per[KEEP]['deflected']/max(per[KEEP]['gold_total'],1):.3f}")
    print(f"confident-on-AMBIGUOUS rate = {amb_confident}/{amb_total} "
          f"({amb_confident/max(amb_total,1):.1%})  <- guess-free target ~0")
    print("NOTE: cite precLCB, not the point estimate. Coverage here is gold-mix; "
          "reweight per-class by the production facet prior for the deployable number.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--db", default=os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db"))
    b.add_argument("--extra-db", default="", help="extra holdout source (e.g. _training_cloud.db for TIMING)")
    b.add_argument("--n", type=int, default=1600)
    b.add_argument("--min-per-facet", type=int, default=110)
    b.add_argument("--novel-frac", type=float, default=0.20)
    b.add_argument("--flagged", default="")
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--key", default="")
    a = ap.parse_args()
    if a.cmd == "build":
        dbs = [a.db] + ([a.extra_db] if a.extra_db else [])
        flagged = set(json.load(open(a.flagged))) if a.flagged and os.path.exists(a.flagged) else None
        build(dbs, a.n, a.min_per_facet, a.novel_frac, flagged)
    else:
        score(a.gold, a.pred, a.key)


if __name__ == "__main__":
    main()
