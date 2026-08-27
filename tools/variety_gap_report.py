"""Rank LABEL and VARIETY coverage gaps between train and holdout splits.

Doctrine: variety-over-volume. Deal-held-out splits expose variety shift —
a class can look "covered" by row count while every train example is the same
variety and the holdout deals carry a different one (the chart class did
exactly this). More volume of the same variety teaches nothing; the cheapest
accuracy comes from grading the varieties the splits DISAGREE about.

Reads one TrainingLog sqlite (path arg), relation ``pdf_image_kind`` by
default, and compares train vs holdout coverage along two axes:

  * label            — the closed gate label set (skip/photo/diagram/...),
  * provenance.variety — the free-form variety tag graders attach
                         (tools/import_image_silver.py writes it; rows without
                         one are counted under ``(untagged)``).

Output: two ranked "grade these next" lists —

  * thin-in-train:   present in holdout, thin/absent in train — the model is
                     EVALUATED on these but barely TRAINED on them (accuracy
                     risk);
  * thin-in-holdout: present in train, thin/absent in holdout — the model is
                     trained on these but the eval cannot SEE them (a blind
                     spot in the gate: eval-gated retrain can neither reward
                     nor catch regressions there).

Pure stdlib + sqlite. Read-only; never writes the db.

Usage:
  python tools/variety_gap_report.py <scratch>/_training_deepseek.db
  python tools/variety_gap_report.py db.sqlite --relation pdf_image_kind --top 15
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

UNTAGGED = "(untagged)"


# ── pure functions (unit-tested) ────────────────────────────────────


def load_rows(db_path: Path, relation: str) -> list[dict[str, Any]]:
    """(label, variety, split) triples for one relation. Read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT label, provenance, split FROM training_rows WHERE relation=?",
            (relation,),
        )
        rows: list[dict[str, Any]] = []
        for label, prov, split in cur.fetchall():
            variety = UNTAGGED
            if prov:
                try:
                    p = json.loads(prov)
                    if isinstance(p, dict):
                        variety = str(p.get("variety") or "").strip() or UNTAGGED
                except Exception:
                    pass
            rows.append({
                "label": str(label or ""),
                "variety": variety,
                "split": "holdout" if str(split or "") == "holdout" else "train",
            })
        return rows
    finally:
        conn.close()


def coverage(rows: list[dict[str, Any]], axis: str) -> dict[str, dict[str, int]]:
    """{value: {"train": n, "holdout": n}} for one axis (label|variety)."""
    out: dict[str, Counter] = {}
    for r in rows:
        v = r[axis]
        if not v:
            continue
        out.setdefault(v, Counter())[r["split"]] += 1
    return {
        v: {"train": c.get("train", 0), "holdout": c.get("holdout", 0)}
        for v, c in out.items()
    }


def rank_gaps(cov: dict[str, dict[str, int]], axis: str) -> list[dict[str, Any]]:
    """Ranked gap rows for one coverage table (pure, deterministic).

    gap_score = rows_on_the_covered_side / (1 + rows_on_the_thin_side): a
    value with 20 holdout rows and 0 train rows scores 20.0 toward
    thin-in-train; balanced coverage scores near 1 and is not a gap. Only
    scores > 1 are reported (i.e. the thin side has strictly fewer rows).
    """
    gaps: list[dict[str, Any]] = []
    for value, c in cov.items():
        tr, ho = c["train"], c["holdout"]
        if tr == ho:
            continue
        thin_side = "train" if tr < ho else "holdout"
        score = round(max(tr, ho) / (1 + min(tr, ho)), 3)
        if score <= 1:
            continue
        gaps.append({
            "axis": axis,
            "value": value,
            "train": tr,
            "holdout": ho,
            "thin_side": thin_side,
            "gap_score": score,
        })
    gaps.sort(key=lambda g: (-g["gap_score"], g["value"]))
    return gaps


# ── report ──────────────────────────────────────────────────────────


def _fmt(gaps: list[dict[str, Any]], top: int) -> str:
    if not gaps:
        return "  (none — coverage is balanced)\n"
    lines = [f"  {'axis':<8} {'value':<28} {'train':>6} {'holdout':>8} {'gap':>7}"]
    for g in gaps[:top] if top else gaps:
        lines.append(
            f"  {g['axis']:<8} {g['value'][:28]:<28} "
            f"{g['train']:>6} {g['holdout']:>8} {g['gap_score']:>7}"
        )
    return "\n".join(lines) + "\n"


def build_report(rows: list[dict[str, Any]], *, relation: str, top: int) -> str:
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_holdout = len(rows) - n_train
    all_gaps = rank_gaps(coverage(rows, "label"), "label") + \
        rank_gaps(coverage(rows, "variety"), "variety")
    all_gaps.sort(key=lambda g: (-g["gap_score"], g["axis"], g["value"]))
    thin_train = [g for g in all_gaps if g["thin_side"] == "train"]
    thin_holdout = [g for g in all_gaps if g["thin_side"] == "holdout"]
    out = [
        f"VARIETY GAP REPORT — relation={relation} "
        f"({len(rows)} rows: {n_train} train / {n_holdout} holdout)",
        "",
        "GRADE THESE NEXT — in holdout, thin in train (eval'd but untrained):",
        _fmt(thin_train, top),
        "EVAL BLIND SPOTS — in train, thin in holdout (trained but uneval'd):",
        _fmt(thin_holdout, top),
        "gap = covered_side / (1 + thin_side); higher = worse imbalance.",
        "Fix thin-in-train by grading that variety (import_image_silver);",
        "fix thin-in-holdout by pulling rows from MORE DEALS, not more rows",
        "from the same deals (the split is deal-hashed).",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db", type=Path, help="TrainingLog sqlite path (read-only)")
    ap.add_argument("--relation", default="pdf_image_kind")
    ap.add_argument("--top", type=int, default=20,
                    help="cap each list at N rows (0 = all)")
    args = ap.parse_args(argv)

    if not args.db.is_file():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 1
    rows = load_rows(args.db, args.relation)
    if not rows:
        print(f"no relation={args.relation!r} rows in {args.db}", file=sys.stderr)
        return 1
    print(build_report(rows, relation=args.relation, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
