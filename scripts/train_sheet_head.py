#!/usr/bin/env python
"""PUR-52: train / evaluate the structural sheet head from hand labels.

Input is the labels file produced by scripts/sheet_fallthrough_labels.py once a
human has filled the `label` column. Features are headers, column types and row
shape only; the sheet name is never used.

  --eval-only                 customer-grouped evaluation, no head written
  --holdout-customer X        (repeatable) evaluate on these unseen customers;
                              default is leave-one-customer-out
  --out PATH                  after evaluating, train on ALL labels and save
                              (serve by setting SHEET_STRUCTURE_HEAD_PATH=PATH or
                              writing app/core/data/sheet_structure_head.json)

Reports: per-class accuracy, fallthrough before (rules) / after (head abstains),
and non-scope sheets the head would route to scope.

Real run (HUMAN ONLY, after labelling; not run by automation):
  python scripts/train_sheet_head.py --labels ./sheet_labels_out/sheet_labels.jsonl \\
      --holdout-customer <customer_key> --report ./sheet_head_eval.json --out ./sheet_structure_head.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.sheet_structure_head import (  # noqa: E402
    DEFAULT_THRESHOLD,
    evaluate_customer_holdout,
    read_labels,
    train_from_records,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--holdout-customer", action="append", default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--report", help="write evaluation JSON here")
    ap.add_argument("--out", help="save a head trained on all labels here")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(argv)

    recs = read_labels(a.labels)
    if len({r["label"] for r in recs}) < 2:
        print(json.dumps({"error": "need labelled sheets of at least two classes; head stays absent (abstains)",
                          "labelled": len(recs)}))
        return 2
    report = evaluate_customer_holdout(recs, holdout_customers=a.holdout_customer, threshold=a.threshold)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if a.out and not a.eval_only:
        head = train_from_records(recs, threshold=a.threshold)
        head.meta = {"labels_file": Path(a.labels).name, "eval": {k: report[k] for k in
                     ("fallthrough_before", "fallthrough_after", "accuracy", "per_class_accuracy")}}
        head.save(a.out)
        print(f"saved head -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
