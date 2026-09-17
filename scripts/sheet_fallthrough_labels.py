#!/usr/bin/env python
"""PUR-51: collect worksheets that reach the classifier's fallthrough, for hand labelling.

Walks a LOCAL directory of .xlsx/.xlsm/.csv workbooks, classifies every sheet
and writes, into --out:

  sheet_labels.jsonl        one record per fallthrough sheet (structure only:
                            headers, column type profile, row shape) with an
                            empty `label` to fill with one of:
                            scope | pricing | bill_of_materials | site_list |
                            schedule | contact_list | boilerplate | junk
  sheet_labels.csv          the same records for spreadsheet labelling
  fallthrough_report.json   fallthrough rate, rules-only vs with the current head

No network / database / cloud access. Cell values beyond the header row are not
written unless --include-samples N is passed.

Synthetic example (what CI/tests do):
  python scripts/sheet_fallthrough_labels.py --root /tmp/synthetic_wbs --out /tmp/labels

Real run (HUMAN ONLY, on an approved local copy; not run by automation):
  python scripts/sheet_fallthrough_labels.py \\
      --root /path/to/local/workbooks --customer-key parent_dir --out ./sheet_labels_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.sheet_fallthrough_census import run_census, write_outputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="local directory of workbooks")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--customer-key", choices=("placeholder", "parent_dir"), default="placeholder",
                    help="placeholder = UNKNOWN_CUSTOMER (fill by hand); parent_dir = first dir under root")
    ap.add_argument("--include-samples", type=int, default=0, metavar="N",
                    help="also write the first N data rows' raw values (off by default)")
    ap.add_argument("--include-positive", action="store_true",
                    help="also write sheets the rules positively matched")
    a = ap.parse_args(argv)
    res = run_census(a.root, customer_key_mode=a.customer_key,
                     include_samples=a.include_samples, include_positive=a.include_positive)
    paths = write_outputs(res, a.out)
    print(json.dumps({**res.report(), "outputs": {k: str(v) for k, v in paths.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
