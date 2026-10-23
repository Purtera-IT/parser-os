"""Admission purity: a base "drop" must rest on a universal reason (arch_review §8 a4 / c5).

``col:admission`` is a universal head: "drop" means the line is not a fact for
any services company (wreckage, boilerplate, noise). A drop that is really
Purtera's choice (a real fact we do not quote) belongs to ``co_action``
(reject / ignore), which ``data._derived`` already keeps off the base. This
check finds the rows the data rule cannot see: the card says drop (the
``rejected`` flag, a noise class, a not-a-fact type) with no company decision,
and its note's only reasoning is the ``[company]`` line, with the universal
part empty. Such a row teaches the base Purtera's policy.

    python -m ml.c3.purity runE/train_presow/*.json runE/train_plan/*.json

Prints counts per file and row keys only (never note text); exits 1 when
any file has a violation and ``--strict`` is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data import _derived, excluded, policy_reject
from .notes import drop_meta, split_note


def admission_impurities(blob: dict[str, Any], company: str | None = None) -> list[str]:
    """Label keys whose derived admission is "drop" on the base while the note's
    reasoning is only in the ``[company]`` line."""
    company = company or str(blob.get("company") or "purtera")
    bad = []
    for lb in blob.get("labels") or []:
        if not isinstance(lb, dict) or excluded(lb) or policy_reject(lb):
            continue
        if _derived(lb).get("admission") != "drop":
            continue
        u, p = split_note(str(lb.get("note") or ""), company)
        if not drop_meta(u) and drop_meta(p):
            bad.append(str(lb.get("label_key") or lb.get("atom_id") or "?"))
    return bad


def check_files(paths: list[str]) -> dict[str, list[str]]:
    out = {}
    for path in paths:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        out[path] = admission_impurities(blob)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any violation")
    a = ap.parse_args(argv)
    found = check_files(a.files)
    total = 0
    for path, keys in found.items():
        total += len(keys)
        if keys:
            print(f"admission purity {Path(path).name}: {len(keys)} drop(s) whose only reason is the "
                  f"company line: {' '.join(keys[:20])}{' ...' if len(keys) > 20 else ''}")
    print(f"admission purity: {total} violation(s) in {len(found)} file(s)")
    return 1 if (a.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
