#!/usr/bin/env python
"""Compile one deal N times and report how much the work_order stage agrees with itself.

Runs entirely locally: point it at a project directory (a local fixture or a dev
checkout of a deal). The teacher is whatever ``TEACHER_API_BASE`` points at in
your environment; with ``--stub-llm`` no network call is made and the extraction
reply is synthesised from the kept documents, which exercises every deterministic
seam (document ranking, atom order, parsing, minting) without an LLM.

    python scripts/work_order_repro.py tests/fixtures/demo_project --runs 3 --stub-llm

Prints one JSON row per deal, shaped for the scorecard (see
``app.core.work_order.deal_agreement``). Exit status is 0 regardless of agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_complete(prompt: str, **_: object) -> str:
    """A deterministic stand-in: one work line per kept document header."""
    docs = re.findall(r"^\[([^\]]+)\]", prompt, flags=re.M)
    return json.dumps(
        {
            "work_lines": [
                {"work": f"deliver the work described in {d}", "object": d, "count": None, "unit": None}
                for d in docs
            ],
            "site_count": None,
            "after_hours": False,
            "no_onsite_hands": False,
            "customer_supplies_equipment": False,
            "one_line_summary": "stub",
        }
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("projects", nargs="+", type=Path, help="deal project directories")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--stub-llm", action="store_true", help="no network; synthesise replies")
    ap.add_argument("--no-cache", action="store_true", help="bypass the compile cache")
    args = ap.parse_args(argv)

    os.environ["SOWSMITH_WORK_ORDER"] = "1"
    from app.core import llm_client, work_order
    from app.core.compiler import compile_project

    if args.stub_llm:
        llm_client.complete = _stub_complete  # type: ignore[assignment]
        os.environ.setdefault("SOWSMITH_WORK_ORDER_MIN_BODY", "1")

    rows = []
    for project in args.projects:
        runs: list[list[str]] = []
        for _ in range(max(1, args.runs)):
            result = compile_project(project, use_cache=not args.no_cache, allow_errors=True)
            runs.append(work_order.work_line_labels(list(result.atoms)))
        row = work_order.deal_agreement(project.name, runs)
        rows.append(row)
        print(json.dumps(row))
    if len(rows) > 1:
        mean = sum(r["agreement"] for r in rows) / len(rows)
        print(json.dumps({"deals": len(rows), "mean_agreement": round(mean, 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
