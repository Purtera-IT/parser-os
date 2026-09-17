#!/usr/bin/env python
"""Compile every deal in a local corpus with SOWSMITH_WORK_ORDER off and on, and diff.

PUR-46. Local only: reads deal directories from disk and writes reports to
``--out``. It does not read from or write to any database or blob store; the
only network it can touch is whatever LLM endpoint your environment already
points at (none with ``--stub-llm``).

    python scripts/work_order_corpus_diff.py CORPUS_DIR --out reports/wo_diff \\
        [--on-runs 2] [--exclude DEAL ...] [--only DEAL ...] [--stub-llm]

``CORPUS_DIR`` holds one sub-directory per deal (the same layout
``compile_project`` takes). Writes:

    OUT/deals/<deal>.json   full per-deal diff
    OUT/rows.jsonl          one row per deal, appended as each deal finishes
    OUT/summary.json        corpus totals and flagged deals
    OUT/report.md           human-readable table + work lines per deal

Flags per deal: lost_atoms (stage was not additive), big_atom_delta,
bad_provenance (a minted line points at no known artifact), collapse (many
atoms -> <=1 line, the Barton Malow shape), unstable_work_lines, error.
Exit status is 1 if any deal lost atoms or has bad provenance, else 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--on-runs", type=int, default=2, help="flag-on compiles per deal (>=2 also measures agreement)")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--stub-llm", action="store_true", help="no network; synthesise extraction replies")
    args = ap.parse_args(argv)

    from app.core import work_order_eval as ev
    from app.core.compiler import compile_project

    if args.stub_llm:
        ev.install_stub_llm()

    deals = ev.list_deals(args.corpus, include=args.only, exclude=args.exclude)
    if not deals:
        print(f"no deals under {args.corpus}", file=sys.stderr)
        return 2
    (args.out / "deals").mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.jsonl"
    rows_path.write_text("")

    def compile_fn(deal: Path):
        return compile_project(deal, use_cache=False, allow_errors=True)

    def on_row(row: dict) -> None:
        (args.out / "deals" / f"{row['deal_id']}.json").write_text(json.dumps(row, indent=2))
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{row['deal_id']}] flags={row.get('flags')}", file=sys.stderr)

    rows = ev.run_corpus_diff(deals, compile_fn, on_runs=args.on_runs, on_row=on_row)
    summary = ev.summarize_diff(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "report.md").write_text(ev.render_markdown(rows, summary))
    print(json.dumps(summary))
    bad = summary["flagged"].get("lost_atoms") or summary["flagged"].get("bad_provenance")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
