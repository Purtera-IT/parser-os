#!/usr/bin/env python
"""Compile deals N times with work_order on and report how much the stage agrees with itself.

Runs entirely locally: point it at deal project directories, or at a corpus
directory with ``--corpus`` (one sub-directory per deal). The teacher is
whatever ``TEACHER_API_BASE`` points at in your environment; with
``--stub-llm`` no network call is made and the extraction reply is synthesised
from the kept documents, which exercises every deterministic seam (document
ranking, atom order, parsing, minting) without an LLM.

    python scripts/work_order_repro.py tests/fixtures/demo_project --runs 3 --stub-llm
    python scripts/work_order_repro.py --corpus CORPUS_DIR --runs 5 --out repro.jsonl  # PUR-31

Prints one JSON row per deal (``agreement`` exact, ``soft_agreement`` by
meaning, ``line_counts`` per run), then a corpus summary (mean/median/min,
deals below ``--threshold``). Exit status is 0 regardless of agreement.
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
    ap.add_argument("projects", nargs="*", type=Path, help="deal project directories")
    ap.add_argument("--corpus", type=Path, help="directory of deal directories")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", type=Path, help="also write rows + summary as JSONL here")
    ap.add_argument("--stub-llm", action="store_true", help="no network; synthesise replies")
    ap.add_argument("--no-cache", action="store_true", help="bypass the compile cache")
    args = ap.parse_args(argv)

    from app.core import work_order_eval as ev
    from app.core.compiler import compile_project

    if args.stub_llm:
        ev.install_stub_llm()

    deals = list(args.projects)
    if args.corpus:
        deals += ev.list_deals(args.corpus, include=args.only, exclude=args.exclude)
    if not deals:
        ap.error("give deal directories or --corpus")

    def compile_fn(deal: Path):
        return compile_project(deal, use_cache=not args.no_cache, allow_errors=True)

    rows = ev.run_repro(deals, compile_fn, runs=args.runs)
    summary = ev.summarize_repro(rows, threshold=args.threshold)
    for row in rows:
        print(json.dumps(row))
    print(json.dumps({"summary": summary}))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.write(json.dumps({"summary": summary}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
