"""Can the dev worker run the base: speed, memory, and the bf16 copy's answers.

    python -m ml.c3.lite_check --checkpoint runC.pt.open_links.slim \\
        --deal train/010043.json [--threads 2] [--budget 90]

Reads the deal the way the live layer does (lines only, 8192-atom windows,
on the CPU), first as trained (fp32 layers over the bf16 encoder), then with
the layers in bf16 (``SOWSMITH_C3_DTYPE=bf16``). ``--threads 2`` stands in
for the worker's 2 vCPU. Prints, for each: load and read time, peak memory,
and for bf16 how often its line type matches fp32's. bf16 may go live only
if it matches at least ``--agree`` (0.995) of lines. Peak memory is the
process's, so the bf16 line includes the fp32 read: ``--only-bf16`` loads
straight to bf16 and reads once, for the true bf16 peak. Counts only.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import asdict

import torch

from .gates import load_checkpoint
from .loop import live_student, load_deal
from .predict import read_deal
from .readiness import digest
from .schema import load_schema

TYPE = "col:label_type"


def _peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3   # bytes on macOS, KB on Linux


def _types(doc) -> dict[str, tuple[str, float]]:
    return {k: (str(r["reads"][TYPE]["answer"]), float(r["reads"][TYPE]["p"]))
            for k, r in doc["lines"].items() if TYPE in r.get("reads", {})}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deal", required=True)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (the dev worker has 2)")
    p.add_argument("--budget", type=float, default=90.0, help="the live SOWSMITH_C3_BUDGET_S")
    p.add_argument("--agree", type=float, default=0.995)
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--only-bf16", action="store_true", help="load in bf16 and read once (true peak)")
    p.add_argument("--only-fp32", action="store_true", help="read once in fp32 (a GPU timing)")
    p.add_argument("--device", default="cpu", help="cpu (the dev worker), mps or cuda (the base service)")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    schema = load_schema()
    deal = load_deal(args.deal)
    t0 = time.monotonic()
    model = live_student(schema, args.encoder).to(args.device)
    load_checkpoint(args.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    if args.only_bf16:
        model.to(torch.bfloat16)
    model.eval()
    load_s = time.monotonic() - t0
    print(f"deal atoms={len(deal.atoms)} device={args.device} threads={torch.get_num_threads()} load_s={load_s:.1f} "
          f"peak_mb={_peak_mb():.0f}", flush=True)
    if args.only_bf16:
        t0 = time.monotonic()
        n = len(_types(read_deal(model, deal, schema, max_atoms=args.max_atoms, device=args.device,
                                 lines_only=True)))
        s = time.monotonic() - t0
        print(f"bf16 read_s={s:.1f} peak_mb={_peak_mb():.0f} lines={n} within_budget={s <= args.budget}")
        return
    if args.only_fp32:
        t0 = time.monotonic()
        n = len(_types(read_deal(model, deal, schema, max_atoms=args.max_atoms, device=args.device,
                                 lines_only=True)))
        s = time.monotonic() - t0
        print(f"fp32 read_s={s:.1f} peak_mb={_peak_mb():.0f} lines={n} within_budget={s <= args.budget}")
        return
    t0 = time.monotonic()
    full = _types(read_deal(model, deal, schema, max_atoms=args.max_atoms, device=args.device, lines_only=True))
    fp32_s = time.monotonic() - t0
    print(f"fp32 read_s={fp32_s:.1f} peak_mb={_peak_mb():.0f} lines={len(full)}"
          f" within_budget={fp32_s <= args.budget}", flush=True)
    model.to(torch.bfloat16)
    t0 = time.monotonic()
    lite = _types(read_deal(model, deal, schema, max_atoms=args.max_atoms, device=args.device, lines_only=True))
    bf16_s = time.monotonic() - t0
    same = sum(full[k][0] == lite.get(k, ("", 0))[0] for k in full) / max(1, len(full))
    dp = sum(abs(full[k][1] - lite[k][1]) for k in full if k in lite) / max(1, len(full))
    print(f"bf16 read_s={bf16_s:.1f} peak_mb={_peak_mb():.0f} agree={same:.4f} mean_dp={dp:.4f}"
          f" within_budget={bf16_s <= args.budget} usable={same >= args.agree}", flush=True)


if __name__ == "__main__":
    main()
