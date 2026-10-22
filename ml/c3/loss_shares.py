"""What a stage's loss is made of, on a saved run, before paying for the next one.

    python -m ml.c3.loss_shares --checkpoint runA.pt.lines --train train/*.json --parts 6

Reads a sample of training parts with the checkpoint (no step is taken) and
prints each loss term as it enters the stage's total, and each question's
share of the heads term. If the heads are a small slice of the total, or line
type a small slice of the heads, the stage spent its steps elsewhere.
Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict

import torch

from . import losses
from .loop import _autocast, gold_weights, live_student, load_split
from .readiness import digest
from .schema import load_schema


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--stage", default="lines")
    p.add_argument("--parts", type=int, default=6)
    p.add_argument("--max-atoms", type=int, default=4096)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    from .gates import load_checkpoint

    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    schema = load_schema()
    model = live_student(schema, args.encoder).to(dev)
    load_checkpoint(args.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    model.eval()
    parts, _ = load_split(args.train, schema, args.max_atoms)
    rng = random.Random(0)
    parts = sorted(rng.sample(parts, min(args.parts, len(parts))), key=len)
    weights, sizes = gold_weights(args.stage)

    per_head: dict[str, list[float]] = {}
    terms: dict[str, list[float]] = {}
    for b in parts:
        kept: dict = {}
        with torch.no_grad(), _autocast(dev):
            total, raw = losses.c3_loss(model, b, weights, judgment_sizes=sizes, keep=kept)
            out, rw = kept["out"], losses.row_weights(b, kept["out"].r.device)
            for opp in schema.select(layer="universal"):
                if opp.key in out.logits and losses.has_answers(b, opp.key):
                    v = losses.answer_loss(out.logits[opp.key].float(), b, opp.key, rw)
                    per_head.setdefault(opp.key, []).append(float(v))
        for k, v in raw.items():
            terms.setdefault(k, []).append(getattr(weights, k) * v)
        terms.setdefault("TOTAL", []).append(float(total))
        print(f"read part {len(b)} lines", flush=True)
    mean = {k: sum(v) / len(v) for k, v in terms.items()}
    total = mean.pop("TOTAL")
    print(f"stage {args.stage} total {total:.3f} over {len(parts)} parts")
    for k, m in sorted(mean.items(), key=lambda kv: -abs(kv[1])):
        if abs(m) > 1e-4:
            print(f"  {k:<20} {m:8.3f}  {m / total:6.1%} of total")
    heads = {k: sum(v) / len(v) for k, v in per_head.items()}
    n = len(heads)
    print(f"heads term averages {n} questions; each is 1/{n} of it unless weighted")
    for k, m in sorted(heads.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k:<32} ce {m:.3f}")


if __name__ == "__main__":
    main()
