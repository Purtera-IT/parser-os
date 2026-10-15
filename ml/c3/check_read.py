"""Read line type on validation deals with any checkpoint, as the run's checks do.

    python -m ml.c3.check_read --deal train/010043.json --checkpoint runC.pt.lines@8 runC.avg.pt

For checkpoints the run never checked (an average, say). Validation deals
only: the holdout is read once, with ``ml.c3.predict --score``. Counts and
rates only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from .gates import load_checkpoint
from .holdout import LABEL_HOLDOUT, deal_number
from .loop import live_student, load_split, type_check
from .readiness import digest
from .schema import load_schema


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--deal", nargs="+", required=True)
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    args = p.parse_args()
    schema = load_schema()
    sha = digest([asdict(o) for o in schema.opportunities])
    parts, _ = load_split(args.deal, schema, args.max_atoms)
    if {deal_number(b.deal_id) for b in parts} & set(LABEL_HOLDOUT):
        raise SystemExit("The holdout is scored once with ml.c3.predict --score, not here")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = live_student(schema, args.encoder)
    for path in args.checkpoint:
        load_checkpoint(path, model, schema_sha256=sha)
        row = type_check(model.to(device), parts, device)
        print(f"read {path} " + " ".join(f"{k}={int(v)}" if k in ("n", "routed_n") else f"{k}={v:.3f}"
                                         for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
