"""Held-out check of a main-heads refit against the checkpoint it started from.

    python -m ml.c3.main_heads_eval --base ~/c3-serve/c3_base/checkpoint.pt \
        --refit <run>/refit_label_type.pt --files <run>/rows/holdout/*.json \
        --keys col:label_type col:supplier --out <run>/holdout.json

The held-out deals are never read by the refit (``labels_from_corrections``
splits them off by deal; ``slot_fit`` picks its epoch on ``val/``), so this is
the number that says whether a refit generalises. Reads both checkpoints the
way they serve: a refit's free head is part of its forward pass. Report-only;
it writes the JSON and nothing else.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def scores(c: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """exact, always-majority and per-answer hits on one key's cached rows."""
    pred = c["base"].argmax(-1)
    y = c["y"]
    n = len(y)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "exact": round(float((pred == y).float().mean()), 4),
        "majority": round(float((y == y.mode().values).float().mean()), 4),
        "per_answer": {names[int(k)]: [int((pred[y == k] == k).sum()), int((y == k).sum())] for k in y.unique()},
    }


def evaluate(checkpoint: str, files: list[str], keys: list[str], encoder: str, device: str,
             max_atoms: int = 8192) -> dict[str, Any]:
    from .gates import load_checkpoint
    from .head_fit import _cache
    from .loop import live_student, load_split
    from .readiness import digest
    from .schema import load_schema

    schema = load_schema()
    sha = digest([asdict(o) for o in schema.opportunities])
    model = live_student(schema, encoder).to(device)
    load_checkpoint(checkpoint, model, schema_sha256=sha)
    model.eval()
    parts, _ = load_split(files, schema, max_atoms)
    cache = _cache(model, parts, keys, device)
    by_key = schema.by_key()
    out = {}
    for k in keys:
        names = [a.value for a in by_key[k].answers]
        out[k] = scores(cache[k], names) if k in cache else {"n": 0}
    return out


def compare(base: dict[str, Any], refit: dict[str, Any]) -> dict[str, Any]:
    """Per key: both scores and the verdict a person reads first."""
    out = {}
    for k in sorted(set(base) | set(refit)):
        b, r = base.get(k) or {"n": 0}, refit.get(k) or {"n": 0}
        if not b.get("n"):
            verdict = "no held-out labels"
        elif r.get("exact", 0) > b.get("exact", 0):
            verdict = "refit better on held-out"
        elif r.get("exact", 0) < b.get("exact", 0):
            verdict = "refit WORSE on held-out"
        else:
            verdict = "no change on held-out"
        out[k] = {"base": b, "refit": r, "verdict": verdict}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True)
    p.add_argument("--refit", required=True)
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if not a.files:
        Path(a.out).write_text(json.dumps({"files": 0, "heads": compare({}, {})}, indent=1))
        print("no held-out files; nothing to evaluate")
        return 0
    base = evaluate(a.base, a.files, a.keys, a.encoder, dev)
    refit = base if Path(a.refit).resolve() == Path(a.base).resolve() else evaluate(a.refit, a.files, a.keys, a.encoder, dev)
    res = {"files": len(a.files), "base": a.base, "refit": a.refit, "heads": compare(base, refit)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res["heads"].items():
        print(f"{k:16s} n={v['base'].get('n', 0):<5} base {v['base'].get('exact', '-')} refit {v['refit'].get('exact', '-')}  {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
