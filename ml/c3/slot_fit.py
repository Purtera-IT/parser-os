"""Refit ONE free head (default read:sow_section) on a Mac, nothing else.

    python -m ml.c3.slot_fit --checkpoint ~/c3-serve/c3_base/checkpoint.pt \
        --train sow_slot/train/*.json ~/c3-local/train/*.json \
        --validation sow_slot/val/*.json --out /tmp/sow_section.pt

The trunk, the encoder and every other head stay exactly as the checkpoint
has them (the 2026-10-18 morning refit retrained line type and collapsed it on
unseen deals). The epoch is picked by this head's exact on the validation
deals, which must not be training deals; the checkpoint's own head is the bar
to beat, so a refit that is not better on held-out deals saves nothing.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .head_fit import _cache, _logits
from .loop import live_student, load_split
from .readiness import digest
from .schema import load_schema


def _exact(model, key, c, free: bool) -> tuple[float, int]:
    """Macro recall over the sections present (scope_of_work is ~3/4 of the
    rows; plain exact rewards answering it everywhere)."""
    if not len(c["y"]):
        return 0.0, 0
    with torch.no_grad():
        lg = _logits(model, key, c) if free else c["base"]
        pred = lg.argmax(-1)
    recs = [float((pred[c["y"] == k] == k).float().mean()) for k in c["y"].unique()]
    return sum(recs) / len(recs), len(c["y"])


def _report(model, key, c, free: bool, names) -> str:
    with torch.no_grad():
        pred = (_logits(model, key, c) if free else c["base"]).argmax(-1)
    exact = float((pred == c["y"]).float().mean())
    maj = float((c["y"] == c["y"].mode().values).float().mean())
    per = " ".join(f"{names[int(k)]}={int((pred[c['y'] == k] == k).sum())}/{int((c['y'] == k).sum())}" for k in c["y"].unique())
    return f"exact {exact:.3f} (always-majority {maj:.3f}) {per}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key", default="read:sow_section")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    from .gates import load_checkpoint, save_checkpoint

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    schema = load_schema()
    sha = digest([asdict(o) for o in schema.opportunities])
    model = live_student(schema, args.encoder).to(dev)
    step = load_checkpoint(args.checkpoint, model, schema_sha256=sha)
    model.eval()
    key = args.key
    train, _ = load_split(args.train, schema, args.max_atoms)
    val, _ = load_split(args.validation, schema, args.max_atoms)
    tr = _cache(model, train, [key], dev)[key]
    va = _cache(model, val, [key], dev)[key]
    print(f"rows train={len(tr['y'])} validation={len(va['y'])}", flush=True)
    model.to("cpu")
    base_exact, n = _exact(model, key, va, free=False)
    print(f"checkpoint {key} validation exact {base_exact:.3f} (n={n})", flush=True)
    names = [a.value for a in schema.by_key()[key].answers]
    print("checkpoint " + _report(model, key, va, False, names), flush=True)
    counts = torch.bincount(tr["y"], minlength=len(names)).float()
    cw = torch.where(counts > 0, counts.sum() / (counts.clamp(min=1) * (counts > 0).sum()), torch.zeros_like(counts))
    model.add_free_head(key)
    opt = torch.optim.AdamW(model.free_heads[key].parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state = base_exact, None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(tr["y"]))
        lg = _logits(model, key, tr, order)
        loss = (F.cross_entropy(lg, tr["y"][order], weight=cw, reduction="none") * tr["w"][order]).sum() / tr["w"][order].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            ex, _ = _exact(model, key, va, free=True)
            print(f"epoch {epoch} loss {float(loss):.3f} validation exact {ex:.3f}", flush=True)
            if ex > best:
                best = ex
                best_state = {n_: t.detach().clone() for n_, t in model.free_heads[key].state_dict().items()}
    if best_state is None:
        print(f"no epoch beat the checkpoint's {base_exact:.3f}; nothing saved", flush=True)
        return
    model.free_heads[key].load_state_dict(best_state)
    print("refit " + _report(model, key, va, True, names), flush=True)
    save_checkpoint(args.out, model, step=step, schema_sha256=sha)
    print(f"saved {args.out}: {key} validation exact {base_exact:.3f} -> {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
