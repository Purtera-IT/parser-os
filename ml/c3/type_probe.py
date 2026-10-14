"""Can line type be learned from the frozen encoder's features at all?

    python -m ml.c3.type_probe --train train/*.json --validation 010087.json 010149.json

Encodes each deal's labeled lines with the frozen line encoder, the features
a frozen-encoder base reads, and reports three things:

1. how alike the type answers' descriptions are, as the base's type head sees them;
2. a plain linear probe with one free weight row per type: its fit on the
   training deals and its exact match and task recall on the validation deals;
3. each line matched to its nearest answer description (where the head starts).

If the probe fits and the base does not, the head is the bottleneck, not
the features. Counts and rates only, never line text. Validation deals only,
never the holdout. Deal Kit and estimating-calc lines are left out.
"""
from __future__ import annotations

import argparse
import collections
import random
import time

import torch
import torch.nn.functional as F

from .data import IGNORE
from .loop import load_deal, training_batches
from .schema import load_schema

SKIP = ("deal_kit", "estimating_calc")


def _rows(path: str, schema, cap: int, rng: random.Random) -> list[tuple[str, int]]:
    deal = load_deal(path)
    kinds = {a.key: (a.doc_kind or "").lower() for a in deal.atoms}
    parts, _ = training_batches(deal, schema, max_atoms=8192)
    seen, out = set(), []
    for b in parts:
        for k, t, y in zip(b.keys, b.texts, b.targets["col:label_type"]):
            if y != IGNORE and k not in seen and kinds.get(k, "") not in SKIP:
                seen.add(k)
                out.append((t, y))
    rng.shuffle(out)
    return out[:cap]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--cap", type=int, default=4000, help="labeled lines per deal")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    from .text import HFEncoder

    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    rng = random.Random(0)
    torch.manual_seed(0)
    schema = load_schema()
    opp = schema.by_key()["col:label_type"]
    names = [a.value for a in opp.answers]
    task = names.index("task")
    t0 = time.time()
    enc = HFEncoder(args.encoder, max_length=8192, dtype=torch.bfloat16).to(dev)
    for q in enc.model.parameters():
        q.requires_grad_(False)

    def embed(texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), 256):
            with torch.no_grad():
                out.append(enc(texts[i:i + 256]).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, enc.dim)

    data = {}
    for path in args.train + args.validation:
        r = _rows(path, schema, args.cap, rng)
        data[path] = (embed([t for t, _ in r]), torch.tensor([y for _, y in r], dtype=torch.long))
        print("encoded", path.rsplit("/", 1)[-1], len(r), f"{time.time() - t0:.0f}s", flush=True)

    answers = F.normalize(embed([a.description for a in opp.answers]), dim=-1)
    sim = answers @ answers.T
    off = sim[~torch.eye(len(answers), dtype=torch.bool)]
    print(f"answer descriptions: mean pairwise cosine {off.mean():.3f} "
          f"min {off.min():.3f} max {off.max():.3f}", flush=True)

    x_tr = torch.cat([data[d][0] for d in args.train])
    y_tr = torch.cat([data[d][1] for d in args.train])
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    probe = torch.nn.Linear(x_tr.shape[1], len(names))
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    xn = (x_tr - mu) / sd
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(xn), y_tr)
        loss.backward()
        opt.step()
    print(f"linear probe: final train loss {loss.item():.3f}", flush=True)
    for d in args.train + args.validation:
        x, y = data[d]
        with torch.no_grad():
            pred = probe((x - mu) / sd).argmax(-1)
        hit = (y == task)
        recall = ((pred == task) & hit).sum().item() / max(1, hit.sum().item())
        near = (F.normalize(x, dim=-1) @ answers.T).argmax(-1)
        print(f"  {'train' if d in args.train else 'VALID'} {d.rsplit('/', 1)[-1]}: "
              f"probe exact {(pred == y).float().mean():.3f} task recall {recall:.2f} | "
              f"nearest description exact {(near == y).float().mean():.3f} (n={len(y)})", flush=True)
    mix = collections.Counter(names[i] for i in y_tr.tolist())
    print("train label mix:", mix.most_common(8))


if __name__ == "__main__":
    main()
